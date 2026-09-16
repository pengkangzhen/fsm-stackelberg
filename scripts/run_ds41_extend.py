"""Extend the ds41 smoke campaign (me_force_zero_sea) from n=20 to n=30.

Freezes new blackboards (s24 upward, backfilled on forward-variance
freeze failures exactly like the s7/s12/s14 precedent) and runs the five
internal arms per usable seed, resuming the identical frozen blackboard:

  freeze: ds41_freeze_sN        --inject me_force_zero_sea --snapshot_dir ...
  arms:   ds41_ea_causal_sN     stackelberg + evidence_rank/hybrid + causal
          ds41_ea_reverse_sN    stackelberg + evidence_rank/hybrid + reverse
          ds41_ea_random_sN     stackelberg + evidence_rank/hybrid + random
                                (+ --probe_seed N)
          ds41_eb_adversarial_sN  adversarial
          ds41_eb_sequential_sN   sequential

External arms (debate/reflexion) are owned by scripts/run_ebext_grid.py;
invoke it with the new seed list once this script finishes.

Restartable: cells whose run_manifest.json exists are skipped. Existing
n=20 seeds are swept first so any incomplete cell is healed before new
seeds are frozen. Spend guard priced at the idle DeepSeek list rate
(in CNY 1/M in, 4/M out -- an upper bound).

Usage:
  uv run --no-sync python scripts/run_ds41_extend.py \
      --target_usable 30 --seed_start 24 [--max_spend_cny 25] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SNAPSHOTS = REPO / "results" / "snapshots"
STATE = REPO / "results" / "ds41_extend"
RESULTS = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
           / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")
PLANT = "me_force_zero_sea"
IN_CNY_PER_M = 1.0
OUT_CNY_PER_M = 4.0

BASE_ARGS = [
    "--dataset", "prob_tslp_ecr_demand",
    "--prob_name", "smoke_H4_Omega5",
    "--provider", "DeepSeek",
    "--model", "deepseek-flash",
    "--knowledge", "progressive",
    "--max_retries", "3",
]


def usable_snapshot_seeds() -> list[int]:
    return sorted(
        int(p.name.rsplit("_s", 1)[1])
        for p in SNAPSHOTS.glob(f"{PLANT}_s*")
        if (p / "snapshot_state.json").exists())


def read_tokens(suffix: str) -> dict | None:
    m = RESULTS / suffix / "run_manifest.json"
    if not m.exists():
        return None
    try:
        man = json.loads(m.read_text())
    except json.JSONDecodeError:
        return None
    cost = man.get("cost") or {}
    return {"prompt_tokens": cost.get("prompt_tokens") or 0,
            "completion_tokens": cost.get("completion_tokens") or 0}


def est_cny(tok: dict | None) -> float:
    if not tok:
        return 0.0
    return (tok["prompt_tokens"] * IN_CNY_PER_M
            + tok["completion_tokens"] * OUT_CNY_PER_M) / 1e6


def run_cell(suffix: str, extra_args: list[str], log_dir: Path) -> dict:
    """Launch one pipeline run; return {'ok', 'tokens', 'rc', 'skipped'}."""
    if (RESULTS / suffix / "run_manifest.json").exists():
        return {"ok": True, "tokens": read_tokens(suffix), "skipped": True,
                "rc": 0}
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MAKO_RESULT_SUFFIX": suffix}
    cmd = ["uv", "run", "--no-sync", "python", "-u", "-m",
           "fsm_stackelberg.main", *BASE_ARGS, *extra_args]
    log_path = log_dir / f"{suffix}.log"
    t0 = time.time()
    with log_path.open("w") as fh:
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=1800)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return {"ok": rc == 0 and read_tokens(suffix) is not None,
            "tokens": read_tokens(suffix), "rc": rc,
            "duration_s": round(time.time() - t0, 1), "skipped": False}


def arm_specs(seed: int, snap: Path) -> list[tuple[str, list[str]]]:
    """(suffix, extra CLI args) for the five internal arms of one seed."""
    resume = ["--resume_from", str(snap)]
    return [
        (f"ds41_ea_causal_s{seed}",
         ["--diagnosis_mode", "stackelberg", "--omega_source",
          "evidence_rank", "--rank_method", "hybrid",
          "--probe_order", "causal", *resume]),
        (f"ds41_ea_reverse_s{seed}",
         ["--diagnosis_mode", "stackelberg", "--omega_source",
          "evidence_rank", "--rank_method", "hybrid",
          "--probe_order", "reverse", *resume]),
        (f"ds41_ea_random_s{seed}",
         ["--diagnosis_mode", "stackelberg", "--omega_source",
          "evidence_rank", "--rank_method", "hybrid",
          "--probe_order", "random", "--probe_seed", str(seed), *resume]),
        (f"ds41_eb_adversarial_s{seed}",
         ["--diagnosis_mode", "adversarial", *resume]),
        (f"ds41_eb_sequential_s{seed}",
         ["--diagnosis_mode", "sequential", *resume]),
    ]


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        STATE.mkdir(parents=True, exist_ok=True)
        self.path = STATE / "state.json"
        if self.path.exists():
            self.d = json.loads(self.path.read_text())
        else:
            self.d = {"cells": {}, "failed_freezes": [], "est_spend_cny": 0.0}

    def record(self, suffix: str, ok: bool, rc: int,
               tok: dict | None) -> None:
        with self.lock:
            self.d["cells"][suffix] = "ok" if ok else f"fail_rc{rc}"
            self.d["est_spend_cny"] = round(
                self.d["est_spend_cny"] + est_cny(tok), 4)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d, indent=1))
            tmp.replace(self.path)

    def mark_freeze_fail(self, seed: int) -> None:
        with self.lock:
            if f"s{seed}" not in self.d["failed_freezes"]:
                self.d["failed_freezes"].append(f"s{seed}")
            self._save()

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.d))

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        tmp.replace(self.path)


def process_seed(state: State, seed: int, max_spend: float,
                 log_dir: Path) -> dict:
    """Freeze one new seed (if needed) then run its five arms serially."""
    snap = SNAPSHOTS / f"{PLANT}_s{seed}"
    summary = {"seed": seed, "freeze": None, "arms": {}}
    if not (snap / "snapshot_state.json").exists():
        res = run_cell(
            f"ds41_freeze_s{seed}",
            ["--diagnosis_mode", "stackelberg", "--probe_order", "causal",
             "--inject", PLANT, "--snapshot_dir", str(snap)],
            log_dir)
        state.record(f"ds41_freeze_s{seed}", res["ok"], res["rc"],
                     res.get("tokens"))
        ok = (snap / "snapshot_state.json").exists()
        summary["freeze"] = "ok" if ok else "fail"
        if not ok:
            state.mark_freeze_fail(seed)
            return summary
    else:
        summary["freeze"] = "ok(skip)"

    for suffix, extra in arm_specs(seed, snap):
        if state.snapshot()["est_spend_cny"] > max_spend:
            summary["arms"][suffix] = "budget_stop"
            break
        res = run_cell(suffix, extra, log_dir)
        state.record(suffix, res["ok"], res["rc"], res.get("tokens"))
        summary["arms"][suffix] = "ok" if res["ok"] else f"fail_rc{res['rc']}"
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_usable", type=int, default=30)
    ap.add_argument("--seed_start", type=int, default=24)
    ap.add_argument("--seed_max", type=int, default=50)
    ap.add_argument("--max_spend_cny", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    state = State()
    log_dir = STATE / "logs"
    taken = set(state.snapshot()["cells"]) | {
        f"ds41_freeze_s{int(s[1:])}" for s in state.d["failed_freezes"]}

    # Sweep 1: heal any incomplete arm on already-usable seeds (resume case).
    todo_existing = [s for s in usable_snapshot_seeds()
                     if any(not (RESULTS / suf / "run_manifest.json").exists()
                            for suf, _ in arm_specs(
                                s, SNAPSHOTS / f"{PLANT}_s{s}"))]

    # Sweep 2: fresh attempt seeds until target usable, with backfill buffer.
    have = len(usable_snapshot_seeds())
    need = max(args.target_usable - have, 0)
    fresh: list[int] = []
    seed = args.seed_start
    while len(fresh) < need + 3 and seed <= args.seed_max:
        if f"ds41_freeze_s{seed}" not in taken:
            fresh.append(seed)
        seed += 1

    print(f"[ext] usable={have} target={args.target_usable} "
          f"heal_existing={todo_existing} fresh={fresh} "
          f"est=CNY{state.snapshot()['est_spend_cny']}", flush=True)

    def report(res: dict) -> None:
        st = state.snapshot()
        print(f"[ext] s{res['seed']}: freeze={res['freeze']} "
              f"arms={ {k.rsplit('_s', 1)[0].split('_', 2)[-1]: v
                        for k, v in res['arms'].items()} } "
              f"usable={len(usable_snapshot_seeds())} "
              f"est=CNY{st['est_spend_cny']}", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    inflight: dict = {}
    fresh_iter = iter(fresh)
    stopped = False
    try:
        for s in todo_existing:
            inflight[pool.submit(process_seed, state, s,
                                 args.max_spend_cny, log_dir)] = f"heal_s{s}"
        while not stopped:
            st = state.snapshot()
            if st["est_spend_cny"] > args.max_spend_cny:
                print(f"[ext] BUDGET STOP est=CNY{st['est_spend_cny']}",
                      flush=True)
                break
            usable = len(usable_snapshot_seeds())
            if usable >= args.target_usable and not inflight:
                break
            fresh_active = sum(1 for f, kind in inflight.items()
                               if not f.done() and kind.startswith("fresh"))
            if (usable + fresh_active < args.target_usable
                    and len(inflight) < args.workers):
                nxt = next(fresh_iter, None)
                if nxt is None:
                    if not inflight:
                        break  # candidate seeds exhausted
                else:
                    inflight[pool.submit(process_seed, state, nxt,
                                         args.max_spend_cny, log_dir)] = \
                        f"fresh_s{nxt}"
                    continue
            if inflight:
                done, _ = wait(set(inflight), timeout=120,
                               return_when=FIRST_COMPLETED)
                for f in done:
                    inflight.pop(f)
                    report(f.result())
                    if any(v == "budget_stop"
                           for v in f.result()["arms"].values()):
                        stopped = True
        for f in list(inflight):
            report(f.result())
    finally:
        pool.shutdown(wait=True)

    st = state.snapshot()
    print(f"[ext] DONE usable_seeds={usable_snapshot_seeds()} "
          f"failed_freezes={st['failed_freezes']} "
          f"est=CNY{st['est_spend_cny']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
