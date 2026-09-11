"""Phase-4 multi-plant attribution grid orchestrator (snapshot mode).

Runs the Exp-A-style sequential ablation for a given fault plant on the
official DeepSeek / deepseek-flash engine:

  per attempt-seed s:  freeze once   (--inject <plant> --snapshot_dir ...)
                       then 3 arms   (--resume_from <snapshot> --probe_order
                                       causal|reverse|random, random also
                                       --probe_seed s)

Freeze-phase contract failures (validator rejects DE output -> no failure
blackboard -> no snapshot_state.json) are recorded as forward variance and
back-filled with the next seed, mirroring the n=20 me_force_zero_sea
campaign (s7/s12/s14 precedent). The orchestrator is restartable: cells
whose run_manifest.json already exists are skipped, and per-plant state is
persisted to results/p4_grid/<short>/state.json after every cell.

Cost guard: accumulated prompt/completion tokens are priced at the idle
DeepSeek list rate (in CNY 1/M, out CNY 4/M -- an upper bound, cache hits
are cheaper) and no new cell launches once the estimate exceeds
--max_spend_cny.

Usage:
  uv run --no-sync python scripts/run_p4_grid.py \
      --plant de_swap_demand_supply_source --short deswap \
      --seed_start 1 --target_usable 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
           / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")
SNAPSHOTS = REPO / "results" / "snapshots"
P4_ROOT = REPO / "results" / "p4_grid"

BASE_ARGS = [
    "--dataset", "prob_tslp_ecr_demand",
    "--prob_name", "smoke_H4_Omega5",
    "--provider", "DeepSeek",
    "--model", "deepseek-flash",
    "--knowledge", "progressive",
    "--max_retries", "3",
    "--omega_source", "evidence_rank",
    "--rank_method", "hybrid",
    "--diagnosis_mode", "stackelberg",
]
ORDERS = ("causal", "reverse", "random")
IDLE_IN_CNY_PER_M = 1.0
IDLE_OUT_CNY_PER_M = 4.0


def cell_dir(suffix: str) -> Path:
    return RESULTS / suffix


def read_tokens(suffix: str) -> dict | None:
    m = cell_dir(suffix) / "run_manifest.json"
    if not m.exists():
        return None
    try:
        manifest = json.loads(m.read_text())
    except json.JSONDecodeError:
        return None
    cost = manifest.get("cost") or {}
    cfg = manifest.get("config") or {}
    return {
        "prompt_tokens": cost.get("prompt_tokens") or 0,
        "completion_tokens": cost.get("completion_tokens") or 0,
        "total_tokens": cost.get("total_tokens") or 0,
        "snapshot_forward_tokens": cfg.get("snapshot_forward_tokens"),
    }


def est_cny(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * IDLE_IN_CNY_PER_M
            + completion_tokens * IDLE_OUT_CNY_PER_M) / 1e6


class PlantCampaign:
    """Restartable per-plant bookkeeping; state mutations hold self.lock."""

    def __init__(self, plant: str, short: str):
        self.plant = plant
        self.short = short
        self.lock = threading.Lock()
        self.state_path = P4_ROOT / short / "state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        (P4_ROOT / short / "logs").mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"plant": plant, "short": short,
                          "usable_seeds": [], "failed_freezes": [],
                          "cells": {}, "est_spend_cny": 0.0}

    def record_cell(self, seed: int, key: str, value: str,
                    tokens: dict | None, suffix: str) -> None:
        with self.lock:
            cell = self.state["cells"].setdefault(f"s{seed}", {})
            cell[key] = value
            cell[f"{key}_suffix"] = suffix
            if tokens:
                self.state["est_spend_cny"] = round(
                    self.state["est_spend_cny"]
                    + est_cny(tokens["prompt_tokens"],
                              tokens["completion_tokens"]), 4)
            self._save()

    def mark_usable(self, seed: int) -> None:
        with self.lock:
            if f"s{seed}" not in self.state["usable_seeds"]:
                self.state["usable_seeds"].append(f"s{seed}")
            self._save()

    def mark_freeze_fail(self, seed: int) -> None:
        with self.lock:
            if f"s{seed}" not in self.state["failed_freezes"]:
                self.state["failed_freezes"].append(f"s{seed}")
            self._save()

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    # -- read helpers (also under lock for consistency) -------------------
    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))


def run_cell(suffix: str, extra_args: list[str], log_path: Path) -> dict:
    """Launch one pipeline run; return {'ok', 'tokens', 'rc', 'skipped'}."""
    if (cell_dir(suffix) / "run_manifest.json").exists():
        return {"ok": True, "tokens": read_tokens(suffix), "skipped": True,
                "rc": 0}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MAKO_RESULT_SUFFIX": suffix}
    cmd = ["uv", "run", "--no-sync", "python", "-u", "-m",
           "fsm_stackelberg.main", *BASE_ARGS, *extra_args]
    t0 = time.time()
    with log_path.open("w") as fh:
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=1800)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return {
        "ok": rc == 0 and read_tokens(suffix) is not None,
        "tokens": read_tokens(suffix),
        "rc": rc,
        "duration_s": round(time.time() - t0, 1),
        "skipped": False,
    }


def process_seed(camp: PlantCampaign, seed: int, max_spend: float) -> dict:
    """Freeze one seed then (if usable) run the three arms serially."""
    snap = SNAPSHOTS / f"{camp.plant}_s{seed}"
    summary = {"seed": seed, "freeze": None, "arms": {}}

    if (snap / "snapshot_state.json").exists():
        summary["freeze"] = "ok(skip)"
    else:
        suffix = f"p4_{camp.short}_freeze_s{seed}"
        res = run_cell(
            suffix,
            ["--probe_order", "causal", "--inject", camp.plant,
             "--snapshot_dir", str(snap)],
            P4_ROOT / camp.short / "logs" / f"freeze_s{seed}.log")
        camp.record_cell(seed, "freeze", "ran" if res["ok"] else f"fail_rc{res['rc']}",
                         res.get("tokens"), suffix)
        summary["freeze"] = "ok" if (snap / "snapshot_state.json").exists() else "fail"

    if (snap / "snapshot_state.json").exists():
        camp.mark_usable(seed)
        for order in ORDERS:
            if camp.snapshot()["est_spend_cny"] > max_spend:
                summary["arms"][order] = "budget_stop"
                break
            suffix = f"p4_{camp.short}_ea_{order}_s{seed}"
            extra = ["--probe_order", order, "--resume_from", str(snap)]
            if order == "random":
                extra += ["--probe_seed", str(seed)]
            res = run_cell(suffix, extra,
                           P4_ROOT / camp.short / "logs"
                           / f"ea_{order}_s{seed}.log")
            camp.record_cell(seed, f"ea_{order}",
                             "ok" if res["ok"] else f"fail_rc{res['rc']}",
                             res.get("tokens"), suffix)
            summary["arms"][order] = "ok" if res["ok"] else "fail"
    else:
        camp.mark_freeze_fail(seed)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True,
                    choices=["de_swap_demand_supply_source",
                             "pd_comment_out_balance"])
    ap.add_argument("--short", required=True,
                    help="short plant tag used in result suffixes")
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--seed_max", type=int, default=40,
                    help="hard cap on attempt seeds (backfill guard)")
    ap.add_argument("--target_usable", type=int, default=10)
    ap.add_argument("--max_spend_cny", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent seed pipelines (arms serial inside each)")
    args = ap.parse_args()

    camp = PlantCampaign(args.plant, args.short)
    st = camp.snapshot()
    print(f"[p4] plant={args.plant} short={args.short} "
          f"usable={len(st['usable_seeds'])} "
          f"failed_freezes={st['failed_freezes']} "
          f"est_spend=CNY{st['est_spend_cny']}", flush=True)

    # 1) finish usable seeds whose arms are incomplete (resume case)
    incomplete = [int(s[1:]) for s in st["usable_seeds"]
                  if any(st["cells"].get(s, {}).get(f"ea_{o}") != "ok"
                         for o in ORDERS)]
    # 2) fresh attempt seeds: enough to reach target + backfill buffer
    taken = set(st["cells"].keys())
    need = max(args.target_usable - len(st["usable_seeds"]), 0)
    fresh, seed = [], args.seed_start
    while len(fresh) < need + 3 and seed <= args.seed_max:
        if f"s{seed}" not in taken:
            fresh.append(seed)
        seed += 1

    def report(res: dict) -> None:
        st_now = camp.snapshot()
        print(f"[p4] seed s{res['seed']}: freeze={res['freeze']} "
              f"arms={res['arms']} usable={len(st_now['usable_seeds'])} "
              f"est=CNY{st_now['est_spend_cny']}", flush=True)

    # Dynamic scheduling: submit a fresh seed only while
    # usable + fresh-in-flight < target, so the target is met without
    # overshoot (freeze failures free a slot and the next seed is submitted).
    pool = ThreadPoolExecutor(max_workers=args.workers)
    inflight: dict[Future, str] = {}
    fresh_iter = iter(fresh)
    stopped = False
    try:
        for s in incomplete:  # resume arms first, unconditional
            inflight[pool.submit(process_seed, camp, s, args.max_spend_cny)] = \
                f"resume_s{s}"
        while not stopped:
            st = camp.snapshot()
            if st["est_spend_cny"] > args.max_spend_cny:
                print(f"[p4] BUDGET STOP est=CNY{st['est_spend_cny']}",
                      flush=True)
                break
            if len(st["usable_seeds"]) >= args.target_usable and not inflight:
                break
            fresh_active = sum(1 for f, kind in inflight.items()
                               if not f.done() and kind.startswith("fresh"))
            if (len(st["usable_seeds"]) + fresh_active < args.target_usable
                    and len(inflight) < args.workers):
                nxt = next(fresh_iter, None)
                if nxt is None:
                    if not inflight:
                        break  # ran out of candidate seeds
                else:
                    inflight[pool.submit(process_seed, camp, nxt,
                                         args.max_spend_cny)] = f"fresh_s{nxt}"
                    continue
            if inflight:
                done, _ = wait(set(inflight), timeout=120,
                               return_when=FIRST_COMPLETED)
                for f in done:
                    kind = inflight.pop(f)
                    res = f.result()
                    report(res)
                    if res["arms"].get("causal") == "budget_stop":
                        stopped = True
        # drain any still-running pipelines
        for f in list(inflight):
            report(f.result())
    finally:
        pool.shutdown(wait=True)

    st = camp.snapshot()
    print(f"[p4] DONE plant={args.short} usable={st['usable_seeds']} "
          f"failed_freezes={st['failed_freezes']} "
          f"est_spend=CNY{st['est_spend_cny']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
