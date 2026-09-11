"""Exp-B external grid runner (Debate / Reflexion) on existing me_force snapshots.

Per usable seed s of the n=20 campaign's me_force_zero_sea snapshots, runs
one cell per external mode, resuming the identical frozen blackboard the
Stackelberg/adversarial arms used (same-board fairness):

  MAKO_RESULT_SUFFIX=ds41_ebext_<mode>_s<s> uv run python -m fsm_stackelberg.main \
    ... --diagnosis_mode <mode> --resume_from results/snapshots/me_force_zero_sea_s<s>

Restartable (skips cells whose manifest exists); budget guard in list-price
upper-bound units (actual spend is typically 0.1-0.3x of it); state in
results/exp_b_external/state.json.

Usage:
  uv run --no-sync python scripts/run_ebext_grid.py [--modes debate,reflexion] \
      [--seeds 1,2,...] [--workers 3] [--max_spend_cny 18.0]
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
RESULTS = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
           / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")
SNAPSHOTS = REPO / "results" / "snapshots"
STATE = REPO / "results" / "exp_b_external"
PLANT = "me_force_zero_sea"

BASE_ARGS = [
    "--dataset", "prob_tslp_ecr_demand",
    "--prob_name", "smoke_H4_Omega5",
    "--provider", "DeepSeek",
    "--model", "deepseek-flash",
    "--knowledge", "progressive",
    "--max_retries", "3",
]


def default_seeds() -> list[int]:
    seeds = []
    for d in SNAPSHOTS.glob(f"{PLANT}_s*"):
        if (d / "snapshot_state.json").exists():
            seeds.append(int(d.name.rsplit("_s", 1)[1]))
    return sorted(seeds)


def read_tokens(suffix: str) -> dict | None:
    m = RESULTS / suffix / "run_manifest.json"
    if not m.exists():
        return None
    try:
        man = json.loads(m.read_text())
    except json.JSONDecodeError:
        return None
    cost = man.get("cost") or {}
    cfg = man.get("config") or {}
    return {"prompt_tokens": cost.get("prompt_tokens") or 0,
            "completion_tokens": cost.get("completion_tokens") or 0,
            "total_tokens": cost.get("total_tokens") or 0,
            "snapshot_forward_tokens": cfg.get("snapshot_forward_tokens") or 0,
            "probe_rounds": cost.get("probe_rounds"),
            "ok": man.get("outcome") is not None}


def run_cell(suffix: str, extra_args: list[str], log_path: Path) -> dict:
    if (RESULTS / suffix / "run_manifest.json").exists():
        return {"ok": True, "tokens": read_tokens(suffix), "skipped": True}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MAKO_RESULT_SUFFIX": suffix}
    cmd = ["uv", "run", "--no-sync", "python", "-u", "-m",
           "fsm_stackelberg.main", *BASE_ARGS, *extra_args]
    with log_path.open("w") as fh:
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=1800)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    tok = read_tokens(suffix)
    return {"ok": rc == 0 and tok is not None, "tokens": tok, "rc": rc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="debate,reflexion")
    ap.add_argument("--seeds", default="",
                    help="comma list; default = all me_force snapshot seeds")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_spend_cny", type=float, default=18.0,
                    help="list-price upper bound; actual is 0.1-0.3x")
    args = ap.parse_args()

    modes = args.modes.split(",")
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else default_seeds())
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "logs").mkdir(exist_ok=True)
    state_f = STATE / "state.json"
    lock = threading.Lock()

    state = (json.loads(state_f.read_text()) if state_f.exists()
             else {"cells": {}, "est_spend_cny": 0.0})

    def est(tok: dict | None) -> float:
        return ((tok["prompt_tokens"] * 1.0
                 + tok["completion_tokens"] * 4.0) / 1e6) if tok else 0.0

    def record(key: str, res: dict) -> None:
        with lock:
            state["cells"][key] = ("ok" if res["ok"] else
                                   f"fail_rc{res.get('rc')}")
            state["est_spend_cny"] = round(
                state["est_spend_cny"] + est(res.get("tokens")), 4)
            tmp = state_f.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(state_f)

    def job(mode: str, seed: int) -> None:
        suffix = f"ds41_ebext_{mode}_s{seed}"
        res = run_cell(
            suffix,
            ["--diagnosis_mode", mode,
             "--resume_from", str(SNAPSHOTS / f"{PLANT}_s{seed}")],
            STATE / "logs" / f"{mode}_s{seed}.log")
        record(suffix, res)
        print(f"[ebext] {mode} s{seed}: {'ok' if res['ok'] else 'FAIL'} "
              f"est=CNY{state['est_spend_cny']}", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = {(mode, s) for mode in modes for s in seeds
               if state["cells"].get(f"ds41_ebext_{mode}_s{s}") != "ok"}
    futures = {}
    stopped = False
    try:
        while pending or futures:
            with lock:
                spend = state["est_spend_cny"]
            if spend > args.max_spend_cny:
                print(f"[ebext] BUDGET STOP est=CNY{spend}", flush=True)
                break
            while pending and len(futures) < args.workers:
                mode, seed = sorted(pending)[0]
                pending.discard((mode, seed))
                futures[pool.submit(job, mode, seed)] = (mode, seed)
            done, _ = wait(set(futures), timeout=180,
                           return_when=FIRST_COMPLETED)
            for f in done:
                futures.pop(f)
            if not pending and not futures and not stopped:
                break
    finally:
        pool.shutdown(wait=True)
    print(f"[ebext] DONE cells={len(state['cells'])} "
          f"est=CNY{state['est_spend_cny']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
