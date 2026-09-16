"""Reflexion K=5 budget-tilted challenger arm on the n=30 frozen blackboards.

Gives Reflexion a larger retry budget (K=5) than every other arm (K=3) to
test the registered reviewer risk that its memory-based revision would
overtake Stackelberg given more rounds. Resumes the identical me_force
snapshots the K=3 arms used (same-board fairness; the tilt is the point):

  MAKO_RESULT_SUFFIX=ds41_r5_reflexion_sN uv run python -m fsm_stackelberg.main \
    ... --diagnosis_mode reflexion --max_retries 5 --resume_from <snapshot>

Restartable (skips cells whose manifest exists); spend guard in DeepSeek
list-price upper-bound units (actual spend is typically 0.2-0.3x).

Usage:
  uv run --no-sync python scripts/run_r5_reflexion.py \
      [--seeds 1,2,...] [--workers 3] [--max_spend_cny 25]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The results tree encodes the retry budget as a path segment
# (prob_tslp_ecr_demand_k{K}); K=5 cells land in _k5.
RESULTS = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
           / "prob_tslp_ecr_demand_k5" / "smoke_H4_Omega5")
SNAPSHOTS = REPO / "results" / "snapshots"
STATE = REPO / "results" / "r5_reflexion"
PLANT = "me_force_zero_sea"

BASE_ARGS = [
    "--dataset", "prob_tslp_ecr_demand",
    "--prob_name", "smoke_H4_Omega5",
    "--provider", "DeepSeek",
    "--model", "deepseek-flash",
    "--knowledge", "progressive",
    "--max_retries", "5",
    "--diagnosis_mode", "reflexion",
]


def default_seeds() -> list[int]:
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
                                  stderr=subprocess.STDOUT, timeout=3600)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    tok = read_tokens(suffix)
    return {"ok": rc == 0 and tok is not None, "tokens": tok, "rc": rc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="",
                    help="comma list; default = all usable me_force seeds")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_spend_cny", type=float, default=25.0,
                    help="list-price upper bound; actual is 0.2-0.3x")
    args = ap.parse_args()

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

    def job(seed: int) -> None:
        suffix = f"ds41_r5_reflexion_s{seed}"
        res = run_cell(
            suffix,
            ["--resume_from", str(SNAPSHOTS / f"{PLANT}_s{seed}")],
            STATE / "logs" / f"r5_s{seed}.log")
        record(suffix, res)
        with lock:
            spend = state["est_spend_cny"]
        print(f"[r5] s{seed}: {'ok' if res['ok'] else 'FAIL'} "
              f"est=CNY{spend}", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = {s for s in seeds
               if state["cells"].get(f"ds41_r5_reflexion_s{s}") != "ok"}
    futures = {}
    try:
        while pending or futures:
            with lock:
                spend = state["est_spend_cny"]
            if spend > args.max_spend_cny:
                print(f"[r5] BUDGET STOP est=CNY{spend}", flush=True)
                break
            while pending and len(futures) < args.workers:
                seed = sorted(pending)[0]
                pending.discard(seed)
                futures[pool.submit(job, seed)] = seed
            done, _ = wait(set(futures), timeout=180,
                           return_when=FIRST_COMPLETED)
            for f in done:
                futures.pop(f)
            if not pending and not futures:
                break
    finally:
        pool.shutdown(wait=True)
    print(f"[r5] DONE cells={len(state['cells'])} "
          f"est=CNY{state['est_spend_cny']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
