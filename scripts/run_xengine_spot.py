"""Cross-engine spot-check: stackelberg order arms on a second engine.

Resumes the me_force snapshots frozen by the primary DeepSeek campaign
(identical failure blackboards) with a different diagnosis-loop engine
(default official Qwen / qwen3.8-flash via QWEN_*), running the three
commitment orders per seed. This isolates the diagnosis-loop engine
variable: the forward prefix is byte-identical by snapshot construction,
so any difference is attributable to the diagnosing/repairing LLM.

Registered scope (manuscript §Exp-B scope conditions): a spot-check, not
a full campaign — engine switch = new campaign freeze; cells are never
pooled with DeepSeek numbers and are reported separately.

Usage:
  uv run --no-sync python scripts/run_xengine_spot.py \
      [--provider Qwen] [--model qwen3.8-flash] [--seeds 1,2,3,4,5] \
      [--workers 3] [--max_spend_cny 12]
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
SNAPSHOTS = REPO / "results" / "snapshots"
PLANT = "me_force_zero_sea"
ORDERS = ("causal", "reverse", "random")
STATE = REPO / "results" / "xengine_spot"


def results_root(provider: str, model: str) -> Path:
    return (REPO / "results" / "mako" / f"{provider}_{model}"
            / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")


def usable_seeds() -> list[int]:
    return sorted(
        int(p.name.rsplit("_s", 1)[1])
        for p in SNAPSHOTS.glob(f"{PLANT}_s*")
        if (p / "snapshot_state.json").exists())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="Qwen")
    ap.add_argument("--model", default="qwen3.8-flash")
    ap.add_argument("--seeds", default="1,2,3,4,5",
                    help="spot-check boards (first five usable by default)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_spend_cny", type=float, default=12.0,
                    help="conservative list-price upper bound")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else usable_seeds()[:5])
    results = results_root(args.provider, args.model)
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "logs").mkdir(exist_ok=True)
    state_f = STATE / "state.json"
    lock = threading.Lock()
    state = (json.loads(state_f.read_text()) if state_f.exists()
             else {"cells": {}, "est_spend_cny": 0.0})

    base_args = [
        "--dataset", "prob_tslp_ecr_demand",
        "--prob_name", "smoke_H4_Omega5",
        "--provider", args.provider,
        "--model", args.model,
        "--knowledge", "progressive",
        "--max_retries", "3",
        "--diagnosis_mode", "stackelberg",
        "--omega_source", "evidence_rank",
        "--rank_method", "hybrid",
    ]

    def read_tokens(suffix: str) -> dict | None:
        m = results / suffix / "run_manifest.json"
        if not m.exists():
            return None
        try:
            man = json.loads(m.read_text())
        except json.JSONDecodeError:
            return None
        cost = man.get("cost") or {}
        return {"prompt_tokens": cost.get("prompt_tokens") or 0,
                "completion_tokens": cost.get("completion_tokens") or 0}

    def run_cell(suffix: str, extra_args: list[str]) -> dict:
        if (results / suffix / "run_manifest.json").exists():
            return {"ok": True, "tokens": read_tokens(suffix),
                    "skipped": True, "rc": 0}
        env = {**os.environ, "MAKO_RESULT_SUFFIX": suffix}
        cmd = ["uv", "run", "--no-sync", "python", "-u", "-m",
               "fsm_stackelberg.main", *base_args, *extra_args]
        log_path = STATE / "logs" / f"{suffix}.log"
        with log_path.open("w") as fh:
            try:
                proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT, timeout=1800)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = -9
        tok = read_tokens(suffix)
        return {"ok": rc == 0 and tok is not None, "tokens": tok, "rc": rc}

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

    def job(order: str, seed: int) -> None:
        suffix = f"xe_qwen38_{order}_s{seed}"
        extra = ["--probe_order", order,
                 "--resume_from", str(SNAPSHOTS / f"{PLANT}_s{seed}")]
        if order == "random":
            extra += ["--probe_seed", str(seed)]
        res = run_cell(suffix, extra)
        record(suffix, res)
        with lock:
            spend = state["est_spend_cny"]
        print(f"[xe] {order} s{seed}: {'ok' if res['ok'] else 'FAIL'} "
              f"est=CNY{spend}", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = {(o, s) for s in seeds for o in ORDERS
               if state["cells"].get(f"xe_qwen38_{o}_s{s}") != "ok"}
    futures = {}
    try:
        while pending or futures:
            with lock:
                spend = state["est_spend_cny"]
            if spend > args.max_spend_cny:
                print(f"[xe] BUDGET STOP est=CNY{spend}", flush=True)
                break
            while pending and len(futures) < args.workers:
                order, seed = sorted(pending)[0]
                pending.discard((order, seed))
                futures[pool.submit(job, order, seed)] = (order, seed)
            done, _ = wait(set(futures), timeout=180,
                           return_when=FIRST_COMPLETED)
            for f in done:
                futures.pop(f)
            if not pending and not futures:
                break
    finally:
        pool.shutdown(wait=True)
    print(f"[xe] DONE cells={len(state['cells'])} "
          f"est=CNY{state['est_spend_cny']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
