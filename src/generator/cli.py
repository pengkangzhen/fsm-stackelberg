"""CLI to export TSLP demand-uncertainty instances for fsm-stackelberg."""

from __future__ import annotations

import argparse
from pathlib import Path

from .serialize import build_smoke_export, write_instance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TSLP ECR two-stage DEP instances for the agentic pipeline"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/prob_tslp_ecr_demand/instances/smoke_H4_Omega5",
        help="Output instance directory",
    )
    parser.add_argument("--T", type=int, default=4, help="Planning-window length H=T")
    parser.add_argument("--scenarios", type=int, default=5, help="|Omega|")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sdr", type=float, default=1.0, help="Supply-demand ratio scale")
    args = parser.parse_args()

    sample, optimal = build_smoke_export(
        T=args.T,
        n_scenarios=args.scenarios,
        seed=args.seed,
        sdr=args.sdr,
    )
    out = write_instance(args.output, sample, optimal)
    print(f"Wrote instance to {out}")
    print(f"  status={optimal['status']} objective={optimal['objective']:.4f}")
    print(f"  nodes={sample['_meta']['instance'].get('n_nodes')} "
          f"arcs={sample['_meta']['instance'].get('n_arcs')} "
          f"|Omega|={sample['_meta']['n_scenarios']}")


if __name__ == "__main__":
    main()
