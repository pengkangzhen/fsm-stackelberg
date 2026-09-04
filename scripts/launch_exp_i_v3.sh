#!/usr/bin/env bash
# Detach Exp-I v3 re-pilot (setsid + nohup). Survives Cursor/SSH disconnect.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs results/exp_i_pilot_v3
ts="$(date +%Y%m%d_%H%M%S)"
out="logs/exp_i_pilot_v3_nohup_${ts}.out"
pidfile="logs/exp_i_pilot_v3_LATEST.pid"
export MASTER_LOG="logs/exp_i_pilot_v3_${ts}.log"
export OUT_DIR="results/exp_i_pilot_v3"
export INJECT="${INJECT:-me_force_zero_sea}"
export N_REPS="${N_REPS:-3}"
export ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-3}"

setsid nohup bash scripts/run_exp_i_pilot_v3.sh < /dev/null > "$out" 2>&1 &
pid=$!
echo "$pid" > "$pidfile"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: v3 job died at start; see $out" >&2
  exit 1
fi

cat <<EOF
Exp-I v3 re-pilot launched (detached).
  PID:        $pid
  Inject:     $INJECT
  Replicates: $N_REPS × {causal, random}
  Kill metric: first_probe_hit (committed ω probes a* first)
  Fixes:      random/reverse no longer Prior-rotated; tight plant; payoff diagnostics
  NoHUP out:  $out
  Master log: $MASTER_LOG
  Status:     $OUT_DIR/STATUS.txt
  Done file:  $OUT_DIR/DONE
  Summary:    $OUT_DIR/summary.md

Check:
  cat results/exp_i_pilot_v3/DONE
  cat results/exp_i_pilot_v3/summary.md
EOF
