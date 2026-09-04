#!/usr/bin/env bash
# Detach Exp-II (setsid + nohup). Survives Cursor/SSH disconnect.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-0}"
N_REPS="${N_REPS:-5}"
OUT_DIR="${OUT_DIR:-results/exp_ii}"
TOKEN_BUDGET="${TOKEN_BUDGET:-2500000}"
ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p logs "$OUT_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
out="logs/exp_ii_nohup_${ts}.out"
pidfile="logs/exp_ii_LATEST.pid"
export MASTER_LOG="logs/exp_ii_${ts}.log"
export OUT_DIR STAGE N_REPS TOKEN_BUDGET ATTEMPTS_PER_CELL SKIP_EXISTING
export INJECT="${INJECT:-me_force_zero_sea}"
export PROVIDER="${PROVIDER:-Qwen}"
export MODEL="${MODEL:-qwen3.7-plus}"
export MODES="${MODES:-adversarial,sequential}"

setsid nohup bash scripts/run_exp_ii.sh < /dev/null > "$out" 2>&1 &
pid=$!
echo "$pid" > "$pidfile"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: Exp-II job died at start; see $out" >&2
  exit 1
fi

cat <<EOF
Exp-II launched (detached).
  PID:           $pid
  Stage:         $STAGE
  N_REPS:        $N_REPS × {adversarial, sequential}
  Skip existing: $SKIP_EXISTING
  Token budget:  $TOKEN_BUDGET (~¥10 hard cap)
  Attempts/cell: $ATTEMPTS_PER_CELL (+1 network-only)
  NoHUP out:     $out
  Master log:    $MASTER_LOG
  Status:        $OUT_DIR/STATUS.txt
  Spend:         $OUT_DIR/TOKEN_SPEND.txt
  Done file:     $OUT_DIR/DONE
  Summary:       $OUT_DIR/summary.md
  Morning note:  results/MORNING.md

Check:
  cat $OUT_DIR/STATUS.txt
  cat $OUT_DIR/TOKEN_SPEND.txt
  cat $OUT_DIR/DONE
  cat $OUT_DIR/summary.md
EOF
