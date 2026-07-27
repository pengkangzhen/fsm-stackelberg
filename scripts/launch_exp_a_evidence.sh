#!/usr/bin/env bash
# Detach redesigned Exp-A (evidence_rank). Survives Cursor/SSH disconnect.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-all}"
N_REPS="${N_REPS:-3}"
OUT_DIR="${OUT_DIR:-results/exp_a_evidence}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1500000}"
ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OMEGA_SOURCE="${OMEGA_SOURCE:-evidence_rank}"
RANK_METHOD="${RANK_METHOD:-llm_rank}"

mkdir -p logs "$OUT_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
out="logs/exp_a_evidence_nohup_${ts}.out"
pidfile="logs/exp_a_evidence_LATEST.pid"
export MASTER_LOG="logs/exp_a_evidence_${ts}.log"
export OUT_DIR STAGE N_REPS TOKEN_BUDGET ATTEMPTS_PER_CELL SKIP_EXISTING
export OMEGA_SOURCE RANK_METHOD
export INJECT="${INJECT:-me_force_zero_sea}"
export PROVIDER="${PROVIDER:-DashScope}"
export MODEL="${MODEL:-deepseek-v4-flash}"

chmod +x scripts/run_exp_a_evidence.sh
setsid nohup bash scripts/run_exp_a_evidence.sh < /dev/null > "$out" 2>&1 &
pid=$!
echo "$pid" > "$pidfile"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: Exp-A evidence job died at start; see $out" >&2
  exit 1
fi

cat <<EOF
Redesigned Exp-A launched (detached).
  PID:           $pid
  Stage:         $STAGE
  N_REPS:        $N_REPS × {causal, reverse, random}
  omega_source:  $OMEGA_SOURCE
  rank_method:   $RANK_METHOD
  Token budget:  $TOKEN_BUDGET (~¥$(python3 -c "print(round($TOKEN_BUDGET/1e6*3.04, 2))"))
  Skip existing: $SKIP_EXISTING
  NoHUP out:     $out
  Master log:    $MASTER_LOG
  Status:        $OUT_DIR/STATUS.txt
  Spend:         $OUT_DIR/TOKEN_SPEND.txt
  Done file:     $OUT_DIR/DONE
  Summary:       $OUT_DIR/summary.md
EOF
