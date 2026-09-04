#!/usr/bin/env bash
# Overnight Exp-II orchestrator: Stage0 smoke → Stage1 full → finalize docs.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs results/exp_ii
ts="$(date +%Y%m%d_%H%M%S)"
orch_log="logs/exp_ii_overnight_${ts}.log"
export PROVIDER="${PROVIDER:-Qwen}"
export MODEL="${MODEL:-qwen3.7-plus}"
export N_REPS="${N_REPS:-5}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-2500000}"
export ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export OUT_DIR="${OUT_DIR:-results/exp_ii}"
export INJECT="${INJECT:-me_force_zero_sea}"

{
  echo "ORCH_START=$(date -Iseconds)"
  echo "Running Stage 0 (ii_adv_r1 + ii_seq_r1)..."
} | tee "$orch_log"

export STAGE=0
export MASTER_LOG="logs/exp_ii_stage0_${ts}.log"
set +e
bash scripts/run_exp_ii.sh
rc0=$?
set -e
echo "STAGE0_RC=$rc0" | tee -a "$orch_log"
cp -f "$OUT_DIR/DONE" "$OUT_DIR/DONE_STAGE0" 2>/dev/null || true
cp -f "$OUT_DIR/TOKEN_SPEND.txt" "$OUT_DIR/TOKEN_SPEND_STAGE0.txt" 2>/dev/null || true

if [[ "$rc0" -ne 0 ]]; then
  echo "Stage 0 failed; NOT launching Stage 1." | tee -a "$orch_log"
  uv run python scripts/finalize_exp_ii_overnight.py --status fail_stage0 --orch-log "$orch_log" \
    >>"$orch_log" 2>&1 || true
  exit "$rc0"
fi

echo "Running Stage 1 (remaining cells, skip existing)..." | tee -a "$orch_log"
export STAGE=1
export MASTER_LOG="logs/exp_ii_stage1_${ts}.log"
set +e
bash scripts/run_exp_ii.sh
rc1=$?
set -e
echo "STAGE1_RC=$rc1" | tee -a "$orch_log"

set +e
uv run python scripts/finalize_exp_ii_overnight.py \
  --status "$([[ $rc1 -eq 0 ]] && echo ok || echo partial)" \
  --orch-log "$orch_log" \
  >>"$orch_log" 2>&1
fin_rc=$?
set -e
echo "FINALIZE_RC=$fin_rc ORCH_END=$(date -Iseconds)" | tee -a "$orch_log"
exit "$rc1"
