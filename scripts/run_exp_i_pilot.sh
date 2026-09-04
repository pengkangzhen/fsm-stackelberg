#!/usr/bin/env bash
# Exp-I pilot (kill criteria): commitment-order ablation vs adversarial.
# Plant: me_drop_stage2_balance on smoke_H4_Omega5 (a*=model_expert).
set -euo pipefail
cd "$(dirname "$0")/.."

PROVIDER="${PROVIDER:-Qwen}"
MODEL="${MODEL:-qwen3.7-plus}"
DATASET="${DATASET:-prob_tslp_ecr_demand}"
PROB="${PROB:-smoke_H4_Omega5}"
INJECT="${INJECT:-me_drop_stage2_balance}"
MAX_RETRIES="${MAX_RETRIES:-3}"
KNOWLEDGE="${KNOWLEDGE:-progressive}"
OUT_DIR="${OUT_DIR:-results/exp_i_pilot}"

mkdir -p "$OUT_DIR" logs
ts="$(date +%Y%m%d_%H%M%S)"
master_log="logs/exp_i_pilot_${ts}.log"
echo "Exp-I pilot start ts=$ts log=$master_log" | tee "$master_log"

# Configs: diagnosis_mode|probe_order|suffix
CONFIGS=(
  "stackelberg|causal|sb_causal"
  "stackelberg|reverse|sb_reverse"
  "stackelberg|random|sb_random"
  "adversarial|causal|adv"
)

set -a
# shellcheck disable=SC1091
source .env
set +a
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r mode order suffix <<<"$spec"
  echo "===== RUN mode=$mode order=$order suffix=$suffix =====" | tee -a "$master_log"
  export MAKO_RESULT_SUFFIX="exp_i_pilot_${suffix}"
  run_log="$OUT_DIR/run_${suffix}.log"
  # Preserve pipeline exit status under set -e + pipefail.
  set +e
  uv run python -u -m fsm_stackelberg.main \
    --algorithm mako \
    --dataset "$DATASET" \
    --prob_name "$PROB" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --diagnosis_mode "$mode" \
    --probe_order "$order" \
    --inject "$INJECT" \
    --true_root_cause model_expert \
    --knowledge "$KNOWLEDGE" \
    --max_retries "$MAX_RETRIES" \
    2>&1 | tee "$run_log" | tee -a "$master_log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "EXIT_$suffix=$rc" | tee -a "$master_log"
  if [[ $rc -ne 0 ]]; then
    echo "WARN: run $suffix exited $rc (continuing grid)" | tee -a "$master_log"
  fi
done

uv run python scripts/summarize_exp_i_pilot.py \
  --out-dir "$OUT_DIR" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prob-name "$PROB" \
  --max-retries "$MAX_RETRIES" \
  2>&1 | tee -a "$master_log"

echo "Exp-I pilot done. Summary: $OUT_DIR/summary.md" | tee -a "$master_log"
