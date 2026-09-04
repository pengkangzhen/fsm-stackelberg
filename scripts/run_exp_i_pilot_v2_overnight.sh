#!/usr/bin/env bash
# Overnight Exp-I re-pilot: strengthened ME plant, causal vs random only.
# Detach-safe: run via scripts/launch_exp_i_overnight.sh (setsid+nohup).
set -uo pipefail
cd "$(dirname "$0")/.."

PROVIDER="${PROVIDER:-Qwen}"
MODEL="${MODEL:-qwen3.7-plus}"
DATASET="${DATASET:-prob_tslp_ecr_demand}"
PROB="${PROB:-smoke_H4_Omega5}"
INJECT="${INJECT:-me_force_zero_sea}"
MAX_RETRIES="${MAX_RETRIES:-3}"
KNOWLEDGE="${KNOWLEDGE:-progressive}"
OUT_DIR="${OUT_DIR:-results/exp_i_pilot_v2}"
ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-3}"
SLEEP_BETWEEN_ATTEMPTS_S="${SLEEP_BETWEEN_ATTEMPTS_S:-30}"

mkdir -p "$OUT_DIR" logs
ts="$(date +%Y%m%d_%H%M%S)"
master_log="${MASTER_LOG:-logs/exp_i_pilot_v2_${ts}.log}"
done_file="$OUT_DIR/DONE"
status_file="$OUT_DIR/STATUS.txt"
rm -f "$done_file"
{
  echo "STARTED=$(date -Iseconds)"
  echo "INJECT=$INJECT"
  echo "PROVIDER=$PROVIDER MODEL=$MODEL"
  echo "MASTER_LOG=$master_log"
} | tee "$status_file" | tee "$master_log"

# Only kill-criteria pair.
CONFIGS=(
  "stackelberg|causal|v2_sb_causal"
  "stackelberg|random|v2_sb_random"
)

set -a
# shellcheck disable=SC1091
source .env
set +a
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

is_network_failure() {
  local logf="$1"
  grep -qiE 'Temporary failure in name resolution|APIConnectionError|ConnectError|Connection error|Name or service not known' "$logf" 2>/dev/null
}

run_one() {
  local mode="$1" order="$2" suffix="$3" attempt="$4"
  local run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
  export MAKO_RESULT_SUFFIX="exp_i_pilot_${suffix}"
  echo "===== RUN mode=$mode order=$order suffix=$suffix attempt=$attempt/$(echo $ATTEMPTS_PER_CELL) =====" \
    | tee -a "$master_log"
  echo "RUNNING=$suffix attempt=$attempt at=$(date -Iseconds)" > "$status_file"

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
    >"$run_log" 2>&1
  local rc=$?
  set -e
  # Also append to master (tail to keep size sane on huge gurobi dumps)
  {
    echo "----- begin $suffix a$attempt rc=$rc -----"
    tail -n 80 "$run_log"
    echo "----- end $suffix a$attempt -----"
  } >> "$master_log"
  echo "EXIT_${suffix}_a${attempt}=$rc" | tee -a "$master_log"
  return "$rc"
}

cell_ok() {
  local suffix="$1"
  local er="results/mako/${PROVIDER}_${MODEL//\//-}/${DATASET}_k${MAX_RETRIES}/${PROB}/exp_i_pilot_${suffix}/experiment_result.json"
  [[ -f "$er" ]] || return 1
  # Treat pure connection crashes (0 tokens, no gurobi) as not OK.
  uv run python - "$er" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
tokens = d.get("total_tokens") or 0
err = (d.get("error_msg") or "") + json.dumps(d.get("error_details") or {})
if tokens <= 0 and ("APIConnectionError" in err or "ConnectError" in err or "name resolution" in err.lower()):
    sys.exit(1)
# Require that inject ran far enough to produce some agent activity OR a real solve/diagnosis.
if tokens <= 0 and not d.get("gurobi_status"):
    sys.exit(1)
sys.exit(0)
PY
}

overall_rc=0
for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r mode order suffix <<<"$spec"
  ok=0
  for attempt in $(seq 1 "$ATTEMPTS_PER_CELL"); do
    set +e
    run_one "$mode" "$order" "$suffix" "$attempt"
    rc=$?
    set -e
    run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
    if cell_ok "$suffix"; then
      echo "CELL_OK $suffix attempt=$attempt" | tee -a "$master_log"
      ok=1
      break
    fi
    if is_network_failure "$run_log" || [[ $rc -ne 0 ]]; then
      echo "RETRYABLE failure on $suffix attempt=$attempt; sleep ${SLEEP_BETWEEN_ATTEMPTS_S}s" \
        | tee -a "$master_log"
      sleep "$SLEEP_BETWEEN_ATTEMPTS_S"
      continue
    fi
    # Non-network failure with a written result: accept and move on.
    if [[ -f "results/mako/${PROVIDER}_${MODEL//\//-}/${DATASET}_k${MAX_RETRIES}/${PROB}/exp_i_pilot_${suffix}/experiment_result.json" ]]; then
      echo "CELL_DONE_WITH_ISSUES $suffix attempt=$attempt" | tee -a "$master_log"
      ok=1
      break
    fi
    sleep "$SLEEP_BETWEEN_ATTEMPTS_S"
  done
  if [[ $ok -ne 1 ]]; then
    echo "CELL_FAILED $suffix after $ATTEMPTS_PER_CELL attempts" | tee -a "$master_log"
    overall_rc=1
  fi
done

set +e
uv run python scripts/summarize_exp_i_pilot.py \
  --out-dir "$OUT_DIR" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prob-name "$PROB" \
  --max-retries "$MAX_RETRIES" \
  --suffixes v2_sb_causal,v2_sb_random \
  >>"$master_log" 2>&1
sum_rc=$?
set -e

{
  echo "FINISHED=$(date -Iseconds)"
  echo "OVERALL_RC=$overall_rc"
  echo "SUMMARY_RC=$sum_rc"
  echo "SUMMARY=$OUT_DIR/summary.md"
} | tee -a "$master_log" | tee "$status_file"

# Atomic-ish done marker for morning check.
{
  echo "FINISHED=$(date -Iseconds)"
  echo "OVERALL_RC=$overall_rc"
  echo "INJECT=$INJECT"
  echo "SUMMARY=$OUT_DIR/summary.md"
  echo "MASTER_LOG=$master_log"
} > "$done_file"

exit "$overall_rc"
