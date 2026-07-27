#!/usr/bin/env bash
# Exp-II internal baselines (cost-safe): adversarial + sequential × N_REPS.
# Same plant/fairness as Exp-I. Primary paper metric: attribution_hit;
# also report first_probe_hit / SSR / tokens for contrast with Exp-I causal.
#
# STAGE:
#   0 — ii_adv_r1 + ii_seq_r1 only
#   1|all — full N_REPS grid (skip existing)
set -uo pipefail
cd "$(dirname "$0")/.."

PROVIDER="${PROVIDER:-Qwen}"
MODEL="${MODEL:-qwen3.7-plus}"
DATASET="${DATASET:-prob_tslp_ecr_demand}"
PROB="${PROB:-smoke_H4_Omega5}"
INJECT="${INJECT:-me_force_zero_sea}"
MAX_RETRIES="${MAX_RETRIES:-3}"
KNOWLEDGE="${KNOWLEDGE:-progressive}"
OUT_DIR="${OUT_DIR:-results/exp_ii}"
N_REPS="${N_REPS:-5}"
MODES="${MODES:-adversarial,sequential}"
ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-1}"
MAX_NETWORK_RETRIES="${MAX_NETWORK_RETRIES:-1}"
SLEEP_BETWEEN_ATTEMPTS_S="${SLEEP_BETWEEN_ATTEMPTS_S:-45}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TOKEN_BUDGET="${TOKEN_BUDGET:-2500000}"
STAGE="${STAGE:-all}"
ONLY_SUFFIXES="${ONLY_SUFFIXES:-}"
PROBE_ORDER="${PROBE_ORDER:-causal}"

mkdir -p "$OUT_DIR" logs
ts="$(date +%Y%m%d_%H%M%S)"
master_log="${MASTER_LOG:-logs/exp_ii_${ts}.log}"
done_file="$OUT_DIR/DONE"
status_file="$OUT_DIR/STATUS.txt"
budget_file="$OUT_DIR/TOKEN_SPEND.txt"
rm -f "$done_file"
{
  echo "STARTED=$(date -Iseconds)"
  echo "STAGE=$STAGE INJECT=$INJECT N_REPS=$N_REPS"
  echo "PROVIDER=$PROVIDER MODEL=$MODEL"
  echo "MODES=$MODES PROBE_ORDER=$PROBE_ORDER"
  echo "SKIP_EXISTING=$SKIP_EXISTING TOKEN_BUDGET=$TOKEN_BUDGET"
  echo "ATTEMPTS_PER_CELL=$ATTEMPTS_PER_CELL"
  echo "MASTER_LOG=$master_log"
  echo "PRIMARY_METRIC=attribution_hit"
  echo "HARD_BUDGET_RMB=10"
} | tee "$status_file" | tee "$master_log"

set -a
# shellcheck disable=SC1091
source .env
set +a
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

RESULT_ROOT="results/mako/${PROVIDER}_${MODEL//\//-}/${DATASET}_k${MAX_RETRIES}/${PROB}"

is_network_failure() {
  local logf="$1"
  grep -qiE 'Temporary failure in name resolution|APIConnectionError|ConnectError|Connection error|Name or service not known' "$logf" 2>/dev/null
}

mode_short() {
  case "$1" in
    adversarial) echo adv ;;
    sequential) echo seq ;;
    *) echo "$1" ;;
  esac
}

mode_from_suffix() {
  local suffix="$1"
  if [[ "$suffix" == ii_adv_* ]]; then
    echo adversarial
  elif [[ "$suffix" == ii_seq_* ]]; then
    echo sequential
  else
    echo unknown
  fi
}

rep_from_suffix() {
  local suffix="$1"
  if [[ "$suffix" =~ _r([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 0
  fi
}

result_dir() {
  local suffix="$1"
  echo "${RESULT_ROOT}/exp_i_pilot_${suffix}"
}

result_json() {
  local suffix="$1"
  echo "$(result_dir "$suffix")/experiment_result.json"
}

cell_ok() {
  local suffix="$1"
  local er
  er="$(result_json "$suffix")"
  [[ -f "$er" ]] || return 1
  uv run python - "$er" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
tokens = d.get("total_tokens") or 0
err = (d.get("error_msg") or "") + json.dumps(d.get("error_details") or {})
if tokens <= 0 and ("APIConnectionError" in err or "ConnectError" in err or "name resolution" in err.lower()):
    sys.exit(1)
ep = d.get("episode_payoff") or {}
if tokens <= 0 and not ep and not d.get("gurobi_status") and not d.get("attributed_layer"):
    sys.exit(1)
sys.exit(0)
PY
}

cell_tokens() {
  local suffix="$1"
  local er
  er="$(result_json "$suffix")"
  [[ -f "$er" ]] || { echo 0; return; }
  uv run python - "$er" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(int(d.get("total_tokens") or 0))
PY
}

est_rmb() {
  local toks="$1"
  uv run python -c "print(round(${toks}/1e6*3.04, 2))"
}

build_suffix_list() {
  local modes=() mode short rep
  IFS=',' read -r -a modes <<< "$MODES"
  SUFFIX_LIST=()
  if [[ -n "$ONLY_SUFFIXES" ]]; then
    IFS=',' read -r -a SUFFIX_LIST <<< "$ONLY_SUFFIXES"
    return
  fi
  case "$STAGE" in
    0)
      SUFFIX_LIST=("ii_adv_r1" "ii_seq_r1")
      ;;
    1|all)
      for rep in $(seq 1 "$N_REPS"); do
        for mode in "${modes[@]}"; do
          mode="$(echo "$mode" | tr -d '[:space:]')"
          [[ -n "$mode" ]] || continue
          short="$(mode_short "$mode")"
          SUFFIX_LIST+=("ii_${short}_r${rep}")
        done
      done
      ;;
    *)
      echo "ERROR: unknown STAGE=$STAGE (use 0|1|all)" | tee -a "$master_log"
      exit 2
      ;;
  esac
}

run_one() {
  local mode="$1" suffix="$2" attempt="$3"
  local run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
  export MAKO_RESULT_SUFFIX="exp_i_pilot_${suffix}"
  echo "===== RUN mode=$mode suffix=$suffix attempt=$attempt =====" | tee -a "$master_log"
  echo "RUNNING=$suffix mode=$mode attempt=$attempt at=$(date -Iseconds)" > "$status_file"

  set +e
  uv run python -u -m fsm_stackelberg.main \
    --algorithm mako \
    --dataset "$DATASET" \
    --prob_name "$PROB" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --diagnosis_mode "$mode" \
    --probe_order "$PROBE_ORDER" \
    --inject "$INJECT" \
    --true_root_cause model_expert \
    --knowledge "$KNOWLEDGE" \
    --max_retries "$MAX_RETRIES" \
    >"$run_log" 2>&1
  local rc=$?
  set -e
  {
    echo "----- begin $suffix a$attempt rc=$rc -----"
    tail -n 100 "$run_log"
    echo "----- end $suffix a$attempt -----"
  } >> "$master_log"
  echo "EXIT_${suffix}_a${attempt}=$rc" | tee -a "$master_log"
  return "$rc"
}

build_suffix_list
echo "PLANNED_SUFFIXES=${SUFFIX_LIST[*]}" | tee -a "$master_log"

TOKENS_SPENT=0
API_CALLS=0
overall_rc=0
budget_stop=0
echo "0" > "$budget_file"

for suffix in "${SUFFIX_LIST[@]}"; do
  mode="$(mode_from_suffix "$suffix")"
  if [[ "$mode" == "unknown" ]]; then
    echo "BAD_SUFFIX $suffix" | tee -a "$master_log"
    overall_rc=1
    continue
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && cell_ok "$suffix"; then
    echo "SKIP_EXISTING $suffix tokens=$(cell_tokens "$suffix")" | tee -a "$master_log"
    continue
  fi

  if [[ "$TOKENS_SPENT" -ge "$TOKEN_BUDGET" ]]; then
    echo "BUDGET_STOP before $suffix spent=$TOKENS_SPENT (~¥$(est_rmb "$TOKENS_SPENT"))" \
      | tee -a "$master_log" | tee "$status_file"
    budget_stop=1
    overall_rc=2
    break
  fi

  ok=0
  max_attempts=$((ATTEMPTS_PER_CELL + MAX_NETWORK_RETRIES))
  for attempt in $(seq 1 "$max_attempts"); do
    if [[ "$attempt" -gt "$ATTEMPTS_PER_CELL" ]]; then
      prev_log="$OUT_DIR/run_${suffix}_a$((attempt - 1)).log"
      if ! is_network_failure "$prev_log"; then
        break
      fi
      echo "NETWORK_RETRY $suffix attempt=$attempt" | tee -a "$master_log"
      sleep "$SLEEP_BETWEEN_ATTEMPTS_S"
    fi

    before_tok="$(cell_tokens "$suffix")"
    set +e
    run_one "$mode" "$suffix" "$attempt"
    rc=$?
    set -e
    run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
    API_CALLS=$((API_CALLS + 1))

    after_tok="$(cell_tokens "$suffix")"
    delta=$((after_tok - before_tok))
    if [[ "$delta" -lt 0 ]]; then
      delta="$after_tok"
    fi
    TOKENS_SPENT=$((TOKENS_SPENT + delta))
    {
      echo "TOKENS_SPENT=$TOKENS_SPENT"
      echo "EST_RMB=$(est_rmb "$TOKENS_SPENT")"
      echo "API_CALLS=$API_CALLS"
      echo "LAST_CELL=$suffix"
      echo "LAST_DELTA=$delta"
      echo "UPDATED=$(date -Iseconds)"
    } | tee "$budget_file" | tee -a "$master_log"

    if [[ "$TOKENS_SPENT" -ge "$TOKEN_BUDGET" ]]; then
      echo "BUDGET_STOP after $suffix spent=$TOKENS_SPENT (~¥$(est_rmb "$TOKENS_SPENT"))" \
        | tee -a "$master_log"
      budget_stop=1
      overall_rc=2
      break
    fi

    if cell_ok "$suffix"; then
      echo "CELL_OK $suffix attempt=$attempt delta_tokens=$delta" | tee -a "$master_log"
      ok=1
      break
    fi
    if is_network_failure "$run_log" || [[ $rc -ne 0 ]]; then
      echo "RETRYABLE $suffix attempt=$attempt rc=$rc" | tee -a "$master_log"
      continue
    fi
    if [[ -f "$(result_json "$suffix")" ]]; then
      echo "CELL_DONE_WITH_ISSUES $suffix attempt=$attempt" | tee -a "$master_log"
      ok=1
      break
    fi
  done

  if [[ $ok -ne 1 ]]; then
    echo "CELL_FAILED $suffix" | tee -a "$master_log"
    overall_rc=1
  fi
  if [[ "$budget_stop" -eq 1 ]]; then
    break
  fi
done

IFS=','
suffix_csv="${SUFFIX_LIST[*]}"
unset IFS

set +e
uv run python scripts/summarize_exp_ii.py \
  --out-dir "$OUT_DIR" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prob-name "$PROB" \
  --max-retries "$MAX_RETRIES" \
  --suffixes "$suffix_csv" \
  --include-exp-i-causal \
  >>"$master_log" 2>&1
sum_rc=$?
set -e

{
  echo "FINISHED=$(date -Iseconds)"
  echo "OVERALL_RC=$overall_rc"
  echo "SUMMARY_RC=$sum_rc"
  echo "BUDGET_STOP=$budget_stop"
  echo "TOKENS_SPENT=$TOKENS_SPENT"
  echo "EST_RMB=$(est_rmb "$TOKENS_SPENT")"
  echo "API_CALLS=$API_CALLS"
  echo "SUMMARY=$OUT_DIR/summary.md"
  echo "SUFFIXES=$suffix_csv"
  echo "STAGE=$STAGE"
} | tee -a "$master_log" | tee "$status_file"

{
  echo "FINISHED=$(date -Iseconds)"
  echo "OVERALL_RC=$overall_rc"
  echo "BUDGET_STOP=$budget_stop"
  echo "INJECT=$INJECT"
  echo "N_REPS=$N_REPS"
  echo "STAGE=$STAGE"
  echo "TOKENS_SPENT=$TOKENS_SPENT"
  echo "EST_RMB=$(est_rmb "$TOKENS_SPENT")"
  echo "API_CALLS=$API_CALLS"
  echo "SUMMARY=$OUT_DIR/summary.md"
  echo "MASTER_LOG=$master_log"
  echo "PRIMARY_METRIC=attribution_hit"
  echo "HARD_BUDGET_RMB=10"
} > "$done_file"

exit "$overall_rc"
