#!/usr/bin/env bash
# Exp-I re-pilot v3: fixed ω ablation + tight plant + first-probe kill metric.
# Grid: stackelberg × {causal, random} × N replicates (default 3).
set -uo pipefail
cd "$(dirname "$0")/.."

PROVIDER="${PROVIDER:-Qwen}"
MODEL="${MODEL:-qwen3.7-plus}"
DATASET="${DATASET:-prob_tslp_ecr_demand}"
PROB="${PROB:-smoke_H4_Omega5}"
INJECT="${INJECT:-me_force_zero_sea}"
MAX_RETRIES="${MAX_RETRIES:-3}"
KNOWLEDGE="${KNOWLEDGE:-progressive}"
OUT_DIR="${OUT_DIR:-results/exp_i_pilot_v3}"
N_REPS="${N_REPS:-3}"
ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-3}"
SLEEP_BETWEEN_ATTEMPTS_S="${SLEEP_BETWEEN_ATTEMPTS_S:-45}"

mkdir -p "$OUT_DIR" logs
ts="$(date +%Y%m%d_%H%M%S)"
master_log="${MASTER_LOG:-logs/exp_i_pilot_v3_${ts}.log}"
done_file="$OUT_DIR/DONE"
status_file="$OUT_DIR/STATUS.txt"
rm -f "$done_file"
{
  echo "STARTED=$(date -Iseconds)"
  echo "INJECT=$INJECT N_REPS=$N_REPS"
  echo "PROVIDER=$PROVIDER MODEL=$MODEL"
  echo "MASTER_LOG=$master_log"
  echo "KILL_METRIC=first_probe_hit"
} | tee "$status_file" | tee "$master_log"

set -a
# shellcheck disable=SC1091
source .env
set +a
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

is_network_failure() {
  local logf="$1"
  grep -qiE 'Temporary failure in name resolution|APIConnectionError|ConnectError|Connection error|Name or service not known' "$logf" 2>/dev/null
}

result_json() {
  local suffix="$1"
  echo "results/mako/${PROVIDER}_${MODEL//\//-}/${DATASET}_k${MAX_RETRIES}/${PROB}/exp_i_pilot_${suffix}/experiment_result.json"
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
# Prefer cells that reached diagnosis commit (have episode_payoff omega) or tokens>0
ep = d.get("episode_payoff") or {}
if tokens <= 0 and not ep.get("committed_omega") and not d.get("gurobi_status"):
    sys.exit(1)
sys.exit(0)
PY
}

run_one() {
  local mode="$1" order="$2" suffix="$3" probe_seed="$4" attempt="$5"
  local run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
  export MAKO_RESULT_SUFFIX="exp_i_pilot_${suffix}"
  echo "===== RUN mode=$mode order=$order suffix=$suffix seed=$probe_seed attempt=$attempt/$ATTEMPTS_PER_CELL =====" \
    | tee -a "$master_log"
  echo "RUNNING=$suffix attempt=$attempt seed=$probe_seed at=$(date -Iseconds)" > "$status_file"

  local seed_args=()
  if [[ "$order" == "random" ]]; then
    seed_args=(--probe_seed "$probe_seed")
  fi

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
    "${seed_args[@]}" \
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

SUFFIX_LIST=()
overall_rc=0

for rep in $(seq 1 "$N_REPS"); do
  for order in causal random; do
    suffix="v3_${order}_r${rep}"
    SUFFIX_LIST+=("$suffix")
    if [[ "$order" == "random" ]]; then
      probe_seed=$((2000 + rep * 31))
    else
      probe_seed=0
    fi
    ok=0
    for attempt in $(seq 1 "$ATTEMPTS_PER_CELL"); do
      set +e
      run_one stackelberg "$order" "$suffix" "$probe_seed" "$attempt"
      rc=$?
      set -e
      run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
      if cell_ok "$suffix"; then
        echo "CELL_OK $suffix attempt=$attempt" | tee -a "$master_log"
        ok=1
        break
      fi
      if is_network_failure "$run_log" || [[ $rc -ne 0 ]]; then
        echo "RETRYABLE $suffix attempt=$attempt; sleep ${SLEEP_BETWEEN_ATTEMPTS_S}s" \
          | tee -a "$master_log"
        sleep "$SLEEP_BETWEEN_ATTEMPTS_S"
        continue
      fi
      if [[ -f "$(result_json "$suffix")" ]]; then
        echo "CELL_DONE_WITH_ISSUES $suffix attempt=$attempt" | tee -a "$master_log"
        ok=1
        break
      fi
      sleep "$SLEEP_BETWEEN_ATTEMPTS_S"
    done
    if [[ $ok -ne 1 ]]; then
      echo "CELL_FAILED $suffix" | tee -a "$master_log"
      overall_rc=1
    fi
  done
done

IFS=','
suffix_csv="${SUFFIX_LIST[*]}"
unset IFS

set +e
uv run python scripts/summarize_exp_i_pilot.py \
  --out-dir "$OUT_DIR" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prob-name "$PROB" \
  --max-retries "$MAX_RETRIES" \
  --suffixes "$suffix_csv" \
  >>"$master_log" 2>&1
sum_rc=$?
set -e

{
  echo "FINISHED=$(date -Iseconds)"
  echo "OVERALL_RC=$overall_rc"
  echo "SUMMARY_RC=$sum_rc"
  echo "SUMMARY=$OUT_DIR/summary.md"
  echo "SUFFIXES=$suffix_csv"
} | tee -a "$master_log" | tee "$status_file"

{
  echo "FINISHED=$(date -Iseconds)"
  echo "OVERALL_RC=$overall_rc"
  echo "INJECT=$INJECT"
  echo "N_REPS=$N_REPS"
  echo "SUMMARY=$OUT_DIR/summary.md"
  echo "MASTER_LOG=$master_log"
  echo "KILL_METRIC=first_probe_hit"
} > "$done_file"

exit "$overall_rc"
