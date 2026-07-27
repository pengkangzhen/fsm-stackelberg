#!/usr/bin/env bash
# Full Exp-I (cost-safe): stackelberg × {causal, reverse, random} × N_REPS.
# Defaults: n=5, reuse pilot v3 causal/random r1-3, skip existing, token budget.
#
# Stages (via STAGE):
#   0 — only full_reverse_r1 (smoke reverse + tooling)
#   1 — remaining cells for N_REPS (skip existing)
#   all — full grid for N_REPS (skip existing)
set -uo pipefail
cd "$(dirname "$0")/.."

PROVIDER="${PROVIDER:-Qwen}"
MODEL="${MODEL:-qwen3.7-plus}"
DATASET="${DATASET:-prob_tslp_ecr_demand}"
PROB="${PROB:-smoke_H4_Omega5}"
INJECT="${INJECT:-me_force_zero_sea}"
MAX_RETRIES="${MAX_RETRIES:-3}"
KNOWLEDGE="${KNOWLEDGE:-progressive}"
OUT_DIR="${OUT_DIR:-results/exp_i_full}"
N_REPS="${N_REPS:-5}"
PROBE_ORDERS="${PROBE_ORDERS:-causal,reverse,random}"
ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-1}"
MAX_NETWORK_RETRIES="${MAX_NETWORK_RETRIES:-1}"
SLEEP_BETWEEN_ATTEMPTS_S="${SLEEP_BETWEEN_ATTEMPTS_S:-45}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REUSE_PILOT="${REUSE_PILOT:-1}"
TOKEN_BUDGET="${TOKEN_BUDGET:-2500000}"
STAGE="${STAGE:-all}"
ONLY_SUFFIXES="${ONLY_SUFFIXES:-}"

mkdir -p "$OUT_DIR" logs
ts="$(date +%Y%m%d_%H%M%S)"
master_log="${MASTER_LOG:-logs/exp_i_full_${ts}.log}"
done_file="$OUT_DIR/DONE"
status_file="$OUT_DIR/STATUS.txt"
budget_file="$OUT_DIR/TOKEN_SPEND.txt"
rm -f "$done_file"
{
  echo "STARTED=$(date -Iseconds)"
  echo "STAGE=$STAGE INJECT=$INJECT N_REPS=$N_REPS"
  echo "PROVIDER=$PROVIDER MODEL=$MODEL"
  echo "PROBE_ORDERS=$PROBE_ORDERS"
  echo "SKIP_EXISTING=$SKIP_EXISTING REUSE_PILOT=$REUSE_PILOT"
  echo "TOKEN_BUDGET=$TOKEN_BUDGET ATTEMPTS_PER_CELL=$ATTEMPTS_PER_CELL"
  echo "MASTER_LOG=$master_log"
  echo "KILL_METRIC=first_probe_hit"
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
if tokens <= 0 and not ep.get("committed_omega") and not d.get("gurobi_status"):
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

sum_tokens_for_suffixes() {
  local total=0 t
  for s in "$@"; do
    t="$(cell_tokens "$s")"
    total=$((total + t))
  done
  echo "$total"
}

# Soft RMB estimate (70/30 in/out mix @ DashScope 8-off: 3.04 RMB / 1M tokens)
est_rmb() {
  local toks="$1"
  uv run python -c "print(round(${toks}/1e6*3.04, 2))"
}

link_reuse_pilot() {
  [[ "$REUSE_PILOT" == "1" ]] || return 0
  local order rep src dst
  for order in causal random; do
    for rep in 1 2 3; do
      [[ "$rep" -le "$N_REPS" ]] || continue
      src="${RESULT_ROOT}/exp_i_pilot_v3_${order}_r${rep}"
      dst="${RESULT_ROOT}/exp_i_pilot_full_${order}_r${rep}"
      if [[ -d "$src" && -f "$src/experiment_result.json" ]]; then
        if [[ -e "$dst" || -L "$dst" ]]; then
          echo "REUSE_EXISTS full_${order}_r${rep}" | tee -a "$master_log"
        else
          ln -s "exp_i_pilot_v3_${order}_r${rep}" "$dst"
          echo "REUSE_LINKED full_${order}_r${rep} -> v3_${order}_r${rep}" | tee -a "$master_log"
        fi
      else
        echo "REUSE_MISS src=$src" | tee -a "$master_log"
      fi
    done
  done
}

build_suffix_list() {
  local orders=() order rep suffix
  IFS=',' read -r -a orders <<< "$PROBE_ORDERS"
  SUFFIX_LIST=()
  if [[ -n "$ONLY_SUFFIXES" ]]; then
    IFS=',' read -r -a SUFFIX_LIST <<< "$ONLY_SUFFIXES"
    return
  fi
  case "$STAGE" in
    0)
      SUFFIX_LIST=("full_reverse_r1")
      ;;
    1|all)
      for rep in $(seq 1 "$N_REPS"); do
        for order in "${orders[@]}"; do
          order="$(echo "$order" | tr -d '[:space:]')"
          [[ -n "$order" ]] || continue
          SUFFIX_LIST+=("full_${order}_r${rep}")
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
  local mode="$1" order="$2" suffix="$3" probe_seed="$4" attempt="$5"
  local run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
  export MAKO_RESULT_SUFFIX="exp_i_pilot_${suffix}"
  echo "===== RUN mode=$mode order=$order suffix=$suffix seed=$probe_seed attempt=$attempt =====" \
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

order_from_suffix() {
  local suffix="$1"
  if [[ "$suffix" =~ _(causal|random|reverse)_ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
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

link_reuse_pilot
build_suffix_list

echo "PLANNED_SUFFIXES=${SUFFIX_LIST[*]}" | tee -a "$master_log"

TOKENS_SPENT=0
API_CALLS=0
overall_rc=0
budget_stop=0

# Pre-count reused/existing tokens toward spend visibility (not API-new, but tracked).
EXISTING_TOKENS="$(sum_tokens_for_suffixes "${SUFFIX_LIST[@]}")"
echo "EXISTING_TOKENS_IN_SCOPE=$EXISTING_TOKENS (~¥$(est_rmb "$EXISTING_TOKENS"))" | tee -a "$master_log"
echo "0" > "$budget_file"

for suffix in "${SUFFIX_LIST[@]}"; do
  order="$(order_from_suffix "$suffix")"
  rep="$(rep_from_suffix "$suffix")"
  if [[ "$order" == "random" ]]; then
    probe_seed=$((2000 + rep * 31))
  else
    probe_seed=0
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && cell_ok "$suffix"; then
    echo "SKIP_EXISTING $suffix tokens=$(cell_tokens "$suffix")" | tee -a "$master_log"
    continue
  fi

  if [[ "$TOKENS_SPENT" -ge "$TOKEN_BUDGET" ]]; then
    echo "BUDGET_STOP before $suffix spent=$TOKENS_SPENT budget=$TOKEN_BUDGET (~¥$(est_rmb "$TOKENS_SPENT"))" \
      | tee -a "$master_log" | tee "$status_file"
    budget_stop=1
    overall_rc=2
    break
  fi

  ok=0
  max_attempts=$((ATTEMPTS_PER_CELL + MAX_NETWORK_RETRIES))
  for attempt in $(seq 1 "$max_attempts"); do
    # Hard cap: only ATTEMPTS_PER_CELL normal tries; extra attempts only after network fail.
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
    run_one stackelberg "$order" "$suffix" "$probe_seed" "$attempt"
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
      echo "BUDGET_STOP after $suffix spent=$TOKENS_SPENT budget=$TOKEN_BUDGET (~¥$(est_rmb "$TOKENS_SPENT"))" \
        | tee -a "$master_log"
      budget_stop=1
      overall_rc=2
      break
    fi

    if cell_ok "$suffix"; then
      # Stage-0 gate fields
      ep_ok="$(uv run python - "$(result_json "$suffix")" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ep = d.get("episode_payoff") or {}
ok = bool(ep.get("committed_omega")) and ("first_probe_hit" in ep)
print("1" if ok else "0")
PY
)"
      if [[ "$ep_ok" != "1" ]]; then
        echo "CELL_OK_BUT_MISSING_DIAGNOSTICS $suffix" | tee -a "$master_log"
        overall_rc=1
      else
        echo "CELL_OK $suffix attempt=$attempt delta_tokens=$delta" | tee -a "$master_log"
      fi
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
  echo "KILL_METRIC=first_probe_hit"
  echo "HARD_BUDGET_RMB=10"
} > "$done_file"

exit "$overall_rc"
