#!/usr/bin/env bash
# Redesigned Exp-A (evidence-informed): stackelberg × {causal, reverse, random} × N_REPS.
# Uses --omega_source evidence_rank (causal tip from rank; reverse/random ignore tip).
# Cost-gated: TOKEN_BUDGET + SKIP_EXISTING. Do not run Debate here.
#
# STAGE: 0 = ea_causal_r1 + ea_reverse_r1 only; 1|all = full N_REPS grid.
set -uo pipefail
cd "$(dirname "$0")/.."

PROVIDER="${PROVIDER:-DashScope}"
MODEL="${MODEL:-deepseek-v4-flash}"
DATASET="${DATASET:-prob_tslp_ecr_demand}"
PROB="${PROB:-smoke_H4_Omega5}"
INJECT="${INJECT:-me_force_zero_sea}"
MAX_RETRIES="${MAX_RETRIES:-3}"
KNOWLEDGE="${KNOWLEDGE:-progressive}"
OUT_DIR="${OUT_DIR:-results/exp_a_evidence}"
N_REPS="${N_REPS:-3}"
PROBE_ORDERS="${PROBE_ORDERS:-causal,reverse,random}"
OMEGA_SOURCE="${OMEGA_SOURCE:-evidence_rank}"
RANK_METHOD="${RANK_METHOD:-llm_rank}"
ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-1}"
MAX_NETWORK_RETRIES="${MAX_NETWORK_RETRIES:-1}"
SLEEP_BETWEEN_ATTEMPTS_S="${SLEEP_BETWEEN_ATTEMPTS_S:-45}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# deepseek-v4-flash is cheaper; ~¥1.5 soft cap for n=3×3 (~9 cells)
TOKEN_BUDGET="${TOKEN_BUDGET:-2500000}"
STAGE="${STAGE:-all}"
ONLY_SUFFIXES="${ONLY_SUFFIXES:-}"

mkdir -p "$OUT_DIR" logs
ts="$(date +%Y%m%d_%H%M%S)"
master_log="${MASTER_LOG:-logs/exp_a_evidence_${ts}.log}"
done_file="$OUT_DIR/DONE"
status_file="$OUT_DIR/STATUS.txt"
budget_file="$OUT_DIR/TOKEN_SPEND.txt"
rm -f "$done_file"
{
  echo "STARTED=$(date -Iseconds)"
  echo "PROTOCOL=evidence_informed_exp_a"
  echo "STAGE=$STAGE INJECT=$INJECT N_REPS=$N_REPS"
  echo "PROVIDER=$PROVIDER MODEL=$MODEL"
  echo "PROBE_ORDERS=$PROBE_ORDERS"
  echo "OMEGA_SOURCE=$OMEGA_SOURCE RANK_METHOD=$RANK_METHOD"
  echo "SKIP_EXISTING=$SKIP_EXISTING TOKEN_BUDGET=$TOKEN_BUDGET"
  echo "MASTER_LOG=$master_log"
  echo "PRIMARY_METRICS=first_probe_hit,verified_attribution_hit"
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
# Evidence-informed: prefer omega_source + verified fields when present.
if ep.get("committed_omega") is None and tokens <= 0:
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
  local orders=() order rep
  IFS=',' read -r -a orders <<< "$PROBE_ORDERS"
  SUFFIX_LIST=()
  if [[ -n "$ONLY_SUFFIXES" ]]; then
    IFS=',' read -r -a SUFFIX_LIST <<< "$ONLY_SUFFIXES"
    return
  fi
  case "$STAGE" in
    0)
      SUFFIX_LIST=("ea_causal_r1" "ea_reverse_r1")
      ;;
    1|all)
      for rep in $(seq 1 "$N_REPS"); do
        for order in "${orders[@]}"; do
          order="$(echo "$order" | tr -d '[:space:]')"
          [[ -n "$order" ]] || continue
          SUFFIX_LIST+=("ea_${order}_r${rep}")
        done
      done
      ;;
    *)
      echo "ERROR: unknown STAGE=$STAGE (use 0|1|all)" | tee -a "$master_log"
      exit 2
      ;;
  esac
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

run_one() {
  local mode="$1" order="$2" suffix="$3" probe_seed="$4" attempt="$5"
  local run_log="$OUT_DIR/run_${suffix}_a${attempt}.log"
  export MAKO_RESULT_SUFFIX="exp_i_pilot_${suffix}"
  echo "===== RUN mode=$mode order=$order suffix=$suffix omega_source=$OMEGA_SOURCE attempt=$attempt =====" \
    | tee -a "$master_log"
  echo "RUNNING=$suffix attempt=$attempt at=$(date -Iseconds)" > "$status_file"

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
    --omega_source "$OMEGA_SOURCE" \
    --rank_method "$RANK_METHOD" \
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
    tail -n 80 "$run_log"
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
  order="$(order_from_suffix "$suffix")"
  rep="$(rep_from_suffix "$suffix")"
  if [[ "$order" == "random" ]]; then
    probe_seed=$((3000 + rep * 37))
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
      echo "BUDGET_STOP after $suffix spent=$TOKENS_SPENT (~¥$(est_rmb "$TOKENS_SPENT"))" \
        | tee -a "$master_log"
      budget_stop=1
      overall_rc=2
      break
    fi

    if cell_ok "$suffix"; then
      ep_ok="$(uv run python - "$(result_json "$suffix")" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ep = d.get("episode_payoff") or {}
ok = bool(ep.get("committed_omega")) and ("first_probe_hit" in ep) and ("verified_attribution_hit" in ep)
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
uv run python scripts/summarize_exp_a_evidence.py \
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
  echo "OMEGA_SOURCE=$OMEGA_SOURCE"
} | tee -a "$master_log" | tee "$status_file"

{
  echo "FINISHED=$(date -Iseconds)"
  echo "OVERALL_RC=$overall_rc"
  echo "BUDGET_STOP=$budget_stop"
  echo "PROTOCOL=evidence_informed_exp_a"
  echo "INJECT=$INJECT"
  echo "N_REPS=$N_REPS"
  echo "STAGE=$STAGE"
  echo "OMEGA_SOURCE=$OMEGA_SOURCE"
  echo "RANK_METHOD=$RANK_METHOD"
  echo "TOKENS_SPENT=$TOKENS_SPENT"
  echo "EST_RMB=$(est_rmb "$TOKENS_SPENT")"
  echo "API_CALLS=$API_CALLS"
  echo "SUMMARY=$OUT_DIR/summary.md"
  echo "MASTER_LOG=$master_log"
  echo "HARD_BUDGET_RMB=10"
} > "$done_file"

exit "$overall_rc"
