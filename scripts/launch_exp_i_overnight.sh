#!/usr/bin/env bash
# Fully detach overnight Exp-I re-pilot from the current terminal/session.
# Survives SSH/Cursor disconnects via setsid + nohup.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs results/exp_i_pilot_v2
ts="$(date +%Y%m%d_%H%M%S)"
out="logs/exp_i_pilot_v2_nohup_${ts}.out"
pidfile="logs/exp_i_pilot_v2_LATEST.pid"
export MASTER_LOG="logs/exp_i_pilot_v2_${ts}.log"
export OUT_DIR="results/exp_i_pilot_v2"
export INJECT="${INJECT:-me_force_zero_sea}"
export ATTEMPTS_PER_CELL="${ATTEMPTS_PER_CELL:-3}"

# setsid: new session, immune to SIGHUP from parent tty.
setsid nohup bash scripts/run_exp_i_pilot_v2_overnight.sh \
  < /dev/null > "$out" 2>&1 &
pid=$!
echo "$pid" > "$pidfile"
# Give child a moment to re-parent / start.
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: overnight job failed to stay alive; see $out" >&2
  exit 1
fi

cat <<EOF
Overnight Exp-I re-pilot launched (detached).
  PID:        $pid  (also in $pidfile)
  Inject:     $INJECT
  NoHUP out:  $out
  Master log: $MASTER_LOG
  Status:     $OUT_DIR/STATUS.txt
  Done file:  $OUT_DIR/DONE   (appears when finished)
  Summary:    $OUT_DIR/summary.md

Morning check:
  cat results/exp_i_pilot_v2/DONE
  cat results/exp_i_pilot_v2/summary.md
  tail -n 50 $MASTER_LOG
EOF
