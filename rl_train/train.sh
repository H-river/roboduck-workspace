#!/usr/bin/env bash
# Train one rl_train config, export ONNX and run the acceptance on checkpoints as they appear.
#
#   rl_train/train.sh <config> [extra mjlab train args...]        # <config> = file stem in rl_train/configs/
#
#   rl_train/train.sh walk_v2_r3                                  # 4096 envs x 3000 iterations, seed 1
#   SEED=2 rl_train/train.sh walk_v2_r3
#   NUM_ENVS=512 ITERATIONS=20 EXPORT_EVERY=10 rl_train/train.sh walk_v2_r3     # smoke test
#   DETACH=1 rl_train/train.sh kick_v2_r1_stand2750               # keeps running after the terminal closes
#
# Environment variables (default):
#   SEED (1)            --agent.seed
#   NUM_ENVS ()         empty = train.num_envs of the config (walk 4096, kick 2048)
#   ITERATIONS ()       empty = train.max_iterations of the config
#   RUN_NAME (<config>_s<SEED>)   run dir = rl_train/runs/logs/rsl_rl/<experiment>/<date>_<RUN_NAME>/
#   EXPORT_EVERY (500)  export <run>/onnx/model_<N>.onnx for every checkpoint N that is a multiple of this, and the last one
#   ACCEPT (quick)      quick | full | none: scripts/accept.py on each exported ONNX -> <run>/acceptance/model_<N>/
#   ACCEPT_FINAL (full) quick | full | none: on the last checkpoint after training ends
#   KICK_WALK (walk_v2_r2_s1_it2750)  walk card that walks / stands around the kick when accepting a kick run
#   LOGGER (tensorboard) or wandb (needs WANDB_API_KEY)
#   MEM_MAX (8G)        RAM cap of the training process (systemd user scope; skipped where systemd is missing, e.g. RunPod)
#   POLL_S (60)         how often to look for new checkpoints
#   DETACH (0)          1 = re-run under setsid + nohup; driver log rl_train/runs/<RUN_NAME>.driver.log
#
# Logs: rl_train/runs/train_<RUN_NAME>.log (mjlab), rl_train/runs/<RUN_NAME>.driver.log (this script, when detached).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$(dirname "$HERE")"
CFG="${1:?usage: rl_train/train.sh <config> [extra mjlab train args]  (configs: $(cd "$HERE/configs" && ls *.yaml | sed 's/\.yaml$//' | tr '\n' ' '))}"
shift
[[ -f "$HERE/configs/$CFG.yaml" ]] || { echo "no such config: rl_train/configs/$CFG.yaml" >&2; exit 2; }

SEED="${SEED:-1}"
RUN_NAME="${RUN_NAME:-${CFG}_s${SEED}}"
EXPORT_EVERY="${EXPORT_EVERY:-500}"
ACCEPT="${ACCEPT:-quick}"
ACCEPT_FINAL="${ACCEPT_FINAL:-full}"
KICK_WALK="${KICK_WALK:-walk_v2_r2_s1_it2750}"
LOGGER="${LOGGER:-tensorboard}"
MEM_MAX="${MEM_MAX:-8G}"
POLL_S="${POLL_S:-60}"
RUNS="$HERE/runs"
MJLAB_PY="$WS/mjlab_microduck/.venv/bin/python"
SIM_PY="$WS/data_collect_sim/.venv/bin/python"
mkdir -p "$RUNS"

if [[ "${DETACH:-0}" == "1" ]]; then
  DETACH=0 setsid nohup "$0" "$CFG" "$@" > "$RUNS/$RUN_NAME.driver.log" 2>&1 < /dev/null &
  echo "started $RUN_NAME in the background (pid $!); driver log $RUNS/$RUN_NAME.driver.log, training log $RUNS/train_$RUN_NAME.log"
  exit 0
fi

log() { echo "[$(date '+%F %T')] $*"; }
[[ -x "$MJLAB_PY" ]] || { echo "missing $MJLAB_PY: run scripts/setup_workspace.sh first" >&2; exit 2; }

args=("RoboDuck-$CFG" --agent.seed "$SEED" --agent.run-name "$RUN_NAME" --agent.logger "$LOGGER")
[[ -n "${NUM_ENVS:-}" ]] && args+=(--env.scene.num-envs "$NUM_ENVS")
[[ -n "${ITERATIONS:-}" ]] && args+=(--agent.max-iterations "$ITERATIONS")
args+=("$@")

scope=()
if command -v systemd-run > /dev/null && systemd-run --user --scope --quiet true 2> /dev/null; then
  scope=(systemd-run --user --scope --quiet -p "MemoryMax=$MEM_MAX" -p MemorySwapMax=0)
else
  log "systemd user scope not available: training runs without a RAM cap"
fi

start=$(date +%s)
log "train RoboDuck-$CFG ${args[*]:1} -> $RUNS/train_$RUN_NAME.log"
( cd "$RUNS" && exec "${scope[@]}" env -u PYTHONPATH MUJOCO_GL=egl "$MJLAB_PY" ../train.py "${args[@]}" ) > "$RUNS/train_$RUN_NAME.log" 2>&1 &
train_pid=$!

run_dir() {  # newest run dir of this run name created after the start
  local d
  d=$(ls -dt "$RUNS"/logs/rsl_rl/*/*_"$RUN_NAME" 2> /dev/null | head -1 || true)
  [[ -n "$d" && $(stat -c %Y "$d") -ge $((start - 5)) ]] && echo "$d" || true
}

accept_one() {  # <run dir> <iteration> <quick|full>
  local run=$1 it=$2 mode=$3 onnx="$1/onnx/model_$2.onnx" out="$1/acceptance/model_$2" a=()
  [[ "$mode" == none || ! -x "$SIM_PY" ]] && return 0
  [[ "$mode" == quick ]] && a+=(--quick)
  if [[ "$CFG" == kick* ]]; then
    a+=(--walk "$KICK_WALK" --kick "$onnx" --kick-config "$run")
  else
    a+=(--walk "$onnx" --config "$run")
  fi
  log "accept ($mode) model_$it -> $out"
  if ( cd "$WS/data_collect_sim" && env -u PYTHONPATH "$SIM_PY" scripts/accept.py "${a[@]}" --rl-config "$CFG" --name "$(basename "$run")_it$it" --out "$out" \
         > "$out.log" 2>&1 ); then
    log "  model_$it: PASS"
  else
    log "  model_$it: $(head -1 "$out/report.md" 2> /dev/null || echo "accept failed, see $out.log")"
  fi
}

process() {  # export (and accept) every due checkpoint not done yet; $1 = final
  local run ck it last
  run=$(run_dir)
  [[ -z "$run" ]] && return 0
  mkdir -p "$run/onnx" "$run/acceptance"
  last=$(ls "$run"/model_*.pt 2> /dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
  for ck in $(ls "$run"/model_*.pt 2> /dev/null | sort -V); do
    it=$(basename "$ck" .pt); it=${it#model_}
    [[ -f "$run/onnx/model_$it.onnx" ]] && continue
    if (( it % EXPORT_EVERY == 0 )) || [[ "$1" == final && "$it" == "$last" ]]; then
      log "export model_$it"
      ( cd "$HERE" && env -u PYTHONPATH MUJOCO_GL=egl "$MJLAB_PY" export_onnx.py --task "RoboDuck-$CFG" --checkpoint "$ck" \
          --out "$run/onnx/model_$it.onnx" > "$run/onnx/model_$it.log" 2>&1 ) || { log "  export failed: $run/onnx/model_$it.log"; continue; }
      [[ "$1" == final && "$it" == "$last" ]] || accept_one "$run" "$it" "$ACCEPT"
    fi
  done
  if [[ "$1" == final && -n "$last" && -f "$run/onnx/model_$last.onnx" ]]; then
    accept_one "$run" "$last" "$ACCEPT_FINAL"
  fi
}

while kill -0 "$train_pid" 2> /dev/null; do
  sleep "$POLL_S"
  process running
done
rc=0; wait "$train_pid" || rc=$?
log "training exited with code $rc ($(( ($(date +%s) - start) / 60 )) min)"
process final
run=$(run_dir)
[[ -n "$run" ]] && log "run dir: $run (ONNX in onnx/, acceptance reports in acceptance/)"
exit "$rc"
