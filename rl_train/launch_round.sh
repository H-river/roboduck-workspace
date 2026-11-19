#!/usr/bin/env bash
# Train one rl_train config for several seeds, one after another, in a memory-capped scope.
#   ./launch_round.sh walk_v2_r1 1 2 [extra mjlab train args...]
set -euo pipefail
cd "$(dirname "$0")/runs"
cfg=$1; shift
seeds=(); while [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; do seeds+=("$1"); shift; done
for s in "${seeds[@]}"; do
  echo "[$(date +%F' '%T)] start $cfg seed $s"
  systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 \
    env -u PYTHONPATH ../../mjlab_microduck/.venv/bin/python ../train.py "RoboDuck-$cfg" \
      --agent.seed "$s" --agent.run-name "${cfg}_s${s}" "$@" > "train_${cfg}_s${s}.log" 2>&1 || echo "seed $s exit $?"
  echo "[$(date +%F' '%T)] end $cfg seed $s"
done
