#!/usr/bin/env bash
# One-time setup after `git clone`: submodules, the two Python environments (uv, locked), self-check.
#
#   scripts/setup_workspace.sh            # from the workspace root; safe to re-run
#
# - mjlab_microduck/.venv   training (mjlab, torch with CUDA 13 -> NVIDIA driver >= 580), also used by rl_train/
# - data_collect_sim/.venv  acceptance / bare-MuJoCo runtime (CPU)
# The submodules already carry our changes as commits (branch roboduck-main); patches/ documents them
# against the clean upstream and patches/check_patches.sh verifies that. Nothing to apply here.
set -euo pipefail
cd "$(dirname "$0")/.."
WS=$(pwd)
t0=$(date +%s)
step() { echo "== $*"; }

command -v uv > /dev/null || { echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh  (pip cannot install this: BAM and mjlab need conflicting protobuf pins that only uv's overrides resolve)" >&2; exit 1; }
# A ROS install on PYTHONPATH (e.g. /opt/ros/jazzy) leaks packages into the venvs and breaks pytest.
unset PYTHONPATH

step "submodules"
git submodule update --init --recursive
for s in mjlab_microduck microduck; do echo "   $s @ $(git -C "$s" rev-parse --short HEAD)"; done

step "mjlab_microduck/.venv (uv sync --locked; first run downloads ~5 GB: torch + CUDA libs)"
(cd mjlab_microduck && uv sync --locked)
step "data_collect_sim/.venv (uv sync --locked)"
(cd data_collect_sim && uv sync --locked --group dev)

step "self-check"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mjlab_microduck/.venv/bin/python - <<'EOF'
import sys
import torch
sys.path.insert(0, "rl_train")
import rl_tasks  # noqa: F401  (registers RoboDuck-* from rl_train/configs)
from mjlab.tasks.registry import list_tasks
tasks = [t for t in list_tasks() if t.startswith(("RoboDuck-", "Mjlab-Velocity-Microduck"))]
print(f"   mjlab: torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -> training needs an NVIDIA GPU with driver >= 580"))
print(f"   tasks: {' '.join(tasks)}")
assert "RoboDuck-walk_v2_r3" in tasks
EOF
(cd data_collect_sim && .venv/bin/python - <<'EOF'
import numpy as np
from duckrt import evaluation
from duckrt.messages import Mode
rt = evaluation.make_rt("walk_v2_r2_s1_it2750")
p = rt.policies["walk_flat"]
print(f"   runtime: card {p.card.name}, ONNX {p.card.onnx.name} loaded")
rt.reset(seed=0)
evaluation.hold(rt, 2.0, Mode.STAND)
evaluation.hold(rt, 2.0, Mode.WALK_FLAT, (0.15, 0.0, 0.0))
x = rt.backend.base_pos()
assert not evaluation.fallen(rt), "robot fell during the self-check"
print(f"   runtime: stood 2 s and walked 2 s at 0.15 m/s -> x = {x[0]:+.3f} m, upright")
EOF
)
echo "== done in $(( $(date +%s) - t0 )) s. Next: HANDOFF.md, section 2 (accept.py --quick on the baseline)."
