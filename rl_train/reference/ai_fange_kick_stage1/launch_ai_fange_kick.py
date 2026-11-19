#!/usr/bin/env python3
"""Register the AI-FanGe ball-kick task and hand over to mjlab's train entry point.
Run with the AI-FanGe venv (/home/jiamo/microduck_ws/ai_fange_mjlab/.venv):
  launch_ai_fange_kick.py --env.scene.num-envs 1024 --agent.max-iterations 3000 ...
The task id is supplied here; do not pass one."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import os
    frac = float(os.environ.get("AF_KICK_GPU_FRACTION", "0"))
    if frac > 0:
        # Cap PyTorch's share of the GPU so the VLA jobs always have room
        # (Warp's simulation buffers are sized by --env.scene.num-envs instead).
        import torch
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    import mjlab_microduck.tasks  # noqa: F401  (AI-FanGe's own tasks)
    from af_kick_cfg import register
    task_id = register()
    from mjlab.scripts.train import main as mjlab_train_main
    sys.argv = [sys.argv[0], task_id] + sys.argv[1:]
    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
