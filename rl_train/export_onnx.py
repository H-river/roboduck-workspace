"""Export a checkpoint's actor to ONNX (runs in mjlab_microduck/.venv).

Adapted from mjlab_microduck/src/mjlab_microduck/scripts/export_onnx.py (workspace copy, no git
commit available; file sha256 in rl_train/NOTES.md): adds --task (so rl_train's v2 tasks work)
and --out. Same exporter, same metadata.

    ../mjlab_microduck/.venv/bin/python export_onnx.py --task Mjlab-Velocity-Microduck-v2 \\
        --checkpoint runs/<run>/model_3000.pt --out /path/to/model_3000.onnx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rl_tasks  # noqa: E402,F401  registers the rl_train tasks
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-Velocity-Microduck")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=args.device))
    cfg = asdict(agent_cfg)
    cfg.get("algorithm", {}).pop("symmetry_cfg", None)  # not needed to load the actor
    runner = load_runner_cls(args.task)(env, cfg, device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    with tempfile.TemporaryDirectory() as tmp:
        runner.export_policy_to_onnx(tmp, "policy.onnx")
        onnx_path = Path(tmp) / "policy.onnx"
        attach_metadata_to_onnx(str(onnx_path), get_base_metadata(env.unwrapped, run_path=args.checkpoint.parent.name))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(onnx_path), args.out)
    print(f"[export] {args.checkpoint} -> {args.out}")
    env.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
