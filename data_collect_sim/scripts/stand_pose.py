"""Measure a walk card's standing pose (zero command) in MuJoCo C and print it relative to HOME.

The output is the `start.joint_offsets` block for rl_train/configs/kick_v2_*.yaml: kick v2 starts
from HOME + U(0, 1) x (standing pose - HOME), so it covers what the walk policy actually stands in.

    uv run python scripts/stand_pose.py --card configs/policies/candidates/<name>.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckrt import evaluation  # noqa: E402
from duckrt.messages import Mode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True)
    ap.add_argument("--stand-s", type=float, nargs="*", default=[2.0, 5.0, 10.0], help="standing times averaged over")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--after-walk", action="store_true", help="walk 2 s at vx 0.15 before standing (also measured)")
    args = ap.parse_args()

    rt = evaluation.make_rt(args.card, stand_card=args.card)
    card = rt.policies["stand"].card
    names, home = card.joint_names, np.asarray(card.default_q)
    poses = []
    for sd in range(args.seeds):
        for walk_first in (False, True) if args.after_walk else (False,):
            for t in args.stand_s:
                rt.reset(seed=sd)
                if walk_first:
                    evaluation.hold(rt, 2.0, Mode.WALK_FLAT, (0.15, 0.0, 0.0))
                evaluation.hold(rt, t, Mode.STAND)
                poses.append(rt.backend.joint_pos())
    poses = np.array(poses)
    mean, spread = poses.mean(0) - home, poses.std(0)
    print(f"# {Path(args.card).stem}: standing pose - HOME (rad), mean of {len(poses)} samples; spread = std")
    print("  joint_offsets:")
    for n, d, s in zip(names, mean, spread):
        print(f"    {n}: {d:+.4f}    # spread {s:.4f}")


if __name__ == "__main__":
    main()
