"""Demo videos of a walk (+ kick) policy in the acceptance runtime (accept.py only records failures).

    uv run python scripts/demo_videos.py [--walk walk_v2_r2_s1_it2750] [--kick kick_v2_r1_stand2750_s1_it3999]
        [--out outputs/demo_videos/<walk>/]

Writes walk.mp4 (stand, forward, turns, sideways, backward, stop), kick_handoff.mp4
(walk -> stand -> kick -> stand -> walk, a few cycles) and ramp_5_up.mp4 / ramp_5_down.mp4. The command
sequence, cycle count and ramp angle live in configs/demo_videos.yaml; fps in configs/acceptance.yaml `videos:`;
the camera is configs/camera.yaml chase_cam.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckrt import evaluation as ev  # noqa: E402
from duckrt.config import CONFIG_DIR, load_yaml  # noqa: E402
from duckrt.messages import Mode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walk", default="walk_v2_r2_s1_it2750")
    ap.add_argument("--kick", default="kick_v2_r1_stand2750_s1_it3999", help="'none' to skip the kick video")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = Path(args.out or f"outputs/demo_videos/{args.walk}")
    out.mkdir(parents=True, exist_ok=True)
    acc = ev.acceptance()
    demo = load_yaml(CONFIG_DIR / "demo_videos.yaml")
    fps = acc["videos"]["fps"]
    cam = load_yaml(CONFIG_DIR / "camera.yaml")["chase_cam"]

    with ev.recording(out / "walk.mp4", fps, cam):
        rt = ev.make_rt(args.walk)
        rt.reset(seed=0)
        for seg in demo["walk_sequence"]:
            mode = Mode.STAND if seg["cmd"] == "stand" else Mode.WALK_FLAT
            ev.hold(rt, seg["s"], mode, (0.0, 0.0, 0.0) if seg["cmd"] == "stand" else tuple(seg["cmd"]))
    print(f"[demo] {out / 'walk.mp4'} (fell: {ev.fallen(rt)})", flush=True)

    if args.kick != "none":
        a = copy.deepcopy(acc)
        a["kick"]["handoff"]["cycles"] = demo["kick_cycles"]
        with ev.recording(out / "kick_handoff.mp4", fps, cam):
            r = ev.kick_handoff(args.walk, args.kick, a)
        print(f"[demo] {out / 'kick_handoff.mp4'} (kicks hit {r['kick_success']}/{r['cycles']}, falls {r['falls']})", flush=True)

    ang = demo["ramp_deg"]
    for d in ("up", "down"):
        with ev.recording(out / f"ramp_{ang:g}_{d}.mp4", fps, cam):
            r = ev.ramps(args.walk, acc, angles=[ang], only=[(float(ang), d, 0)])
        print(f"[demo] {out / f'ramp_{ang:g}_{d}.mp4'} (success: {r['results'][f'{ang:g}_{d}']['success']}/1)", flush=True)


if __name__ == "__main__":
    main()
