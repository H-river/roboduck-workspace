"""Build a layout, check it, and render a top view + the head-camera view at the start.

Checks: module schema / ranges, modules inside the arena and not overlapping each other or
the robot start (at build time); then 5 s of STAND: no NaN, no deep penetration, no
explosion, the robot stays up and loose objects stay put.

    uv run python scripts/build_scene.py --layout demo_v1
    uv run python scripts/build_scene.py --layout ramp_test --set ramp.angle_deg=7
    uv run python scripts/build_scene.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from duckrt.camera import HEAD_CAM, TOP_CAM  # noqa: E402
from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml, output_dir  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.scene.builder import load_layout, parse_overrides  # noqa: E402
from duckrt.scene.modules import LAYOUT_DIR  # noqa: E402
from duckrt.testing import md_table  # noqa: E402

STAND_S = 5.0
MAX_PENETRATION_M = 0.005
MAX_SPEED = 5.0           # any dof speed above this = explosion
OBJECT_DRIFT_M = 0.01     # loose objects (ball) must not move by themselves


def check_layout(name: str, overrides: dict, sim_cfg: dict, cam_cfg: dict, out: Path) -> dict:
    layout = load_layout(name, overrides)
    rt = SimRuntime(sim_cfg, layout=layout, camera_cfg=cam_cfg)
    b = rt.backend
    mods0 = {s["id"]: s for s in b.module_states()}
    tag = name + "".join(f"_{k}.{p}={v}" for k, d in overrides.items() for p, v in d.items())
    hc, tc = cam_cfg["head_camera"], cam_cfg["top_cam"]
    imageio.imwrite(out / f"{tag}_top.png", b.render(TOP_CAM, tc["width"], tc["height"]))
    imageio.imwrite(out / f"{tag}_head_start.png", b.render(HEAD_CAM, hc["width"], hc["height"]))
    max_pen, max_speed, nan = 0.0, 0.0, False
    for _ in range(int(STAND_S * 50)):
        if rt.ticks % 5 == 0:
            rt.send(Mode.STAND)
        rt.tick()
        d = b.data
        if d.ncon:
            max_pen = max(max_pen, float(-min(d.contact[i].dist for i in range(d.ncon))))
        max_speed = max(max_speed, float(np.abs(d.qvel).max()))
        nan |= b.has_nan()
    imageio.imwrite(out / f"{tag}_head_after_stand.png", b.render(HEAD_CAM, hc["width"], hc["height"]))
    drift = {}
    for s in b.module_states():
        if s["type"] == "ball":
            drift[s["id"]] = float(np.linalg.norm(s["pos"] - mods0[s["id"]]["pos"]))
    ok = (not nan and max_pen < MAX_PENETRATION_M and max_speed < MAX_SPEED and not rt.switcher.fall_flag
          and all(v < OBJECT_DRIFT_M for v in drift.values()))
    res = {"layout": name, "overrides": overrides, "ok": ok, "nan": nan, "max_penetration_m": max_pen,
           "max_dof_speed": max_speed, "fell": rt.switcher.fall_flag, "object_drift_m": drift,
           "robot_start": {"pos": b.world.robot_start.pos.tolist(), "yaw_deg": float(np.degrees(b.world.robot_start.yaw)),
                           "ground_z": b.ground_height(*b.world.robot_start.pos[:2])},
           "modules": [{"id": s["id"], "type": s["type"], "definition": s["definition"],
                        "pos": np.round(s["pos"], 3).tolist(), "yaw_deg": round(float(np.degrees(s["yaw"])), 1),
                        "size": {k: v for k, v in s["params"].items() if k.endswith("_m") or k in ("angle_deg",)},
                        "rgba": s["rgba"], "sites": {k: np.round(v, 3).tolist() for k, v in s["sites"].items()}}
                       for s in mods0.values()],
           "images": [str(out / f"{tag}_{x}.png") for x in ("top", "head_start", "head_after_stand")]}
    print(f"[{'ok' if ok else 'FAIL'}] {tag}: penetration {max_pen * 1000:.2f} mm, max dof speed {max_speed:.2f}, "
          f"nan {nan}, fell {rt.switcher.fall_flag}, drift {drift}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="demo_v1")
    ap.add_argument("--set", nargs="*", default=[], help="overrides <module_id>.<param>=<value>")
    ap.add_argument("--all", action="store_true", help="every layout in assets/layouts")
    args = ap.parse_args()
    sim_cfg, cam_cfg = load_sim_cfg(), load_yaml(CONFIG_DIR / "camera.yaml")
    out = output_dir("scenes")
    names = sorted(p.stem for p in LAYOUT_DIR.glob("*.yaml")) if args.all else [args.layout]
    results = [check_layout(n, parse_overrides(args.set) if not args.all else {}, sim_cfg, cam_cfg, out) for n in names]
    for r in results:
        print(f"\n{r['layout']} modules:")
        print(md_table(["id", "type", "definition", "pos (m)", "yaw °", "sizes"],
                       [[m["id"], m["type"], m["definition"], m["pos"], m["yaw_deg"],
                         ", ".join(f"{k}={v}" for k, v in m["size"].items())] for m in r["modules"]]))
    (out / ("all" if args.all else names[0]) ).with_suffix(".json").write_text(json.dumps(results, indent=2, default=float))
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
