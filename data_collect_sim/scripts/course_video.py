"""Course video: the walk policy driven through a layout (ramp, flags, around the zone / obstacle), then
lining up behind the ball and kicking it at the goal. Three views side by side: chase camera, top
camera (with the trail), head camera.

    uv run python scripts/course_video.py [--walk walk_v2_r2_s1_it2750] [--kick kick_v2_r1_stand2750_s1_it3999]
        [--out outputs/demo_videos/<walk>/course_<layout>.mp4] [--seed 0]

Route, controller gains and views: configs/demo_videos.yaml `course:`. The driver uses privileged
poses (robot, ball, goal) like the oracle would; it is a demo, not a test, and has no gates.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from duckrt import evaluation as ev  # noqa: E402
from duckrt.camera import HEAD_CAM, TOP_CAM  # noqa: E402
from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.scene.builder import load_layout  # noqa: E402
from duckrt.testing import VideoWriter  # noqa: E402


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Course:
    def __init__(self, args, cfg: dict):
        self.cfg = cfg
        cam_cfg = load_yaml(CONFIG_DIR / "camera.yaml")
        self.layout = load_layout(cfg["layout"])
        roles = {"walk_flat": args.walk, "stand": args.walk, "kick": None if args.kick == "none" else args.kick}
        self.rt = SimRuntime(load_sim_cfg(), roles=roles, layout=self.layout, camera_cfg=cam_cfg, seed=args.seed)
        self.rt.reset(seed=args.seed)
        self.b = self.rt.backend
        v = cfg["view"]
        self.v = v
        self.writer = VideoWriter(args.out, v["fps"])
        self.next_frame = 0.0
        self.label = ""
        self.cmd = (0.0, 0.0, 0.0)
        self.mode = Mode.STAND
        self.trail: list[tuple[float, float]] = []
        # top camera: world (x, y) -> pixel, from its pose and fovy
        m = self.b.model
        cid = m.camera(TOP_CAM).id
        self.top_z = float(m.cam_pos[cid][2])
        self.top_c = m.cam_pos[cid][:2].copy()
        self.top_f = (v["top"]["height"] / 2) / math.tan(math.radians(float(m.cam_fovy[cid])) / 2)
        self.fell = False
        self.goal_watch: dict | None = None  # set during the kick: ball crossing the goal line

    # -- video ----------------------------------------------------------------------------
    def top_px(self, xy) -> tuple[float, float]:
        t = self.v["top"]
        s = self.top_f / self.top_z  # ground plane z = 0
        return t["width"] / 2 + (xy[0] - self.top_c[0]) * s, t["height"] / 2 - (xy[1] - self.top_c[1]) * s

    def frame(self) -> None:
        v, b = self.v, self.b
        c = v["chase"]
        chase = Image.fromarray(b.render_chase(c["width"], c["height"], c["distance"], c["azimuth"], c["elevation"]))
        top = Image.fromarray(b.render(TOP_CAM, v["top"]["width"], v["top"]["height"]))
        head = Image.fromarray(b.render(HEAD_CAM, v["head"]["width"], v["head"]["height"]))
        d = ImageDraw.Draw(top)
        if len(self.trail) > 1:
            d.line([self.top_px(p) for p in self.trail], fill=(255, 220, 0), width=2)
        W = c["width"] + v["top"]["width"]
        H = max(c["height"], v["top"]["height"] + v["head"]["height"])
        img = Image.new("RGB", (W, H), (20, 20, 24))
        img.paste(chase, (0, 0))
        img.paste(top, (c["width"], 0))
        hx = c["width"] + (v["top"]["width"] - v["head"]["width"]) // 2
        img.paste(head, (hx, v["top"]["height"]))
        d = ImageDraw.Draw(img)
        d.text((10, 8), f"t {self.rt.t:5.1f} s   {self.mode.name:<9}  cmd vx {self.cmd[0]:+.2f} vy {self.cmd[1]:+.2f} wz {self.cmd[2]:+.2f}",
               fill=(255, 255, 255))
        d.text((10, 24), self.label, fill=(255, 220, 0))
        d.text((c["width"] + 6, 4), "top camera", fill=(255, 255, 255))
        d.text((hx + 4, v["top"]["height"] + 4), "head camera", fill=(255, 255, 255))
        self.writer.add(np.asarray(img))

    def step(self) -> None:
        """One control tick; records a frame at the video rate and keeps the trail."""
        self.rt.tick()
        p = self.b.base_pos()
        if not self.trail or math.dist(self.trail[-1], p[:2]) > 0.02:
            self.trail.append((float(p[0]), float(p[1])))
        if self.rt.t >= self.next_frame - 1e-9:
            self.frame()
            self.next_frame += 1.0 / self.v["fps"]
        self.fell |= ev.fallen(self.rt)
        gw = self.goal_watch
        if gw is not None and not gw["scored"]:
            # goal frame: +x into the goal; scored once the whole ball is past the line between the posts
            bp = self.b.module_state(gw["ball"])["pos"]
            dx, dy = bp[0] - gw["pos"][0], bp[1] - gw["pos"][1]
            c, s = math.cos(gw["yaw"]), math.sin(gw["yaw"])
            gx, gy = c * dx + s * dy, -s * dx + c * dy
            gw["scored"] = gx > gw["r"] and abs(gy) < gw["half_w"]
            if gw["scored"]:
                self.label = "GOAL!"

    def drive(self, seconds: float, fn) -> bool:
        """fn() -> (mode, cmd) or None (done), polled at the upper-layer rate. False on timeout / fall."""
        t_end = self.rt.t + seconds
        next_cmd = self.rt.t
        while self.rt.t < t_end:
            if self.rt.t >= next_cmd - 1e-9:
                r = fn()
                if r is None:
                    return True
                self.mode, self.cmd = r
                self.rt.send(*r)
                next_cmd += self.rt.upper_period
            self.step()
            if self.fell:
                return False
        return False

    # -- behaviours -----------------------------------------------------------------------
    def goto(self, xy) -> bool:
        f = self.cfg["follow"]
        xy = np.asarray(xy, float)

        def fn():
            p, yaw = self.b.base_pos()[:2], self.b.base_yaw()
            d = xy - p
            if np.linalg.norm(d) < f["arrive_m"]:
                return None
            bearing = wrap(math.atan2(d[1], d[0]) - yaw)
            wz = float(np.clip(f["k_yaw"] * bearing, -f["wz_max"], f["wz_max"]))
            vx = f["vx_max"] * max(0.0, math.cos(bearing)) if abs(math.degrees(bearing)) < f["slow_bearing_deg"] else 0.0
            return Mode.WALK_FLAT, (vx, 0.0, wz)

        return self.drive(f["timeout_s"], fn)

    def stand(self, seconds: float) -> None:
        self.drive(seconds, lambda: (Mode.STAND, (0.0, 0.0, 0.0)))

    def kick_at_goal(self) -> dict:
        k = self.cfg["kick"]
        acc = ev.acceptance()
        card = self.rt.policies["kick"].card
        foot_y = card.task.get("foot_line_y", card.task["spawn"]["offset_xy"][1] if "spawn" in card.task
                               else acc["kick"]["card_defaults"]["foot_line_y"])
        kx = float(np.mean(acc["kick"]["window"]["x"]))
        ball = self.b.module_state(k["ball"])["pos"][:2]
        goal = self.b.module_state(k["goal"])["pos"][:2]
        yaw_k = math.atan2(goal[1] - ball[1], goal[0] - ball[0])
        R = np.array([[math.cos(yaw_k), -math.sin(yaw_k)], [math.sin(yaw_k), math.cos(yaw_k)]])
        pose = ball - R @ np.array([kx, foot_y])
        u = np.array([math.cos(yaw_k), math.sin(yaw_k)])
        self.label = "kick: approach behind the ball"
        if not self.goto(pose - k["approach_back_m"] * u):
            return {"approach": False}
        self.label = "kick: line up (ball in the kick window)"
        g, mx = k["align_gain"], k["align_max"]

        def align():
            e = ev.ball_in_robot_frame(self.rt) - np.array([kx, foot_y])
            ey = wrap(yaw_k - self.b.base_yaw())
            if np.all(np.abs(e) < k["align_tol_m"]) and abs(math.degrees(ey)) < k["align_tol_deg"]:
                return None
            cmd = np.clip([g[0] * e[0], g[1] * e[1], g[2] * ey], -np.asarray(mx), mx)
            v = float(np.linalg.norm(cmd[:2]))
            if 1e-9 < v < k["align_min_speed"]:
                cmd[:2] *= k["align_min_speed"] / v
            return Mode.WALK_FLAT, tuple(float(c) for c in cmd)

        aligned = self.drive(k["align_timeout_s"], align)
        self.label = "kick: stand, measure the ball"
        self.stand(acc["kick"]["stand_before_s"])
        meas = ev.ball_in_robot_frame(self.rt)
        self.label = f"kick: KICK (ball at x {meas[0]:+.3f}, y {meas[1]:+.3f} m)"
        self.mode, self.cmd = Mode.KICK, (0.0, 0.0, 0.0)
        gs = self.b.module_state(k["goal"])
        self.goal_watch = {"ball": k["ball"], "pos": gs["pos"], "yaw": gs["yaw"], "half_w": gs["params"]["width_m"] / 2,
                           "r": ev._ball(self.rt)[2], "scored": False}
        self.rt.send(Mode.KICK, kick_ball_xy=tuple(float(c) for c in meas))
        while True:
            self.step()
            if self.rt.switcher.kick is None or self.fell:
                break
            if self.rt.ticks % 5 == 0:
                self.rt.send(Mode.STAND)
        self.label = "GOAL!" if self.goal_watch["scored"] else "kick: done"
        self.stand(k["watch_after_s"])
        scored = self.goal_watch["scored"]
        self.goal_watch = None
        return {"approach": True, "aligned": aligned, "ball_at_kick": meas.round(3).tolist(), "scored": bool(scored)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walk", default="walk_v2_r2_s1_it2750")
    ap.add_argument("--kick", default="kick_v2_r1_stand2750_s1_it3999", help="'none' to skip the kick")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_yaml(CONFIG_DIR / "demo_videos.yaml")["course"]
    args.out = Path(args.out or f"outputs/demo_videos/{args.walk}/course_{cfg['layout']}.mp4")
    c = Course(args, cfg)
    c.label = "start"
    c.stand(1.0)
    log = []
    for wp in cfg["waypoints"]:
        c.label = f"-> {wp['name']}"
        ok = c.goto(wp["xy"])
        log.append((wp["name"], ok, round(c.rt.t, 1)))
        print(f"[course] {wp['name']}: {'reached' if ok else 'FAILED'} at t {c.rt.t:.1f} s", flush=True)
        if not ok:
            break
    else:
        if args.kick != "none":
            r = c.kick_at_goal()
            print(f"[course] kick: {r}", flush=True)
    c.label = "end" + ("  (FELL)" if c.fell else "")
    c.stand(1.0)
    c.writer.close()
    print(f"[course] {args.out}: {c.rt.t:.1f} s sim, fell {c.fell}", flush=True)


if __name__ == "__main__":
    main()
