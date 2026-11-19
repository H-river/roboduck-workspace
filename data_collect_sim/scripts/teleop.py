"""Keyboard teleop viewer: head camera (main view), top_cam inset, HUD.

One pygame window. Keys are read as state, so holding and several keys at once work:
  W/S  +/- vx     A/D  +/- vy     Q/E  +/- wz        (held = on, released = 0, 0 -> full in 0.3 s)
  Shift  toggle low / high gear (50 % / 100 % of the walk card's command_range)
  1  WALK_FLAT    0  STAND    J  KICK (one-shot)    R  reset the layout    Esc  quit

The upper layer sends a CommandMsg at 10 Hz, the head camera renders at 15 fps and the
policy runs at its own rate, all on the sim clock; the sim is paced to the wall clock and
a warning is printed when it falls behind. on_tick(state, frame, command) is the hook for
the phase-2 recorder (does nothing now).

    uv run python scripts/teleop.py --layout demo_v1
    uv run python scripts/teleop.py --layout demo_v1 --autopilot     # headless check: drive to flag A
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from duckrt.camera import TOP_CAM  # noqa: E402
from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml, output_dir  # noqa: E402
from duckrt.messages import CommandMsg, FrameMsg, Mode, StateMsg  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.scene.builder import load_layout, parse_overrides  # noqa: E402

AXES = {"vx": ("w", "s"), "vy": ("a", "d"), "wz": ("q", "e")}


def on_tick(state: StateMsg, frame: FrameMsg | None, command: CommandMsg | None) -> None:
    """Phase-2 recorder hook: called every control tick with the robot-side state, the new
    head-camera frame (None if no frame was exposed during this tick) and the last command."""


class KeyState:
    """Which virtual keys are down (from pygame or from the autopilot)."""

    def __init__(self):
        self.down: set[str] = set()


class Teleop:
    def __init__(self, args, sim_cfg, cam_cfg, tcfg):
        self.args, self.tcfg = args, tcfg
        self.layout_name, self.overrides = args.layout, parse_overrides(args.set)
        self.rt = SimRuntime(sim_cfg, layout=load_layout(self.layout_name, self.overrides), camera_cfg=cam_cfg,
                             render_head=True, on_tick=on_tick, seed=args.seed)
        self.cam_cfg = cam_cfg
        self.card = self.rt.policies["walk_flat"].card
        self.gear = 0
        self.mode = Mode.WALK_FLAT  # W works immediately; 0 = STAND (walk_flat cannot stop by itself)
        self.vel = np.zeros(3)
        self.pending_kick = False
        self.keys = KeyState()
        self.top_img = None
        self.next_top = 0.0
        self.rtf_hist: list[tuple[float, float]] = []
        self.behind_warn_t = -1e9

    # -- upper layer (10 Hz) --------------------------------------------------------------------
    def target_velocity(self) -> np.ndarray:
        r = self.card.command_range
        scale = self.tcfg["command"]["gears"][self.gear]
        tgt = np.zeros(3)
        for i, (ax, (pos, neg)) in enumerate(AXES.items()):
            lo, hi = r[ax]
            if pos in self.keys.down and neg not in self.keys.down:
                tgt[i] = hi * scale
            elif neg in self.keys.down and pos not in self.keys.down:
                tgt[i] = lo * scale
        if tgt[0] == 0 and tgt[1] == 0 and tgt[2] != 0 and "turn_in_place_wz_max" in self.card.command_rules:
            tgt[2] = math.copysign(self.card.command_rules["turn_in_place_wz_max"] * scale, tgt[2])
        return tgt

    def upper_tick(self) -> None:
        r = self.card.command_range
        full = np.array([max(abs(v) for v in r[a]) for a in ("vx", "vy", "wz")])
        if "turn_in_place_wz_max" in self.card.command_rules:
            full[2] = max(full[2], self.card.command_rules["turn_in_place_wz_max"])
        step = full * self.rt.upper_period / self.tcfg["command"]["ramp_time_s"]
        self.vel += np.clip(self.target_velocity() - self.vel, -step, step)
        if self.pending_kick:
            self.rt.send(Mode.KICK)
            self.pending_kick = False
        else:
            self.rt.send(self.mode, tuple(self.vel) if self.mode == Mode.WALK_FLAT else (0.0, 0.0, 0.0))

    def handle_key(self, key: str) -> bool:
        if key == "escape":
            return False
        if key == "shift":
            self.gear = 1 - self.gear
        elif key == "1":
            self.mode = Mode.WALK_FLAT
        elif key == "0":
            self.mode = Mode.STAND
        elif key == "j":
            self.pending_kick = True
        elif key == "r":
            self.rt.reset()
            self.vel[:] = 0
            self.mode = Mode.WALK_FLAT
        return True

    # -- sim stepping ---------------------------------------------------------------------------
    def advance(self, until_t: float) -> None:
        n = 0
        while self.rt.t < until_t - 1e-9 and n < self.tcfg["realtime"]["max_catchup_ticks"]:
            if self.rt.t >= self.next_upper - 1e-9:
                self.upper_tick()
                self.next_upper += self.rt.upper_period
            self.rt.tick()
            n += 1
        if self.rt.t >= self.next_top - 1e-9:
            tc = self.cam_cfg["top_cam"]
            self.top_img = self.rt.backend.render(TOP_CAM, tc["width"], tc["height"])
            self.next_top = self.rt.t + 1.0 / self.tcfg["window"]["inset_fps"]

    def hud_lines(self) -> list[str]:
        b, sw = self.rt.backend, self.rt.switcher
        v = b.base_vel_robot()
        p = b.base_pos()
        fr = self.rt.last_frame
        pitch = None
        if fr is not None:
            f = fr.cam_extrinsics.T_world_cam[:3, 2]
            pitch = -math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
        kick = f" [{sw.kick.phase}]" if sw.kick else ""
        rtf = self.realtime_factor()
        hint = ""
        if sw.mode == Mode.WALK_FLAT and not self.target_velocity().any() and math.hypot(v[0], v[1]) > 0.03:
            hint = "   <- keys released but still stepping: press 0 (STAND) to stop"
        elif sw.mode == Mode.STAND and self.target_velocity().any():
            hint = "   <- STAND ignores W/A/S/D/Q/E: press 1 to walk"
        return [
            f"mode {sw.mode.value}{kick}   policy {sw.active.card.name if sw.active else '-'}   gear {'HIGH 100%' if self.gear else 'LOW 50%'}"
            f"   fall_flag {'YES' if sw.fall_flag else 'no'}{'   WATCHDOG' if sw.watchdog_tripped else ''}",
            f"cmd sent  vx {self.vel[0]:+.3f}  vy {self.vel[1]:+.3f}  wz {self.vel[2]:+.3f}     "
            f"clipped  vx {sw.cmd[0]:+.3f}  vy {sw.cmd[1]:+.3f}  wz {sw.cmd[2]:+.3f}",
            f"actual    vx {v[0]:+.3f}  vy {v[1]:+.3f}  wz {v[2]:+.3f}     pose x {p[0]:+.2f}  y {p[1]:+.2f}  yaw {math.degrees(b.base_yaw()):+6.1f}°",
            (f"head cam pitch {pitch:+.1f}° down" if pitch is not None else "head cam -") + hint,
            f"sim t {self.rt.t:6.1f} s   real-time factor {rtf:.2f}   W/S A/D Q/E move, Shift gear, 1 walk, 0 stand, J kick, R reset, Esc quit",
        ]

    def realtime_factor(self) -> float:
        h = self.rtf_hist
        if len(h) < 2 or h[-1][0] - h[0][0] < 1e-6:
            return 0.0
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])

    # -- main loops -----------------------------------------------------------------------------
    def run_window(self) -> None:
        import pygame

        w = self.tcfg["window"]
        hc = self.cam_cfg["head_camera"]
        W, H = hc["width"] * w["head_scale"], hc["height"] * w["head_scale"]
        pygame.init()
        screen = pygame.display.set_mode((W, H + w["hud_height"]))
        pygame.display.set_caption(f"RoboDuck teleop — {self.layout_name}")
        font = pygame.font.SysFont("monospace", w["font_size"])
        keymap = {pygame.K_w: "w", pygame.K_s: "s", pygame.K_a: "a", pygame.K_d: "d", pygame.K_q: "q", pygame.K_e: "e"}
        events = {pygame.K_ESCAPE: "escape", pygame.K_LSHIFT: "shift", pygame.K_RSHIFT: "shift", pygame.K_1: "1",
                  pygame.K_0: "0", pygame.K_j: "j", pygame.K_r: "r"}
        self.next_upper = self.rt.t
        wall0, sim0 = time.perf_counter(), self.rt.t
        start_wall = wall0
        running = True
        while running:
            if self.args.smoke and time.perf_counter() - start_wall > self.args.smoke:
                out = output_dir("teleop")
                pygame.image.save(screen, str(out / "window_smoke.png"))
                print(f"[smoke] {self.args.smoke} s wall -> sim t {self.rt.t:.2f} s, real-time factor {self.realtime_factor():.2f}, "
                      f"frames {self.rt.frames}; screenshot {out / 'window_smoke.png'}")
                break
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key in events:
                    running = self.handle_key(events[ev.key])
                    if events[ev.key] == "r":
                        wall0, sim0 = time.perf_counter(), self.rt.t
                        self.next_upper = self.rt.t
            pressed = pygame.key.get_pressed()
            self.keys.down = {k for code, k in keymap.items() if pressed[code]}
            target = sim0 + (time.perf_counter() - wall0)
            self.advance(target)
            behind = target - self.rt.t
            now = time.perf_counter()
            self.rtf_hist = [x for x in self.rtf_hist if now - x[0] < 1.0] + [(now, self.rt.t)]
            if behind > self.tcfg["realtime"]["warn_behind_s"] and now - self.behind_warn_t > 1.0:
                print(f"[warning] sim {behind * 1000:.0f} ms behind the wall clock", flush=True)
                self.behind_warn_t = now
                wall0, sim0 = time.perf_counter(), self.rt.t  # do not try to catch up forever
            self.draw(pygame, screen, font, W, H)
            pygame.display.flip()
            time.sleep(0.002)
        pygame.quit()

    def draw(self, pygame, screen, font, W, H) -> None:
        w = self.tcfg["window"]
        screen.fill((20, 20, 24))
        fr = self.rt.last_frame
        if fr is not None:
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(fr.image.transpose(1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, (W, H)), (0, 0))
        if self.top_img is not None:
            iw = w["inset_width"]
            ih = int(iw * self.top_img.shape[0] / self.top_img.shape[1])
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(self.top_img.transpose(1, 0, 2)))
            screen.blit(pygame.transform.smoothscale(surf, (iw, ih)), (W - iw - 6, 6))
            pygame.draw.rect(screen, (240, 240, 240), (W - iw - 7, 5, iw + 2, ih + 2), 1)
        for i, line in enumerate(self.hud_lines()):
            col = (255, 90, 90) if ("fall_flag YES" in line or "WATCHDOG" in line) else (230, 230, 230)
            screen.blit(font.render(line, True, col), (8, H + 6 + i * (w["font_size"] + 6)))

    def run_autopilot(self) -> int:
        """Headless: press the same keys a person would to reach the target flag; saves screenshots."""
        import pygame

        ac = self.tcfg["autopilot"]
        out = output_dir("teleop")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        w = self.tcfg["window"]
        hc = self.cam_cfg["head_camera"]
        W, H = hc["width"] * w["head_scale"], hc["height"] * w["head_scale"]
        screen = pygame.display.set_mode((W, H + w["hud_height"]))
        font = pygame.font.SysFont("monospace", w["font_size"])
        b = self.rt.backend
        target = b.module_state(ac["target"])["pos"][:2]
        self.next_upper = self.rt.t
        self.handle_key("1")
        if ac.get("gear", "low") == "high":
            self.handle_key("shift")
        next_shot, wall0 = 0.0, time.perf_counter()
        reached = None
        while self.rt.t < ac["timeout_s"]:
            p = b.base_pos()[:2]
            d = target - p
            dist = float(np.linalg.norm(d))
            if dist < ac["arrive_m"]:
                reached = self.rt.t
                break
            bearing = math.degrees(math.atan2(math.sin(math.atan2(d[1], d[0]) - b.base_yaw()), math.cos(math.atan2(d[1], d[0]) - b.base_yaw())))
            keys = set()
            if abs(bearing) > ac["turn_deadband_deg"]:
                keys.add("q" if bearing > 0 else "e")
            if abs(bearing) < ac["walk_max_bearing_deg"]:
                keys.add("w")
            self.keys.down = keys
            self.advance(self.rt.t + 0.02)
            now = time.perf_counter()
            self.rtf_hist = [x for x in self.rtf_hist if now - x[0] < 1.0] + [(now, self.rt.t)]
            if self.rt.switcher.fall_flag:
                print(f"[autopilot] fell at t {self.rt.t:.1f}")
                break
            if self.rt.t >= next_shot:
                self.draw(pygame, screen, font, W, H)
                pygame.image.save(screen, str(out / f"autopilot_t{int(self.rt.t):03d}.png"))
                next_shot += ac["screenshot_every_s"]
        self.keys.down = set()
        self.advance(self.rt.t + 1.0)
        self.draw(pygame, screen, font, W, H)
        pygame.image.save(screen, str(out / "autopilot_final.png"))
        wall = time.perf_counter() - wall0
        print(f"[autopilot] {'reached ' + ac['target'] + ' at sim t %.1f s' % reached if reached else 'did NOT reach the target'}; "
              f"final pose {b.base_pos()[:2].round(2).tolist()}, fall_flag {self.rt.switcher.fall_flag}; wall {wall:.1f} s; screenshots in {out}")
        for t, e in self.rt.switcher.events:
            if "watchdog" in e or "fall" in e:
                print(f"   event t={t:.2f} {e}")
        pygame.quit()
        return 0 if reached else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="demo_v1")
    ap.add_argument("--set", nargs="*", default=[], help="module overrides <id>.<param>=<value>")
    ap.add_argument("--autopilot", action="store_true", help="headless key-pressing check (drive to the target flag)")
    ap.add_argument("--smoke", type=float, default=0.0, help="run the real-time window loop headless for N s and exit")
    ap.add_argument("--seed", type=int, default=0, help="sim seed (actuator-delay RNG)")
    args = ap.parse_args()
    if args.autopilot or args.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    sim_cfg, cam_cfg = load_sim_cfg(), load_yaml(CONFIG_DIR / "camera.yaml")
    tcfg = load_yaml(CONFIG_DIR / "teleop.yaml")
    app = Teleop(args, sim_cfg, cam_cfg, tcfg)
    if args.autopilot:
        sys.exit(app.run_autopilot())
    app.run_window()


if __name__ == "__main__":
    main()
