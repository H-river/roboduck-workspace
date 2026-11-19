"""Ramp test with the flat walk policy (step 4). Nothing is trained.

For each angle (0 / 3 / 5 / 7 deg) and direction (up: ramp_test, start 0.5 m before the
foot; down: ramp_test_down, start on the top platform), `runs` runs with a fixed forward
command at 60 % of the card's vx limit (+ one run at the limit). Recorded: success, falls
(where), traverse time, trunk pitch over time, joint torque peaks and how often the XL330
current limit is hit. One video per angle and direction.

    uv run python scripts/ramp_test.py [--angles 0 3 5 7] [--runs 10]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml, output_dir  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.scene.builder import load_layout  # noqa: E402
from duckrt.testing import VideoWriter, md_table  # noqa: E402

OUT = output_dir("ramp_test")
LEG_GROUPS = {"hip_yaw": [0, 9], "hip_roll": [1, 10], "hip_pitch": [2, 11], "knee": [3, 12], "ankle": [4, 13]}


def trunk_pitch_down_deg(q) -> float:
    w, x, y, z = q
    r20 = 2 * (x * z - w * y)  # world z of the body x axis
    return -math.degrees(math.asin(max(-1.0, min(1.0, r20))))


def ramp_frame(rt):
    s = rt.backend.module_state("ramp")
    return s["pos"][:2], s["yaw"], s["params"]


def along(p_xy, origin, yaw) -> float:
    return float((p_xy[0] - origin[0]) * math.cos(yaw) + (p_xy[1] - origin[1]) * math.sin(yaw))


def steer_cmd(rt, cmd_vx, origin, ryaw, direction, sc, wz_max) -> tuple[float, float, float]:
    """Executor-like hold: steer back to the ramp centre line and along it (privileged pose)."""
    b = rt.backend
    p = b.base_pos()
    path_yaw = ryaw if direction == "up" else ryaw + math.pi
    lat = -(p[0] - origin[0]) * math.sin(path_yaw) + (p[1] - origin[1]) * math.cos(path_yaw)  # left of the line > 0
    corr = max(-math.radians(sc["max_heading_corr_deg"]), min(math.radians(sc["max_heading_corr_deg"]), -sc["k_lateral_rad_per_m"] * lat))
    err = math.atan2(math.sin(path_yaw + corr - b.base_yaw()), math.cos(path_yaw + corr - b.base_yaw()))
    return cmd_vx, 0.0, max(-wz_max, min(wz_max, sc["k_heading"] * err))


def run_one(rt: SimRuntime, direction: str, cmd_vx: float, seed: int, tc: dict, video: Path | None = None, cam_cfg=None,
            steer: bool = False) -> dict:
    rng = np.random.default_rng(10_000 + seed)
    b = rt.backend
    base = b.world.robot_start
    start = copy.deepcopy(base)
    start.pos[:2] += rng.uniform(-1, 1, 2) * tc["start_noise"]["xy_m"]
    start.yaw += math.radians(rng.uniform(-1, 1) * tc["start_noise"]["yaw_deg"])
    rt.reset(seed=seed, start=start)
    b.actuator.reset_stats()
    origin, ryaw, rp = ramp_frame(rt)
    L, T = rp["slope_length_m"], rp["top_length_m"]
    goal_u = L + T / 2 if direction == "up" else -0.25
    wz_max = rt.policies["walk_flat"].card.command_range["wz"][1]
    vw = VideoWriter(video, tc["video_fps"]) if video else None
    next_frame = 0.0
    log, fall = [], None
    while rt.t < tc["timeout_s"]:
        if rt.ticks % 5 == 0:
            cmd = steer_cmd(rt, cmd_vx, origin, ryaw, direction, tc["steer"], wz_max) if steer else (cmd_vx, 0.0, 0.0)
            rt.send(Mode.WALK_FLAT, cmd)
        rt.tick()
        p = b.base_pos()
        u = along(p, origin, ryaw)
        ground = b.ground_height(p[0], p[1])
        log.append((rt.t, u, trunk_pitch_down_deg(b.base_quat()), p[2] - ground, p[1]))
        if vw is not None and rt.t >= next_frame - 1e-9:
            vw.add(b.render_chase(640, 480, 1.1, 90.0, -12.0))
            next_frame += 1.0 / tc["video_fps"]
        if rt.switcher.fall_flag or p[2] - ground < tc["fall_trunk_height_m"] or b.has_nan():
            fall = {"t": rt.t, "u_m": u, "xy": p[:2].round(3).tolist()}
            break
        if (direction == "up" and u >= goal_u) or (direction == "down" and u <= goal_u):
            break
    if vw is not None:
        vw.close()
    a = np.array(log)
    success = fall is None and ((direction == "up" and a[-1, 1] >= goal_u) or (direction == "down" and a[-1, 1] <= goal_u))
    on_slope = (a[:, 1] >= 0) & (a[:, 1] <= L) if direction == "up" else (a[:, 1] >= 0) & (a[:, 1] <= L)
    act = b.actuator
    return {"direction": direction, "cmd_vx": cmd_vx, "seed": seed, "steer": steer, "success": bool(success), "fall": fall,
            "time_s": float(a[-1, 0]) if success else None,
            "slope_time_s": float(on_slope.sum() * rt.switcher.period),
            "pitch_on_slope_mean_deg": float(a[on_slope, 2].mean()) if on_slope.any() else None,
            "pitch_max_abs_deg": float(np.abs(a[:, 2]).max()),
            "torque_peak_nm": {g: float(act.stat_peak_torque[idx].max()) for g, idx in LEG_GROUPS.items()},
            "current_limited_pct": {g: float(100 * act.stat_current_limited[idx].max() / max(act.stat_steps, 1)) for g, idx in LEG_GROUPS.items()},
            "lateral_drift_m": float(a[-1, 4] - a[0, 4]),
            "trace": a[:, :3].round(4).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angles", type=float, nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--card", default=None, help="walk policy card (default: sim.yaml runtime.policies.walk_flat)")
    ap.add_argument("--variants", choices=["A", "B", "both"], default="both", help="A fixed command, B heading hold")
    ap.add_argument("--tag", default=None, help="output subdirectory (default: the card name)")
    args = ap.parse_args()
    global OUT
    if args.card or args.tag:
        OUT = output_dir("ramp_test", args.tag or Path(args.card).stem)
    sim_cfg, cam_cfg = load_sim_cfg(), load_yaml(CONFIG_DIR / "camera.yaml")
    tc = load_yaml(CONFIG_DIR / "tests.yaml")["ramp_test"]
    angles = args.angles if args.angles is not None else tc["angles_deg"]
    runs = args.runs or tc["runs"]
    results = []
    torque_limit = None
    for ang in angles:
        for direction in ("up", "down"):
            layout = load_layout(tc[direction]["layout"], {"ramp": {"angle_deg": float(ang)}})
            rt = SimRuntime(sim_cfg, layout=layout, roles={"walk_flat": args.card} if args.card else None)
            torque_limit = rt.backend.actuator.cfg.max_current * rt.backend.actuator.kt
            vmax = rt.policies["walk_flat"].card.command_range["vx"][1]
            cmds = [tc["speed_fraction"] * vmax] * runs + ([vmax] if tc["extra_full_speed_run"] else [])
            for steer in {"A": (False,), "B": (True,), "both": (False, True)}[args.variants]:
                for i, v in enumerate(cmds):
                    video = OUT / f"ramp_{ang:g}deg_{direction}{'_steer' if steer else ''}.mp4" if i == 0 else None
                    r = run_one(rt, direction, v, i, tc, video, cam_cfg, steer=steer)
                    r["angle_deg"] = ang
                    r["ramp_height_m"] = float(rt.backend.module_state("ramp")["params"]["height_m"])
                    results.append(r)
                ok = [r for r in results if r["angle_deg"] == ang and r["direction"] == direction and r["steer"] == steer]
                print(f"  {ang:g} deg {direction} {'B steer' if steer else 'A fixed'}: {sum(r['success'] for r in ok[:runs])}/{runs} "
                      f"at vx {cmds[0]:.3f}, full-speed run {'ok' if ok[-1]['success'] else 'FAIL'}; falls {sum(r['fall'] is not None for r in ok)}")

    # pitch plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, direction in zip(axes, ("up", "down")):
        for ang, col in zip(angles, plt.cm.viridis(np.linspace(0, 0.9, len(angles)))):
            for r in [r for r in results if r["angle_deg"] == ang and r["direction"] == direction and r["steer"]][:3]:
                tr = np.array(r["trace"])
                ax.plot(tr[:, 0], tr[:, 2], color=col, lw=0.8, label=f"{ang:g}°" if r["seed"] == 0 else None)
        ax.set_title(f"trunk pitch ({direction}hill, variant B), + = nose down")
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("pitch [deg]")
    fig.tight_layout()
    fig.savefig(OUT / "pitch.png", dpi=110)

    (OUT / "report.json").write_text(json.dumps(results, indent=2, default=float))
    rows = []
    for steer in {"A": (False,), "B": (True,), "both": (False, True)}[args.variants]:
     for ang in angles:
        for direction in ("up", "down"):
            rs = [r for r in results if r["angle_deg"] == ang and r["direction"] == direction and r["steer"] == steer]
            main_rs, full = rs[:runs], rs[runs:]
            t = [r["time_s"] for r in main_rs if r["success"]]
            falls = [r["fall"] for r in rs if r["fall"]]
            rows.append(["B 保持航向" if steer else "A 固定命令", f"{ang:g}°", "上坡" if direction == "up" else "下坡", f"{rs[0]['ramp_height_m'] * 100:.1f} cm",
                         f"{sum(r['success'] for r in main_rs)}/{len(main_rs)}",
                         ("成功" if full[0]["success"] else "失败") if full else "-",
                         f"{np.mean(t):.1f} ± {np.std(t):.1f}" if t else "-",
                         f"{np.mean([r['pitch_on_slope_mean_deg'] for r in main_rs if r['pitch_on_slope_mean_deg'] is not None]):+.1f}",
                         len(falls), "; ".join(f"u={f['u_m']:.2f} m @ {f['t']:.1f} s" for f in falls[:3]),
                         " / ".join(f"{max(r['torque_peak_nm'][g] for r in rs):.2f}" for g in ("hip_pitch", "knee", "ankle")),
                         " / ".join(f"{max(r['current_limited_pct'][g] for r in rs):.0f}" for g in ("hip_pitch", "knee", "ankle"))])
    L = [f"# ramp_test {date.today()} — walk_flat，固定前进命令 vx = {tc['speed_fraction']:.0%} × 上限（另加 1 次上限速度）", "",
         f"坡长 0.8 m（水平），平台 0.6 m，宽 0.4 m，两侧 4 cm 护栏，摩擦 1.0。上坡从坡底前 0.5 m 起步，到达平台中心算成功；"
         f"下坡从平台起步，越过坡底 0.25 m 算成功。起点加 ±2 cm / ±3° 扰动。u = 沿坡方向的坐标（坡底 = 0）。"
         f"XL330 电流限制对应力矩 {torque_limit:.2f} N·m。", "",
         "A = 题目要求的固定前进命令（不转向）；B = 同样的 vx，加执行器式的 10 Hz 航向 / 中线保持（ωz = 2 × 航向误差，"
         "航向目标按横向偏差修正 3 rad/m，ωz 限 ±0.375）。A 的失败几乎都是航向漂移把机器人带出 0.4 m 宽的坡道。", "",
         md_table(["变体", "坡角", "方向", "坡高", f"成功（vx {tc['speed_fraction'] * 0.25:.2f}）", "上限速度那次", "通过时间 s",
                   "坡上平均俯仰 °", "摔倒", "摔倒位置", "力矩峰值 N·m 髋俯仰/膝/踝", "电流限幅占比 % 髋俯仰/膝/踝"], rows), "",
         "俯仰曲线（变体 B）：pitch.png；视频：ramp_<角度>deg_<up|down>[_steer].mp4"]
    md = "\n".join(L) + "\n"
    (OUT / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
