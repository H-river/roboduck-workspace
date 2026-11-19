"""Walk policy acceptance in the bare-MuJoCo runtime (step 1).

1. random:    N seeds x 60 s, piecewise-random commands in the card's command_range
              (segments U(2, 5) s). Target: >= 19/20 seeds without a fall. Falls are
              re-run with a chase camera and saved as video with the command at the fall.
2. tracking:  held commands (configs/tests.yaml), actual vs commanded velocity, mean / std.
3. zero:      10 s zero command: stands or steps in place? (video)
4. settle:    walk -> zero command, time until standing still (-> card handoff.settle_s).
5. videos:    turn in place, walking while turning.
6. realtime:  sim seconds per wall second on one core, without rendering and with the head
              camera rendered every 1/15 s.

    uv run python scripts/walk_test.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml, output_dir  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.policy_card import CARD_DIR, update_card_fields  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.testing import VideoWriter, fmt, foot_floor_contacts, md_table  # noqa: E402

OUT = output_dir("walk_test")  # replaced by a per-card subdir in main()


CARD = "walk_flat"  # set from --card


def make_rt(sim_cfg, camera_cfg=None, render_head=False, seed=0) -> SimRuntime:
    return SimRuntime(sim_cfg, roles={"walk_flat": CARD}, camera_cfg=camera_cfg, render_head=render_head, seed=seed)


def segment_commands(rng, card, duration, seg_range):
    r = card.command_range
    segs, t = [], 0.0
    while t < duration:
        d = float(rng.uniform(*seg_range))
        cmd = [float(rng.uniform(*r["vx"])), float(rng.uniform(*r["vy"])), float(rng.uniform(*r["wz"]))]
        segs.append((t, min(t + d, duration), cmd))
        t += d
    return segs


def cmd_at(segs, t):
    for a, b, c in segs:
        if a <= t < b:
            return c
    return segs[-1][2]


def fell(rt, tc) -> bool:
    return rt.switcher.fall_flag or rt.backend.base_pos()[2] < tc["fall"]["trunk_z_min"] or rt.backend.has_nan()


def random_runs(sim_cfg, cam_cfg, tc, quick: bool) -> dict:
    rc = tc["random"]
    n_seeds = 3 if quick else rc["seeds"]
    duration = 10.0 if quick else rc["duration_s"]
    rt = make_rt(sim_cfg)
    card = rt.policies["walk_flat"].card
    results, seg_stats = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        segs = segment_commands(rng, card, duration, rc["segment_s"])
        rt.reset(seed=seed)
        vel_log, fall_t = [], None
        rt_fn = lambda t: (Mode.WALK_FLAT, cmd_at(segs, t))  # noqa: E731
        next_cmd = 0.0
        while rt.t < duration - 1e-9:
            if rt.t >= next_cmd - 1e-9:
                rt.send(*rt_fn(rt.t))
                next_cmd += rt.upper_period
            rt.tick()
            vel_log.append((rt.t, *rt.switcher.cmd, *rt.backend.base_vel_robot()))
            if fell(rt, tc):
                fall_t = rt.t
                break
        vel = np.array(vel_log)
        for a, b, c in segs:  # per-segment tracking (second half of each segment)
            m = (vel[:, 0] >= (a + b) / 2) & (vel[:, 0] < b)
            if m.sum() > 10:
                seg_stats.append((*vel[m, 1:4].mean(0), *vel[m, 4:7].mean(0)))
        res = {"seed": seed, "fell": fall_t is not None, "fall_t": fall_t,
               "fall_cmd": cmd_at(segs, fall_t) if fall_t is not None else None,
               "final_pos": rt.backend.base_pos().round(3).tolist()}
        if fall_t is not None:
            res["video"] = str(fall_video(sim_cfg, cam_cfg, tc, seed, segs, fall_t))
        results.append(res)
        print(f"  seed {seed:2d}: {'FALL at %.2f s cmd %s' % (fall_t, np.round(res['fall_cmd'], 3).tolist()) if fall_t else 'ok'}")
    ss = np.array(seg_stats)
    gains = {}
    for i, ax in enumerate(("vx", "vy", "wz")):
        c, a = ss[:, i], ss[:, 3 + i]
        gains[ax] = {"gain": float(np.dot(c, a) / np.dot(c, c)), "mae": float(np.abs(a - c).mean())}
    return {"runs": results, "no_fall": sum(not r["fell"] for r in results), "n": n_seeds,
            "duration_s": duration, "segment_tracking": gains}


def fall_video(sim_cfg, cam_cfg, tc, seed, segs, fall_t) -> Path:
    vc, cc = tc["videos"], cam_cfg["chase_cam"]
    rt = make_rt(sim_cfg)
    rt.reset(seed=seed)
    path = OUT / f"fall_seed{seed}.mp4"
    vw = VideoWriter(path, vc["fps"])
    next_cmd, next_frame = 0.0, max(0.0, fall_t - vc["fall_pre_s"])
    while rt.t < fall_t + vc["fall_post_s"]:
        if rt.t >= next_cmd - 1e-9:
            rt.send(Mode.WALK_FLAT, cmd_at(segs, rt.t))
            next_cmd += rt.upper_period
        rt.tick()
        if rt.t >= next_frame - 1e-9:
            vw.add(rt.backend.render_chase(cc["width"], cc["height"], cc["distance"], cc["azimuth"], cc["elevation"]))
            next_frame += 1.0 / vc["fps"]
    vw.close()
    return path


def tracking(sim_cfg, tc, quick: bool) -> list[dict]:
    t = tc["tracking"]
    rt = make_rt(sim_cfg)
    rows = []
    for cmd in t["commands"]:
        per_seed = []
        for seed in range(1 if quick else t["seeds"]):
            rt.reset(seed=seed)
            log = []
            rt.run(t["hold_s"], command_fn=lambda _t: (Mode.WALK_FLAT, cmd),
                   per_tick=lambda r: log.append((r.t, *r.switcher.cmd, *r.backend.base_vel_robot())))
            a = np.array(log)
            a = a[a[:, 0] >= t["hold_s"] - t["measure_last_s"]]
            per_seed.append(a)
        win = 25  # 0.5 s windows: std of window means = drift / wobble slower than one step
        wmeans = np.concatenate([x[: len(x) // win * win, 4:7].reshape(-1, win, 3).mean(1) for x in per_seed])
        a = np.concatenate(per_seed)
        rows.append({"cmd": cmd, "clipped_cmd": a[0, 1:4].round(4).tolist(), "mean": a[:, 4:7].mean(0).tolist(),
                     "std": a[:, 4:7].std(0).tolist(), "std_0p5s": wmeans.std(0).tolist(), "fell": bool(rt.switcher.fall_flag)})
        print(f"  cmd {cmd}: actual mean {np.round(rows[-1]['mean'], 3).tolist()} std {np.round(rows[-1]['std'], 3).tolist()}")
    return rows


def zero_command(sim_cfg, cam_cfg, tc) -> dict:
    zc, vc, cc = tc["zero_command"], tc["videos"], cam_cfg["chase_cam"]
    rt = make_rt(sim_cfg)
    b = rt.backend
    m = b.model
    feet = [m.site("left_foot").id, m.site("right_foot").id]
    vw = VideoWriter(OUT / "zero_command.mp4", vc["fps"])
    z0 = None
    lifts, speeds, yaws, contacts = [], [], [], []
    next_frame, start_pos, start_yaw = 0.0, b.base_pos(), b.base_yaw()
    while rt.t < zc["duration_s"] - 1e-9:
        if rt.ticks % 5 == 0:
            rt.send(Mode.STAND)
        rt.tick()
        fz = b.data.site_xpos[feet, 2].copy()
        if rt.t >= 1.0 and z0 is None:
            z0 = fz.copy()
        if z0 is not None:
            lifts.append(fz - z0)
            speeds.append(np.linalg.norm(b.base_lin_vel_world()[:2]))
            contacts.append(foot_floor_contacts(b, sim_cfg["robot"]["foot_geoms"]))
        if rt.t >= next_frame - 1e-9:
            vw.add(b.render_chase(cc["width"], cc["height"], cc["distance"], cc["azimuth"], cc["elevation"]))
            next_frame += 1.0 / vc["fps"]
    vw.close()
    lifts = np.array(lifts)
    thr = zc["lift_threshold_m"]
    lift_events = [int(np.sum(np.diff((lifts[:, i] > thr).astype(int)) == 1)) for i in range(2)]
    contacts = np.array(contacts)
    stepping = bool(max(lift_events) >= 2)
    return {"duration_s": zc["duration_s"], "max_foot_lift_m": lifts.max(0).tolist(), "lift_events_per_foot": lift_events,
            "both_feet_contact_fraction": float(contacts.all(1).mean()),
            "base_displacement_m": float(np.linalg.norm(b.base_pos()[:2] - start_pos[:2])),
            "yaw_drift_deg": float(np.degrees(b.base_yaw() - start_yaw)),
            "base_speed_mean_m_s": float(np.mean(speeds)), "base_speed_max_m_s": float(np.max(speeds)),
            "verdict": "原地踏步" if stepping else "站定", "video": str(OUT / "zero_command.mp4")}


def settle(sim_cfg, tc, quick: bool) -> dict:
    s = tc["settle"]
    rt = make_rt(sim_cfg)
    times = []
    for seed in range(2 if quick else s["seeds"]):
        rt.reset(seed=seed)
        rt.run(s["walk_s"], command_fn=lambda _t: (Mode.WALK_FLAT, s["walk_cmd"]))
        t0, ok_since, done = rt.t, None, None
        while rt.t - t0 < s["timeout_s"]:
            if rt.ticks % 5 == 0:
                rt.send(Mode.STAND)
            rt.tick()
            v = rt.backend.base_vel_robot()
            still = (np.hypot(v[0], v[1]) < s["speed_max"] and abs(v[2]) < s["yaw_rate_max"]
                     and all(foot_floor_contacts(rt.backend, sim_cfg["robot"]["foot_geoms"])))
            ok_since = (ok_since if ok_since is not None else rt.t) if still else None
            if ok_since is not None and rt.t - ok_since >= s["hold_s"]:
                done = ok_since - t0
                break
        times.append(done)
    valid = [t for t in times if t is not None]
    return {"settle_times_s": times, "timeouts": sum(t is None for t in times),
            "p50": float(np.median(valid)) if valid else None, "max": float(max(valid)) if valid else None}


def videos(sim_cfg, cam_cfg, tc) -> dict:
    vc, cc = tc["videos"], cam_cfg["chase_cam"]
    out = {}
    for name in ("turn_in_place", "walk_turn"):
        spec = vc[name]
        rt = make_rt(sim_cfg)
        path = OUT / f"{name}.mp4"
        vw = VideoWriter(path, vc["fps"])
        next_frame = 0.0
        yaw0 = rt.backend.base_yaw()
        rates = []

        def per_tick(r):
            nonlocal next_frame
            rates.append(r.backend.base_vel_robot())
            if r.t >= next_frame - 1e-9:
                vw.add(r.backend.render_chase(cc["width"], cc["height"], cc["distance"], cc["azimuth"], cc["elevation"]))
                next_frame += 1.0 / vc["fps"]

        rt.run(spec["duration_s"], command_fn=lambda _t: (Mode.WALK_FLAT, spec["cmd"]), per_tick=per_tick)
        vw.close()
        r = np.array(rates)[len(rates) // 3:]
        out[name] = {"cmd": spec["cmd"], "clipped_cmd": rt.switcher.cmd.round(4).tolist(), "mean_vel": r.mean(0).round(4).tolist(),
                     "net_yaw_deg": float(np.degrees(np.unwrap([yaw0, rt.backend.base_yaw()])[1] - yaw0)),
                     "fell": bool(rt.switcher.fall_flag), "video": str(path)}
        print(f"  {name}: {out[name]}")
    return out


def realtime(sim_cfg, cam_cfg, tc) -> dict:
    rc = tc["realtime"]
    try:
        os.sched_setaffinity(0, {int(rc["cpu"])})
        pinned = True
    except (AttributeError, OSError):
        pinned = False
    res = {"pinned_cpu": rc["cpu"] if pinned else None}
    for label, render in (("no_render", False), ("head_cam_15fps", True)):
        rt = make_rt(sim_cfg, camera_cfg=cam_cfg if render else None, render_head=render)
        rt.run(1.0, command_fn=lambda _t: (Mode.WALK_FLAT, rc["cmd"]))  # warm-up (renderer init)
        f0 = rt.frames
        t0 = time.perf_counter()
        rt.run(rc["duration_s"], command_fn=lambda _t: (Mode.WALK_FLAT, rc["cmd"]))
        wall = time.perf_counter() - t0
        frames = rt.frames - f0
        res[label] = {"sim_s": rc["duration_s"], "wall_s": wall, "realtime_factor": rc["duration_s"] / wall}
        if render:
            res[label]["frames_rendered"] = frames
            res[label]["resolution"] = [rt.camera.width, rt.camera.height]
        print(f"  {label}: x{res[label]['realtime_factor']:.1f} real time ({wall:.2f} s wall for {rc['duration_s']} s)")
    return res


def write_report(rep: dict, card) -> str:
    lines = [f"# walk_test {date.today()} — {card.name} ({card.onnx.name})", ""]
    r = rep["random"]
    lines += [f"## 随机命令：{r['no_fall']}/{r['n']} 个 seed 在 {r['duration_s']:.0f} s 内没有摔倒", ""]
    falls = [x for x in r["runs"] if x["fell"]]
    if falls:
        lines += [md_table(["seed", "摔倒时刻 s", "当时命令", "视频"],
                           [[x["seed"], fmt(x["fall_t"], 2), np.round(x["fall_cmd"], 3).tolist(), Path(x["video"]).name] for x in falls]), ""]
    g = r["segment_tracking"]
    lines += ["分段统计（每段后半段的平均速度对命令做过原点回归）：" + "，".join(
        f"{k} 增益 {v['gain']:.2f}、MAE {v['mae']:.3f}" for k, v in g.items()), ""]
    lines += ["## 命令跟踪（保持 8 s，取后 5 s，3 个 seed；均值 ± 标准差，括号里是 0.5 s 窗口平均后的标准差）", ""]
    rows = []
    for t in rep["tracking"]:
        m, s, w, c = t["mean"], t["std"], t["std_0p5s"], t["clipped_cmd"]
        rows.append([str(t["cmd"]), str(c)] + [f"{m[i]:+.3f} ± {s[i]:.3f} ({w[i]:.3f})" for i in range(3)]
                    + [f"{m[i] - c[i]:+.3f}" for i in range(3)])
    lines += [md_table(["命令 (vx, vy, wz)", "裁剪后", "实际 vx", "实际 vy", "实际 wz", "误差 vx", "误差 vy", "误差 wz"], rows), ""]
    z = rep["zero_command"]
    lines += [f"## 零命令 {z['duration_s']:.0f} s：**{z['verdict']}**", "",
              f"- 脚抬起最大值 L/R：{z['max_foot_lift_m'][0] * 1000:.1f} / {z['max_foot_lift_m'][1] * 1000:.1f} mm；抬脚次数 L/R：{z['lift_events_per_foot']}",
              f"- 双脚同时着地比例 {z['both_feet_contact_fraction'] * 100:.0f} %；机身位移 {z['base_displacement_m'] * 1000:.1f} mm；偏航漂移 {z['yaw_drift_deg']:+.2f}°",
              f"- 机身水平速度均值 {z['base_speed_mean_m_s'] * 1000:.1f} mm/s，最大 {z['base_speed_max_m_s'] * 1000:.1f} mm/s", ""]
    st = rep["settle"]
    if st["p50"] is None:
        lines += [f"## 走 → 零命令后站稳：**{st['timeouts']}/{len(st['settle_times_s'])} 次在 3 s 内都没停下**", ""]
    else:
        lines += [f"## 走 → 零命令后站稳所需时间：中位 {st['p50']:.2f} s，最大 {st['max']:.2f} s，超时 {st['timeouts']} 次", ""]
    for k, v in rep["videos"].items():
        lines += [f"- {k}: 命令 {v['cmd']}（裁剪后 {v['clipped_cmd']}），稳态速度 {v['mean_vel']}，{'摔倒' if v['fell'] else '未摔'}，视频 {Path(v['video']).name}"]
    rt = rep["realtime"]
    lines += ["", "## 单核实时倍率", "",
              md_table(["情况", "仿真 s", "墙钟 s", "倍率"],
                       [[k, rt[k]["sim_s"], fmt(rt[k]["wall_s"], 2), f"x{rt[k]['realtime_factor']:.1f}"] for k in ("no_render", "head_cam_15fps")]),
              f"\n（固定在 CPU {rt['pinned_cpu']}；头部相机 {rt['head_cam_15fps'].get('resolution')}，{rt['head_cam_15fps'].get('frames_rendered')} 帧，EGL 渲染在 GPU 上）"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 short seeds, 1 tracking seed (smoke test)")
    ap.add_argument("--no-card-update", action="store_true")
    ap.add_argument("--card", default="walk_flat", help="policy card to test as walk_flat")
    args = ap.parse_args()
    global CARD, OUT
    CARD = args.card
    OUT = output_dir("walk_test", CARD)
    sim_cfg, cam_cfg = load_sim_cfg(), load_yaml(CONFIG_DIR / "camera.yaml")
    tc = load_yaml(CONFIG_DIR / "tests.yaml")["walk_test"]
    rep = {}
    print("[random]"); rep["random"] = random_runs(sim_cfg, cam_cfg, tc, args.quick)
    print("[tracking]"); rep["tracking"] = tracking(sim_cfg, tc, args.quick)
    print("[zero command]"); rep["zero_command"] = zero_command(sim_cfg, cam_cfg, tc); print("  ", rep["zero_command"]["verdict"])
    print("[settle]"); rep["settle"] = settle(sim_cfg, tc, args.quick); print("  ", rep["settle"])
    print("[videos]"); rep["videos"] = videos(sim_cfg, cam_cfg, tc)
    print("[realtime]"); rep["realtime"] = realtime(sim_cfg, cam_cfg, tc)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    rt = make_rt(sim_cfg)
    card = rt.policies["walk_flat"].card
    md = write_report(rep, card)
    (OUT / "report.md").write_text(md)
    print(md)
    r = rep["random"]
    if not args.quick and not args.no_card_update:
        settle_s = float(np.ceil(rep["settle"]["max"] * 10) / 10) if rep["settle"]["max"] is not None else None
        update_card_fields(CARD_DIR / f"{CARD}.yaml", {
            "validated_on": f"仿真平地（bare MuJoCo + BAM，{date.today()}）：{r['no_fall']}/{r['n']} 个 seed × {r['duration_s']:.0f} s 随机命令不摔；"
                            f"静止起步零命令{rep['zero_command']['verdict']}；行走后零命令"
                            f"{'能停' if rep['settle']['timeouts'] == 0 else '停不下来（%d/%d 次超时）' % (rep['settle']['timeouts'], len(rep['settle']['settle_times_s']))}；"
                            f"见 outputs/walk_test/{CARD}/report.md",
            "handoff": {"needs_stand_before": True, "settle_s": settle_s},
        })
        print(f"card updated: settle_s {settle_s}")
    ok = r["no_fall"] >= r["n"] - 1
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
