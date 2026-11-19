"""Kick policy test through the switcher's KICK sequence (stand -> kick -> stand).

The ball is teleported into the kick window (card task.spawn, relative to the trunk root
in its yaw frame) when the kick starts playing, as the training reset does. Measured:
  * success over a grid of ball offsets around the window centre (-> kick window),
  * ball speed / direction, robot falls, hip_yaw near the walk-model limit,
  * when the kick motion is over (max joint speed < threshold) -> card duration_s,
  * kicks requested right after walking (hot handoff walk -> stand -> kick).

    uv run python scripts/kick_test.py [--no-card-update]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml, output_dir  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.policy_card import CARD_DIR, update_card_fields  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.scene.builder import load_layout  # noqa: E402
from duckrt.testing import VideoWriter, md_table  # noqa: E402

OUT = output_dir("kick_test")


def place_ball(rt: SimRuntime, offset_xy) -> np.ndarray:
    b = rt.backend
    m, d = b.model, b.data
    j = m.joint("mod_ball_free").id
    qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
    root, yaw = b.base_pos(), b.base_yaw()
    c, s = math.cos(yaw), math.sin(yaw)
    r = m.geom_size[m.geom("mod_ball_sphere").id][0]
    p = np.array([root[0] + c * offset_xy[0] - s * offset_xy[1], root[1] + s * offset_xy[0] + c * offset_xy[1], r])
    d.qpos[qa:qa + 3] = p
    d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    d.qvel[va:va + 6] = 0
    mujoco.mj_forward(m, d)
    return p


def ball_state(rt):
    b = rt.backend
    m, d = b.model, b.data
    j = m.joint("mod_ball_free").id
    return d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3].copy(), d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 3].copy()


def one_kick(rt: SimRuntime, card, offset, seed, tc, walk=None, video: Path | None = None, cam_cfg=None) -> dict:
    rt.reset(seed=seed)
    b = rt.backend
    if walk is not None:
        rt.run(walk["walk_s"], command_fn=lambda _t: (Mode.WALK_FLAT, walk["cmd"]))
    else:
        rt.run(0.5, command_fn=lambda _t: (Mode.STAND, (0, 0, 0)))
    rt.send(Mode.KICK)
    vw = VideoWriter(video, tc["video_fps"]) if video else None
    cc = cam_cfg["chase_cam"] if cam_cfg else None
    next_frame = rt.t
    sw = rt.switcher
    play_t0 = ball0 = heading = None
    jmax, bspeed, bvel, hipyaw_margin = [], [], [], np.inf
    hy = [b.model.joint(n).id for n in ("left_hip_yaw", "right_hip_yaw")]
    hy_rng = b.model.jnt_range[hy]
    settle_ok = None
    while True:
        rt.tick()
        if vw is not None and rt.t >= next_frame - 1e-9:
            vw.add(b.render_chase(cc["width"], cc["height"], 0.7, 120.0, -25.0))
            next_frame += 1.0 / tc["video_fps"]
        phase = sw.kick.phase if sw.kick else "done"
        if phase == "play" and play_t0 is None:
            play_t0 = rt.t - rt.switcher.period
            v = b.base_vel_robot()
            settle_ok = bool(np.hypot(v[0], v[1]) < 0.03)
            ball0 = place_ball(rt, np.asarray(card.task["spawn"]["offset_xy"]) + offset)
            heading = b.base_yaw()
        if play_t0 is not None and phase == "play":
            jmax.append((rt.t - play_t0, float(np.abs(b.joint_vel()).max())))
            bv = ball_state(rt)[1][:2]
            bspeed.append(float(np.linalg.norm(bv)))
            bvel.append(bv.copy())
            q = b.joint_pos()[[0, 9]]
            hipyaw_margin = min(hipyaw_margin, float(np.min(np.abs(np.concatenate([q - hy_rng[:, 0], hy_rng[:, 1] - q])))))
        if phase == "done":
            break
    rt.run(1.0, command_fn=lambda _t: (Mode.STAND, (0, 0, 0)))
    if vw is not None:
        vw.close()
    bp, _ = ball_state(rt)
    disp = bp[:2] - ball0[:2]
    dist = float(np.linalg.norm(disp))
    peak = max(bspeed) if bspeed else 0.0
    # kick direction = ball velocity direction at peak speed (before any wall bounce), vs heading
    if peak > 1e-3:
        v = bvel[int(np.argmax(bspeed))]
        a = math.atan2(v[1], v[0]) - heading
        dir_err = float(np.degrees(math.atan2(math.sin(a), math.cos(a))))
    else:
        dir_err = float("nan")
    t_peak = jmax[int(np.argmax(bspeed))][0] if bspeed else None
    # motion over: last time the joint speed exceeded the threshold during play
    over = [t for t, v in jmax if v > tc["settle_joint_speed"]]
    t_motion_end = max(over) if over else 0.0
    success = peak >= tc["success"]["min_ball_speed"] and abs(dir_err) <= tc["success"]["max_dir_err_deg"]
    return {"offset": list(map(float, offset)), "seed": seed, "after_walk": walk is not None, "settled_before_play": settle_ok,
            "ball_peak_speed": peak, "t_peak_s": t_peak, "ball_travel_m": dist, "dir_err_deg": dir_err,
            "motion_end_s": t_motion_end, "fell": bool(sw.fall_flag or b.up_z() < 0.5), "success": bool(success),
            "hip_yaw_min_margin_rad": hipyaw_margin}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-card-update", action="store_true")
    args = ap.parse_args()
    sim_cfg, cam_cfg = load_sim_cfg(), load_yaml(CONFIG_DIR / "camera.yaml")
    tc = load_yaml(CONFIG_DIR / "tests.yaml")["kick_test"]
    rt = SimRuntime(sim_cfg, layout=load_layout(tc["layout"]))
    card = rt.policies["kick"].card
    card.duration_s = tc["record_s"]  # measure with the full training episode, then shorten
    res = []
    for ox in tc["offsets_m"]:
        for oy in tc["offsets_m"]:
            for seed in range(tc["seeds"]):
                res.append(one_kick(rt, card, np.array([ox, oy]), seed, tc))
        print(f"  offset x {ox:+.3f}: " + " ".join(
            f"y{r['offset'][1]:+.3f}:{'Y' if r['success'] else 'n'}" for r in res if r['offset'][0] == ox))
    walk = [one_kick(rt, card, np.zeros(2), s, tc, walk=tc["walk_before"]) for s in range(tc["walk_before"]["seeds"])]
    one_kick(rt, card, np.zeros(2), 0, tc, video=OUT / "kick_centre.mp4", cam_cfg=cam_cfg)
    one_kick(rt, card, np.zeros(2), 0, tc, walk=tc["walk_before"], video=OUT / "kick_after_walk.mp4", cam_cfg=cam_cfg)

    centre = [r for r in res if r["offset"] == [0.0, 0.0]]
    allr = res + walk
    motion_end = float(np.percentile([r["motion_end_s"] for r in allr if r["success"]] or [tc["record_s"]], 95))
    duration = float(math.ceil((motion_end + 0.2) * 10) / 10)
    # verify the shortened play time: kick -> stand handover right after the motion
    card.duration_s = duration
    verify = [one_kick(rt, card, np.zeros(2), 100 + s, tc) for s in range(tc["verify_seeds"])]
    verify += [one_kick(rt, card, np.zeros(2), 200 + s, tc, walk=tc["walk_before"]) for s in range(tc["verify_seeds"])]
    rep = {"grid": res, "after_walk": walk, "motion_end_p95_s": motion_end, "duration_s": duration, "verify": verify}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))

    offs = tc["offsets_m"]
    win = card.task["spawn"]["offset_xy"]
    rows = []
    for ox in offs:
        row = [f"x {win[0] + ox:.3f}"]
        for oy in offs:
            rs = [r for r in res if r["offset"] == [ox, oy]]
            row.append(f"{sum(r['success'] for r in rs)}/{len(rs)}")
        rows.append(row)
    L = [f"# kick_test {date.today()} — {card.name} ({card.onnx.name})", "",
         f"球放在 trunk 根节点 yaw 系下 (x, y)，窗口中心 {win}，训练噪声 ±{card.task['spawn']['noise_xy']} m。"
         f"成功 = 球最大速度 ≥ {tc['success']['min_ball_speed']} m/s 且方向偏差 ≤ {tc['success']['max_dir_err_deg']}°。", "",
         "## 成功率（行 = x，列 = y）", "", md_table(["", *[f"y {win[1] + oy:.3f}" for oy in offs]], rows), ""]
    def summ(rs, label):
        ok = [r for r in rs if r["success"]]
        return [label, f"{len(ok)}/{len(rs)}", f"{np.mean([r['ball_peak_speed'] for r in rs]):.2f}",
                f"{np.mean([r['ball_travel_m'] for r in rs]):.2f}",
                f"{np.nanmean([r['dir_err_deg'] for r in rs]):+.1f} ± {np.nanstd([r['dir_err_deg'] for r in rs]):.1f}",
                f"{np.mean([r['t_peak_s'] for r in rs]):.2f}", f"{np.mean([r['motion_end_s'] for r in rs]):.2f}",
                sum(r["fell"] for r in rs), f"{min(r['hip_yaw_min_margin_rad'] for r in rs):.3f}"]
    L += [md_table(["工况", "成功", "球峰值速度 m/s", "球位移 m", "方向偏差 °（峰值速度方向 − 航向，负 = 偏右）", "踢中时刻 s", "动作结束 s", "摔倒", "hip_yaw 离限位最小余量 rad"],
                   [summ(centre, "站立起步，窗口中心"), summ(res, "站立起步，全部 25 个位置"),
                    summ(walk, f"走 {tc['walk_before']['walk_s']} s 后立即 KICK"),
                    summ([r for r in verify if not r["after_walk"]], f"验证：duration_s={duration}，站立起步"),
                    summ([r for r in verify if r["after_walk"]], f"验证：duration_s={duration}，走后 KICK")]), "",
          f"走后 KICK：play 开始时机身已静止（< 3 cm/s）的比例 {sum(bool(r['settled_before_play']) for r in walk)}/{len(walk)}。",
          f"动作结束时刻（关节最大速度 < {tc['settle_joint_speed']} rad/s）p95 = {motion_end:.2f} s → duration_s = {duration}。",
          "视频：kick_centre.mp4、kick_after_walk.mp4"]
    md = "\n".join(L) + "\n"
    (OUT / "report.md").write_text(md)
    print(md)
    if not args.no_card_update:
        n_ok = sum(r["success"] for r in verify)
        centre = verify
        update_card_fields(CARD_DIR / "kick.yaml", {
            "duration_s": duration,
            "validated_on": f"仿真平地（bare MuJoCo + BAM，{date.today()}）：duration_s={duration} 时窗口中心 {n_ok}/{len(verify)} 成功"
                            f"（站立起步与走后各半），摔倒 {sum(r['fell'] for r in verify)}；见 outputs/kick_test/report.md",
        })
        print(f"card updated: duration_s {duration}")


if __name__ == "__main__":
    main()
