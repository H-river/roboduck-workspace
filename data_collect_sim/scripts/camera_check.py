"""Head camera check (step 2).

1. projection: small markers at known ground positions; render a segmentation image, take
   each marker's pixel centroid, pixel_to_ground() it and compare with the truth (< 1 cm);
   ground_to_pixel() of the truth vs the centroid as the reverse check. Standing, for the
   default camera and a tilted one, plus one frame while walking (head moving).
2. visibility: 40 x 40 cm flag mats at 0.5 / 1 / 2 / 3 m and the ball at the kick window
   (kick card spawn centre) and a few distances ahead: pixel width, fully visible or not.
   A pitch x fovy sweep shows what it would take to see the kick-window ball.
3. jitter: camera roll / pitch while walking at 3 speeds: std, peak-to-peak, main frequency.

    uv run python scripts/camera_check.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from duckrt.camera import HEAD_CAM, TOP_CAM, ground_to_pixel, intrinsics, pixel_to_ground, world_to_robot  # noqa: E402
from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml, output_dir  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.policy_card import load_card  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.testing import md_table  # noqa: E402

OUT = output_dir("camera_check")
MIN_PIXELS = 10


def add_markers(points: list[tuple[str, tuple[float, float, float], float, list[float]]]):
    """extra_spec_fn adding flat discs / boxes / spheres that do not collide."""

    def fn(spec: mujoco.MjSpec) -> None:
        # no MSAA: multisampling blends the id colours of the segmentation pass at edges,
        # which corrupts the ids of small markers
        spec.visual.quality.offsamples = 0
        for name, pos, size, rgba in points:
            g = spec.worldbody.add_geom()
            g.name = name
            if name.startswith("mat"):
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = [size / 2, size / 2, 0.001]
            elif name.startswith("ball"):
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [size, 0, 0]
            else:
                g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                g.size = [size, 0.0005, 0]
            g.pos = list(pos)
            g.rgba = rgba
            g.contype = 0
            g.conaffinity = 0
    return fn


ROLES = None  # set from --card


def make_rt(sim_cfg, cam_cfg, extra=None, pitch=None, fovy=None) -> SimRuntime:
    cc = copy.deepcopy(cam_cfg)
    if pitch is not None:
        cc["head_camera"]["pitch_down_deg"] = float(pitch)
    if fovy is not None:
        cc["head_camera"]["fovy_deg"] = float(fovy)
    return SimRuntime(sim_cfg, camera_cfg=cc, render_head=False, extra_spec_fn=extra, roles=ROLES)


def stand(rt: SimRuntime, seconds: float) -> None:
    rt.run(seconds, command_fn=lambda _t: (Mode.STAND, (0, 0, 0)))


def centroids(seg: np.ndarray, gids: dict[str, int]) -> dict[str, tuple[float, float, int]]:
    out = {}
    for name, gid in gids.items():
        v, u = np.nonzero(seg == gid)
        if len(u):
            out[name] = (float(u.mean()), float(v.mean()), int(len(u)))
    return out


def cam_angles(T_wc: np.ndarray) -> tuple[float, float]:
    """(pitch_down_deg, roll_deg) of an OpenCV camera pose (x right, y down, z forward)."""
    R = T_wc[:3, :3]
    f, r, d = R[:, 2], R[:, 0], R[:, 1]
    pitch_down = -math.degrees(math.asin(np.clip(f[2], -1, 1)))
    roll = math.degrees(math.atan2(-r[2], -d[2]))  # + = image turned clockwise
    return pitch_down, roll


# -- 1. projection ----------------------------------------------------------------------------
def projection(sim_cfg, cam_cfg, tc) -> dict:
    pc = tc["projection"]
    res = {"variants": []}
    for var in pc["variants"]:
        pts = [(f"m_{x:.2f}_{y:+.2f}", (x, y, 0.0005), pc["marker_radius_m"], [1, 0.1, 0.8, 1])
               for x in var["xs"] for y in var["ys"]]
        rt = make_rt(sim_cfg, cam_cfg, add_markers(pts), pitch=var["pitch_down_deg"])
        stand(rt, tc["settle_s"])
        lab = f"pitch{var['pitch_down_deg'] or 0:g}_standing"
        for sc in (pc["verify_scale"], 1):
            res["variants"].append(check_frame(rt, pts, lab, sc))
        if var["pitch_down_deg"] is not None:  # same markers, one frame mid-walk
            wf = pc["walking_frame"]
            rt.run(wf["at_s"], command_fn=lambda _t: (Mode.WALK_FLAT, wf["cmd"]))
            for sc in (pc["verify_scale"], 1):
                res["variants"].append(check_frame(rt, pts, f"pitch{var['pitch_down_deg']:g}_walking", sc))
    verify = [v for v in res["variants"] if v["scale"] == pc["verify_scale"] and v["n"]]
    res["max_ground_err_m"] = max(v["max_ground_err_m"] for v in verify)
    res["pass"] = res["max_ground_err_m"] < pc["tolerance_m"]
    return res


def check_frame(rt: SimRuntime, pts, label: str, scale: int = 1) -> dict:
    """Markers whose blob touches the image border or has < min_pixels are skipped
    (their centroid is biased / quantization-limited)."""
    b, cam = rt.backend, rt.camera
    m = b.model
    W, H = cam.width * scale, cam.height * scale
    K = intrinsics(cam.fovy, W, H)
    gids = {name: m.geom(name).id for name, *_ in pts}
    seg = b.render(HEAD_CAM, W, H, segmentation=True)
    img = b.render(HEAD_CAM, W, H)
    ext = cam.extrinsics()
    rows, gerr, perr = [], [], []
    skipped = 0
    for name, pos, *_ in pts:
        v_, u_ = np.nonzero(seg == gids[name])
        if len(u_) == 0:
            continue
        if len(u_) < MIN_PIXELS or u_.min() == 0 or v_.min() == 0 or u_.max() == W - 1 or v_.max() == H - 1:
            skipped += 1
            continue
        u, v, npx = float(u_.mean()), float(v_.mean()), int(len(u_))
        p = pixel_to_ground(u, v, K, ext.T_world_cam)
        uv = ground_to_pixel(np.array([pos[0], pos[1], 0.0]), K, ext.T_world_cam)
        e = float(np.linalg.norm(p[:2] - np.array(pos[:2]))) if p is not None else float("nan")
        pe = float(np.hypot(uv[0] - u, uv[1] - v)) if uv else float("nan")
        gerr.append(e)
        perr.append(pe)
        rows.append({"marker": name, "true_xy": pos[:2], "pixel": (round(u, 2), round(v, 2)), "pixels": npx,
                     "ground_err_m": e, "pixel_err_px": pe,
                     "robot_frame_xy": world_to_robot(p, ext.T_world_robot)[:2].round(4).tolist() if p is not None else None})
        img[int(round(v)), max(0, int(round(u)) - 3):int(round(u)) + 4] = [0, 255, 0]
        img[max(0, int(round(v)) - 3):int(round(v)) + 4, int(round(u))] = [0, 255, 0]
    if scale == 1:
        imageio.imwrite(OUT / f"projection_{label}.png", img)
    pd, roll = cam_angles(ext.T_world_cam)
    return {"label": label, "scale": scale, "resolution": [W, H], "n": len(rows), "of": len(pts), "skipped_edge_or_small": skipped,
            "max_ground_err_m": max(gerr) if gerr else None,
            "mean_ground_err_m": float(np.mean(gerr)) if gerr else None, "max_pixel_err_px": max(perr) if perr else None,
            "cam_height_m": float(ext.T_world_cam[2, 3]), "cam_pitch_down_deg": pd, "cam_roll_deg": roll,
            "markers": rows, "image": str(OUT / f"projection_{label}.png")}


# -- 2. visibility ----------------------------------------------------------------------------
def visibility_scene(tc, kick_card):
    vc = tc["visibility"]
    ball = kick_card.task["ball"]
    r = ball["radius"]
    spawn = kick_card.task["spawn"]["offset_xy"]
    pts = [(f"mat_{d:g}", (d, 0.0, 0.001), vc["flag_mat_m"], [0.85, 0.1, 0.1, 1]) for d in vc["flag_distances_m"]]
    pts += [("ball_kick", (spawn[0], spawn[1], r), r, ball["rgba"])]
    # extra balls off to the left (y = +0.3) so they overlap neither the mats nor the kick ball
    pts += [(f"ball_{d:g}", (d, 0.3, r), r, ball["rgba"]) for d in vc["ball_extra_distances_m"]]
    return pts


def visibility_frame(rt: SimRuntime, pts, tc, save: str | None) -> dict:
    b, cam = rt.backend, rt.camera
    m = b.model
    seg = b.render(HEAD_CAM, cam.width, cam.height, segmentation=True)
    ext = cam.extrinsics()
    half = tc["visibility"]["flag_mat_m"] / 2
    out = {}
    for name, pos, size, _ in pts:
        gid = m.geom(name).id
        v, u = np.nonzero(seg == gid)
        entry = {"pixels": int(len(u)), "width_px": int(u.max() - u.min() + 1) if len(u) else 0}
        if name.startswith("mat"):
            corners = [np.array([pos[0] + sx * half, pos[1] + sy * half, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
            inside = [ground_to_pixel(c, cam.K, ext.T_world_cam, cam.width, cam.height) is not None for c in corners]
            entry["corners_in_image"] = int(sum(inside))
            entry["fully_visible"] = bool(all(inside) and len(u) > 0)
        else:
            entry["visible"] = bool(len(u) > 0)
        out[name] = entry
    if save:
        img = b.render(HEAD_CAM, cam.width, cam.height)
        imageio.imwrite(OUT / f"{save}.png", img)
        top = b.render(TOP_CAM, 480, 320)
        imageio.imwrite(OUT / f"{save}_top.png", top)
    pd, roll = cam_angles(ext.T_world_cam)
    out["_camera"] = {"height_m": float(ext.T_world_cam[2, 3]), "pitch_down_deg": pd, "roll_deg": roll,
                      "near_ground_m": near_ground(cam, ext), "pos_robot": ext.T_robot_cam[:3, 3].round(4).tolist()}
    return out


def near_ground(cam, ext) -> float | None:
    """Robot-frame distance (x) of the closest visible ground point on the image centre column."""
    p = pixel_to_ground((cam.width - 1) / 2, cam.height - 1, cam.K, ext.T_world_cam)
    return None if p is None else float(world_to_robot(p, ext.T_world_robot)[0])


def visibility(sim_cfg, cam_cfg, tc) -> dict:
    kick_card = load_card("kick")
    pts = visibility_scene(tc, kick_card)
    rt = make_rt(sim_cfg, cam_cfg, add_markers(pts))
    stand(rt, tc["settle_s"])
    default = visibility_frame(rt, pts, tc, save="visibility_default")
    sweep = []
    vc = tc["visibility"]
    for fovy in vc["sweep_fovy_deg"]:
        for pitch in vc["sweep_pitch_down_deg"]:
            rt = make_rt(sim_cfg, cam_cfg, add_markers(pts), pitch=pitch, fovy=fovy)
            stand(rt, tc["settle_s"])
            save = f"sweep_fovy{fovy}_pitch{pitch}" if (fovy, pitch) in ((45, 0), (45, 60), (90, 30), (90, 40), (75, 40)) else None
            f = visibility_frame(rt, pts, tc, save)
            sweep.append({"fovy": fovy, "pitch_down": pitch, "ball_kick_visible": f["ball_kick"]["visible"],
                          "ball_kick_px": f["ball_kick"]["pixels"],
                          "mats_fully_visible": [d for d in vc["flag_distances_m"] if f[f"mat_{d:g}"]["fully_visible"]],
                          "mat_width_px": {d: f[f"mat_{d:g}"]["width_px"] for d in vc["flag_distances_m"]},
                          "near_ground_m": f["_camera"]["near_ground_m"], "cam_pitch_down_deg": f["_camera"]["pitch_down_deg"]})
    return {"default": default, "sweep": sweep, "kick_window_centre": kick_card.task["spawn"]["offset_xy"]}


# -- 3. jitter --------------------------------------------------------------------------------
def jitter(sim_cfg, cam_cfg, tc) -> list[dict]:
    jc = tc["jitter"]
    out = []
    for speed in [0.0] + list(jc["speeds"]):
        rt = make_rt(sim_cfg, cam_cfg)
        stand(rt, tc["settle_s"])
        mode = Mode.STAND if speed == 0 else Mode.WALK_FLAT
        log = []
        rt.run(jc["warmup_s"], command_fn=lambda _t: (mode, (speed, 0, 0)))
        rt.run(jc["duration_s"], command_fn=lambda _t: (mode, (speed, 0, 0)),
               per_tick=lambda r: log.append((*cam_angles(r.camera.extrinsics().T_world_cam), r.camera.extrinsics().T_world_cam[2, 3])))
        a = np.array(log)
        fs = 1.0 / rt.policies["walk_flat"].period
        row = {"cmd_vx": speed, "fell": bool(rt.switcher.fall_flag), "samples": len(a), "rate_hz": fs}
        for i, k in enumerate(("pitch", "roll")):
            x = a[:, i] - a[:, i].mean()
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            freqs = np.fft.rfftfreq(len(x), 1 / fs)
            spec[freqs < 0.3] = 0  # ignore slow drift
            row[k] = {"mean_deg": float(a[:, i].mean()), "std_deg": float(a[:, i].std()),
                      "p2p_deg": float(a[:, i].max() - a[:, i].min()), "main_freq_hz": float(freqs[np.argmax(spec)])}
        row["height"] = {"mean_m": float(a[:, 2].mean()), "std_mm": float(a[:, 2].std() * 1000),
                         "p2p_mm": float((a[:, 2].max() - a[:, 2].min()) * 1000)}
        out.append(row)
        print(f"  vx {speed}: pitch std {row['pitch']['std_deg']:.2f} p2p {row['pitch']['p2p_deg']:.2f} f {row['pitch']['main_freq_hz']:.2f} Hz | "
              f"roll std {row['roll']['std_deg']:.2f} p2p {row['roll']['p2p_deg']:.2f} f {row['roll']['main_freq_hz']:.2f} Hz")
    return out


def write_report(rep: dict, cam_cfg) -> str:
    hc = cam_cfg["head_camera"]
    L = [f"# camera_check — head camera {hc['width']}x{hc['height']}, fovy {hc['fovy_deg']}°, pitch_down {hc['pitch_down_deg']}° (占位值)", ""]
    p = rep["projection"]
    L += [f"## 1. 投影验证：{'PASS' if p['pass'] else 'FAIL'}（最大地面误差 {p['max_ground_err_m'] * 1000:.2f} mm，阈值 10 mm）", ""]
    L += ["验收用 4 倍分辨率渲染（投影函数和外参与分辨率无关，K 按比例缩放）；320×240 原分辨率的结果一并列出：远处 1 cm 标记只有几个像素，"
          "相机离地约 0.23 m 时 2.5 m 处 1 像素约对应 0.1 m 地面，误差主要来自像素量化。贴边或少于 10 像素的标记不计。", ""]
    L += [md_table(["工况", "分辨率", "计入标记", "最大地面误差 mm", "平均 mm", "反投影最大像素误差 px", "相机高度 m", "相机俯角 °", "横滚 °"],
                   [[v["label"], f"{v['resolution'][0]}×{v['resolution'][1]}", f"{v['n']}/{v['of']}", f"{v['max_ground_err_m'] * 1000:.2f}" if v["n"] else "-",
                     f"{v['mean_ground_err_m'] * 1000:.2f}" if v["n"] else "-", f"{v['max_pixel_err_px']:.2f}" if v["n"] else "-",
                     f"{v['cam_height_m']:.3f}", f"{v['cam_pitch_down_deg']:+.1f}", f"{v['cam_roll_deg']:+.1f}"] for v in p["variants"]]), ""]
    d = rep["visibility"]["default"]
    L += ["## 2. 可见性（默认相机，站立）", "",
          f"相机离地 {d['_camera']['height_m']:.3f} m，俯角 {d['_camera']['pitch_down_deg']:+.1f}°，"
          f"画面中下沿对应地面 {d['_camera']['near_ground_m'] if d['_camera']['near_ground_m'] is None else round(d['_camera']['near_ground_m'], 2)} m（机器人系）", ""]
    rows = []
    for k, v in d.items():
        if k.startswith("mat"):
            rows.append([f"旗垫 40×40 cm @ {k[4:]} m", v["width_px"], v["pixels"], f"{v['corners_in_image']}/4", "完整" if v["fully_visible"] else "不完整"])
        elif k.startswith("ball"):
            label = f"球 @ kick 窗口 {rep['visibility']['kick_window_centre']}" if k == "ball_kick" else f"球 @ 前方 {k[5:]} m"
            rows.append([label, v["width_px"], v["pixels"], "-", "可见" if v["visible"] else "**不可见**"])
    L += [md_table(["目标", "像素宽", "像素数", "角点在画面内", "结论"], rows), ""]
    L += ["### 俯角 × fovy 扫描（站立；kick 窗口球是否可见、哪些旗垫完整可见、画面下沿的地面距离）", ""]
    sw = rep["visibility"]["sweep"]
    L += [md_table(["fovy °(竖直)", "附加俯角 °", "kick 球可见(px)", "完整可见的旗垫 m", "下沿地面 m", "1 m 旗垫像素宽"],
                   [[s["fovy"], s["pitch_down"], f"{'是' if s['ball_kick_visible'] else '否'} ({s['ball_kick_px']})",
                     s["mats_fully_visible"], "-" if s["near_ground_m"] is None else f"{s['near_ground_m']:.2f}", s["mat_width_px"][1.0]] for s in sw]), ""]
    p2 = rep.get("phase2")
    if p2:
        L += ["## 2b. 站立姿态下的旗垫、站 / 走相机俯仰差（第二阶段新增）", "",
              md_table(["旗垫距离 m", "完整可见", "像素宽"], [[d, "是" if v["fully_visible"] else "否", v["width_px"]] for d, v in p2["flags"].items()]), "",
              f"站立相机俯角 {p2['stand_pitch_deg']:+.1f}°，行走（vx 0.2）{p2['walk_pitch_deg']:+.1f}°，差 {p2['stand_minus_walk_pitch_deg']:+.1f}°", ""]
    L += ["## 3. 步态抖动（walk_flat，相机相对重力的俯仰 / 横滚）", ""]
    L += [md_table(["vx 命令", "俯仰 std °", "俯仰 峰峰 °", "俯仰主频 Hz", "横滚 std °", "横滚 峰峰 °", "横滚主频 Hz", "高度 std mm", "高度峰峰 mm"],
                   [[j["cmd_vx"], f"{j['pitch']['std_deg']:.2f}", f"{j['pitch']['p2p_deg']:.2f}", f"{j['pitch']['main_freq_hz']:.2f}",
                     f"{j['roll']['std_deg']:.2f}", f"{j['roll']['p2p_deg']:.2f}", f"{j['roll']['main_freq_hz']:.2f}",
                     f"{j['height']['std_mm']:.1f}", f"{j['height']['p2p_mm']:.1f}"] for j in rep["jitter"]]), ""]
    return "\n".join(L) + "\n"


def main() -> None:
    import argparse

    global ROLES, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default=None, help="walk card that walks and stands (default: sim.yaml roles)")
    ap.add_argument("--quick", action="store_true", help="only the phase-2 additions + jitter (skip projection / sweep)")
    args = ap.parse_args()
    if args.card:
        ROLES = {"walk_flat": args.card, "stand": args.card}
        OUT = output_dir("camera_check", Path(args.card).stem)
    sim_cfg, cam_cfg = load_sim_cfg(), load_yaml(CONFIG_DIR / "camera.yaml")
    tc = load_yaml(CONFIG_DIR / "tests.yaml")["camera_check"]
    rep = {}
    from duckrt import evaluation

    print("[stand / walk camera, flags 1-2.5 m]")
    rep["phase2"] = evaluation.camera(args.card or sim_cfg["runtime"]["policies"]["walk_flat"])
    if args.quick:
        print(json.dumps(rep["phase2"], indent=1, default=float))
        (OUT / "report_phase2.json").write_text(json.dumps(rep["phase2"], indent=2, default=float))
        return
    print("[projection]")
    rep["projection"] = projection(sim_cfg, cam_cfg, tc)
    for v in rep["projection"]["variants"]:
        print(f"  {v['label']}: {v['n']}/{v['of']} markers, max ground err {v['max_ground_err_m']} m, max px err {v['max_pixel_err_px']}")
    print("[visibility]")
    rep["visibility"] = visibility(sim_cfg, cam_cfg, tc)
    print("[jitter]")
    rep["jitter"] = jitter(sim_cfg, cam_cfg, tc)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    md = write_report(rep, cam_cfg)
    (OUT / "report.md").write_text(md)
    print(md)
    sys.exit(0 if rep["projection"]["pass"] else 1)


if __name__ == "__main__":
    main()
