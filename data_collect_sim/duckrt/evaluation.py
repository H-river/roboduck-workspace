"""Skill acceptance tests (phase 2), shared by scripts/accept.py, eval_candidates.py and the *_test.py scripts.

Every function builds its own SimRuntime from card names / paths, runs the scenario on the
sim clock and returns plain metrics plus `pass` where configs/acceptance.yaml defines a gate.
Definitions are documented in configs/acceptance.yaml and rl_train/NOTES.md §1.1. No threshold or
test size lives in this file: everything comes from the `acc` dict (configs/acceptance.yaml,
optionally with its `quick:` section merged in, see acceptance()).

`only=` (trial keys) re-runs single trials; together with recording() that is how accept.py
makes videos of failed trials. `stand_card` is the policy used in STAND mode (default: the walk
card itself, i.e. walk with a zero command).
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from duckrt.camera import HEAD_CAM, ground_to_pixel  # noqa: E402
from duckrt.config import CONFIG_DIR, load_sim_cfg, load_yaml  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402
from duckrt.scene.builder import load_layout  # noqa: E402


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def acceptance(quick: bool = False, path: str | Path | None = None) -> dict:
    """configs/acceptance.yaml; quick=True merges its `quick:` section (smaller sizes, same gates)."""
    acc = load_yaml(path or CONFIG_DIR / "acceptance.yaml")
    return deep_merge(acc, acc.get("quick", {})) if quick else acc


def _fall_height() -> float:
    global _FALL_H
    if _FALL_H is None:
        _FALL_H = float(acceptance()["common"]["fall_trunk_height_m"])
    return _FALL_H


_FALL_H: float | None = None


# -- failure videos ---------------------------------------------------------------------------
_REC: dict | None = None


@contextlib.contextmanager
def recording(path: str | Path, fps: float, chase_cam: dict):
    """While active, every SimRuntime made by make_rt() writes a chase-camera video to `path`."""
    global _REC
    _REC = {"path": Path(path), "fps": float(fps), "cam": chase_cam, "writers": []}
    try:
        yield _REC
    finally:
        for w in _REC["writers"]:
            w.close()
        _REC = None


def _install_recorder(rt: SimRuntime) -> None:
    from duckrt.testing import VideoWriter

    rec = _REC
    w = VideoWriter(rec["path"] if not rec["writers"] else rec["path"].with_stem(f"{rec['path'].stem}_{len(rec['writers'])}"),
                    rec["fps"])
    rec["writers"].append(w)
    st = {"next": 0.0, "last": -1.0}
    cam = rec["cam"]

    def on_tick(*_):
        if rt.t < st["last"]:  # reset() rewound the clock
            st["next"] = rt.t
        st["last"] = rt.t
        if rt.t >= st["next"] - 1e-9:
            w.add(rt.backend.render_chase(cam["width"], cam["height"], cam["distance"], cam["azimuth"], cam["elevation"]))
            st["next"] += 1.0 / rec["fps"]

    rt.on_tick = on_tick


@contextlib.contextmanager
def _not_recorded(rt: SimRuntime):
    saved, rt.on_tick = rt.on_tick, None
    try:
        yield
    finally:
        rt.on_tick = saved


def _stand(stand_card):
    return "same" if stand_card in (None, "same") else stand_card


def make_rt(walk_card: str, sim_cfg: dict | None = None, stand_card: str | None = "same", kick_card: str | None = None,
            layout=None, camera_cfg=None, extra_spec_fn=None, seed: int = 0) -> SimRuntime:
    sim_cfg = sim_cfg or load_sim_cfg()
    roles = {"walk_flat": walk_card, "stand": walk_card if _stand(stand_card) == "same" else stand_card, "kick": kick_card}
    rt = SimRuntime(sim_cfg, roles=roles, layout=layout, camera_cfg=camera_cfg, extra_spec_fn=extra_spec_fn, seed=seed)
    if _REC is not None:
        _install_recorder(rt)
    return rt


def fallen(rt: SimRuntime, min_height: float | None = None) -> bool:
    b = rt.backend
    p = b.base_pos()
    h = _fall_height() if min_height is None else min_height
    return bool(rt.switcher.fall_flag or b.has_nan() or p[2] - b.ground_height(p[0], p[1]) < h)


def tilt_deg(rt: SimRuntime) -> float:
    return float(np.degrees(np.arccos(np.clip(rt.backend.up_z(), -1.0, 1.0))))


def hold(rt: SimRuntime, seconds: float, mode: Mode, cmd=(0.0, 0.0, 0.0), per_tick=None, **kw) -> None:
    rt.run(seconds, command_fn=lambda _t: (mode, cmd), per_tick=per_tick) if not kw else _run_kw(rt, seconds, mode, cmd, per_tick, kw)


def _run_kw(rt, seconds, mode, cmd, per_tick, kw):
    t_end = rt.t + seconds
    next_cmd = rt.t
    while rt.t < t_end - 1e-9:
        if rt.t >= next_cmd - 1e-9:
            rt.send(mode, cmd, **kw)
            next_cmd += rt.upper_period
        rt.tick()
        if per_tick is not None:
            per_tick(rt)


def _keep(only, key) -> bool:
    return only is None or key in [tuple(k) if isinstance(k, list) else k for k in only]


# -- random commands ------------------------------------------------------------------------
def random_walk(walk_card: str, acc: dict | None = None, seeds: int | None = None, only=None, stand_card=None) -> dict:
    """only: seeds."""
    acc = acc or acceptance()
    c = acc["walk"]["random"]
    n = seeds or c["seeds"]
    rt = make_rt(walk_card)
    r = rt.policies["walk_flat"].card.command_range
    falls, ran = [], 0
    for seed in range(n):
        rng = np.random.default_rng(1000 + seed)
        segs, t = [], 0.0
        while t < c["duration_s"]:
            d = float(rng.uniform(*c["segment_s"]))
            segs.append((t, t + d, [float(rng.uniform(*r[a])) for a in ("vx", "vy", "wz")]))
            t += d
        if not _keep(only, seed):
            continue
        ran += 1
        rt.reset(seed=seed)
        next_cmd = 0.0
        while rt.t < c["duration_s"] - 1e-9:
            if rt.t >= next_cmd - 1e-9:
                rt.send(Mode.WALK_FLAT, next(cm for a, b, cm in segs if a <= rt.t < b))
                next_cmd += rt.upper_period
            rt.tick()
            if fallen(rt, c["fall_trunk_height_m"]):
                falls.append({"seed": seed, "t": rt.t, "cmd": rt.switcher.cmd.round(4).tolist()})
                break
    return {"no_fall": ran - len(falls), "n": ran, "falls": falls, "pass": len(falls) == 0}


# -- stop -----------------------------------------------------------------------------------
def stop(walk_card: str, acc: dict | None = None, only=None, stand_card=None) -> dict:
    """only: (command index, seed) pairs."""
    acc = acc or acceptance()
    c = acc["walk"]["stop"]
    st = c["stopped"]
    rt = make_rt(walk_card)
    b = rt.backend
    trials = []
    for ci, cmd in enumerate(c["commands"]):
        for seed in range(c["seeds"]):
            if not _keep(only, (ci, seed)):
                continue
            rt.reset(seed=seed)
            hold(rt, c["walk_s"], Mode.WALK_FLAT, cmd)
            clipped = rt.switcher.cmd.round(4).tolist()
            t0 = rt.t
            log: list[tuple[float, np.ndarray]] = []
            since = t_stop = drift = None
            fell = False
            while rt.t - t0 < c["timeout_s"] + c["drift_window_s"]:
                hold(rt, rt.switcher.period, Mode.WALK_FLAT, (0.0, 0.0, 0.0))
                log.append((rt.t, b.base_pos()[:2].copy()))
                if fallen(rt):
                    fell = True
                    break
                if t_stop is None:
                    v = b.base_vel_robot()
                    still = math.hypot(v[0], v[1]) < st["speed_max_m_s"] and abs(v[2]) < st["yaw_rate_max_rad_s"]
                    since = (since if since is not None else rt.t) if still else None
                    if since is not None and rt.t - since >= st["hold_s"] - 1e-9:
                        t_stop = since
                    elif rt.t - t0 > c["timeout_s"]:
                        break
                elif rt.t >= t_stop + c["drift_window_s"] - 1e-9:
                    p_stop = min(log, key=lambda x: abs(x[0] - t_stop))[1]
                    drift = float(np.linalg.norm(log[-1][1] - p_stop))
                    break
            stop_s = None if t_stop is None else t_stop - t0
            ok = (not fell and stop_s is not None and stop_s <= c["max_stop_s"] + 1e-9
                  and drift is not None and drift < c["max_drift_m"])
            trials.append({"ci": ci, "cmd": cmd, "clipped": clipped, "seed": seed, "stop_s": stop_s, "drift_m": drift,
                           "fell": fell, "pass": bool(ok)})
    worst = max((t["stop_s"] if t["stop_s"] is not None else 99.0) for t in trials)
    return {"trials": trials, "max_stop_s": worst, "max_drift_m": max((t["drift_m"] or 0.0) for t in trials),
            "n_pass": sum(t["pass"] for t in trials), "n": len(trials), "pass": all(t["pass"] for t in trials)}


# -- symmetry -------------------------------------------------------------------------------
def _mean_vel(rt, cmd, duration, warmup, seed, axis):
    rt.reset(seed=seed)
    hold(rt, warmup, Mode.WALK_FLAT, cmd)
    vals = []
    hold(rt, duration, Mode.WALK_FLAT, cmd, per_tick=lambda r: vals.append(r.backend.base_vel_robot()[axis]))
    return float(np.mean(vals)), fallen(rt)


def symmetry(walk_card: str, acc: dict | None = None, only=None, stand_card=None) -> dict:
    acc = acc or acceptance()
    c = acc["walk"]["symmetry"]
    rt = make_rt(walk_card)
    s = c["straight"]
    straight = [_mean_vel(rt, s["cmd"], s["duration_s"], c["warmup_s"], seed, 2)[0] for seed in range(s["seeds"])]
    straight_ok = all(abs(w) < s["max_abs_yaw_rate"] for w in straight)
    turns = []
    for cmd in c["turn"]["cmds"]:
        neg = [cmd[0], cmd[1], -cmd[2]]
        wp = np.mean([_mean_vel(rt, cmd, c["turn"]["duration_s"], c["warmup_s"], sd, 2)[0] for sd in range(c["turn"]["seeds"])])
        wn = np.mean([_mean_vel(rt, neg, c["turn"]["duration_s"], c["warmup_s"], sd, 2)[0] for sd in range(c["turn"]["seeds"])])
        ratio = abs(wp) / max(abs(wn), 1e-6)
        lo, hi = c["turn"]["ratio_range"]
        turns.append({"cmd": cmd, "w_pos": float(wp), "w_neg": float(wn), "ratio": float(ratio), "pass": bool(lo <= ratio <= hi)})
    lat = c["lateral"]
    vp = np.mean([_mean_vel(rt, lat["cmd"], lat["duration_s"], c["warmup_s"], sd, 1)[0] for sd in range(lat["seeds"])])
    neg = [lat["cmd"][0], -lat["cmd"][1], lat["cmd"][2]]
    vn = np.mean([_mean_vel(rt, neg, lat["duration_s"], c["warmup_s"], sd, 1)[0] for sd in range(lat["seeds"])])
    return {"straight_yaw_rates": straight, "straight_pass": straight_ok, "turns": turns,
            "lateral": {"v_pos": float(vp), "v_neg": float(vn), "ratio": float(abs(vp) / max(abs(vn), 1e-6))},
            "pass": straight_ok and all(t["pass"] for t in turns)}


def walk_clip(walk_card: str, cmd, seconds: float, seed: int = 0) -> None:
    """Walk at `cmd` for `seconds` (video helper for aggregate tests such as symmetry)."""
    rt = make_rt(walk_card)
    rt.reset(seed=seed)
    hold(rt, seconds, Mode.WALK_FLAT, cmd)


# -- command tracking (recorded only) ---------------------------------------------------------
def tracking(walk_card: str, acc: dict | None = None, only=None, stand_card=None) -> dict:
    acc = acc or acceptance()
    t = acc["walk"]["tracking"]
    rt = make_rt(walk_card)
    rows = []
    for cmd in t["commands"]:
        means, fell, clipped = [], False, None
        for seed in range(t["seeds"]):
            rt.reset(seed=seed)
            log = []
            hold(rt, t["hold_s"], Mode.WALK_FLAT, cmd, per_tick=lambda r: log.append((r.t, *r.backend.base_vel_robot())))
            clipped = rt.switcher.cmd.round(4).tolist()
            a = np.array(log)
            means.append(a[a[:, 0] >= t["hold_s"] - t["measure_last_s"], 1:4].mean(0))
            fell |= fallen(rt)
        m = np.mean(means, 0)
        rows.append({"cmd": cmd, "clipped": clipped, "mean": m.round(4).tolist(), "err": (m - np.array(clipped)).round(4).tolist(),
                     "fell": bool(fell)})
    return {"rows": rows, "pass": None}


# -- ramps (heading hold) -------------------------------------------------------------------
def steer_cmd(rt: SimRuntime, cmd_vx: float, origin, ryaw: float, direction: str, sc: dict, wz_max: float):
    """Executor-like hold: steer back to the ramp centre line and along it (privileged pose)."""
    b = rt.backend
    p = b.base_pos()
    path_yaw = ryaw if direction == "up" else ryaw + math.pi
    lat = -(p[0] - origin[0]) * math.sin(path_yaw) + (p[1] - origin[1]) * math.cos(path_yaw)
    lim = math.radians(sc["max_heading_corr_deg"])
    corr = max(-lim, min(lim, -sc["k_lateral_rad_per_m"] * lat))
    err = math.atan2(math.sin(path_yaw + corr - b.base_yaw()), math.cos(path_yaw + corr - b.base_yaw()))
    return cmd_vx, 0.0, max(-wz_max, min(wz_max, sc["k_heading"] * err))


def ramp_run(rt: SimRuntime, direction: str, cmd_vx: float, seed: int, rc: dict, steer: bool = True) -> dict:
    b = rt.backend
    rng = np.random.default_rng(10_000 + seed)
    start = copy.deepcopy(b.world.robot_start)
    start.pos[:2] += rng.uniform(-1, 1, 2) * rc["start_noise"]["xy_m"]
    start.yaw += math.radians(rng.uniform(-1, 1) * rc["start_noise"]["yaw_deg"])
    rt.reset(seed=seed, start=start)
    b.actuator.reset_stats()
    s = b.module_state("ramp")
    origin, ryaw, rp = s["pos"][:2], s["yaw"], s["params"]
    L, T = rp["slope_length_m"], rp["top_length_m"]
    goal_u = L + T / 2 if direction == "up" else -0.25
    wz_max = rt.policies["walk_flat"].card.command_range["wz"][1]
    fall, t_done = None, None
    while rt.t < rc["timeout_s"]:
        if rt.ticks % 5 == 0:
            cmd = steer_cmd(rt, cmd_vx, origin, ryaw, direction, rc["steer"], wz_max) if steer else (cmd_vx, 0.0, 0.0)
            rt.send(Mode.WALK_FLAT, cmd)
        rt.tick()
        p = b.base_pos()
        u = (p[0] - origin[0]) * math.cos(ryaw) + (p[1] - origin[1]) * math.sin(ryaw)
        if fallen(rt, rc["fall_trunk_height_m"]):
            fall = {"t": rt.t, "u_m": float(u)}
            break
        if (direction == "up" and u >= goal_u) or (direction == "down" and u <= goal_u):
            t_done = rt.t
            break
    return {"success": t_done is not None, "time_s": t_done, "fall": fall,
            "current_limited_pct": float(100 * b.actuator.stat_current_limited.max() / max(b.actuator.stat_steps, 1))}


def ramps(walk_card: str, acc: dict | None = None, angles=None, runs: int | None = None, only=None, stand_card=None) -> dict:
    """only: (angle, direction, run) triples."""
    acc = acc or acceptance()
    a = acc["walk"]["ramp"]
    rc = load_yaml(CONFIG_DIR / "tests.yaml")["ramp_test"]
    n = runs or a["runs"]
    out = {}
    for ang in (angles if angles is not None else a["gate_angles_deg"] + a["record_angles_deg"]):
        for direction in ("up", "down"):
            idx = [i for i in range(n) if _keep(only, (float(ang), direction, i)) or _keep(only, (ang, direction, i))]
            if not idx:
                continue
            layout = load_layout(rc[direction]["layout"], {"ramp": {"angle_deg": float(ang)}})
            rt = make_rt(walk_card, layout=layout)
            vx = a["speed_fraction"] * rt.policies["walk_flat"].card.command_range["vx"][1]
            res = [dict(ramp_run(rt, direction, vx, i, rc), run=i) for i in idx]
            t = [r["time_s"] for r in res if r["success"]]
            out[f"{ang:g}_{direction}"] = {"angle": ang, "direction": direction, "success": sum(r["success"] for r in res), "n": len(res),
                                            "falls": sum(r["fall"] is not None for r in res), "time_mean_s": float(np.mean(t)) if t else None,
                                            "current_limited_max_pct": max(r["current_limited_pct"] for r in res),
                                            "failed_runs": [r["run"] for r in res if not r["success"]]}
    gate = [out[f"{g:g}_{d}"] for g in a["gate_angles_deg"] for d in ("up", "down") if f"{g:g}_{d}" in out]
    return {"results": out, "pass": all(r["success"] == r["n"] for r in gate) if gate else None}


# -- head takeover --------------------------------------------------------------------------
def _walk_speed(rt, wa) -> float:
    vals = []
    hold(rt, wa["duration_s"], Mode.WALK_FLAT, wa["cmd"], per_tick=lambda r: vals.append(r.backend.base_vel_robot()[0]))
    k = int(wa["measure_last_s"] / rt.switcher.period)
    return float(np.mean(vals[-k:]))


def head_override(walk_card: str, acc: dict | None = None, trials: int | None = None, cam_cfg: dict | None = None,
                  only=None, stand_card=None) -> dict:
    """only: trial indices."""
    acc = acc or acceptance()
    c = acc["walk"]["head_override"]
    n = trials or c["trials"]
    cam_cfg = cam_cfg or load_yaml(CONFIG_DIR / "camera.yaml")
    rt = make_rt(walk_card, stand_card=_stand(stand_card), camera_cfg=cam_cfg)
    wa = c["walk_after"]
    base = []
    with _not_recorded(rt):
        for seed in range(min(n, wa["baseline_seeds"])):
            rt.reset(seed=seed)
            hold(rt, c["settle_s"], Mode.STAND)
            base.append(_walk_speed(rt, wa))
    baseline = float(np.mean(base))
    res = []
    for seed in range(n):
        if not _keep(only, seed):
            continue
        rt.reset(seed=100 + seed)
        hold(rt, c["settle_s"], Mode.STAND)
        p0 = rt.backend.base_pos()[:2].copy()
        mx = {"tilt": 0.0, "disp": 0.0, "fell": False}
        pitches = []

        def mon(r):
            mx["tilt"] = max(mx["tilt"], tilt_deg(r))
            mx["disp"] = max(mx["disp"], float(np.linalg.norm(r.backend.base_pos()[:2] - p0)))
            mx["fell"] |= fallen(r)

        def mon_cam(r):
            mon(r)
            f = r.camera.extrinsics().T_world_cam[:3, 2]
            pitches.append(-math.degrees(math.asin(max(-1.0, min(1.0, f[2])))))

        hold(rt, c["hold_s"], Mode.STAND, per_tick=mon_cam, head_pitch_down_deg=c["pitch_down_deg"])
        hold(rt, c["after_restore_s"], Mode.STAND, per_tick=mon)
        stable = not mx["fell"] and mx["tilt"] < c["stable"]["max_tilt_deg"] and mx["disp"] < c["stable"]["max_displacement_m"]
        v = _walk_speed(rt, wa)
        fell_after = fallen(rt)
        ratio = v / baseline if baseline > 1e-6 else float("nan")
        start_ok = (not fell_after) and wa["min_ratio"] <= ratio <= wa["max_ratio"]
        res.append({"seed": seed, "stable": bool(stable), "fell": bool(mx["fell"] or fell_after), "max_tilt_deg": mx["tilt"],
                    "max_disp_m": mx["disp"], "cam_pitch_during_deg": float(np.mean(pitches[len(pitches) // 2:])),
                    "walk_after_vx": v, "ratio": ratio, "start_ok": bool(start_ok), "pass": bool(stable and start_ok)})
    return {"baseline_vx": baseline, "trials": res, "n_pass": sum(r["pass"] for r in res), "n": len(res),
            "pass": all(r["pass"] for r in res)}


# -- walk <-> stand hot switch ----------------------------------------------------------------
def walk_stand_switch(walk_card: str, acc: dict | None = None, only=None, stand_card=None) -> dict:
    """STAND -> WALK_FLAT (random command, random duration) -> STAND -> WALK_FLAT. only: trial indices."""
    acc = acc or acceptance()
    c = acc["walk"]["switch"]
    rt = make_rt(walk_card, stand_card=_stand(stand_card))
    r = rt.policies["walk_flat"].card.command_range
    trials = []
    for i in range(c["trials"]):
        rng = np.random.default_rng(2000 + i)
        cmd1, cmd2 = ([float(rng.uniform(*r[a]) * c["cmd_fraction"]) for a in ("vx", "vy", "wz")] for _ in range(2))
        tw, ts = float(rng.uniform(*c["walk_s"])), float(rng.uniform(*c["stand_s"]))
        if not _keep(only, i):
            continue
        rt.reset(seed=i)
        mx = {"tilt": 0.0, "fell": False, "t": None}

        def mon(rr):
            mx["tilt"] = max(mx["tilt"], tilt_deg(rr))
            if not mx["fell"] and fallen(rr):
                mx["fell"], mx["t"] = True, rr.t

        for seconds, mode, cmd in ((c["stand_first_s"], Mode.STAND, (0.0, 0.0, 0.0)), (tw, Mode.WALK_FLAT, cmd1),
                                   (ts, Mode.STAND, (0.0, 0.0, 0.0)), (c["walk_again_s"], Mode.WALK_FLAT, cmd2)):
            hold(rt, seconds, mode, cmd, per_tick=mon)
            if mx["fell"]:
                break
        trials.append({"trial": i, "cmd1": np.round(cmd1, 3).tolist(), "cmd2": np.round(cmd2, 3).tolist(), "walk_s": tw,
                       "stand_s": ts, "fell": mx["fell"], "fall_t": mx["t"], "max_tilt_deg": mx["tilt"], "pass": not mx["fell"]})
    return {"trials": trials, "n_pass": sum(t["pass"] for t in trials), "n": len(trials), "pass": all(t["pass"] for t in trials)}


# -- camera ---------------------------------------------------------------------------------
def _mats(distances, size):
    def fn(spec: mujoco.MjSpec) -> None:
        spec.visual.quality.offsamples = 0
        for d in distances:
            g = spec.worldbody.add_geom()
            g.name = f"mat_{d:g}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [size / 2, size / 2, 0.001]
            g.pos = [d, 0.0, 0.001]
            g.rgba = [0.85, 0.1, 0.1, 1]
            g.contype = g.conaffinity = 0
    return fn


def cam_angles(T_wc: np.ndarray) -> tuple[float, float]:
    R = T_wc[:3, :3]
    f, r, d = R[:, 2], R[:, 0], R[:, 1]
    return -math.degrees(math.asin(np.clip(f[2], -1, 1))), math.degrees(math.atan2(-r[2], -d[2]))


def camera(walk_card: str, acc: dict | None = None, cam_cfg: dict | None = None, only=None, stand_card=None,
           save_png: str | Path | None = None) -> dict:
    acc = acc or acceptance()
    c = acc["walk"]["camera"]
    cam_cfg = cam_cfg or load_yaml(CONFIG_DIR / "camera.yaml")
    size = load_yaml(CONFIG_DIR / "tests.yaml")["camera_check"]["visibility"]["flag_mat_m"]
    rt = make_rt(walk_card, stand_card=_stand(stand_card), camera_cfg=cam_cfg, extra_spec_fn=_mats(c["flag_distances_m"], size))
    b, cam = rt.backend, rt.camera
    hold(rt, 2.0, Mode.STAND)
    seg = b.render(HEAD_CAM, cam.width, cam.height, segmentation=True)
    if save_png is not None:
        import imageio.v2 as imageio

        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(save_png), b.render(HEAD_CAM, cam.width, cam.height))
    ext = cam.extrinsics()
    vis = {}
    for d in c["flag_distances_m"]:
        gid = b.model.geom(f"mat_{d:g}").id
        v, u = np.nonzero(seg == gid)
        corners = [np.array([d + sx * size / 2, sy * size / 2, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
        inside = sum(ground_to_pixel(p, cam.K, ext.T_world_cam, cam.width, cam.height) is not None for p in corners)
        vis[f"{d:g}"] = {"fully_visible": bool(inside == 4 and len(u) > 0), "corners_in_image": inside,
                         "width_px": int(u.max() - u.min() + 1) if len(u) else 0}
    stand_pitch = []
    hold(rt, 2.0, Mode.STAND, per_tick=lambda r: stand_pitch.append(cam_angles(r.camera.extrinsics().T_world_cam)[0]))
    jitter = {}
    for sp in c["jitter_speeds"]:
        rt.reset(seed=0)
        hold(rt, 2.0, Mode.WALK_FLAT, (sp, 0, 0))
        a = []
        hold(rt, c["jitter_s"], Mode.WALK_FLAT, (sp, 0, 0),
             per_tick=lambda r: a.append((*cam_angles(r.camera.extrinsics().T_world_cam), r.camera.extrinsics().T_world_cam[2, 3])))
        a = np.array(a)
        jitter[f"{sp:g}"] = {"pitch_mean_deg": float(a[:, 0].mean()), "pitch_std_deg": float(a[:, 0].std()),
                             "pitch_p2p_deg": float(np.ptp(a[:, 0])), "roll_std_deg": float(a[:, 1].std()),
                             "roll_p2p_deg": float(np.ptp(a[:, 1])), "height_std_mm": float(a[:, 2].std() * 1000)}
    walk_pitch = jitter[f"{0.2:g}"]["pitch_mean_deg"] if f"{0.2:g}" in jitter else float("nan")
    gate = vis[f"{c['gate_flag_m']:g}"]["fully_visible"]
    return {"flags": vis, "stand_pitch_deg": float(np.mean(stand_pitch)), "walk_pitch_deg": walk_pitch,
            "stand_minus_walk_pitch_deg": float(np.mean(stand_pitch) - walk_pitch), "jitter": jitter, "pass": bool(gate)}


TESTS = {"random": random_walk, "stop": stop, "symmetry": symmetry, "ramp": ramps, "head_override": head_override,
         "camera": camera, "tracking": tracking, "switch": walk_stand_switch}
LEGACY_WALK_TESTS = ["random", "stop", "symmetry", "ramp", "head_override", "camera"]   # eval_candidates.py's set


def walk_suite(walk_card: str, which: list[str] | None = None, acc: dict | None = None, stand_card=None) -> dict:
    acc = acc or acceptance()
    out = {}
    for name in which or LEGACY_WALK_TESTS:
        out[name] = TESTS[name](walk_card, acc, stand_card=stand_card)
    out["all_pass"] = all(v.get("pass") is not False for v in out.values() if isinstance(v, dict))
    return out


# -- kick ------------------------------------------------------------------------------------
def _ball(rt: SimRuntime):
    m = rt.backend.model
    j = m.joint("mod_ball_free").id
    return m.jnt_qposadr[j], m.jnt_dofadr[j], float(m.geom_size[m.geom("mod_ball_sphere").id][0])


def place_ball_robot_frame(rt: SimRuntime, xy) -> np.ndarray:
    b = rt.backend
    qa, va, r = _ball(rt)
    root, yaw = b.base_pos(), b.base_yaw()
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.array([root[0] + c * xy[0] - s * xy[1], root[1] + s * xy[0] + c * xy[1], r])
    b.data.qpos[qa:qa + 3] = p
    b.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    b.data.qvel[va:va + 6] = 0
    mujoco.mj_forward(b.model, b.data)
    return p


def ball_in_robot_frame(rt: SimRuntime) -> np.ndarray:
    b = rt.backend
    qa, _, _ = _ball(rt)
    d = b.data.qpos[qa:qa + 2] - b.base_pos()[:2]
    yaw = b.base_yaw()
    return np.array([math.cos(yaw) * d[0] + math.sin(yaw) * d[1], -math.sin(yaw) * d[0] + math.cos(yaw) * d[1]])


def run_kick(rt: SimRuntime, ball_xy, rng: np.random.Generator, noise: float, succ: dict, head_down: dict | None = None) -> dict:
    """Ball at ball_xy (robot frame) -> [optional head takeover: measure, release] -> KICK with the
    measured position -> sequence done. Kick direction from the ball velocity at its peak speed."""
    b = rt.backend
    place_ball_robot_frame(rt, ball_xy)
    if head_down:
        hold(rt, head_down["hold_s"], Mode.STAND, head_pitch_down_deg=head_down["pitch_down_deg"])
        hold(rt, head_down["restore_s"], Mode.STAND)
    meas = ball_in_robot_frame(rt) + rng.uniform(-noise, noise, 2)
    heading = b.base_yaw()
    _, va, _ = _ball(rt)
    rt.send(Mode.KICK, kick_ball_xy=tuple(meas))
    peak, vpk, fell = 0.0, None, False
    while True:
        rt.tick()
        v = b.data.qvel[va:va + 2].copy()
        if np.linalg.norm(v) > peak:
            peak, vpk = float(np.linalg.norm(v)), v
        fell |= fallen(rt)
        if rt.switcher.kick is None or fell:
            break
        if rt.ticks % 5 == 0:
            rt.send(Mode.STAND)  # ignored during the sequence (keeps the watchdog fed afterwards)
    err = None
    if vpk is not None and peak > 1e-3:
        a = math.atan2(vpk[1], vpk[0]) - heading
        err = math.degrees(math.atan2(math.sin(a), math.cos(a)))
    ok = (not fell) and peak >= succ["min_ball_speed"] and err is not None and abs(err) <= succ["max_dir_err_deg"]
    return {"ball_xy": list(map(float, ball_xy)), "measured_xy": meas.round(4).tolist(), "peak_speed": peak,
            "dir_err_deg": err, "fell": fell, "success": bool(ok)}


def _kick_setup(walk_card, kick_card, acc, stand_card=None):
    acc = acc or acceptance()
    c = acc["kick"]
    rt = make_rt(walk_card, stand_card=_stand(stand_card), kick_card=kick_card, layout=load_layout("kick_test"))
    card = rt.policies["kick"].card
    dflt = c["card_defaults"]
    foot_y = card.task.get("foot_line_y", card.task["spawn"]["offset_xy"][1] if "spawn" in card.task else dflt["foot_line_y"])
    noise = float(card.task.get("obs_noise_m", dflt["obs_noise_m"]))
    xs = np.linspace(*c["window"]["x"], c["window"]["grid"])
    return c, rt, foot_y, noise, xs


def kick_window(walk_card: str, kick_card: str, acc: dict | None = None, quick: bool = False, only=None, stand_card=None) -> dict:
    """only: grid cell indices (x-major, then y, then seed). quick=True: the `quick:` sizes of acceptance.yaml."""
    acc = acc or acceptance(quick=quick)
    c, rt, foot_y, noise, xs = _kick_setup(walk_card, kick_card, acc, stand_card)
    ys = foot_y + np.linspace(*c["window"]["y_offset"], c["window"]["grid"])
    grid, k = [], -1
    for x in xs:
        for y in ys:
            for sd in range(c["window"]["seeds"]):
                k += 1
                if not _keep(only, k):
                    continue
                rt.reset(seed=sd)
                hold(rt, c["stand_before_s"], Mode.STAND)
                grid.append(dict(run_kick(rt, (x, y), np.random.default_rng(sd * 1000 + k), noise, c["success"]), cell=k))
    succ = [g for g in grid if g["success"]]
    errs = [g["dir_err_deg"] for g in succ]
    rate = len(succ) / len(grid)
    bias = float(np.mean(errs)) if errs else float("nan")
    return {"n": len(grid), "success_rate": rate, "bias_deg": bias,
            "mean_abs_err_deg": float(np.mean(np.abs(errs))) if errs else float("nan"),
            "falls": sum(g["fell"] for g in grid), "grid": grid,
            "pass_rate": bool(rate >= c["gate_success_rate"]), "pass_dir": bool(abs(bias) < c["gate_mean_abs_dir_deg"]),
            "pass": bool(rate >= c["gate_success_rate"] and abs(bias) < c["gate_mean_abs_dir_deg"])}


def kick_head_down(walk_card: str, kick_card: str, acc: dict | None = None, quick: bool = False, only=None, stand_card=None) -> dict:
    """Same positions (foot line, x across the window), direct kick vs head takeover -> release -> kick."""
    acc = acc or acceptance(quick=quick)
    c, rt, foot_y, noise, xs = _kick_setup(walk_card, kick_card, acc, stand_card)
    hc = c["head_down_then_kick"]
    direct, hd = [], []
    for i, x in enumerate(xs):
        for sd in range(hc["seeds"]):
            for lst, head in ((direct, None), (hd, hc)):
                rt.reset(seed=100 + sd)
                hold(rt, c["stand_before_s"], Mode.STAND)
                lst.append(run_kick(rt, (x, foot_y), np.random.default_rng(i * 10 + sd), noise, c["success"], head_down=head))
    r_d = sum(g["success"] for g in direct) / len(direct)
    r_h = sum(g["success"] for g in hd) / len(hd)
    return {"direct_rate": r_d, "head_down_rate": r_h, "n": len(direct), "pass": bool(r_h >= r_d)}


def kick_handoff(walk_card: str, kick_card: str, acc: dict | None = None, quick: bool = False, only=None, stand_card=None) -> dict:
    """walk -> stand -> kick -> stand -> walk, `cycles` times in one continuous run."""
    acc = acc or acceptance(quick=quick)
    c, rt, foot_y, noise, xs = _kick_setup(walk_card, kick_card, acc, stand_card)
    h = c["handoff"]
    rt.reset(seed=7)
    rng = np.random.default_rng(7)
    log = []
    for i in range(h["cycles"]):
        hold(rt, h["walk_s"], Mode.WALK_FLAT, h["walk_cmd"])
        hold(rt, h["stand_s"], Mode.STAND)
        r = run_kick(rt, (float(np.mean(xs)), foot_y), rng, noise, c["success"])
        hold(rt, h["after_s"], Mode.STAND)
        fell = r["fell"] or fallen(rt)
        log.append({"cycle": i, "kick_success": r["success"], "fell": bool(fell)})
        if fell:
            rt.reset(seed=100 + i)
    falls = sum(x["fell"] for x in log)
    return {"cycles": h["cycles"], "falls": falls, "kick_success": sum(x["kick_success"] for x in log), "log": log, "pass": falls == 0}


KICK_TESTS = {"kick_window": kick_window, "kick_head_down": kick_head_down, "kick_handoff": kick_handoff}


def kick_suite(walk_card: str, kick_card: str, acc: dict | None = None, quick: bool = False, stand_card=None) -> dict:
    acc = acc or acceptance(quick=quick)
    out = {"window": kick_window(walk_card, kick_card, acc, stand_card=stand_card),
           "head_down": kick_head_down(walk_card, kick_card, acc, stand_card=stand_card),
           "handoff": kick_handoff(walk_card, kick_card, acc, stand_card=stand_card)}
    out["pass"] = all(v["pass"] for v in out.values())
    return out
