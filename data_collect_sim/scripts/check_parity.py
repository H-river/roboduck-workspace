"""Parity between the mjlab training environment and our runtime (DR off).

1. obs replay   -- put each recorded mjlab state into our MjData, build the observation
                   with our pipeline (IMU synthesized as the BNO08x would report it, deploy
                   formulas, card layout) and compare term by term.      criterion < 1e-5
2. one step     -- from each recorded state (+ BAM carried torque), apply the recorded
                   action through our BAM port and physics for one policy step and compare
                   the next joint state with mjlab's.
3. closed loop  -- start from mjlab's initial state, run our full runtime (switcher +
                   onnxruntime) with the same command and lag, compare obs / joints over time.

mjlab reference rollouts come from scripts/parity_mjlab_rollout.py (mjlab venv; run
automatically if missing). Our side uses the TRAINING model (robot_walk.xml, feet-only
contact) for the comparison; the closed loop is repeated on the project model
(allcollisions geoms) to show the effect of that choice.

    uv run python scripts/check_parity.py
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckrt.config import ROOT, load_sim_cfg, output_dir  # noqa: E402
from duckrt.messages import Mode  # noqa: E402
from duckrt.policy_card import load_card  # noqa: E402
from duckrt.runtime import SimRuntime  # noqa: E402

MJLAB_PY = ROOT.parent / "mjlab_microduck" / ".venv" / "bin" / "python"
# name: (command, trunk reset z, gravity z). "float" = no gravity, no contact: isolates the
# actuator port + articulated dynamics from the contact model.
CASES = {
    "vx0.2": ((0.2, 0.0, 0.0), 0.12, -9.81),
    "zero": ((0.0, 0.0, 0.0), 0.12, -9.81),
    "wz0.5": ((0.0, 0.0, 0.5), 0.12, -9.81),
    "float": ((0.2, 0.0, 0.0), 0.5, 0.0),
}
TOL_OBS = 1e-5


def term_slices(card) -> dict[str, slice]:
    out, i = {}, 0
    for t in card.obs_layout:
        out[t["name"]] = slice(i, i + t["dim"])
        i += t["dim"]
    return out


def ensure_rollout(name: str, cmd, z: float, gravity: float, steps: int, lag: int, vin: float, drop: float) -> Path:
    path = output_dir("parity") / f"mjlab_{name}.npz"
    if path.exists():
        d = np.load(path)
        if int(d["lag"]) == lag and len(d["obs"]) >= steps + 1 and abs(d["qpos"][0][2] - z) < 1e-6:
            return path
    card = load_card("walk_flat")
    subprocess.run([str(MJLAB_PY), str(ROOT / "scripts" / "parity_mjlab_rollout.py"), "--onnx", str(card.onnx),
                    "--cmd", *map(str, cmd), "--steps", str(steps), "--lag", str(lag), "--vin", str(vin),
                    "--drop-gain", str(drop), "--z", str(z), "--gravity", str(gravity), "--out", str(path)],
                   check=True, cwd=ROOT)
    return path


def make_runtime(sim_cfg: dict, training_model: bool, gravity: float = -9.81) -> SimRuntime:
    cfg = copy.deepcopy(sim_cfg)
    cfg["physics"]["gravity"] = [0.0, 0.0, gravity]
    if training_model:
        cfg["robot"]["mjcf"] = "../mjlab_microduck/src/mjlab_microduck/robot/microduck/robot_walk.xml"
        cfg["robot"]["joint_ranges_from"] = None
    return SimRuntime(cfg, roles={"walk_flat": "walk_flat"})


def set_state(rt: SimRuntime, qpos, qvel) -> None:
    b = rt.backend
    b.data.qpos[:] = qpos
    b.data.qvel[:] = qvel
    mujoco.mj_forward(b.model, b.data)


def obs_replay(rt: SimRuntime, rec) -> dict:
    pol = rt.policies["walk_flat"]
    sl = term_slices(pol.card)
    b = rt.backend
    errs = {k: 0.0 for k in sl}
    worst = {k: -1 for k in sl}
    n = len(rec["obs"])
    for k in range(n):
        set_state(rt, rec["qpos"][k], rec["qvel"][k])
        last = rec["action"][k - 1] if k > 0 else np.zeros(14, np.float32)
        ours = pol.obs_builder.build(b.imu_quat(), b.imu_gyro(), b.joint_pos(), b.joint_vel(), last, rec["command"][k])
        diff = np.abs(ours.astype(np.float64) - rec["obs"][k].astype(np.float64))
        for name, s in sl.items():
            if diff[s].max() > errs[name]:
                errs[name], worst[name] = float(diff[s].max()), k
    return {"steps": n, "max_abs_err": errs, "worst_step": worst, "pass": all(e < TOL_OBS for e in errs.values())}


def one_step(rt: SimRuntime, rec) -> dict:
    """Teacher-forced: state k (+ BAM delay history and carried torque) -> step -> compare k+1."""
    b = rt.backend
    pol = rt.policies["walk_flat"]
    dec = rt.decimation(pol)
    lag = b.actuator.delay.max_lag
    jp_err, jv_err, base_err = [], [], []
    for k in range(len(rec["obs"]) - 1):
        set_state(rt, rec["qpos"][k], rec["qvel"][k])
        b.actuator.delay.reset()
        if k == 0:  # mjlab's reset leaves zeros in the delay line
            b.actuator.delay.push(np.zeros(len(pol.default_q)))
        else:  # history of physics-step targets before this policy step
            prev_t = pol.default_q + rec["action"][k - 1].astype(np.float64)
            for _ in range(min(lag, dec * k)):
                b.actuator.delay.push(prev_t)
        b.actuator._prev_motor_torque = rec["prev_motor_torque"][k].copy()
        tgt = pol.default_q + rec["action"][k].astype(np.float64)
        for _ in range(dec):
            b.step_physics(tgt)
        b.forward()
        jp_err.append(np.abs(b.joint_pos() - rec["qpos"][k + 1][b.qadr]).max())
        jv_err.append(np.abs(b.joint_vel() - rec["qvel"][k + 1][b.vadr]).max())
        base_err.append(np.abs(b.base_pos() - rec["qpos"][k + 1][:3]).max())
    jp, jv, bp = map(np.asarray, (jp_err, jv_err, base_err))
    return {"joint_pos_err_rad": {"median": float(np.median(jp)), "p95": float(np.percentile(jp, 95)), "max": float(jp.max())},
            "joint_vel_err_rad_s": {"median": float(np.median(jv)), "p95": float(np.percentile(jv, 95)), "max": float(jv.max())},
            "base_pos_err_m": {"median": float(np.median(bp)), "max": float(bp.max())}}


def closed_loop(rt: SimRuntime, rec) -> dict:
    b = rt.backend
    rt.reset(seed=0)
    set_state(rt, rec["qpos"][0], rec["qvel"][0])
    b.actuator.reset()  # reset compute at mjlab's initial state (zeros target, as mjlab)
    cmd = tuple(float(c) for c in rec["command"][0])
    pol = rt.policies["walk_flat"]
    obs_err, jp_err = [], []
    n = len(rec["obs"]) - 1
    for k in range(n):
        rt.send(Mode.WALK_FLAT, cmd)
        rt.tick()
        obs_err.append(float(np.abs(pol.last_obs.astype(np.float64) - rec["obs"][k]).max()))
        jp_err.append(float(np.abs(b.joint_pos() - rec["qpos"][k + 1][b.qadr]).max()))
    first_1e5 = next((k for k, e in enumerate(obs_err) if e > 1e-5), None)
    first_1e2 = next((k for k, e in enumerate(obs_err) if e > 1e-2), None)
    base_final = (b.base_pos() - rec["qpos"][n][:3]).tolist()
    travel = (rec["qpos"][n][:3] - rec["qpos"][0][:3]).tolist()
    return {"steps": n, "obs_err_at": {str(k): obs_err[k] for k in (0, 1, 5, 10, 25, 50, 100, n - 1) if k < n},
            "first_step_obs_err_gt_1e-5": first_1e5, "first_step_obs_err_gt_1e-2": first_1e2,
            "joint_pos_err_at": {str(k): jp_err[k] for k in (0, 10, 50, 100, n - 1) if k < n},
            "base_pos_diff_final_m": base_final, "mjlab_travel_m": travel,
            "ours_travel_m": (b.base_pos() - rec["qpos"][0][:3]).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    sim_cfg = load_sim_cfg()
    a = sim_cfg["actuator"]
    lag = int(a["delay"]["steps"])
    # DR off on both sides: fixed lag, fixed vin (mjlab then sizes forcerange from that vin),
    # and mjlab's reset artifact (zeros in the delay line) reproduced.
    a["delay"] = {"mode": "fixed", "steps": lag}
    a["vin_max_for_forcerange"] = float(a["vin"])
    a["reset_target"] = "zeros"
    vin, drop = float(a["vin"]), float(a["vin_drop_gain"])
    report = {"lag_steps": lag, "vin": vin, "drop_gain": drop, "cases": {}}
    runtimes: dict[tuple[bool, float], SimRuntime] = {}
    for name, (cmd, z, gravity) in CASES.items():
        for tm in (True, False):
            if (tm, gravity) not in runtimes:
                runtimes[(tm, gravity)] = make_runtime(sim_cfg, training_model=tm, gravity=gravity)
        train_rt, proj_rt = runtimes[(True, gravity)], runtimes[(False, gravity)]
        rec = dict(np.load(ensure_rollout(name, cmd, z, gravity, args.steps, lag, vin, drop)))
        r = {"command": list(cmd), "reset_z": z, "gravity": gravity,
             "mjlab_mujoco": str(rec["mujoco_version"]), "ours_mujoco": mujoco.__version__}
        r["obs_replay"] = obs_replay(train_rt, rec)
        r["one_step"] = one_step(train_rt, rec)
        r["closed_loop_training_model"] = closed_loop(train_rt, rec)
        r["closed_loop_project_model"] = closed_loop(proj_rt, rec)
        report["cases"][name] = r
        o = r["obs_replay"]
        print(f"\n== {name} cmd={cmd}")
        print(f"  obs replay ({o['steps']} states): " + ", ".join(f"{k} {v:.1e}" for k, v in o["max_abs_err"].items())
              + f"  -> {'PASS' if o['pass'] else 'FAIL'} (< {TOL_OBS:g})")
        s = r["one_step"]
        print(f"  one step: joint pos err median {s['joint_pos_err_rad']['median']:.1e} p95 {s['joint_pos_err_rad']['p95']:.1e} "
              f"max {s['joint_pos_err_rad']['max']:.1e} rad; joint vel err median {s['joint_vel_err_rad_s']['median']:.1e} "
              f"max {s['joint_vel_err_rad_s']['max']:.1e} rad/s")
        for key in ("closed_loop_training_model", "closed_loop_project_model"):
            c = r[key]
            print(f"  {key}: obs err " + ", ".join(f"@{k} {v:.1e}" for k, v in c["obs_err_at"].items())
                  + f"; first >1e-5 at step {c['first_step_obs_err_gt_1e-5']}, >1e-2 at {c['first_step_obs_err_gt_1e-2']}; "
                  f"travel mjlab {np.round(c['mjlab_travel_m'][:2], 3).tolist()} ours {np.round(c['ours_travel_m'][:2], 3).tolist()}")
    out = output_dir("parity") / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    ok = all(c["obs_replay"]["pass"] for c in report["cases"].values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
