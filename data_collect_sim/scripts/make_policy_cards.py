"""Generate configs/policies/<name>.yaml from the training configs.

Every value is read from the dumped training config (params/env.yaml), the training
source files (parsed with ast, not imported) or the ONNX metadata, and is written next
to a comment naming its source. Fields filled by tests later (validated_on, handoff,
duration_s) keep their current value if the card already exists.

    uv run python scripts/make_policy_cards.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import onnxruntime as ort
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckrt.config import CONFIG_DIR, ROOT, load_yaml, resolve  # noqa: E402
from duckrt.policy_card import CARD_DIR  # noqa: E402
from duckrt.upstream import OBSERVATION_DOF_ORDER  # noqa: E402

OBS_FUNCS = {
    "base_ang_vel": "mjlab_microduck.tasks.mdp.mounted_body_ang_vel",
    "projected_gravity": "mjlab_microduck.tasks.mdp.mounted_body_projected_gravity",
    "joint_pos": "mjlab.envs.mdp.observations.joint_pos_rel",
    "joint_vel": "mjlab.envs.mdp.observations.joint_vel_rel",
    "actions": "mjlab.envs.mdp.observations.last_action",
    "command": "mjlab.envs.mdp.observations.generated_commands",
    "kick_ball_xy": "rl_tasks.kick_v2.kick_ball_xy",
}


class _PyTagLoader(yaml.SafeLoader):
    """Reads mjlab's dumped env.yaml: python tags become plain lists / dicts / strings."""


def _python_tag(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return f"<{suffix}>{loader.construct_scalar(node)}"


_PyTagLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)


def load_env_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_PyTagLoader)


def find_line(path: Path, pattern: str, after: str | None = None) -> int:
    """1-based line of the first match of `pattern` (after the first match of `after`)."""
    rx = re.compile(pattern)
    lines = path.read_text().splitlines()
    start = 0 if after is None else find_line(path, after)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            return i + 1
    raise ValueError(f"{pattern!r} not found in {path}")


def ast_assignments(path: Path) -> dict[str, object]:
    """Literal assignments `name = <literal>` and `obj.attr = <literal>` (last one wins)."""
    out: dict[str, object] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            key = ast.unparse(t)
            try:
                out[key] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        try:
            return "../" + str(p.relative_to(ROOT.parent))
        except ValueError:
            return str(p)


def resolve_joint_regex(patterns: dict[str, float], joints: list[str]) -> list[float]:
    vals = []
    for j in joints:
        hits = [v for pat, v in patterns.items() if re.fullmatch(pat, j)]
        if len(hits) != 1:
            raise ValueError(f"joint {j}: {len(hits)} init_state patterns match")
        vals.append(float(hits[0]))
    return vals


def obs_layout(env: dict, env_yaml: Path, n_joints: int) -> list[tuple[str, int, str]]:
    terms = env["observations"]["actor"]["terms"]
    dims = {"base_ang_vel": 3, "projected_gravity": 3, "joint_pos": n_joints, "joint_vel": n_joints,
            "actions": n_joints, "command": 3, "kick_ball_xy": 2}
    out = []
    for name, t in terms.items():
        func = t["func"].removeprefix("<name:").removesuffix(">")
        if OBS_FUNCS.get(name) != func:
            raise ValueError(f"unexpected obs term {name}: {func}")
        if t.get("scale") is not None or t.get("clip") is not None or t.get("history_length", 0) != 0:
            raise ValueError(f"obs term {name} has scale/clip/history; card writer does not handle it")
        noise = t.get("noise")
        noise_s = f"noise U({noise['n_min']}, {noise['n_max']})" if noise else "no noise"
        delay_s = f"delay {t['delay_min_lag']}-{t['delay_max_lag']} steps"
        biased = t["params"].get("biased")
        extra = f", biased={biased}" if biased is not None else ""
        line = find_line(env_yaml, rf"^      {name}:$", after=r"^observations:")
        out.append((name, dims[name], f"{func.split('.')[-1]}{extra}; train {noise_s}, {delay_s}; env.yaml:{line}"))
    return out


def card_text(c: dict) -> str:
    """Emit the card with comments. `c` holds (value, comment) pairs."""
    lines = [
        f"# Policy card: {c['name']}",
        "# Generated by scripts/make_policy_cards.py from the training configs; do not hand-edit",
        "# except the test-filled fields (validated_on, handoff, duration_s).",
        "",
    ]

    def emit(key, value, comment=None):
        if isinstance(value, (dict, list)):
            dumped = yaml.safe_dump({key: value}, default_flow_style=None, allow_unicode=True, width=10_000,
                                    sort_keys=False).rstrip()
        else:
            scalar = yaml.safe_dump(value, allow_unicode=True, width=10_000).strip().removesuffix("...").strip()
            dumped = f"{key}: {scalar}"
        if comment:
            lines.append(f"# {comment}")
        lines.append(dumped)

    for key, value, comment in c["fields"]:
        if key == "obs_layout":
            lines.append(f"# {comment}")
            lines.append("obs_layout:")
            for name, dim, note in value:
                lines.append(f"  - {{name: {name}, dim: {dim}}}   # {note}")
            continue
        emit(key, value, comment)
    return "\n".join(lines) + "\n"


def keep_test_fields(path: Path, defaults: dict) -> dict:
    if not path.exists():
        return defaults
    old = yaml.safe_load(path.read_text())
    return {k: old.get(k, v) for k, v in defaults.items()}


def ckpt_iteration(path: Path) -> int | None:
    m = re.search(r"model_(\d+)\.pt$", str(path))
    return int(m.group(1)) if m else None


def make_walk(src: dict, name: str = "walk_flat", out_path: Path | None = None) -> str:
    env_yaml, cfg_py = resolve(src["env_yaml"]), resolve(src["train_cfg_py"])
    vc_py, consts_py = resolve(src["mjlab_velocity_command_py"]), resolve(src["robot_constants_py"])
    env = load_env_yaml(env_yaml)
    agent = load_env_yaml(resolve(src["agent_yaml"]))
    onnx = resolve(src["onnx"])
    meta = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    joints = meta["joint_names"].split(",")
    if joints != OBSERVATION_DOF_ORDER:
        raise ValueError("ONNX joint order != OBSERVATION_DOF_ORDER")
    dt, dec = float(env["sim"]["mujoco"]["timestep"]), int(env["decimation"])
    default_q = resolve_joint_regex(env["scene"]["entities"]["robot"]["init_state"]["joint_pos"], joints)
    act = env["actions"]["joint_pos"]
    assert act["use_default_offset"] and act["clip"] is None
    twist = env["commands"]["twist"]
    rng = twist["ranges"]
    asg = ast_assignments(cfg_py) if cfg_py.suffix == ".py" else {}
    # v2 tasks use a proper cfg dataclass, so the rotation fields are in env.yaml; v1 set them as
    # plain attributes (not dumped) -> read the training source instead
    if "rotation_env_ang_vel_range" in twist:
        rot_range, rot_min, rot_frac = twist["rotation_env_ang_vel_range"], twist["rotation_min_ang_vel"], twist["rel_rotation_envs"]
        rot_src = f"env.yaml:{find_line(env_yaml, r'rotation_env_ang_vel_range')}"
    else:
        rot_range, rot_min, rot_frac = (asg["command.rotation_env_ang_vel_range"], asg["command.rotation_min_ang_vel"],
                                        asg["command.rel_rotation_envs"])
        rot_src = f"{rel(cfg_py)}:{find_line(cfg_py, r'rotation_env_ang_vel_range')}"
        assert list(rng["lin_vel_x"]) == list(asg["command.ranges.lin_vel_x"])
    fwd_line = find_line(vc_py, r"clamp\(min=0\.3\)")
    fwd_vx = float(re.search(r"clamp\(min=([0-9.]+)\)", vc_py.read_text().splitlines()[fwd_line - 1]).group(1))
    stages = (env.get("curriculum") or {}).get("staged_curriculum")
    stage_step = int(stages["params"]["stages"][0]["step"]) if stages else None
    it = ckpt_iteration(resolve(src["checkpoint"]))
    steps_seen = None if it is None else it * int(agent["num_steps_per_env"])
    a = env["scene"]["entities"]["robot"]["articulation"]["actuators"][0]
    test = keep_test_fields(out_path or CARD_DIR / f"{name}.yaml", {
        "validated_on": "未测（第 1 步 walk_test 之后填写）",
        "handoff": {"needs_stand_before": True, "settle_s": 0.5},
        "duration_s": None,
    })
    L = lambda p, pat: f"{rel(p)}:{find_line(p, pat)}"  # noqa: E731
    fields = [
        ("name", name, None),
        ("onnx", rel(onnx), f"exported from {rel(resolve(src['checkpoint']))} (NOTES §0.3); sha256 {sha256(onnx)[:16]}…"),
        ("control_hz", 1.0 / (dt * dec), f"1 / (sim.mujoco.timestep {dt} x decimation {dec}), env.yaml:1 and :{find_line(env_yaml, r'^    timestep:')}"),
        ("obs_layout", obs_layout(env, env_yaml, len(joints)), "observations.actor.terms, in order (env.yaml). IMU terms use BNO080 mount quat "
         f"{env['observations']['actor']['terms']['base_ang_vel']['params']['mount_quat']} on body trunk_base"),
        ("obs_history", 1, "history_length 0 on every term, no group history (env.yaml)"),
        ("obs_scales", {}, "scale null and clip null on every term (env.yaml)"),
        ("action", {"type": "joint_pos_offset", "scale": float(act["scale"])},
         f"actions.joint_pos: target = default_q + scale * a, use_default_offset true, clip null (env.yaml:{find_line(env_yaml, r'^actions:')})"),
        ("joint_names", joints, "ONNX metadata joint_names == microduck/src/constants.py OBSERVATION_DOF_ORDER"),
        ("default_q", default_q, "scene.entities.robot.init_state.joint_pos regexes resolved per joint (= HOME_FRAME "
         f"{L(consts_py, r'^HOME_FRAME')}, = STAND keyframe); full precision, the ONNX metadata copy is rounded to 3 decimals"),
        ("command_range", {"vx": list(map(float, rng["lin_vel_x"])), "vy": list(map(float, rng["lin_vel_y"])), "wz": list(map(float, rng["ang_vel_z"]))},
         f"commands.twist.ranges (env.yaml:{find_line(env_yaml, r'^      lin_vel_x:')}"
         + (f"; {L(cfg_py, r'command.ranges.lin_vel_x')})" if "command.ranges.lin_vel_x" in cfg_py.read_text() else ")")),
        ("command_rules", {"turn_in_place_wz_max": float(rot_range[1]), "turn_in_place_wz_min": float(rot_min),
                           "forward_only_vx": fwd_vx, "turn_in_place_fraction": float(rot_frac),
                           "forward_only_fraction": float(env["commands"]["twist"]["rel_forward_envs"]),
                           "standing_fraction": float(env["commands"]["twist"]["rel_standing_envs"])},
         f"turn-in-place envs: vx=vy=0, |wz| in [{rot_min}, {rot_range[1]}] ({rot_src}); "
         f"forward-only envs: vx = {fwd_vx}, vy = wz = 0 ({L(vc_py, r'clamp\(min=0\.3\)')}). "
         + (f"Curriculum stage at env step {stage_step} {'not reached' if steps_seen < stage_step else 'reached'}: "
            f"checkpoint iteration {it} x {agent['num_steps_per_env']} = {steps_seen}" if stage_step and steps_seen
            else "No staged curriculum." if not stage_step else "Checkpoint iteration unknown (ONNX only).")),
        ("validated_on", test["validated_on"], "filled by scripts/walk_test.py"),
        ("handoff", test["handoff"], "updated after walk_test / kick tests"),
        ("duration_s", test["duration_s"], "continuous policy"),
        ("actuator", {"motor": a["motor_name"], "model": a["model"], "kp_fw": a["kp_fw"], "vin_range": a["vin_range"],
                      "vin_drop_gain_range": a["vin_drop_gain_range"], "vin_min": a["vin_min"], "max_current": a["max_current"],
                      "delay_steps": [a["delay_min_lag"], a["delay_max_lag"]]},
         f"scene.entities.robot.articulation.actuators[0] (env.yaml:{find_line(env_yaml, r'motor_name: xl330')}); {L(consts_py, r'^actuators = BamActuatorCfg')}"),
        ("sim", {"mjcf": rel(resolve(src["mjcf"])), "sim_dt": dt, "decimation": dec, "commit": None,
                 "checkpoint": rel(resolve(src["checkpoint"])), "checkpoint_sha256": sha256(resolve(src["checkpoint"])),
                 "env_yaml": rel(env_yaml), "env_yaml_sha256": sha256(env_yaml)},
         "training model = microduck_constants.get_spec() -> robot_walk.xml; workspace is not a git repo, so files are pinned by sha256"),
    ]
    return card_text({"name": name, "fields": fields})


def make_kick(src: dict, name: str = "kick") -> str:
    env_yaml, cfg_py = resolve(src["env_yaml"]), resolve(src["train_cfg_py"])
    env = load_env_yaml(env_yaml)
    agent = load_env_yaml(resolve(src["agent_yaml"]))
    onnx = resolve(src["onnx"])
    onnx_src = Path(src["onnx_source"])
    if not onnx.exists() or sha256(onnx) != sha256(onnx_src):
        onnx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_src, onnx)
    meta = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    joints = meta["joint_names"].split(",")
    if joints != OBSERVATION_DOF_ORDER:
        raise ValueError("ONNX joint order != OBSERVATION_DOF_ORDER")
    dt, dec = float(env["sim"]["mujoco"]["timestep"]), int(env["decimation"])
    default_q = resolve_joint_regex(env["scene"]["entities"]["robot"]["init_state"]["joint_pos"], joints)
    act = env["actions"]["joint_pos"]
    assert act["use_default_offset"] and act["clip"] is None
    rng = env["commands"]["twist"]["ranges"]
    asg = ast_assignments(cfg_py)
    ball_ev = env["events"]["reset_ball"]["params"]
    ball_el = ET.parse(resolve(src["ball_xml"])).getroot().find(".//body[@name='ball']")
    geom, inertial = ball_el.find("geom"), ball_el.find("inertial")
    a = env["scene"]["entities"]["robot"]["articulation"]["actuators"][0]
    test = keep_test_fields(CARD_DIR / "kick.yaml", {
        "validated_on": "未测（第 1 步 kick 测试之后填写）",
        "handoff": {"needs_stand_before": True, "settle_s": 0.5},
        "duration_s": None,
    })
    L = lambda p, pat: f"{rel(p)}:{find_line(p, pat)}"  # noqa: E731
    fields = [
        ("name", "kick", None),
        ("onnx", rel(onnx), f"copy of {onnx_src} (exported at the end of the run = model_3999); sha256 {sha256(onnx)[:16]}…"),
        ("control_hz", 1.0 / (dt * dec), f"1 / ({dt} x {dec}) (env.yaml)"),
        ("obs_layout", obs_layout(env, env_yaml, len(joints)), "observations.actor.terms (env.yaml); same 51-D layout as walk_flat. "
         "The ball is NOT observed (blind kick); joint_pos biased=True only matters with encoder-bias DR"),
        ("obs_history", 1, "history_length 0 (env.yaml)"),
        ("obs_scales", {}, "scale null, clip null (env.yaml)"),
        ("action", {"type": "joint_pos_offset", "scale": float(act["scale"])}, "actions.joint_pos (env.yaml)"),
        ("joint_names", joints, "ONNX metadata == OBSERVATION_DOF_ORDER"),
        ("default_q", default_q, "init_state.joint_pos (env.yaml) = AF HOME_FRAME; identical to walk_flat"),
        ("command_range", {"vx": [0.0, 0.0], "vy": [0.0, 0.0], "wz": [0.0, 0.0]},
         f"trained with twist held near 0 (vx {list(rng['lin_vel_x'])}, vy {list(rng['lin_vel_y'])}, wz {list(rng['ang_vel_z'])}; "
         f"{L(cfg_py, r'command.ranges.lin_vel_x')}); the runtime feeds exactly 0"),
        ("command_rules", {}, None),
        ("validated_on", test["validated_on"], "filled after the kick test"),
        ("handoff", test["handoff"], "walk -> zero command -> settle -> kick -> stand -> walk; measured in step 1"),
        ("duration_s", test["duration_s"], f"how long the kick is played; training episodes are {env['episode_length_s']} s "
         f"(env.yaml episode_length_s; {L(cfg_py, r'^EPISODE_LENGTH_S')})"),
        ("actuator", {"motor": a["motor_name"], "model": a["model"], "kp_fw": a["kp_fw"], "vin_range": a["vin_range"],
                      "vin_drop_gain_range": a["vin_drop_gain_range"], "vin_min": a["vin_min"], "max_current": a["max_current"],
                      "delay_steps": [a["delay_min_lag"], a["delay_max_lag"]]}, "same BamActuatorCfg as walk (env.yaml)"),
        ("task", {
            "kick_foot": asg["KICK_FOOT"],
            "ball": {"radius": float(geom.get("size")), "mass": float(inertial.get("mass")),
                     "diaginertia": [float(x) for x in inertial.get("diaginertia").split()],
                     "friction": [float(x) for x in geom.get("friction").split()],
                     "rgba": [float(x) for x in geom.get("rgba").split()]},
            "spawn": {"frame": "robot root (trunk_base) position, yaw-only frame",
                      "offset_xy": [float(x) for x in ball_ev["offset"]], "noise_xy": float(ball_ev["noise_xy"]),
                      "z": "ball radius above the floor"},
            "start_state": "standing only (z 0.11-0.12, tilt <= 5 deg, joints HOME +-0.05)",
            "episode_length_s": float(env["episode_length_s"]),
        }, f"ball: {rel(resolve(src['ball_xml']))}; spawn: events.reset_ball (env.yaml:{find_line(env_yaml, r'^  reset_ball:')}), "
           f"{L(cfg_py, r'^BALL_OFFSET_X')}; kick foot {L(cfg_py, r'^KICK_FOOT')}"),
        ("sim", {"mjcf": rel(resolve(src["mjcf"])), "sim_dt": dt, "decimation": dec, "commit": None,
                 "checkpoint": str(resolve(src["checkpoint"])), "checkpoint_sha256": sha256(resolve(src["checkpoint"])),
                 "env_yaml": str(env_yaml), "env_yaml_sha256": sha256(env_yaml),
                 "training_iterations": int(agent["max_iterations"])},
         "trained on robot_allcollisions.xml (hip_yaw inner limit +-0.436, walk model +-0.175; NOTES §0.1-3b); "
         "robot dir and tasks identical to this repo's mjlab_microduck (diff -rq)"),
    ]
    return card_text({"name": "kick", "fields": fields})


def make_kick_v2(src: dict, name: str = "kick_v2", out_path: Path | None = None) -> str:
    """Card for an rl_train kick v2 run (ball observation). Task data from the kick YAML config."""
    env_yaml = resolve(src["env_yaml"])
    kc_path = resolve(src["kick_config_yaml"])
    sys.path.insert(0, str(kc_path.parents[1]))
    from rl_tasks_config import load_kick_config  # noqa: E402  (tiny loader, no mjlab import)

    kc = load_kick_config(kc_path)
    env = load_env_yaml(env_yaml)
    onnx = resolve(src["onnx"])
    meta = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    joints = meta["joint_names"].split(",")
    if joints != OBSERVATION_DOF_ORDER:
        raise ValueError("ONNX joint order != OBSERVATION_DOF_ORDER")
    dt, dec = float(env["sim"]["mujoco"]["timestep"]), int(env["decimation"])
    default_q = resolve_joint_regex(env["scene"]["entities"]["robot"]["init_state"]["joint_pos"], joints)
    k, ball = kc["kick"], kc["ball"]
    foot_y = float(sum(k["ball_y_range"]) / 2)
    test = keep_test_fields(out_path or CARD_DIR / f"{name}.yaml", {
        "validated_on": "未测", "handoff": {"needs_stand_before": True, "settle_s": 0.5},
        "duration_s": float(k["still_after_s"]) + 0.3,
    })
    a = env["scene"]["entities"]["robot"]["articulation"]["actuators"][0]
    fields = [
        ("name", name, None),
        ("onnx", rel(onnx), f"exported from {rel(resolve(src['checkpoint']))}; sha256 {sha256(onnx)[:16]}…"),
        ("control_hz", 1.0 / (dt * dec), f"1 / ({dt} x {dec}) (env.yaml)"),
        ("obs_layout", obs_layout(env, env_yaml, len(joints)), "observations.actor.terms (env.yaml); walk layout + kick_ball_xy "
         "(ball (x, y) in the robot yaw frame, measured once before the kick, constant during it)"),
        ("obs_history", 1, "history_length 0 (env.yaml)"),
        ("obs_scales", {}, "scale null, clip null (env.yaml)"),
        ("action", {"type": "joint_pos_offset", "scale": float(env["actions"]["joint_pos"]["scale"])}, "actions.joint_pos (env.yaml)"),
        ("joint_names", joints, "ONNX metadata == OBSERVATION_DOF_ORDER"),
        ("default_q", default_q, "init_state.joint_pos (env.yaml) = HOME"),
        ("command_range", {"vx": [0.0, 0.0], "vy": [0.0, 0.0], "wz": [0.0, 0.0]}, "twist held at ~0 in training; the runtime feeds 0"),
        ("command_rules", {}, None),
        ("validated_on", test["validated_on"], "filled after the kick acceptance"),
        ("handoff", test["handoff"], "stand (walk v2, zero command) before and after"),
        ("duration_s", test["duration_s"], f"play time: still_after_s {k['still_after_s']} + 0.3 ({rel(kc_path)}); verified by the kick acceptance"),
        ("actuator", {"motor": a["motor_name"], "model": a["model"], "kp_fw": a["kp_fw"], "vin_range": a["vin_range"],
                      "max_current": a["max_current"], "delay_steps": [a["delay_min_lag"], a["delay_max_lag"]]}, "(env.yaml)"),
        ("task", {"kick_foot": k["foot"], "ball": {"radius": ball["radius_m"], "mass": ball["mass_kg"], "friction": ball["friction"],
                                                   "rgba": [1.0, 0.55, 0.0, 1.0]},
                  "ball_x_range": k["ball_x_range"], "ball_y_range": k["ball_y_range"], "foot_line_y": foot_y,
                  "obs_noise_m": k["ball_obs_noise_m"], "obs_defaults": {"kick_ball_xy": [float(sum(k["ball_x_range"]) / 2), foot_y]},
                  "episode_length_s": float(env["episode_length_s"])},
         f"{rel(kc_path)}"),
        ("sim", {"mjcf": "robot_allcollisions.xml geometry + robot_walk.xml joint ranges (rl_train/rl_tasks/kick_v2.py)",
                 "sim_dt": dt, "decimation": dec, "commit": None, "checkpoint": rel(resolve(src["checkpoint"])),
                 "checkpoint_sha256": sha256(resolve(src["checkpoint"])), "env_yaml": rel(env_yaml)}, None),
    ]
    return card_text({"name": name, "fields": fields})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="card name (key in configs/policy_sources.yaml)")
    args = ap.parse_args()
    sources = load_yaml(CONFIG_DIR / "policy_sources.yaml")
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    makers = {"walk": make_walk, "kick": make_kick, "kick_v2": make_kick_v2}
    for name, src in sources.items():
        if args.only and name != args.only:
            continue
        text = makers[src["kind"]](src, name)
        out = CARD_DIR / f"{name}.yaml"
        out.write_text(text)
        print(f"[ok] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
