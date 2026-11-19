"""Assemble the simulation model with MjSpec. Upstream XML files are only read.

build_world() = robot (robot_allcollisions.xml + walk joint ranges + training foot contact
+ BAM motors) + environment (plain floor, or a layout of modules) + cameras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import yaml

from duckrt.actuation import BamActuator
from duckrt.config import resolve

_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
}
_CONES = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}
_SOLVERS = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG, "pgs": mujoco.mjtSolver.mjSOL_PGS}


@dataclass
class RobotStart:
    pos: np.ndarray            # trunk position (x, y, z) in world
    yaw: float                 # heading [rad]
    joint_pos: dict[str, float]


@dataclass
class BuiltWorld:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    robot_start: RobotStart
    modules: list[Any] = field(default_factory=list)   # filled by the layout code (step 3)
    layout: dict[str, Any] | None = None


def read_keyframe(sim_cfg: dict, name: str | None = None) -> tuple[float, dict[str, float]]:
    """Trunk z and joint angles of a keyframe from the upstream scene file, by joint name."""
    rc = sim_cfg["robot"]
    m = mujoco.MjModel.from_xml_path(str(resolve(rc["keyframes_from"])))
    k = m.key(name or rc["init_keyframe"])
    qpos = m.key_qpos[k.id]
    joints = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            joints[m.joint(j).name] = float(qpos[m.jnt_qposadr[j]])
    return float(qpos[2]), joints


def make_robot_spec(sim_cfg: dict, actuator: BamActuator) -> mujoco.MjSpec:
    rc, pc = sim_cfg["robot"], sim_cfg["physics"]
    spec = mujoco.MjSpec.from_file(str(resolve(rc["mjcf"])))

    if rc.get("joint_ranges_from"):
        ref = mujoco.MjModel.from_xml_path(str(resolve(rc["joint_ranges_from"])))
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.range = ref.jnt_range[ref.joint(j.name).id].copy()

    fc = rc["foot_contact"]
    for g in spec.geoms:
        if g.name in rc["foot_geoms"]:
            g.condim = int(fc["condim"])
            g.priority = int(fc["priority"])
            g.friction[0] = float(fc["friction_slide"])

    o = spec.option
    o.timestep = float(pc["timestep"])
    o.integrator = _INTEGRATORS[pc["integrator"]]
    o.cone = _CONES[pc["cone"]]
    o.solver = _SOLVERS[pc["solver"]]
    o.iterations = int(pc["iterations"])
    o.ls_iterations = int(pc["ls_iterations"])
    o.ccd_iterations = int(pc["ccd_iterations"])
    o.tolerance = float(pc["tolerance"])
    o.ls_tolerance = float(pc["ls_tolerance"])
    o.impratio = float(pc["impratio"])
    o.gravity = np.asarray(pc["gravity"], dtype=np.float64)

    actuator.edit_spec(spec)
    return spec


def add_sky_and_light(spec: mujoco.MjSpec) -> None:
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0.0, 0.0, 0.0], width=512, height=3072)
    light = spec.worldbody.add_light()
    light.pos = [0.0, 0.0, 3.5]
    light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.diffuse = [0.55, 0.55, 0.55]


def add_plain_floor(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                     width=300, height=300)
    mat = spec.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5], reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    g = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                material="groundplane")
    g.contype, g.conaffinity, g.condim = 1, 1, 3


def build_world(
    sim_cfg: dict,
    actuator: BamActuator,
    layout: dict | None = None,
    camera_cfg: dict | None = None,
    extra_spec_fn=None,
) -> BuiltWorld:
    """extra_spec_fn(spec) may add test-only geoms (e.g. projection markers) before compile."""
    spec = make_robot_spec(sim_cfg, actuator)
    add_sky_and_light(spec)
    trunk_z, joints = read_keyframe(sim_cfg)
    start = RobotStart(pos=np.array([0.0, 0.0, trunk_z]), yaw=0.0, joint_pos=joints)
    modules: list[Any] = []
    if layout is None:
        add_plain_floor(spec)
    else:
        modules, start = apply_layout(spec, layout, start)
    if camera_cfg is not None:
        from duckrt.camera import add_cameras  # step 2

        add_cameras(spec, camera_cfg, layout)
    if extra_spec_fn is not None:
        extra_spec_fn(spec)
    model = spec.compile()
    return BuiltWorld(spec=spec, model=model, robot_start=start, modules=modules, layout=layout)


# -- layouts --------------------------------------------------------------------------------
ROBOT_START_RADIUS_M = 0.12   # keep-out circle around the robot start for the overlap check


class LayoutError(ValueError):
    pass


def load_layout(name_or_path: str, overrides: dict[str, dict] | None = None) -> dict:
    """Read assets/layouts/<name>.yaml; overrides = {instance_id: {param: value}}."""
    from duckrt.scene.modules import LAYOUT_DIR

    p = resolve(name_or_path) if str(name_or_path).endswith(".yaml") else LAYOUT_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        layout = yaml.safe_load(f)
    layout["_file"] = str(p)
    for inst_id, params in (overrides or {}).items():
        hits = [m for m in layout["modules"] if m["id"] == inst_id]
        if not hits:
            raise LayoutError(f"override for unknown module id {inst_id!r}")
        hits[0].setdefault("params", {}).update(params)
    return layout


def parse_overrides(items: list[str] | None) -> dict[str, dict]:
    """['ramp.angle_deg=5', ...] -> {'ramp': {'angle_deg': 5.0}}."""
    out: dict[str, dict] = {}
    for it in items or []:
        key, val = it.split("=", 1)
        inst, param = key.split(".", 1)
        out.setdefault(inst, {})[param] = yaml.safe_load(val)
    return out


def _cell_xy(cell, grid: float, arena_size) -> np.ndarray:
    W, H = arena_size
    return np.array([-W / 2 + (cell[0] + 0.5) * grid, -H / 2 + (cell[1] + 0.5) * grid])


def apply_layout(spec: mujoco.MjSpec, layout: dict, start: RobotStart):
    from duckrt.scene.modules import (ModuleInstance, build_module, circle_rect_overlap, load_module_defs,
                                      load_palette, rects_overlap, validate_params)

    defs, palette = load_module_defs(), load_palette()
    grid = float(layout.get("grid_m", 0.6))

    def make(entry: dict, default_id: str) -> ModuleInstance:
        name = entry["module"]
        if name not in defs:
            raise LayoutError(f"{layout['name']}: unknown module definition {name!r}")
        d = defs[name]
        raw = {k: v for k, v in d.items() if k not in ("type", "_file")}
        raw.update(entry.get("params") or {})
        inst_id = entry.get("id", default_id)
        params = validate_params(d["type"], raw, f"{layout['name']}/{inst_id} ({name})")
        return ModuleInstance(id=inst_id, type=d["type"], definition=name, params=params,
                              pos=np.zeros(2), yaw=np.radians(float(entry.get("yaw_deg", 0.0))))

    arena = make({"module": layout["arena"], "id": "arena"}, "arena")
    if arena.type != "arena":
        raise LayoutError("layout.arena must refer to a module of type arena")
    W, H = arena.params["size_m"]
    layout["_arena_size"], layout["_arena_centre"] = [W, H], [0.0, 0.0]

    def place(entry: dict) -> np.ndarray:
        if ("cell" in entry) == ("pos" in entry):
            raise LayoutError(f"{layout['name']}: give exactly one of cell / pos for {entry}")
        if "cell" in entry:
            return _cell_xy(entry["cell"], grid, (W, H)) + np.asarray(entry.get("offset_m", [0.0, 0.0]), float)
        return np.asarray(entry["pos"], float)

    instances = [arena]
    ids = {"arena"}
    for k, entry in enumerate(layout.get("modules") or []):
        inst = make(entry, f"m{k}")
        if inst.id in ids:
            raise LayoutError(f"{layout['name']}: duplicate module id {inst.id!r}")
        ids.add(inst.id)
        inst.pos = place(entry)
        instances.append(inst)

    rs = layout["robot_start"]
    start_xy = place(rs)
    start = RobotStart(pos=np.array([start_xy[0], start_xy[1], start.pos[2]]), yaw=float(np.radians(rs.get("yaw_deg", 0.0))),
                       joint_pos=start.joint_pos)

    # checks: inside the arena, no overlaps, robot start clear
    inner = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]])
    problems = []
    fps = {m.id: m.footprint() for m in instances if m.type != "arena"}
    for i, a in fps.items():
        if (a.min(0) < inner[0] - 1e-9).any() or (a.max(0) > inner[2] + 1e-9).any():
            problems.append(f"{i} sticks out of the arena")
        if i not in rs.get("on_modules", []) and circle_rect_overlap(start_xy, ROBOT_START_RADIUS_M, a):
            problems.append(f"{i} overlaps the robot start")
    keys = list(fps)
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            if rects_overlap(fps[keys[x]], fps[keys[y]], margin=1e-6):
                problems.append(f"{keys[x]} overlaps {keys[y]}")
    if not (abs(start_xy[0]) < W / 2 - ROBOT_START_RADIUS_M and abs(start_xy[1]) < H / 2 - ROBOT_START_RADIUS_M):
        problems.append("robot start too close to the arena wall")
    if problems:
        raise LayoutError(f"{layout['name']}: " + "; ".join(problems))

    for inst in instances:
        build_module(spec, inst, palette, np.zeros(2))
    return instances, start
