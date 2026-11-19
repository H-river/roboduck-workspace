"""Scene modules: schema, validation, geometry (MjSpec) and 2-D footprints.

A module definition (assets/modules/*.yaml) is a named parameter set with a `type`.
A layout places module definitions (by grid cell or exact position, with a yaw) and may
override parameters per instance. Every module becomes one body `mod_<id>` (static, or with
a free joint for the ball); geoms are named `mod_<id>_<part>`.

Local frames (yaw rotates them about world z):
  arena           centre of the arena
  flag            centre of the mat; pole near the mat's +x edge
  forbidden_zone  centre
  obstacle        centre of the base
  ramp            centre of the ramp foot (start of the incline); +x = uphill
  ball            centre
  goal            centre of the goal line; +x = into the goal (shooting direction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from duckrt.config import CONFIG_DIR, ROOT, load_yaml, resolve

MODULE_DIR = ROOT / "assets" / "modules"
LAYOUT_DIR = ROOT / "assets" / "layouts"


@dataclass
class P:
    kind: str                      # float | int | bool | str | color | vec2 | vec3 | path
    required: bool = False
    default: Any = None
    lo: Any = None
    hi: Any = None
    note: str = ""


SCHEMA: dict[str, dict[str, P]] = {
    "arena": {
        "size_m": P("vec2", True, lo=1.0, hi=10.0),
        "wall_height_m": P("float", False, 0.05, 0.02, 0.30),
        "wall_thickness_m": P("float", False, 0.02, 0.005, 0.10),
        "floor_color": P("color", False, "floor"),
        "floor_color_alt": P("color", False, "floor_alt"),
        "checker_m": P("float", False, 0.30, 0.05, 2.0, "checker square size"),
        "wall_color": P("color", False, "wall"),
        "floor_friction": P("float", False, 1.0, 0.1, 2.0),
        "texture": P("path", False, None),
    },
    "flag": {
        "id": P("str", True),
        "color": P("color", True),
        "pole_height_m": P("float", True, lo=0.25, hi=0.30),
        "pole_radius_m": P("float", False, 0.006, 0.003, 0.02),
        "mat_size_m": P("vec2", False, [0.40, 0.40], 0.20, 1.0),
        "pole_offset_m": P("vec2", False, [0.17, 0.0], -0.5, 0.5, "pole position on the mat"),
        "flag_size_m": P("vec2", False, [0.10, 0.06], 0.02, 0.3, "cloth width x height"),
        "texture": P("path", False, None),
    },
    "forbidden_zone": {
        "size_m": P("vec2", True, lo=0.1, hi=3.0),
        "color": P("color", False, "forbidden_zone"),
        "texture": P("path", False, None),
    },
    "obstacle": {
        "size_m": P("vec3", True, lo=0.05, hi=2.0),
        "color": P("color", False, "obstacle"),
        "texture": P("path", False, None),
    },
    "ramp": {
        "angle_deg": P("float", True, lo=0.0, hi=20.0),
        "slope_length_m": P("float", False, 0.60, 0.60, 3.0, "horizontal length of the incline"),
        "top_length_m": P("float", False, 0.40, 0.40, 3.0),
        "width_m": P("float", False, 0.40, 0.25, 1.5),
        "down_slope": P("bool", False, False, note="add a descending incline after the platform"),
        "rail_height_m": P("float", False, 0.04, 0.0, 0.15),
        "rail_thickness_m": P("float", False, 0.015, 0.005, 0.05),
        "friction": P("float", False, 1.0, 0.2, 2.0),
        "thickness_m": P("float", False, 0.02, 0.005, 0.1),
        "color": P("color", False, "ramp"),
        "rail_color": P("color", False, "ramp_rail"),
        "texture": P("path", False, None),
    },
    "ball": {
        "radius_m": P("float", False, None, 0.02, 0.15, "default: kick card task.ball"),
        "mass_kg": P("float", False, None, 0.005, 1.0),
        "friction": P("vec3", False, None, 0.0, 2.0),
        "color": P("color", False, "ball"),
    },
    "goal": {
        "width_m": P("float", True, lo=0.3, hi=1.5, note="inner width between posts"),
        "height_m": P("float", False, 0.20, 0.10, 0.50),
        "depth_m": P("float", False, 0.20, 0.08, 0.50),
        "post_size_m": P("float", False, 0.025, 0.01, 0.06, "square post / bar section"),
        "color": P("color", False, "goal"),
        "back_color": P("color", False, "goal_back"),
    },
}


@dataclass
class ModuleInstance:
    id: str
    type: str
    definition: str
    params: dict[str, Any]
    pos: np.ndarray            # (x, y) of the module origin
    yaw: float
    body: str = ""
    geoms: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)
    rgba: dict[str, list[float]] = field(default_factory=dict)

    def footprint(self) -> np.ndarray | None:
        """2-D corners (4, 2) of the oriented bounding rectangle in world, or None (arena)."""
        rect = local_rect(self.type, self.params)
        if rect is None:
            return None
        (x0, x1), (y0, y1) = rect
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return pts @ R.T + self.pos


def local_rect(mtype: str, p: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if mtype == "arena":
        return None
    if mtype == "flag":
        hx, hy = p["mat_size_m"][0] / 2, p["mat_size_m"][1] / 2
        return (-hx, hx), (-hy, hy)
    if mtype in ("forbidden_zone", "obstacle"):
        hx, hy = p["size_m"][0] / 2, p["size_m"][1] / 2
        return (-hx, hx), (-hy, hy)
    if mtype == "ramp":
        L = ramp_total_length(p)
        hy = p["width_m"] / 2 + p["rail_thickness_m"]
        return (0.0, L), (-hy, hy)
    if mtype == "ball":
        r = p["radius_m"]
        return (-r, r), (-r, r)
    if mtype == "goal":
        hy = p["width_m"] / 2 + p["post_size_m"]
        return (0.0, p["depth_m"] + p["post_size_m"] / 2), (-hy, hy)
    raise ValueError(mtype)


def ramp_total_length(p: dict) -> float:
    return p["slope_length_m"] * (2 if p["down_slope"] else 1) + p["top_length_m"]


# -- loading / validation -----------------------------------------------------------------------
class ModuleError(ValueError):
    pass


def load_palette() -> dict[str, list[float]]:
    return load_yaml(CONFIG_DIR / "palette.yaml")


def load_module_defs() -> dict[str, dict]:
    defs: dict[str, dict] = {}
    for f in sorted(MODULE_DIR.glob("*.yaml")):
        for name, d in (yaml.safe_load(f.read_text()) or {}).items():
            if name in defs:
                raise ModuleError(f"module definition {name!r} defined twice ({f.name})")
            defs[name] = dict(d, _file=f.name)
    return defs


def _ball_defaults() -> dict:
    from duckrt.policy_card import load_card

    ball = load_card("kick").task["ball"]
    return {"radius_m": ball["radius"], "mass_kg": ball["mass"], "friction": ball["friction"],
            "diaginertia": ball["diaginertia"]}


def resolve_color(value, palette: dict) -> list[float]:
    if isinstance(value, str):
        if value not in palette:
            raise ModuleError(f"colour {value!r} not in configs/palette.yaml")
        return [float(x) for x in palette[value]]
    v = [float(x) for x in value]
    if len(v) not in (3, 4) or not all(0.0 <= x <= 1.0 for x in v):
        raise ModuleError(f"bad rgba {value!r}")
    return v if len(v) == 4 else v + [1.0]


def validate_params(mtype: str, raw: dict, where: str) -> dict:
    if mtype not in SCHEMA:
        raise ModuleError(f"{where}: unknown module type {mtype!r} (known: {sorted(SCHEMA)})")
    schema = SCHEMA[mtype]
    unknown = set(raw) - set(schema) - {"type", "_file", "diaginertia"}
    if unknown:
        raise ModuleError(f"{where}: unknown parameters {sorted(unknown)} for {mtype}")
    out: dict[str, Any] = {}
    if mtype == "ball":
        out.update(_ball_defaults())
    for name, spec in schema.items():
        if name in raw:
            v = raw[name]
            if v is None and spec.required:
                raise ModuleError(f"{where}: required parameter {name!r} is null ({mtype})")
        elif spec.required:
            raise ModuleError(f"{where}: missing required parameter {name!r} ({mtype})")
        elif spec.default is not None or name not in out:
            v = spec.default
        else:
            continue
        out[name] = _check(name, spec, v, where)
    return out


def _check(name: str, spec: P, v, where: str):
    if v is None:
        return None
    k = spec.kind
    try:
        if k == "float":
            v = float(v)
            vals = [v]
        elif k == "int":
            v = int(v)
            vals = [v]
        elif k == "bool":
            if not isinstance(v, bool):
                raise TypeError
            return v
        elif k in ("str", "path"):
            if not isinstance(v, str):
                raise TypeError
            if k == "path" and not resolve(v).exists():
                raise ModuleError(f"{where}: {name} file not found: {v}")
            return v
        elif k == "color":
            return v
        elif k in ("vec2", "vec3"):
            n = 2 if k == "vec2" else 3
            if isinstance(v, (int, float)) and n == 2:
                v = [float(v), float(v)]
            v = [float(x) for x in v]
            if len(v) != n:
                raise TypeError
            vals = v
        else:
            raise ModuleError(f"schema kind {k}")
    except (TypeError, ValueError) as e:
        if isinstance(e, ModuleError):
            raise
        raise ModuleError(f"{where}: {name} must be {k}, got {v!r}") from None
    for x in vals:
        if spec.lo is not None and x < spec.lo - 1e-12 or spec.hi is not None and x > spec.hi + 1e-12:
            raise ModuleError(f"{where}: {name}={v} out of range [{spec.lo}, {spec.hi}]")
    return v


# -- geometry ---------------------------------------------------------------------------------
def _quat_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _quat_pitch(angle: float) -> list[float]:
    """Rotation about +y by `angle` (positive tilts +x down... used with negative for uphill)."""
    return [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]


def _material(spec: mujoco.MjSpec, name: str, rgba: list[float], texture: str | None, repeat=(1, 1)) -> str:
    mat = spec.add_material(name=name, rgba=rgba)
    if texture:
        tex = spec.add_texture(name=name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(resolve(texture)))
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        mat.texrepeat = list(repeat)
        mat.rgba = [1, 1, 1, rgba[3]]
    return name


def _geom(body, inst: ModuleInstance, part: str, gtype, size, pos, quat=None, rgba=None, material=None,
          collide=True, friction=None, group=0):
    g = body.add_geom()
    g.name = f"{body.name}_{part}"
    g.type = gtype
    g.size = list(size) + [0.0] * (3 - len(size))
    g.pos = list(pos)
    if quat is not None:
        g.quat = list(quat)
    if material:
        g.material = material
    elif rgba is not None:
        g.rgba = rgba
    g.contype = 1 if collide else 0
    g.conaffinity = 1 if collide else 0
    g.condim = 3
    g.group = group
    if friction is not None:
        g.friction = list(friction)
    inst.geoms.append(g.name)
    return g


BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
PLANE = mujoco.mjtGeom.mjGEOM_PLANE


def build_module(spec: mujoco.MjSpec, inst: ModuleInstance, palette: dict, centre: np.ndarray) -> None:
    p, t = inst.params, inst.type
    col = lambda key: resolve_color(p[key], palette)  # noqa: E731
    inst.body = f"mod_{inst.id}"
    body = spec.worldbody.add_body(name=inst.body, pos=[inst.pos[0], inst.pos[1], 0.0], quat=_quat_yaw(inst.yaw))

    if t == "arena":
        W, H = p["size_m"]
        tex = spec.add_texture(name="arena_floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                               rgb1=col("floor_color")[:3], rgb2=col("floor_color_alt")[:3], width=256, height=256)
        mat = spec.add_material(name="arena_floor", texuniform=True,
                                texrepeat=[1.0 / (2 * p["checker_m"])] * 2, reflectance=0.05)
        if p["texture"]:  # real mat photo (future)
            tex = spec.add_texture(name="arena_floor_img", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(resolve(p["texture"])))
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        g = spec.worldbody.add_geom(name="floor", type=PLANE, size=[0, 0, 0.05], material="arena_floor")
        g.contype, g.conaffinity, g.condim = 1, 1, 3
        g.friction = [p["floor_friction"], 0.005, 0.0001]
        inst.geoms.append("floor")
        wh, wt = p["wall_height_m"], p["wall_thickness_m"]
        wc = col("wall_color")
        for part, pos, size in (("wall_n", (0, H / 2 + wt / 2, wh / 2), (W / 2 + wt, wt / 2, wh / 2)),
                                ("wall_s", (0, -H / 2 - wt / 2, wh / 2), (W / 2 + wt, wt / 2, wh / 2)),
                                ("wall_e", (W / 2 + wt / 2, 0, wh / 2), (wt / 2, H / 2, wh / 2)),
                                ("wall_w", (-W / 2 - wt / 2, 0, wh / 2), (wt / 2, H / 2, wh / 2))):
            _geom(body, inst, part, BOX, size, pos, rgba=wc)
        inst.rgba = {"floor": col("floor_color"), "wall": wc}

    elif t == "flag":
        mx, my = p["mat_size_m"]
        c = col("color")
        mat = _material(spec, f"{inst.body}_mat", c, p["texture"])
        _geom(body, inst, "mat", BOX, (mx / 2, my / 2, 0.0005), (0, 0, 0.0005), material=mat, collide=False)
        h, r = p["pole_height_m"], p["pole_radius_m"]
        ox, oy = p["pole_offset_m"]
        _geom(body, inst, "pole", CYL, (r, h / 2), (ox, oy, h / 2), rgba=resolve_color("flag_pole", palette))
        fw, fh = p["flag_size_m"]
        # cloth hangs from the pole top towards -y (visible from the front and the side)
        _geom(body, inst, "cloth", BOX, (0.0015, fw / 2, fh / 2), (ox, oy - r - fw / 2, h - fh / 2), rgba=c, collide=False)
        s = body.add_site(name=f"{inst.body}_centre", pos=[0, 0, 0], size=[0.01, 0, 0])
        inst.sites.append(s.name)
        inst.rgba = {"flag": c}

    elif t == "forbidden_zone":
        sx, sy = p["size_m"]
        c = col("color")
        mat = _material(spec, f"{inst.body}_mat", c, p["texture"])
        _geom(body, inst, "zone", BOX, (sx / 2, sy / 2, 0.0006), (0, 0, 0.0006), material=mat, collide=False)
        inst.rgba = {"zone": c}

    elif t == "obstacle":
        sx, sy, sz = p["size_m"]
        c = col("color")
        mat = _material(spec, f"{inst.body}_mat", c, p["texture"])
        _geom(body, inst, "box", BOX, (sx / 2, sy / 2, sz / 2), (0, 0, sz / 2), material=mat)
        inst.rgba = {"obstacle": c}

    elif t == "ramp":
        th = math.radians(p["angle_deg"])
        L, T, w, d = p["slope_length_m"], p["top_length_m"], p["width_m"], p["thickness_m"]
        h = L * math.tan(th)
        c, rc = col("color"), col("rail_color")
        mat = _material(spec, f"{inst.body}_mat", c, p["texture"])
        fr = [p["friction"], 0.005, 0.0001]
        rh, rt = p["rail_height_m"], p["rail_thickness_m"]

        def incline(part: str, x0: float, up: bool):
            # top surface from (x0, z=0) to (x0 + L, z=h) (or reverse for the down slope)
            slen = L / math.cos(th)
            ang = -th if up else th
            cx = x0 + L / 2
            cz = h / 2 - (d / 2) * math.cos(th)
            ox = (d / 2) * math.sin(th) * (1 if up else -1)
            q = _quat_pitch(ang)
            _geom(body, inst, part, BOX, (slen / 2, w / 2, d / 2), (cx + ox, 0, cz), quat=q, material=mat, friction=fr)
            if rh > 0:
                for side, sy in (("l", 1), ("r", -1)):
                    rz = h / 2 + rh / 2 * math.cos(th)
                    _geom(body, inst, f"{part}_rail_{side}", BOX, (slen / 2, rt / 2, rh / 2),
                          (cx - (rh / 2) * math.sin(th) * (1 if up else -1), sy * (w / 2 + rt / 2), rz), quat=q, rgba=rc,
                          group=1)  # group 1: not "ground" for SimBackend.ground_height()

        incline("up", 0.0, True)
        # platform: top at z = h; slab down to the floor so nothing can slide under it
        ph = max(h, 1e-3)
        _geom(body, inst, "top", BOX, (T / 2, w / 2, ph / 2), (L + T / 2, 0, ph / 2), material=mat, friction=fr)
        if rh > 0:
            for side, sy in (("l", 1), ("r", -1)):
                _geom(body, inst, f"top_rail_{side}", BOX, (T / 2, rt / 2, (ph + rh) / 2), (L + T / 2, sy * (w / 2 + rt / 2), (ph + rh) / 2),
                      rgba=rc, group=1)
        if p["down_slope"]:
            incline("down", L + T, False)
        s = body.add_site(name=f"{inst.body}_top_centre", pos=[L + T / 2, 0, h], size=[0.01, 0, 0])
        inst.sites.append(s.name)
        inst.params = dict(p, height_m=h, total_length_m=ramp_total_length(p))
        inst.rgba = {"ramp": c, "rail": rc}

    elif t == "ball":
        r = p["radius_m"]
        body.pos = [inst.pos[0], inst.pos[1], r]
        body.add_freejoint(name=f"{inst.body}_free")
        c = col("color")
        _geom(body, inst, "sphere", SPHERE, (r,), (0, 0, 0), rgba=c, friction=p["friction"])
        body.explicitinertial = True
        body.ipos = [0.0, 0.0, 0.0]   # must be set explicitly, the MjSpec default is not the body origin
        body.iquat = [1.0, 0.0, 0.0, 0.0]
        body.mass = p["mass_kg"]
        body.inertia = list(p.get("diaginertia") or [2.0 / 3.0 * p["mass_kg"] * r * r] * 3)
        inst.rgba = {"ball": c}

    elif t == "goal":
        W, Hh, D, ps = p["width_m"], p["height_m"], p["depth_m"], p["post_size_m"]
        c, bc = col("color"), col("back_color")
        hp = ps / 2
        for side, sy in (("l", 1), ("r", -1)):
            _geom(body, inst, f"post_{side}", BOX, (hp, hp, Hh / 2), (0, sy * (W / 2 + hp), Hh / 2), rgba=c)
            _geom(body, inst, f"side_{side}", BOX, (D / 2, 0.002, Hh / 2), (D / 2, sy * (W / 2 + hp), Hh / 2), rgba=bc)
        _geom(body, inst, "crossbar", BOX, (hp, W / 2 + ps, hp), (0, 0, Hh + hp), rgba=c)
        _geom(body, inst, "back", BOX, (0.004, W / 2 + ps, Hh / 2), (D, 0, Hh / 2), rgba=bc)
        s = body.add_site(name=f"{inst.body}_goal_line", pos=[0, 0, 0.001], size=[0.002, W / 2, 0.001],
                          type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[1, 1, 1, 0.6])
        inst.sites.append(s.name)
        inst.rgba = {"goal": c, "back": bc}
    else:
        raise ModuleError(t)


# -- overlap check ------------------------------------------------------------------------------
def rects_overlap(a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> bool:
    """Separating-axis test for two convex quads (4, 2); margin shrinks the test (touching is ok)."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            n /= np.linalg.norm(n)
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() + margin or pb.max() <= pa.min() + margin:
                return False
    return True


def circle_rect_overlap(c: np.ndarray, r: float, rect: np.ndarray) -> bool:
    sq = np.array([c + [-r, -r], c + [r, -r], c + [r, r], c + [-r, r]])
    return rects_overlap(sq, rect)
