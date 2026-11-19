"""MuJoCo backend: build, reset, step, sensors, ground truth, rendering.

Stepping follows mjlab (envs/manager_based_rl_env.py:412-457): per policy tick the
runtime calls step_physics() `decimation` times and then forward() once; observations
are read after forward(). Do not call mj_forward between physics steps -- BAM reads
qfrc_bias, and mjlab's actuator sees it one substep stale on substeps 2..4.

Rendering uses a second MjData with kinematics only, so cameras can be rendered at any
physics step without touching the dynamics.

Set MUJOCO_GL (e.g. egl) before importing this module to render headless.
"""

from __future__ import annotations

import mujoco
import numpy as np

from duckrt.actuation import ActuatorCfg, BamActuator
from duckrt.messages import Mode, StateMsg
from duckrt.scene.builder import BuiltWorld, build_world
from duckrt.upstream import OBSERVATION_DOF_ORDER


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class SimBackend:
    def __init__(
        self,
        sim_cfg: dict,
        layout: dict | None = None,
        camera_cfg: dict | None = None,
        seed: int = 0,
        extra_spec_fn=None,
    ):
        self.cfg = sim_cfg
        self.joint_names = list(OBSERVATION_DOF_ORDER)
        self.rng = np.random.default_rng(seed)
        self.actuator = BamActuator(ActuatorCfg.from_dict(sim_cfg["actuator"]), self.joint_names, rng=self.rng)
        self.world: BuiltWorld = build_world(sim_cfg, self.actuator, layout=layout, camera_cfg=camera_cfg,
                                             extra_spec_fn=extra_spec_fn)
        self.model = self.world.model
        self.data = mujoco.MjData(self.model)
        self.rdata = mujoco.MjData(self.model)
        self.actuator.bind(self.model, self.data)

        m = self.model
        rc = sim_cfg["robot"]
        self.trunk_id = m.body(rc["trunk_body"]).id
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == self.trunk_id]
        self.free_qadr = int(m.jnt_qposadr[free[0]])
        self.free_vadr = int(m.jnt_dofadr[free[0]])
        jids = [m.joint(n).id for n in self.joint_names]
        self.qadr = m.jnt_qposadr[jids].copy()
        self.vadr = m.jnt_dofadr[jids].copy()
        self.head_qadr = m.jnt_qposadr[[m.joint(n).id for n in rc["head_joints"]]].copy()
        self.mount_quat = np.asarray(rc["imu_mount_quat"], dtype=np.float64)
        self.mount_quat /= np.linalg.norm(self.mount_quat)
        self._accel_adr = m.sensor_adr[m.sensor("imu_accel").id] if _has_sensor(m, "imu_accel") else None

        self.physics_dt = float(m.opt.timestep)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._scene_option = mujoco.MjvOption()
        self.reset(seed)

    # -- reset ----------------------------------------------------------------------------
    def reset(self, seed: int | None = None, start=None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.actuator.delay.rng = self.rng
        start = start or self.world.robot_start
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        d.qpos[self.free_qadr:self.free_qadr + 3] = start.pos
        d.qpos[self.free_qadr + 2] += self.ground_height(start.pos[0], start.pos[1])
        d.qpos[self.free_qadr + 3:self.free_qadr + 7] = [np.cos(start.yaw / 2), 0.0, 0.0, np.sin(start.yaw / 2)]
        for name, q in start.joint_pos.items():
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
        mujoco.mj_forward(m, d)
        self.actuator.reset()

    def ground_height(self, x: float, y: float) -> float:
        """Height of the static scene (group-0 geoms: floor, ramps, ...) under (x, y)."""
        geomid = np.zeros(1, dtype=np.int32)
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        dist = mujoco.mj_ray(self.model, self.data, np.array([x, y, 2.0]), np.array([0.0, 0.0, -1.0]),
                             groups, 1, -1, geomid)
        return 0.0 if dist < 0 else float(2.0 - dist)

    # -- stepping -------------------------------------------------------------------------
    def step_physics(self, q_target: np.ndarray) -> None:
        self.actuator.apply(q_target)
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    # -- robot-side measurements (what the real robot has) --------------------------------
    def base_quat(self) -> np.ndarray:
        q = self.data.qpos[self.free_qadr + 3:self.free_qadr + 7].copy()
        return q / np.linalg.norm(q)

    def base_ang_vel_body(self) -> np.ndarray:
        return self.data.qvel[self.free_vadr + 3:self.free_vadr + 6].copy()

    def imu_quat(self) -> np.ndarray:
        """BNO08x quaternion: sensor frame = body (x) conj(mount) (NOTES §0.1-1b)."""
        return quat_mul(self.base_quat(), quat_conj(self.mount_quat))

    def imu_gyro(self) -> np.ndarray:
        """Gyro in the sensor frame = R_mount * omega_body (training mounted_body_ang_vel)."""
        return quat_rotate(self.mount_quat, self.base_ang_vel_body())

    def imu_acc(self) -> np.ndarray:
        """Specific force in the sensor frame from the XML accelerometer (site frame = body
        (x) mount), re-expressed in the physical sensor frame: a_s = R_m R_m a_site."""
        if self._accel_adr is None:
            return np.zeros(3)
        a_site = self.data.sensordata[self._accel_adr:self._accel_adr + 3].copy()
        return quat_rotate(self.mount_quat, quat_rotate(self.mount_quat, a_site))

    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    def head_q(self) -> np.ndarray:
        return self.data.qpos[self.head_qadr].copy()

    def read_state(self, mode: Mode, fall_flag: bool) -> StateMsg:
        return StateMsg(
            t=self.time, imu_quat=self.imu_quat(), gyro=self.imu_gyro(), acc=self.imu_acc(),
            joint_pos=self.joint_pos(), joint_vel=self.joint_vel(), mode=mode, fall_flag=fall_flag,
        )

    # -- privileged ground truth ----------------------------------------------------------
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[self.free_qadr:self.free_qadr + 3].copy()

    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[self.free_vadr:self.free_vadr + 3].copy()

    def base_yaw(self) -> float:
        return yaw_of(self.base_quat())

    def base_vel_robot(self) -> np.ndarray:
        """(vx, vy, wz) in the robot (yaw) frame -- comparable to the velocity command."""
        v = self.base_lin_vel_world()
        yaw = self.base_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        w_world = quat_rotate(self.base_quat(), self.base_ang_vel_body())
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], w_world[2]])

    def up_z(self) -> float:
        """z of the trunk's up axis in world (1 = upright)."""
        return float(quat_rotate(self.base_quat(), np.array([0.0, 0.0, 1.0]))[2])

    def actuator_torques(self) -> np.ndarray:
        return self.actuator.last_output_torque.copy()

    def has_nan(self) -> bool:
        return bool(np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any())

    def module_states(self) -> list[dict]:
        """Privileged scene query (oracle / relabelling): every module's id, type, current world
        pose (the ball's from its free joint), parameters (sizes), colours and named sites."""
        out = []
        m, d = self.model, self.data
        for inst in self.world.modules:
            bid = m.body(inst.body).id
            pos = d.xpos[bid].copy()
            quat = d.xquat[bid].copy()
            out.append({
                "id": inst.id, "type": inst.type, "definition": inst.definition,
                "pos": pos, "yaw": yaw_of(quat), "params": dict(inst.params), "rgba": dict(inst.rgba),
                "sites": {s: d.site_xpos[m.site(s).id].copy() for s in inst.sites},
                "footprint": inst.footprint() if inst.type != "ball" else None,
            })
        return out

    def module_state(self, module_id: str) -> dict:
        for s in self.module_states():
            if s["id"] == module_id:
                return s
        raise KeyError(module_id)

    # -- rendering ------------------------------------------------------------------------
    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def sync_render_data(self) -> mujoco.MjData:
        """Kinematics of the current state in the render copy (dynamics untouched)."""
        rd = self.rdata
        rd.qpos[:] = self.data.qpos
        rd.qvel[:] = self.data.qvel
        rd.time = self.data.time
        mujoco.mj_kinematics(self.model, rd)
        mujoco.mj_comPos(self.model, rd)
        mujoco.mj_camlight(self.model, rd)
        return rd

    def render(self, camera: str, width: int, height: int, segmentation: bool = False) -> np.ndarray:
        """RGB image, or with segmentation=True an (H, W) array of geom ids (-1 = none)."""
        rd = self.sync_render_data()
        r = self._renderer(width, height)
        r.update_scene(rd, camera=camera, scene_option=self._scene_option)
        if not segmentation:
            return r.render().copy()
        r.enable_segmentation_rendering()
        try:
            seg = r.render().copy()
        finally:
            r.disable_segmentation_rendering()
        ids = seg[..., 0].astype(np.int64)
        ids[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
        return ids

    def render_chase(self, width: int, height: int, distance: float, azimuth: float, elevation: float) -> np.ndarray:
        """Third-person view tracking the trunk (test videos only)."""
        rd = self.sync_render_data()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.trunk_id
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r = self._renderer(width, height)
        r.update_scene(rd, camera=cam, scene_option=self._scene_option)
        return r.render().copy()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


def _has_sensor(m: mujoco.MjModel, name: str) -> bool:
    try:
        m.sensor(name)
        return True
    except KeyError:
        return False
