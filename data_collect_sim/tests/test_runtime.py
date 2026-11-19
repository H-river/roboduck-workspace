"""Runtime contract tests: IMU convention, switcher modes, watchdog, clipping, kick sequence."""

import logging

import numpy as np
import pytest

from duckrt.config import load_sim_cfg
from duckrt.messages import CommandMsg, Mode
from duckrt.runtime import SimRuntime
from duckrt.upstream import quat_apply_inverse


@pytest.fixture(scope="module")
def rt():
    return SimRuntime(load_sim_cfg())


def _rotvec(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ v


def test_imu_matches_training_formula(rt):
    """Deploy formula on the synthesized sensor quat == training mounted_body_* terms."""
    b = rt.backend
    rng = np.random.default_rng(0)
    mount = b.mount_quat
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        b.data.qpos[3:7] = q
        b.data.qvel[3:6] = rng.normal(size=3)
        R_b = np.array([_rotvec(q, e) for e in np.eye(3)]).T
        up_b = R_b.T @ np.array([0, 0, 1.0])
        train_grav = _rotvec(mount, up_b)                      # quat_apply(mount, -g_body)
        train_gyro = _rotvec(mount, b.data.qvel[3:6])          # quat_apply(mount, omega_body)
        ours_grav = np.array(quat_apply_inverse(b.imu_quat(), [0, 0, 1.0]))
        assert np.allclose(ours_grav, train_grav, atol=1e-9)
        assert np.allclose(b.imu_gyro(), train_gyro, atol=1e-9)
    rt.reset(seed=0)
    assert np.allclose(quat_apply_inverse(rt.backend.imu_quat(), [0, 0, 1.0]), [1, 0, 0], atol=1e-6)


def test_watchdog_zeroes_command_keeps_mode(rt):
    rt.reset(seed=0)
    rt.send(Mode.WALK_FLAT, (0.2, 0.0, 0.0))
    for _ in range(5):
        rt.tick()
    assert rt.switcher.cmd[0] == pytest.approx(0.2)
    for _ in range(15):  # 0.3 s without commands
        rt.tick()
    assert rt.switcher.mode == Mode.WALK_FLAT
    assert np.allclose(rt.switcher.cmd, 0.0)
    assert rt.switcher.watchdog_tripped


def test_unimplemented_mode_is_ignored(rt, caplog):
    rt.reset(seed=0)
    rt.send(Mode.WALK_FLAT, (0.1, 0.0, 0.0))
    rt.tick()
    with caplog.at_level(logging.WARNING):
        rt.send(Mode.GET_UP, (0.0, 0.0, 0.0))
    rt.tick()
    assert rt.switcher.mode == Mode.WALK_FLAT
    assert rt.switcher.cmd[0] == pytest.approx(0.1)
    assert any("not implemented" in r.getMessage() for r in caplog.records)


def test_command_clipping(rt):
    rt.reset(seed=0)
    rt.send(Mode.WALK_FLAT, (0.7, -0.3, 1.5))
    rt.tick()
    assert np.allclose(rt.switcher.cmd, [0.25, -0.15, 0.375])
    rt.send(Mode.WALK_FLAT, (0.0, 0.0, 1.5))  # turn in place -> wider wz
    rt.tick()
    assert np.allclose(rt.switcher.cmd, [0.0, 0.0, 0.75])


def test_stale_seq_ignored(rt):
    rt.reset(seed=0)
    rt.submit(CommandMsg(t_send=0.0, mode=Mode.WALK_FLAT, cmd=(0.2, 0, 0), seq=10))
    rt.submit(CommandMsg(t_send=0.0, mode=Mode.WALK_FLAT, cmd=(0.0, 0, 0), seq=9))
    rt.tick()
    assert rt.switcher.cmd[0] == pytest.approx(0.2)


def test_kick_sequence_and_command_lock(rt):
    rt.reset(seed=0)
    sw = rt.switcher
    rt.send(Mode.STAND)
    rt.tick()
    rt.send(Mode.KICK)
    phases = []
    for _ in range(int(12 / 0.02)):
        rt.send(Mode.WALK_FLAT, (0.2, 0, 0))  # must be ignored during the sequence
        rt.tick()
        phases.append(sw.kick.phase if sw.kick else "done")
        if sw.kick is None:
            break
    order = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert order == ["settle_before", "play", "settle_after", "done"]
    assert sw.mode == Mode.STAND
    names = [e[1] for e in sw.events if e[1].startswith("policy")]
    assert any("-> kick" in n for n in names) and any("kick ->" in n for n in names)


def test_fall_flag(rt):
    rt.reset(seed=0)
    b = rt.backend
    b.data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # lying on its side
    b.data.qpos[2] = 0.05
    b.forward()
    for _ in range(int(0.4 / 0.02)):
        rt.tick()
    assert rt.switcher.fall_flag
