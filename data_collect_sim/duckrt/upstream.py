"""Read-only imports from the deployment runtime (microduck/src).

The deploy code is imported, not copied, so the sim runtime and the real robot share
the exact same joint tables and IMU math. microduck/src is appended to sys.path (not
prepended) so nothing here can shadow other packages.
"""

from __future__ import annotations

import sys

from duckrt.config import ROOT

MICRODUCK_SRC = (ROOT.parent / "microduck" / "src").resolve()
if str(MICRODUCK_SRC) not in sys.path:
    sys.path.append(str(MICRODUCK_SRC))

_dwb = sys.dont_write_bytecode
sys.dont_write_bytecode = True  # keep microduck/src read-only (no __pycache__ there)
try:
    import constants as deploy_constants  # noqa: E402  microduck/src/constants.py
    import imu_reader as deploy_imu  # noqa: E402  microduck/src/imu_reader.py
finally:
    sys.dont_write_bytecode = _dwb

OBSERVATION_DOF_ORDER: list[str] = list(deploy_constants.OBSERVATION_DOF_ORDER)
MOTOR_TO_ID: dict[str, int] = dict(deploy_constants.MOTOR_TO_ID)
NEUTRAL_POSE: dict[str, float] = dict(deploy_constants.NEUTRAL_POSE)
IMU_MOUNT_QUAT: tuple[float, float, float, float] = tuple(deploy_constants.IMU_MOUNT_QUAT)

# Deploy IMU helpers: quat_apply_inverse(q, v) and imu_quat_to_body(q) = q (x) IMU_MOUNT_QUAT.
quat_apply_inverse = deploy_imu.quat_apply_inverse
imu_quat_to_body = deploy_imu.imu_quat_to_body
