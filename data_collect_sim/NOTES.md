# data_collect_sim — NOTES

按时间顺序追加。路径都相对 workspace 根目录 `/home/jiamo/Microduck-build-tutorial-main`（`mjlab_microduck/.venv/...` 里的 mjlab / bam 源码写作 `site-packages/...`）。

---

## 第 0 步：调研（2026-10-24，只读）

主要证据：训练时自动落盘的**已解析配置** `mjlab_microduck/checkpoints/walk_stable_5s/params/env.yaml`（与 `logs/rsl_rl/mjlab_microduck_velocity/2026-10-16_17-43-09/params/env.yaml` 逐字节相同）。下面每一项都和源码交叉核对过。

### 0.1 "已知情况"逐条核实

| # | 说法 | 结论 | 证据 / 说明 |
|---|---|---|---|
| 1a | 能用的模型是 `robot_walk.xml`，14 个关节，名字和顺序与 `MOTOR_TO_ID` / `OBSERVATION_DOF_ORDER` 一致 | ✔ | 编译后关节顺序 = `OBSERVATION_DOF_ORDER`（`microduck/src/constants.py:97-112`）；mjlab 编译出的训练场景关节顺序相同。 |
| 1b | imu site 的朝向与 `IMU_MOUNT_QUAT` 一致 | **数值相同，语义相反** | site quat = (0.7071, 0, 0.7071, 0) = `IMU_MOUNT_QUAT`。但训练观测是 `quat_apply(mount, v_body)`（`mjlab_microduck/src/mjlab_microduck/tasks/mdp.py:142-179`），部署端是 `imu_quat_to_body(q) = q ⊗ mount`（`microduck/src/imu_reader.py:69-73`），所以物理传感器系 = 机身系 ⊗ conj(mount)，**正好是 site 的逆**。直接读 site 上的 `framequat`/`gyro` 传感器会让 gravity / gyro 的 x、z 翻号（站立时 gravity 观测从 (1,0,0) 变成 (−1,0,0)）。策略观测从不读这个 site。→ 我们的后端用 trunk_base 的机身姿态 + mount 四元数构造 IMU，**不用 XML 里的 imu 传感器**。（`~/microduck_ws/rl_fange/NOTES.md:229-236` 也独立得出了同样结论。） |
| 1c | 有 INIT / STAND / SIT / FOLD 四个 keyframe | ✔（位置更正） | keyframe 在 `scene_walk.xml` / `scene.xml` 里，不在 `robot_walk.xml` 里。STAND = "STAND2" 姿态，与训练 HOME_FRAME 相同（见 0.2）。 |
| 1d | 总质量 0.768 kg | ✔ | `body_subtreemass[1]` = 0.7680。 |
| 1e | 带 `head_camera` | ✔ 存在，但**朝向是错的** | 相机挂在 `jaw_soft` 上，`quat="0 0 -1 0"`（由 `config_mjcf_walk.json` 的 post_import sed 生成）。STAND 时它的光轴朝**后**、画面滚转 90°，渲染出来是镜头网格内壁（灰色圆盘）。同位置的 `head_camera` **site** 的 +x 轴朝前，`lens` / `m12_lens_holder` 网格也说明镜头朝 site +x。→ 我们用 site 帧构造相机：光轴 = site +x，画面上方 = site +z（见 0.6、第 2 步）。 |
| 2 | `microduck/src/model/mjcf/robot.xml` 是旧版 19 关节机器人 | ✔ | 加载得 nu=19（手臂、`head`、踝关节拆成 pitch/roll）。 |
| 3a | `scene_walk.xml` 只有脚和地面碰撞 | ✔ | 另有 3 个 contype=2 的几何（`power_support`、两条 `leg`），只做自碰撞。 |
| 3b | `scene.xml`（`robot_allcollisions.xml`）里躯干、髋、腿、下巴都碰地 | ✔，**但它还改了关节限位** | 编译后逐项比较，质量、惯量、site、相机全同，只有关节范围不同：`left_hip_yaw` 下限 −0.4363（walk 版 −0.1745），`right_hip_yaw` 上限 0.4363（walk 版 0.1745），`head_yaw` ±0.35（walk 版 ±0.3491）。walk 策略是在 walk 版限位下训练的。→ **决定**：用 allcollisions 的碰撞几何 + walk 版关节限位（`configs/sim.yaml: robot.joint_ranges_from`，可切回原版）。 |
| 3c | — | 补充 | 训练里的 `FULL_COLLISION`（`robot/microduck_constants.py:41-46`，`disable_other_geoms=True`）只能按名字禁用其他几何；allcollisions 里多出来的碰撞几何**都没有名字**，所以不会被禁用（用 mjlab `CollisionCfg.edit_spec` 实际编译验证）。训练里两只脚：condim 3、priority 1、friction 1.0。我们的场景照搬这三项。 |
| 4 | XML position 执行器（kp=0.55）不是真实动力学；mjlab 和 MuJoCoController 都用 BAM | ✔，**但两者用的是 BAM 里两套不同的实现** | mjlab：`bam.mjlab.BamActuator`，摩擦在 `compute()` 里以力矩形式加入，带 BAM Algorithm-1 静摩擦截断（`site-packages/bam/mjlab.py:429-525`）。MuJoCoController：`bam.mujoco.MujocoController`，每步改写 MuJoCo 的 `dof_frictionloss` / `dof_damping`，由求解器截断（`site-packages/bam/mujoco.py:111-190`）。→ 我们按训练用的 `bam.mjlab` 逐行移植到 numpy（第 1 步）。 |
| 5 | `sim_main.py:34` 写死旧模型路径 | ✔ | 另外两点让它即使换了路径也复现不了训练：IMU 读的是 XML site 传感器（见 1b，帧反了），BAM 用的是 `bam.mujoco` 那一套（见 4）。我们不用 MuJoCoController。 |
| 6 | mjlab venv 有 bam、没有 onnxruntime | ✔ | 新环境 `data_collect_sim/.venv` 装了 onnxruntime 1.23.2，bam 锁到同一 commit `0007411bd82c…`（`mjlab_microduck/uv.lock:111-113`）。 |
| 7 | 现成 walk 策略在 `checkpoints/walk_stable_5s/` | ✔ | 位置：`mjlab_microduck/checkpoints/walk_stable_5s/`，含 `model_1300.pt`、`params/`、`eval.json`（64 个 DR 环境 vx=0.2 走 5 s 全部通过）、一段视频。**没有 ONNX** → 已导出（0.3）。 |

### 0.2 `Mjlab-Velocity-Microduck` 环境（`model_1300.pt` 用的配置）

源码：`mjlab_microduck/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`（下称 cfg），`tasks/mdp.py`，`robot/microduck_constants.py`，mjlab 1.3.0。

#### 观测（actor，51 维，按顺序拼接）

| 维度 | 名字 | 定义 | 噪声（仅训练） | 延迟 | 出处 |
|---|---|---|---|---|---|
| 0–2 | `base_ang_vel` | `quat_apply(mount, ω_trunk)`，ω 是 `trunk_base` 机身系角速度，mount = `BNO080_MOUNT_QUAT` (0.7071068, 0, 0.7071068, 0) | U(±0.03) | 0–1 个策略步，每 64 步重采样 | cfg:82, 206-228；mdp.py:164-179；env.yaml:274-350 |
| 3–5 | `projected_gravity` | `quat_apply(mount, −g_trunk)`，是"向上"的单位向量，站立 ≈ (1, 0, 0) | U(±0.01) | 同上 | cfg:207-213, 226-228；mdp.py:142-161；env.yaml:351-427 |
| 6–19 | `joint_pos` | `q − default_q`，`biased=False`（不加 encoder_bias） | U(±0.001) | 无 | cfg:185-191；`site-packages/mjlab/envs/mdp/observations.py:51-61` |
| 20–33 | `joint_vel` | `q̇ − 0` | U(±0.25) | 0–1 个策略步，每步重采样 | cfg:193-199 |
| 34–47 | `actions` | 上一步策略的**原始输出**（缩放前） | 无 | 无 | env.yaml:572-585；observations.py:80-83；`managers/action_manager.py:114-117` |
| 48–50 | `command` | twist 命令 (vx, vy, ωz)，机器人系 | 无 | 无 | env.yaml:586-600 |

- 没有 scale、没有 clip、没有观测历史（`history_length` 0），没有步态相位 / 时钟（ONNX 输入 51 维，`microduck/src/moves/walk.py:50-54` 只有输入 > 51 才加相位）。
- 处理顺序：func → noise → clip → scale → delay（`managers/observation_manager.py:329-350`）。噪声只在 `enable_corruption=True` 时加（训练 True，play False，cfg:456）。
- 归一化在 ONNX 里：`(obs − mean) / (std + 0.01)` 再进 MLP 512-256-128 ELU（已用 checkpoint 权重核对，最大误差 2.4e-6）。

#### 动作
- `JointPositionAction`，`scale = 1.0`（cfg:177），偏移 = `default_joint_pos`（`use_default_offset`，`envs/mdp/actions/actions.py:218-219`），不裁剪（env.yaml:801 `clip: null`）。
- 目标 = `a + default_q − encoder_bias`（actions.py:221-224）；DR 关闭时 encoder_bias = 0。
- `default_q` = `HOME_FRAME`（`robot/microduck_constants.py:21-39`，注释"STAND2 pose"）= `scene*.xml` 的 STAND keyframe：
  `[0, −0.0873, −0.4579, −0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, −0.4530]`（`OBSERVATION_DOF_ORDER`）。
  ONNX metadata 里的 `default_joint_pos` 被 mjlab 导出器四舍五入到 3 位小数（`site-packages/mjlab/rl/exporter_utils.py:12-20`），部署端 walk.py 直接用它，所以部署端参考姿态偏 ≤ 5e-4 rad。**policy card 用配置里的全精度值。**

#### 频率与步进
- `timestep = 0.005`（cfg:107-116），`decimation = 4`（env.yaml:1）→ 策略 50 Hz。
- 求解器：implicitfast、newton、pyramidal、iterations 10、ls_iterations 20、ccd_iterations 100、tolerance 1e-8（env.yaml:1253-1270）。
- 每个策略步的顺序（`envs/manager_based_rl_env.py:412-457`）：处理动作 → 4 × [写目标 → 执行器 compute（含延迟）→ `mj_step`] → 终止 / 奖励 → **一次 `mj_forward`** → 命令更新 → 事件 → 观测。所以 4 个子步里只有第 1 个子步 BAM 读到的 `qfrc_bias` 是新鲜的，后 3 个落后一个子步。我们照这个顺序实现。

#### 速度命令（`model_1300.pt` 实际见到的分布）
- 基础范围：vx U(−0.25, 0.25)，vy U(−0.15, 0.15)，ωz U(−0.375, 0.375)（cfg:337-339）。
- 10 % standing 环境：全零（cfg:333）。
- 10 % 原地转向环境：vx = vy = 0，ωz ~ U(−0.75, 0.75) 且 |ωz| ≥ 0.5（cfg:335, 341-342；mdp.py:43-77）。
- 20 % forward-only 环境（mjlab 默认 `rel_forward_envs=0.2`，env.yaml:1794）：vx = max(|vx|, 0.3)，vy = ωz = 0（`tasks/velocity/mdp/velocity_command.py:92-99`）。基础范围上限只有 0.25，所以这些环境的 vx **恰好是 0.3**。
- heading 环境 0 %；每 3–8 s 重采样（env.yaml:1784-1792）。
- 课程（cfg:387-412）在 env step 72000 = 第 3000 次迭代才放宽到 vx ±0.35、ωz ±0.75、原地转 ±1.5。**model_1300 在第 1300 次迭代，没到这一步**。（另外，2026-10-16 检查时发现 `command.build` 的 lambda 抓住的是 deepcopy 前的 cfg，课程放宽可能根本传不到采样器；对 model_1300 无影响。）
- → policy card：`command_range` = vx [−0.25, 0.25]、vy [−0.15, 0.15]、ωz [−0.375, 0.375]；附加规则：vx = vy = 0 时 |ωz| 上限 0.75。

#### BAM 执行器
- 配置：`BamActuatorCfg(motor_name="xl330", model="m6", kp_fw=125, vin_range=(7,8), vin_drop_gain_range=(0,0.2), vin_min=6.0, max_current=1.75, delay_min_lag=3, delay_max_lag=6)`（`robot/microduck_constants.py:50-61`；env.yaml:127-152）。
- 参数文件：`site-packages/bam/params/xl330/m6.json`（kt 0.36601、R 2.81139、armature 1.8077e-3、摩擦项若干）。
- XL330 固件常数：error_gain = 4096/(2π) / (256 × 885) = 2.8774e-3，max_pwm 1.0（`bam/dynamixel/actuator.py:79-101`）。
- 改模型（`bam/mjlab.py:186-235`）：position 执行器改成 motor，forcerange ±(max vin)·kt/R = ±1.042 N·m，关节 armature = 1.8077e-3，damping = frictionloss = 0。
- 每个物理步（`bam/mjlab.py:429-525`）：duty = clip((q* − q)·kp·error_gain, ±1)；τ_m = kt·vin·duty/R − kt²·q̇/R，按 ±max_current·kt 截断；vin 每步减去 `drop_gain·Σ|τ_m(上一步)|`、下限 vin_min；摩擦预算 = m6 模型（Coulomb + Stribeck + 方向性负载摩擦 + 二次项）+ 粘滞；Algorithm-1：τ_stop = (I/dt)·q̇ + τ_m − τ_ext，I = 1/`dof_invweight0`，τ_ext = −`qfrc_bias`；输出 τ_m − sign(τ_stop)·min(|τ_stop|, 预算)。
- 延迟：目标位置经过 mjlab `DelayBuffer`，每个物理步从 {3,4,5,6} 均匀抽一个滞后（15–30 ms），历史用第一个值回填（`site-packages/mjlab/utils/buffers/delay_buffer.py`；`actuator/actuator.py:270-302`）。

#### 训练时的域随机化（本阶段全部关闭，列给第二阶段）

| 项 | 范围 | 时机 | 出处 |
|---|---|---|---|
| 脚底摩擦 | abs U(0.3, 1.2) | startup | env.yaml:915-981 |
| encoder bias（只影响执行目标） | U(±0.015) rad | startup | env.yaml:982-1044 |
| 躯干 CoM 偏移 | ±5 mm / 轴 | startup | cfg:357-362 |
| 关节 armature 缩放 | ×U(0.9, 1.1) | startup | cfg:364-372 |
| 关节 frictionloss 缩放 | ×U(0.5, 2.0) | startup | cfg:374-382 —— **实际无效**：BAM 已把 frictionloss 置 0 |
| 推机器人 | 每 1–3 s，xy 速度 ±0.5 m/s | interval | cfg:347-350 |
| 电池电压 | vin U(7, 8) V | startup | constants:55 |
| 电压跌落增益 | U(0, 0.2) V/N·m | startup | constants:56 |
| 执行器延迟 | 3–6 物理步，每步重采样 | 每步 | constants:59-60 |
| 观测噪声 / 观测延迟 | 见观测表 | — | — |
| 初始状态 | xy ±0.5 m，yaw 任意，z 0.12–0.13，关节 = HOME | reset | env.yaml:811-831 |

**关闭 DR 时的标称值（本项目的取法）**：vin 7.5 V（= 部署端 `BAM_VIN`，`microduck/src/constants.py:56`，也是训练区间中点）；跌落增益 0.1（训练区间中点；部署端 MuJoCoController 用 0.2，记为差异）；vin_min 6.0；max_current 1.75；脚底摩擦 1.0（XML / 训练 CollisionCfg）；其余随机项取 0。执行器延迟不算 DR，它是在模拟真机的通信延迟，默认按训练分布保留（每物理步从 {3..6} 中抽，RNG 可设种子），parity 测试里固定。观测延迟默认 0。

### 0.3 walk 策略 ONNX
- `walk_stable_5s/` 只有 `.pt`，用仓库自带脚本导出（上游文件没改）：
  ```
  cd data_collect_sim/outputs/policies
  ../../../mjlab_microduck/.venv/bin/python ../../../mjlab_microduck/src/mjlab_microduck/scripts/export_onnx.py \
      --checkpoint ../../../mjlab_microduck/checkpoints/walk_stable_5s/model_1300.pt --device cpu
  mv policy.onnx walk_flat_model1300.onnx
  ```
- 结果：`outputs/policies/walk_flat_model1300.onnx`，sha256 `e2677036…690f25d`，输入 `obs`[1,51] → 输出 `actions`[1,14]，metadata 里 `run_path=walk_stable_5s`。
- 核对：64 个随机输入，ONNX 和用 checkpoint 权重手写的前向最大差 2.4e-6。
- 仓库里另外两个 51 维 walk ONNX：`microduck/src/agents/walk.onnx`（上游自带，run `2026-09-04_00-06-48`，不在本机）和训练 run 结束时自动导出的 `logs/.../2026-10-16_17-43-09.onnx`（最后一个 checkpoint，不是 1300）。本项目按要求用 model_1300。

### 0.4 部署代码（`microduck/src`）和训练环境的差异

我们的运行时以后要上真机，所以先把部署端的做法列清楚。冲突处都能判断以哪边为准，不需要停下来问。

| # | 项 | 部署端 | 训练端 | 我们怎么做 |
|---|---|---|---|---|
| D1 | gyro | BNO08x 原始 gyro（传感器系）（`observer.py:78`） | `R_mount · ω_body` | 两边等价（传感器系 = body ⊗ conj(mount)）。仿真里按训练公式算，真机后端读原始 gyro。 |
| D2 | gravity | `quat_apply_inverse(q_sensor, [0,0,1])`（`observer.py:84-85`） | `R_mount · (−g_body)` | 等价；仿真后端先合成 `q_sensor = q_body ⊗ conj(mount)` 再走部署公式，让真机和仿真走同一条代码路径。parity 测试验证。 |
| D3 | default_q | ONNX metadata，3 位小数（`walk.py:39-47`） | 全精度 | 用 card 里的全精度值。 |
| D4 | 启动 | 1.5 s smoothstep 从当前姿态插值到 default，然后 kp 400→125（`walk.py:65-90`） | 直接从 HOME 落地，kp 一直 125 | 仿真直接从 STAND keyframe 起步，kp 125。真机交接流程第二阶段再接。 |
| D5 | 倾倒保护 | 机身重力 z > −0.5（倾角 > 60°）时跳过策略，目标回到 `NEUTRAL_POSE`（`walk.py:101-107`；scheduler 每 tick 新建 `MotorCommand`） | 无（高度 < 0.10 m 终止） | switcher 里的摔倒检测另设阈值（`sim.yaml`），fall_flag 置位后的行为第二阶段定。 |
| D6 | 速度命令上限 | `scale_velocity`：vx 0.7 / −0.5，vy 0.3，ωz 1.5（走）/ 3.0（原地）（`input/input_source.py:19-34`，`constants.py:84-88`） | vx ±0.25（forward-only 0.3），vy ±0.15，ωz ±0.375（原地 0.75） | **部署端上限是训练范围的 2–4 倍**。switcher 按 card 的 `command_range` 裁剪。 |
| D7 | 命令变化 | 无平滑，50 Hz 直通 | 每 3–8 s 跳变一次 | 上层 10 Hz，遥控加变化率限制。 |
| D8 | joint_vel | Dynamixel present_velocity（0.229 rpm 量化） | 仿真真值 | 仿真用真值；量化可作为第二阶段的增强。 |
| D9 | 执行器延迟 / 观测延迟 | 真机自带 | 3–6 物理步；IMU 0–1 步，joint_vel 0–1 步 | 见 0.2 的标称值说明。 |
| D10 | 目标 kp | `sync_write_kp(KP_RL=125)`（`walk.py:85-87`） | kp_fw 125 | 一致。 |
| D11 | 动作裁剪 | 无 | 无 | 一致（不裁剪）。 |
| D12 | 头部关节 | 策略自己输出头部 4 个关节的目标 | 同 | 一致；头部相机因此随策略晃动（第 2 步测）。 |

### 0.5 其他策略

在本 workspace 里没找到 stand / sit / kick / get_up 的任何文件。在 `~/microduck_ws`（另一个 workspace）里找到：

| 策略 | 结论 | 说明 |
|---|---|---|
| **kick** | **找到，可用于本机器人**：`~/microduck_ws/runs/ai_fange_kick/models/2026-10-06_13-14-13_af_kick_stage1/`（`model_3999.pt` + 同名 `.onnx`，16:14 导出，即 model_3999） | 训练配置 `~/microduck_ws/duck_soccer/training/ai_fange_kick/af_kick_cfg.py`，任务 `Mjlab-BallKick-Right-AiFange`，2048 env × 4000 iter。机器人目录和 tasks 与本仓库 `mjlab_microduck` **逐文件相同**（`diff -rq` 验证）；BAM 同一 commit；mujoco 3.15.0（我们 3.16.0）。 |
| kick 观测 | 51 维，和 walk **完全相同的布局**；**观测里没有球**（盲踢），twist 命令训练时保持 ≈0（vx、vy ±0.01，ωz ±0.05，af_kick_cfg.py:211-213） | `joint_pos` 用 `biased=True`（:197），DR 关闭时等价。critic 才有球的位置 / 速度。 |
| kick 球 | 半径 0.035 m，质量 0.015 kg，惯量 1.225e-5，摩擦 (0.5, 0.005, 0.0001)，橙色 | `ball.xml`（af_kick_cfg.py:64；同目录 `ball.xml`） |
| kick 球的初始分布 | 球心在**机器人根节点（trunk_base）的 yaw 坐标系**下 (0.09, −0.042) ± 0.015 m（每轴均匀），z = 半径；右脚踢 | af_kick_cfg.py:65-67, 226-229；`pollen_kick_mdp.py:608-648` |
| kick 起始状态 | 只从站立开始：z 0.11–0.12，倾角 ±5°，关节 HOME ±0.05；回合 5 s；没有"踢完回站立"的显式要求，奖励里有站立姿态 / 高度项 | af_kick_cfg.py:63, 219-225 |
| kick 机器人模型 | `robot_allcollisions.xml`（**hip_yaw 内侧限位 ±0.436**，见 0.1-3b），FULL_COLLISION，BAM 同 walk | af_kick_cfg.py:81, 87-95 |
| kick 状态 | 只训了 stage 1；Pollen 原版 kick 还需要 refine / precise 两个阶段和独立验收，这一版**没有做独立验收** | `~/microduck_ws/duck_soccer/NOTES.md:1163-1185` |
| stand | **无独立策略**。用 walk 零命令代替（第 1 步实测站不站得住） | — |
| sit | **无**。`~/microduck_ws/rl_fange` 计划训练 sit↔stand，只跑过 smoke，没有正式模型（`rl_fange/NOTES.md:255-272`） | — |
| get_up | **对本机器人无**。`~/microduck_ws/runs/standup/*` 是在 Pollen 的 microduck_rl 模型上训练的（61 维观测、kp 200、不同 MJCF），不能直接用 | `duck_soccer/NOTES.md:969-973` |
| 关键帧 | 只有 MJCF 里的 SIT / FOLD keyframe（关节姿态，不是动作序列） | `scene*.xml` |

### 0.6 head_camera

- 挂在 `jaw_soft`（head_roll 关节之后的头部最末端刚体）上，所以随 neck_pitch / head_pitch / head_yaw / head_roll 四个关节运动。walk 策略本身会输出这四个关节的目标。
- 相对 `jaw_soft`：pos (0.0155, −9.14e-5, −0.0733)；site quat (0.7071, 0, 0.7071, 0)；XML 相机 quat (0, 0, −1, 0)（**错**，见 0.1-1e）；`fovy` 没写，用 MuJoCo 默认值 45°（竖直方向）。
- STAND keyframe 落地站稳后（XML 执行器静持 2 s，躯干 z 0.115 m）：镜头离地 **0.242 m**，在躯干前方 6.3 cm；site 帧的光轴几乎水平，**俯仰 +0.4°（微微朝上）**。
- 后果：竖直 FOV 45° 时下视场边缘只到水平线以下 22.5°，离地 0.242 m → **0.58 m 以内的地面看不见**。kick 窗口的球（前方约 9 cm）肯定不在画面里。真机相机是 M12 镜头座，实际 FOV 未知，相机参数先写占位值（第 2 步）。
- （`~/microduck_ws/duck_soccer` 量到的"俯仰 −43.9°"来自他们自己定义的相机，不是这个 XML。）

### 0.7 对实现的决定（小结）
1. 机器人模型：`robot_allcollisions.xml` 的碰撞几何 + `robot_walk.xml` 的关节限位 + 训练 CollisionCfg 的脚底参数；不改上游 XML，用 MjSpec 在加载时改。
2. 执行器：把 `bam/mjlab.py` 的 `BamActuator` 移植成 numpy（含 `edit_spec` 的改模型和 mjlab `DelayBuffer` 语义）。
3. IMU：由 trunk_base 机身姿态和 `BNO080_MOUNT_QUAT` 合成"传感器四元数"和传感器系 gyro，观测按部署公式构造。
4. 步进顺序和 mjlab 一样：4 个子步只在策略步末尾做一次 `mj_forward`。
5. 相机：用 `head_camera` site 帧重建，fovy 等参数放 `configs/camera.yaml`。
6. 有 kick（盲踢、右脚、球在根节点系 (0.09, −0.042)）→ 实现 `KICK`；STAND = walk 零命令；GET_UP / CELEBRATE / WALK_SLOPE / WALK_ROUGH 注册为未实现。
7. 本步没有需要停下来问的情况：walk checkpoint 找到了；训练 / 部署的观测差异都能判断以哪边为准。

---

## 第 1 步：运行时底座（2026-10-24）

### 1.1 做了什么
- 独立环境 `data_collect_sim/.venv`（uv，Python 3.13.5，`uv.lock`）：mujoco 3.16.0（= mjlab venv）、onnxruntime 1.23.2、bam 锁到 `0007411bd82c…`（= mjlab_microduck），numpy 2.3.5（= mjlab venv），pygame 2.6.1、imageio 2.37.0、pyyaml 6.0.3。
- `duckrt/`：`messages.py`、`policy_card.py`、`obs.py`、`policy.py`、`actuation.py`（`bam/mjlab.py` 的 numpy 移植）、`backend_sim.py`、`switcher.py`，另加 `runtime.py`（统一仿真时钟的调度）、`upstream.py`（直接 import 部署端的 `constants.py` / `imu_reader.py`，不复制）、`config.py`、`testing.py`。
- 部署端代码复用方式：`OBSERVATION_DOF_ORDER`、`MOTOR_TO_ID`、`IMU_MOUNT_QUAT`、`quat_apply_inverse`、`imu_quat_to_body` 都从 `microduck/src` import（`upstream.py` 把它追加到 `sys.path` 末尾）。
- policy card：`scripts/make_policy_cards.py` 读 `configs/policy_sources.yaml` 指向的 `params/env.yaml`（用容忍 `!!python` 标签的 loader）、训练源码（ast 解析，不 import）和 ONNX metadata，生成 `configs/policies/{walk_flat,walk_flat_2600,kick}.yaml`，每个值旁边写出处和行号。

### 1.2 parity（`scripts/check_parity.py`，报告 `outputs/parity/report.json`）
方法：`scripts/parity_mjlab_rollout.py` 在 mjlab 自己的 venv 里跑 `Mjlab-Velocity-Microduck`（play 配置、1 个 env、CPU）：删掉所有 DR 事件，观测延迟 0，BAM vin = 7.5、跌落增益 = 0.1、执行器延迟固定 4 步，复位姿态固定（x = y = yaw = 0，z = 0.12），命令直接写进 command term（这个任务的 `command.build` lambda 不认 cfg 修改，见 0.2）。策略用同一个 ONNX（`onnx.reference`）。每个策略步记录 qpos、qvel、观测、动作、BAM 携带力矩。我们这边用**训练模型**（robot_walk.xml，只有脚碰地）对比，再用项目模型（allcollisions 几何）跑一遍闭环。

| 工况 | ① 观测重放：逐项最大误差 | ② 单步动力学：关节位置误差 中位 / p95 / 最大 (rad) | ③ 闭环：观测误差首次 > 1e-5 | 闭环 4 s 位移 mjlab / 我们 (m) |
|---|---|---|---|---|
| vx 0.2 | gyro 6.0e-7，gravity 2.4e-7，q 3.0e-8，q̇ / 动作 / 命令 0 | 1.2e-3 / 4.5e-3 / 7.6e-3 | 第 2 步 | (0.534, 0.096) / (0.500, 0.157) |
| 零命令 | ≤ 2.4e-7 | 8.6e-4 / 1.6e-3 / 2.0e-3 | 第 2 步 | (0.002, −0.001) / (0.001, −0.001) |
| ωz 0.5 | ≤ 7.2e-7 | 1.4e-3 / 6.7e-3 / 8.6e-3 | 第 2 步 | (0.178, 0.091) / (0.166, 0.109) |
| 悬空（重力 0，z 0.5，无接触） | ≤ 2.4e-7 | **3.2e-8 / 5.7e-8 / 8.1e-8** | 第 16 步（第 199 步 2.7e-4） | — |

结论：
- **观测构造 PASS**：四个工况、每项都 < 1e-6（float32 精度），满足 < 1e-5。IMU 用"传感器四元数 = 机身 ⊗ conj(mount)"再走部署公式，和训练的 `mounted_body_*` 一致；`tests/test_runtime.py::test_imu_matches_training_formula` 用随机姿态再验证一遍。
- **执行器 + 关节动力学一致**：悬空工况单步误差 ≤ 8e-8 rad，闭环 15 步内 < 1e-5，之后只是 float32（mjwarp）和 float64 的累积误差。BAM 移植、延迟线、步进顺序、`qfrc_bias` 的新旧都对上了。
- **有地面接触时轨迹会发散**，从第一次脚着地开始。原因已定位，是**接触生成算法不同**，不是参数：同一状态下，mjwarp（训练）在脚底生成 4 个分散约 1 cm、深度各不相同的接触点；MuJoCo C 的 mesh–plane 碰撞把 4 个点都放在最深顶点附近 1.5 mm 以内（等于点接触，脚可以绕着它晃）。两边编译后的模型逐字段比较完全相同（只有 forcerange 因 vin 固定不同，见下）。试过：把脚底凸包 `maxhullvert` 降到 64 / 32 / 16 / 8（更差，改了几何），`multiccd` 在 3.16 默认已开。没有再改几何去凑。统计上两边接近（4 s 位移差 < 7 %，见 1.3 的跟踪表；`walk_stable_5s/eval.json` 在 vx 0.2 下 DR 环境平均 0.141 m/s，我们 0.143）。

排查中找到并修掉的不一致：
1. **mjlab 复位时往延迟线里写的是全 0 目标**：`reset()` 末尾 `scene.write_data_to_sim()` 用初值为 0 的 `joint_pos_target` 跑一次执行器，所以每个训练 episode 开头 3–6 个物理步里所有关节都被命令到 0 rad，这是训练里的一个伪影。现在 `actuator.reset_target` 可选：`zeros`（复现 mjlab，parity 用）或 `hold`（当前姿态，物理上合理，默认）。复位时也像 mjlab 那样跑一次执行器计算，让电压跌落模型的"上一步力矩"状态一致。
2. mjlab 的 `observation_manager.compute()` 会返回缓存，复位后改了命令要用 `compute_group()` 重新算（只影响参考数据的第 0 步）。
3. vin 固定时 mjlab 用这个 vin 算 forcerange（±0.976 N·m）；训练时 vin ∈ [7, 8]，用 8 V（±1.042）。parity 里按 vin 设，默认配置保持训练的 8 V。

另外：项目模型（allcollisions 几何 + walk 版限位）和训练模型在这些 4 s 工况里轨迹逐位相同，多出来的碰撞几何在正常行走时没有碰到任何东西。

### 1.3 walk_test（`outputs/walk_test/<card>/report.md`）
两张卡都跑了（原因见 1.4）。默认配置：执行器延迟按训练分布每物理步从 {3..6} 抽，观测延迟 0，DR 关闭。

| | walk_flat（model_1300，指定的那个） | walk_flat_2600（同一 run 最后一个 checkpoint） |
|---|---|---|
| 20 seed × 60 s 随机命令 | **20/20 不摔** | **20/20 不摔** |
| 分段跟踪增益 vx / vy / ωz | 0.64 / 0.26 / 0.68 | 0.62 / 0.23 / 0.43 |
| vx 0.10 / 0.15 / 0.20 / 0.25 → 实际 | 0.085 / 0.116 / 0.143 / 0.188 | **0.000** / 0.104 / 0.136 / 0.165 |
| vx −0.15 / −0.25 → 实际 | −0.052 / −0.134 | −0.034 / −0.138 |
| vy ±0.15 → 实际 | +0.055 / −0.020 | +0.046 / −0.030 |
| 原地 ωz ±0.375 / ±0.6 → 实际 | +0.19 / −0.27，+0.49 / −0.40 | +0.21 / **−0.01**，+0.28 / **−0.05** |
| 直走 vx 0.2 时的航向漂移 ωz | **+0.13 rad/s（向左）** | −0.003 |
| 边走边转 vx 0.2、ωz ±0.3 → ωz | +0.30 / −0.18 | +0.07 / −0.24 |
| 静止起步零命令 10 s | 站定（抬脚 0，位移 6 mm，偏航 −0.8°） | 站定（位移 3 mm，偏航 −1.7°） |
| 行走后给零命令 | **停不下来**：10/10 次 3 s 内都没站稳，继续以约 0.1 m/s 踏步漂移 | 0.44–0.52 s 站稳 |
| 单核实时倍率（不渲染 / 每 1/15 s 渲染 320×240 头部相机） | ×61 / ×42 | ×62 / ×42 |

（瞬时速度标准差很大，ωz 约 0.4 rad/s、vy 约 0.13 m/s，这是步态里机身的摆动；按 0.5 s 窗口平均后 ωz 只有 0.03–0.1，见报告。）视频：`zero_command.mp4`、`turn_in_place.mp4`、`walk_turn.mp4`（每张卡各一套）。实时倍率测的时候进程固定在 CPU 2，EGL 渲染走 GPU。

### 1.4 ⚠️ model_1300 行走后停不下来；STAND 的处理
- 现象：静止起步给零命令能站住；**走起来以后再给零命令，会一直踏步并以约 0.1 m/s 漂移**。在 mjlab 训练环境里复现了（8 个 env，DR 开着，vx 0.2 走 4 s 后归零，6 s 后速度仍是 0.07–0.19 m/s），所以这是策略本身的行为，不是仿真器差异。
- 原因（推测，和训练配置一致）：`no_stepping` 惩罚权重在课程到第 3000 次迭代之前一直是 0（cfg:290-298, 400-405），model_1300 还在第 1300 次迭代。
- 试过的替代：不用策略、以 kp 400（部署端空闲时的 KP_DEFAULT）或 kp 125 保持 default_q：两种都会在 4 s 内倒下；把命令在 1 s 内渐变到 0：不倒，但也停不下。
- 同一 run 的 model_2600 和上游 `walk.onnx` 都能停住（3 s 后 1–2 mm/s）。
- **决定**：WALK_FLAT 仍用指定的 model_1300（低速跟踪和转向明显更好）；**STAND 用 model_2600 配零命令**（`sim.yaml runtime.policies.stand`）。热切换实测 25/25：5 种行走命令 × 5 个 seed，1300 走 4 s → 切 2600 零命令 → 0.40–0.60 s 站稳、不摔 → 再切回 1300 走 vx 0.2，正常（0.11–0.16 m/s）。两者同一训练 run、同一观测格式，切换时重置动作历史。KICK 的前后站立阶段也用 stand 角色。
- 这条需要 RL 同学确认（见总结）。

### 1.5 switcher 行为（`tests/test_runtime.py`，7 个测试全过）
- WALK_FLAT / STAND / KICK 已实现；WALK_SLOPE / WALK_ROUGH / CELEBRATE / GET_UP 收到时记 warning，模式保持，命令不生效，**也不算看门狗心跳**（否则执行器一直发未实现模式时机器人会按旧命令一直走）。
- 命令按 card 的 `command_range` 裁剪；vx = vy = 0 时 ωz 上限放宽到 0.75（训练里的原地转向环境）。
- 看门狗 200 ms：速度归零、模式保持。seq 不递增的消息丢弃。
- fall_flag：IMU 算出的机身 up 向量 z < 0.5（倾角 > 60°）持续 0.3 s。倾倒保护（同部署端 walk.py:101-107）：倾角 > 60° 时跳过策略、保持 default_q。
- KICK：stand 站 `settle_s` → kick 策略播放 `duration_s` → stand 站 `settle_s` → 回到之前的模式；整个过程中忽略新命令；每次换策略都重置动作历史。kick 的 `duration_s`、`settle_s` 在第 3 步有了球之后实测（见第 3 步之后的 kick 一节）。

### 1.6 调了哪些参数、为什么
- 没有调任何策略相关参数。所有改动都是为了和训练对齐：执行器复位行为（`reset_target`）、forcerange 的 vin 来源、碰撞几何 / 关节限位的组合方式（0.7）。
- `sim.yaml runtime.policies.stand = walk_flat_2600`：见 1.4。
- 注意：shell 里的 `PYTHONPATH=/opt/ros/jazzy/...` 会把 ROS 的 pytest 插件带进来让 pytest 崩，所有命令都建议加 `env -u PYTHONPATH`（README 里写了）。

---

## 第 2 步：头部相机（2026-10-24，`outputs/camera_check/report.md`）

### 2.1 做了什么
- `configs/camera.yaml`：挂载 body `jaw_soft`；位姿取自 `head_camera` **site**（XML 里那个 camera 朝后，见 0.1-1e）；`pos_offset_m`、`pitch_down_deg`、`fovy_deg` 都是 `# 占位，待真机实测`；320×240、15 fps。加载时用 MjSpec 加一个新相机 `head_cam`（光轴 = site +x，画面上方 = site +z），XML 不动。另加俯视相机 `top_cam`（高度按场地尺寸自动算）和测试视频用的跟随相机。
- `duckrt/camera.py`：`intrinsics()`（fovy 是竖直方向，方形像素，像素中心在整数坐标，同 OpenCV）；`HeadCamera.extrinsics()` 给出相机在世界系和机器人系下的位姿（OpenCV 相机系：x 右、y 下、z 前；机器人系原点在躯干投影到地面处，x 为航向）；`pixel_to_ground()` / `ground_to_pixel()`，射线不朝下或点在相机后面时返回 None。`FrameMsg` 带曝光时刻的外参。
- 渲染用单独的 MjData，只算运动学（`mj_kinematics` + `mj_camlight`），所以相机可以在任意物理步曝光，不影响动力学（见 1.2，`qfrc_bias` 的新旧会影响 BAM）。

### 2.2 投影验证：PASS
- 地面放 15 个 1 cm 半径的薄圆片（不参与碰撞），渲染**分割图**（geom id），取每个标记像素质心 → `pixel_to_ground` → 和真值比；再把真值 `ground_to_pixel` 和质心比。
- 结果（4 倍分辨率渲染，K 按比例缩放）：默认相机（标记在 0.7–2.5 m）最大 8.2 mm（远处，约 0.34 像素的质心误差）；下俯 35° 时（0.25–1 m）站立 0.5 mm、**行走中的一帧 1.2 mm**。320×240 原分辨率下近处标记 < 2.4 mm。
- 排查中踩的坑：分割渲染要关 MSAA（`visual.quality.offsamples = 0`），否则小标记边缘的 id 颜色被混合，id 会错；贴着画面边缘的标记质心有偏差，不计。

### 2.3 头部姿态不是 STAND keyframe 那样，而且站和走不一样
| 策略 / 状态 | 相机俯角 | 离地高度 | neck_pitch（默认 0.349） |
|---|---|---|---|
| walk_flat（1300）静止零命令 | 16.5° 向下 | 0.229 m | 0.156 |
| walk_flat（1300）vx 0.2 | 2.5° 向下 | 0.243 m | 0.354 |
| walk_flat_2600 静止零命令 | 20.9° 向下 | 0.224 m | 0.130 |
| walk_flat_2600 vx 0.2 | 6.4° 向下 | 0.241 m | 0.268 |

策略把头当作平衡用的质量在动：**站着时低头约 15–20°，走起来抬回来**。所以起步 / 停下时画面俯角会变约 15°。§0.6 里 STAND keyframe 下"俯仰 +0.4°"只是关键帧姿态，策略运行时不是这样。`FrameMsg` 每帧带外参，投影本身不受影响；但这对 VLA 的输入分布和真机标定有影响，列进给 RL 同学的问题。

### 2.4 可见性（默认相机：fovy 45°、附加俯角 0，站立时实际俯角 ≈ 20°）
| 目标 | 像素宽 | 是否完整可见 |
|---|---|---|
| 40×40 cm 旗垫 @ 0.5 m | 320（超出画面） | 不完整（2/4 角点） |
| 旗垫 @ 1 m / 2 m / 3 m | 157 / 69 / 44 px | 完整 |
| 球 @ kick 窗口 (0.09, −0.042) | — | **不可见** |
| 球 @ 0.15 / 0.3 / 0.5 m | — | 不可见（画面下沿对应地面 0.33 m） |
| 球 @ 1 m | 24 px | 可见 |

**俯角 × fovy 扫描**（附加俯角是在当前 site 帧基础上再往下；站立时实际俯角 ≈ 附加值 + 20°）：
- fovy 45°：附加 ≥ 40° 才能看到 kick 窗口的球，但那时 0.5 m 以外的旗垫一个都不完整。看球和看导航目标不能兼顾。
- fovy 60°：附加 30° 起能看到球（31 px，刚进画面），同样看不到完整的旗垫。
- fovy 75°：附加 30°：球 1612 px + 0.5 m 旗垫完整。
- **fovy 90°（竖直，4:3 时水平约 106°）、附加 20–30°**：球 802–2576 px，0.5–2(3) m 的旗垫全部完整可见。这是扫描里唯一同时满足"踢球距离看得到球"和"0.5–2 m 看得到旗"的区域。

**建议（不改默认值）**：真机若用 M12 镜头，选竖直视场 ≥ 90° 的广角，安装时比 `head_camera` site 再向下 20–30°；或者接受踢球阶段看不到球（盲踢策略本来也不看球），只要求在约 0.3 m 外能看到球并对准。确定后改 `camera.yaml` 的 `fovy_deg` / `pitch_down_deg`，`camera_check.py` 重跑。注意广角镜头有畸变，这里是针孔模型。

### 2.5 步态抖动（walk_flat，相机相对重力，20 s，50 Hz 采样）
| vx 命令 | 俯仰 std / 峰峰 / 主频 | 横滚 std / 峰峰 / 主频 | 高度 std / 峰峰 |
|---|---|---|---|
| 0（stand） | 0.07° / 0.38° / — | 0.25° / 1.45° / 1.9 Hz | 0.1 / 0.5 mm |
| 0.1 | 3.31° / 15.9° / 2.30 Hz | 1.94° / 7.6° / 2.30 Hz | 3.6 / 16.2 mm |
| 0.2 | 1.54° / 6.7° / 2.15 Hz | 2.61° / 8.0° / 2.15 Hz | 2.6 / 10.8 mm |
| 0.25 | 1.26° / 6.7° / 2.15 Hz | 2.82° / 8.7° / 2.15 Hz | 2.6 / 11.7 mm |

主频约 2.15–2.3 Hz（= 步频）。vx 0.1 的俯仰抖动最大（峰峰 16°），低速时步态最不稳。这组数要和真机对比，用来设训练增强（旋转 ±3–8°、平移 ±1 cm 量级）。

### 2.6 调了哪些参数、为什么
- 没改相机默认值（都是占位）。`offsamples = 0` 只在 camera_check 的场景里设，正常 RGB 渲染保留抗锯齿。

---

## 第 3 步：模块化场景（2026-10-24，`outputs/scenes/`）

### 3.1 做了什么
- `duckrt/scene/modules.py`：7 种模块（arena / flag / forbidden_zone / obstacle / ramp / ball / goal）的 schema（必填、类型、取值范围），加载时校验，未知参数、缺必填项、越界、未知颜色都会报错并指出是哪个布局的哪个模块。每个模块生成一个 body `mod_<id>`，geom 命名为 `mod_<id>_<部件>`。碰撞按表格要求：旗杆、围挡、障碍、坡道、球门、球碰撞；旗垫、禁区不碰撞（contype = conaffinity = 0）。预留 `texture` 字段（arena / flag / zone / obstacle / ramp），本阶段为空。
- 颜色统一来自 `configs/palette.yaml`（也接受显式 RGBA）。球的半径、质量、摩擦、惯量默认取 kick card（= 训练用的 ball.xml）。
- `assets/modules/*.yaml`：命名参数集（`flag_red`、`ramp_up_down`、`goal_small` ……）。`assets/layouts/*.yaml`：用 60 cm 网格（`cell`，原点在场地西南角，可加 `offset_m`）或精确坐标（`pos`）放模块，带 `yaw_deg` 和逐实例参数覆盖；命令行 `--set ramp.angle_deg=7` 也能覆盖。
- `duckrt/scene/builder.py`：MjSpec 上依次加机器人（allcollisions 几何 + walk 限位）、灯光、arena 地面（`floor` 平面）和模块、相机（`head_cam`、`top_cam`）；构建时检查：模块都在场地内、两两不重叠（有向矩形分离轴判定）、不压机器人起点（12 cm 圆；布局可以用 `on_modules` 声明起点就在某个模块上，比如坡顶平台）。机器人起点高度 = 关键帧躯干高度 + 起点下方静态场景的高度（射线只打 group 0 的场景几何）。
- 特权查询：`SimBackend.module_states()` / `module_state(id)` 返回每个模块的 id、类型、定义名、当前世界位姿（球取自由关节）、尺寸参数、颜色和命名 site（旗中心、坡顶中心、球门线）。
- 布局：`flat_single_flag`（红旗在起点前 1.5 m）、`ramp_test`（坡底在起点前 0.5 m，坡长 0.8 m，平台 0.6 m）、`ramp_test_down`（同一坡道镜像，起点在平台上）、`demo_v1`（3.6 × 2.4 m：起点 → 上坡 / 平台 / 下坡 → 旗 A → 中间禁区 + 障碍 → 旗 B → 球 → 球门；软垫位置 cell (5, 1) 留空并注释）、`kick_test`（空场 + 球）。

### 3.2 检查结果（`scripts/build_scene.py --all`）
5 个布局全部通过：站立 5 s 无 NaN，最大穿透 1.7 mm（就是脚底正常接触），最大自由度速度 1.37 rad/s，不摔，球 5 s 漂移 0.4 mm。每个布局输出俯视图、起点头部相机图、站立 5 s 后的头部相机图。

### 3.3 发现 / 修掉的问题
- **MjSpec 的 `body.ipos` 默认值是 `[nan, 0, 0]`**（"未设置"的标记），只设 `explicitinertial = True` + mass / inertia 时，球的质心被编译到了世界原点，于是球自己滚走（5 s 0.73 m）。必须显式设 `ipos`、`iquat`。
- 俯视图偏亮，把 headlight 漫反射从 0.6 降到 0.4、环境光 0.3 → 0.25，方向光 0.55。

## KICK（第 1 步的 KICK 模式，用第 3 步的球做验证；`scripts/kick_test.py`，`outputs/kick_test/report.md`）
- 流程走 switcher 的真实 KICK 序列：stand（model_2600 零命令）站 0.5 s → kick 播放 → stand 站 0.5 s → 回到原模式。球在 kick 开始播放时按训练方式瞬移到窗口里（trunk 根节点 yaw 系下的偏移）。
- **窗口**（站立起步，每格 4 次；成功 = 球峰值速度 ≥ 0.3 m/s 且方向偏差 ≤ 30°）：x 0.060–0.090 基本 100 %，x 0.105 时 y 两端开始失败，x 0.120 全部失败；y −0.072 到 −0.027 可靠，−0.012 不行。也就是**球要在根节点前 6–10 cm、右脚中线 ±3 cm 内**。训练里只覆盖了 (0.09, −0.042) ± 0.015。
- 窗口中心：球峰值速度 1.14–1.17 m/s（训练目标 1.0），滚出 2.2–2.5 m；**方向比机身朝向偏右 6–8°**（峰值速度方向）。在 kick 训练用的 allcollisions 关节范围下偏右约 −14°～−22°（看位移方向），说明偏差是策略本身的，不是我们的关节限位造成的。
- 踢的时候 hip_yaw 会顶到 walk 模型的限位（余量 0）；换成 kick 训练的 ±0.436 范围也会顶到另一端（余量 0.001），球速 1.30 vs 1.17 m/s，成功率相同。保留 walk 限位。
- **duration_s = 1.1 s**：关节最大速度回到 < 0.5 rad/s 的时刻 p95 = 0.86 s，加 0.2 s 余量。用 1.1 s 重新验证：站立起步 10/10，走 3 s 后直接 KICK 10/10（stand 0.5 s 后机身都已静止），无摔倒。已写回 `configs/policies/kick.yaml` 的 `duration_s` 和 `validated_on`。
- 视频：`kick_centre.mp4`、`kick_after_walk.mp4`。

---

## 第 4 步：坡道测试（2026-10-24，`outputs/ramp_test/report.md`，更陡的补充在 `outputs/ramp_test/extra_steep/`）

### 4.1 设置
- 只用平地步态 walk_flat（model_1300），不训练任何东西。坡道：水平坡长 0.8 m，平台 0.6 m，宽 0.4 m，两侧 4 cm 护栏，摩擦 1.0。上坡从坡底前 0.5 m 起步，到平台中心算成功；下坡从平台起步，越过坡底 0.25 m 算成功。每个角度、每个方向 10 次（起点 ±2 cm / ±3° 扰动，执行器延迟按训练分布随机），vx = 0.6 × 0.25 = 0.15 m/s，另加 1 次 vx 0.25。
- **两个变体**：A = 题目要求的固定前进命令；B = 同样的 vx，加执行器式的 10 Hz 航向 / 中线保持（用真值位姿，ωz 限在 ±0.375）。原因：A 的失败**几乎全是航向漂移把机器人带出 0.4 m 宽的坡道**（摔倒点都在 |y| ≈ 0.21–0.29 m，也就是护栏上），**连 0° 的平坦"坡道"上坡也只有 5/10**，所以 A 测的主要是航向漂移，不是爬坡能力。

### 4.2 结果
| 坡角（坡高） | A 上坡 | A 下坡 | **B 上坡** | **B 下坡** | B 上坡时坡面水平速度 | B 上坡 / 下坡通过时间 |
|---|---|---|---|---|---|---|
| 0°（0） | 5/10 | 9/10 | **10/10** | **10/10** | 0.116 m/s | 14.1 / 11.4 s |
| 3°（4.2 cm） | 9/10 | 1/10 | **10/10** | **10/10** | 0.076 m/s | 17.8 / 9.6 s |
| 5°（7.0 cm） | 8/10 | 1/10 | **10/10** | **10/10** | 0.057 m/s | 21.3 / 8.3 s |
| 7°（9.8 cm） | 9/10 | 9/10 | **10/10** | **10/10** | 0.044 m/s | 25.4 / 7.5 s |
| 上限速度 vx 0.25 那一次 | 0–7° 上坡全失败，下坡只有 7° 成功 | | 0–7° 全成功 | 0–7° 全成功 | | |

补充（只跑了 5 次 + 1 次上限速度，`extra_steep/`）：
- **10° / 12° / 15° 上坡：vx 0.15 时 0/5**，A、B 都一样。不是摔倒，是**卡在坡底**：躯干停在 u ≈ 0.02–0.04 m，原地踏步、没有前进，电流也没饱和。10° 时用 vx 0.25 的那一次（B）能上去；12° 用 0.25 也不行。
- 下坡：10° 5/5，12° 4/5，15° B 4/5、A 1/5（A 摔 5 次）。

其他观察：
- 机身俯仰（`pitch.png`，变体 B）：坡上平均 +0.2°～+1.4°（略低头），步态内 −1°～+3° 波动。下坡 7° 时波动最大（到 +3°）。机身保持接近竖直，没有顺着坡面倾斜。
- 关节力矩：变体 B 峰值髋俯仰 0.25–0.77 N·m（下坡最大）、膝 0.57、踝 0.55–0.61；**电流限幅（0.64 N·m）占比都是 0 %**。峰值超过 0.64 是因为 BAM 的输出力矩里含摩擦项。A 变体里撞护栏、绊倒时会有 1–3 % 的步数在电流限幅上。这一版的 XL330 在 ≤ 7° 时没有过载迹象。
- 上坡越陡越慢：坡面水平速度 0.116 → 0.044 m/s（7° 只剩平地的 38 %），10° 附近降到 0。

### 4.3 给 RL 同学的结论（这份表会直接交过去）
- **≤ 7° 的坡，平地步态够用**：只要上层做航向保持，上下坡 10/10、不摔、不过载；代价是上坡很慢（7° 时 0.044 m/s）。
- **≥ 10° 上坡需要单独处理**：低速命令下卡在坡底；要么训练坡道 / 地形课程，要么让执行器在坡底给更大的 vx（10° 时 0.25 能上，12° 不行）。
- 下坡到 10° 都还好，12–15° 开始摔。
- 仅凭固定命令走窄坡道不可靠，**根源是 walk_flat 的航向漂移**（walk_test 里直走 vx 0.2 时 ωz ≈ +0.13 rad/s），和坡度本身无关。

---

## 第 5 步：遥控查看器（2026-10-24，`scripts/teleop.py`）

### 5.1 做了什么
- 一个 pygame 窗口：主画面是头部相机 320×240 放大 3 倍，右上角小窗是 `top_cam`，底部 5 行 HUD：模式、当前策略、档位、fall_flag、看门狗、发出 / 裁剪后的命令、实际机身速度、位姿、相机俯角、仿真时间、实时倍率，外加按键提示。不用 `mujoco.viewer`。
- 按键用 `pygame.key.get_pressed()` 读状态，可以同时按。W/S、A/D、Q/E 按住生效、松开归零，变化率限制为 0.3 s 从 0 到满速（上层 10 Hz 时每 tick 最多变满速的 1/3）。Shift 切低 / 高档（card `command_range` 的 50 % / 100 %），1 = WALK_FLAT，0 = STAND，J = KICK（单次），R 重置布局，Esc 退出。`--layout`、`--set`（模块参数覆盖）、`--seed`。
- 调度：上层 10 Hz 发 CommandMsg，相机 15 fps（`SimRuntime` 在物理步级别按时刻曝光），策略按 card 的频率；全部在同一个仿真时钟下推进，按墙钟节拍跑；仿真落后墙钟超过 0.1 s 时打印 warning，并重设节拍基准（不会无限追赶）。
- `on_tick(state, frame, command)`：每个控制 tick 调用一次，参数类型就是 `messages.py` 里的 `StateMsg` / `FrameMsg`（这一 tick 内没曝光就是 None）/ `CommandMsg`。现在是空函数，第二阶段的记录器挂在这里。
- 启动直接进 WALK_FLAT（按 W 就走）。走过之后松开按键，walk_flat 会继续踏步（见 1.4），HUD 会提示"按 0 停下"。

### 5.2 自测
- `--smoke 4`（无窗口驱动跑真实节拍的主循环）：墙钟 4.0 s → 仿真 4.00 s，实时倍率 0.98，60 帧头部画面（15 fps）。
- `--autopilot`（无窗口，按人会按的虚拟键：朝旗 A 的方位角 > 5° 就按 Q/E，方位角 < 90° 时一直按 W，低档）：demo_v1 从起点 → 上坡 → 平台 → 下坡 → 旗 A，**9 个 seed 里 8 个到达**（27.9–29.4 s），1 个在下坡段摔倒。截图在 `outputs/teleop/`。
- 试过的失败版本也说明了操作上的问题：
  1. 方位角 > 35° 就停下原地右转：坡顶平台上命令 ωz −0.75，机器人反而向左转并往后滑。walk_flat 的原地右转本来就弱（平地 −0.6 → −0.40），在坡上会失效。
  2. 高档（vx 0.25）：向左的航向漂移（约 +0.2 rad/s）和按住 E 能得到的右转（约 −0.18 rad/s）差不多大，从下坡侧面掉下去。
  → 在 demo_v1 里遥控：**用低档，边走边转**。

### 5.3 没法在这里做的
- 真窗口（X11）里的手动操作需要你来试；窗口代码路径和无头 smoke 是同一套，只是 SDL 驱动不同。
