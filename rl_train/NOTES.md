# rl_train — NOTES（RoboDuck 第二阶段：RL 技能定版）

按时间追加。路径相对 workspace 根目录；`site-packages/...` 指 `mjlab_microduck/.venv/lib/python3.13/site-packages/...`；
`cfg` = `mjlab_microduck/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`；
`env.yaml` = `mjlab_microduck/checkpoints/walk_stable_5s/params/env.yaml`（训练时落盘的已解析配置）。

---

## 第 0 步：调研（2026-10-24）

### 0.1 IMU 朝向：训练和部署一致，不需要停

第一阶段的两条说法并不矛盾：一条说的是**数值**相同，另一条说的是**语义**相反。逐项看：

| 位置 | 内容 | 含义 |
|---|---|---|
| XML `imu` site（`robot_walk.xml`，trunk_base 下） | `quat="0.7071068 0 0.7071068 0"` | site 帧 = 机身帧 ⊗ q_m，即 R_site = R_b·R_m |
| `IMU_MOUNT_QUAT`（`microduck/src/constants.py:94`，注释 "Rotation from trunk frame (body) to IMU sensor frame"） | (0.7071068, 0, 0.7071068, 0) = 绕 y 轴 +90° | 同上的 q_m |
| 训练观测（`cfg:82, 206-228`；`mjlab_microduck/src/mjlab_microduck/tasks/mdp.py:142-179`） | gyro = `quat_apply(q_m, ω_body)`，gravity = `quat_apply(q_m, −g_body)`（向上的单位向量） | 传感器读数 = R_m·v_body，也就是传感器帧 = 机身 ⊗ conj(q_m) |
| 部署观测（`microduck/src/observer.py:78, 84-85`；`imu_reader.py:69-83`） | gyro = BNO08x 原始 gyro；gravity = `quat_apply_inverse(q_sensor, [0,0,1])`；`imu_quat_to_body(q) = q ⊗ q_m` | `imu_quat_to_body` 说明 q_body = q_sensor ⊗ q_m，即 q_sensor = q_body ⊗ conj(q_m)，和训练一致 |

- **训练 ↔ 部署一致**：传感器帧都等于"机身 ⊗ conj(q_m)"。站立时两边的 gravity 观测都是 (1, 0, 0)，gyro 都等于 R_m·ω_body。第一阶段的 `check_parity.py` 用部署公式（由机身姿态合成传感器四元数）重放 mjlab 状态，观测误差 < 1e-6，这一点已经实测过。
- **XML 的 imu site 是 R_b·R_m，正好是上面传感器帧的逆**，但训练和部署都不读这个 site。它只在"有人拿 XML 的 site 传感器去模拟 BNO08x"时才会出错（microduck 的 MuJoCoController 就是这样做的，x、z 会反号）。
- 真机上剩下要确认的只有一件事：BNO08x 实际的安装方向是否满足 `imu_quat_to_body` 的假设（`microduck/src/imu_orientation_check.py` 就是干这个的）。这属于硬件测量，不是训练 / 部署之间的冲突。

### 0.2 `walk_stable_5s`（2026-10-16_17-43-09 run）完整训练配置

**规模**：num_envs 4096（env.yaml:3），每次迭代 24 步，max_iterations 15000（`agent.yaml`），save_interval 100；实际训到 2600 次迭代（`logs/.../model_2600.pt`），walk_stable_5s 取的是 model_1300。PPO：MLP 512-256-128 ELU，观测归一化，lr 1e-3 adaptive（KL 0.01），entropy 0.01，clip 0.2，γ 0.99，λ 0.95，5 epoch × 4 minibatch（cfg:485-520）。

**仿真 / 终止 / 地形**：dt 0.005 × decimation 4 = 50 Hz（cfg:107-116）；平地 plane（cfg:169-170）；episode 20 s；终止条件：超时、`root_height_below_minimum` 0.10 m（cfg:415-418）、`out_of_terrain_bounds`（env.yaml:1768-1781）。

**奖励**（env.yaml:1296-1767；权重在 cfg:231-326 设置）：

| 项 | 权重 | 参数 | 说明 |
|---|---|---|---|
| track_linear_velocity | 2.0 | std √0.1 = 0.316 | exp(−(‖v_xy−cmd‖² + v_z²)/std²)（`site-packages/mjlab/tasks/velocity/mdp/rewards.py:27-44`） |
| track_angular_velocity | 2.0 | std √0.5 = 0.707 | exp(−((ω_z−cmd)² + ω_xy²)/std²)（:47-64）。**很宽松**：ω_z 误差 0.13 rad/s 只损失 3.3 % |
| upright | 2.0 | std √0.05，pitch 0 | mjlab_microduck `mdp.upright` |
| pose（variable_posture） | 1.0 | 站立 / 行走两组逐关节 std，walking_threshold 0.01 | （:385-466） |
| air_time | 3.0 | 0.125–0.3 s，只在命令 > 0.01 时 | （:209-238） |
| foot_clearance | −2.0 | 目标 2 cm | |
| foot_swing_height | −0.25 | 目标 2 cm | |
| foot_slip | −1.0 | | |
| **no_stepping** | **0.0** | 命令 < 0.01 时数腾空的脚 | **课程到 env step 72000 = 第 3000 次迭代才改成 −1.0**（cfg:290-298, 387-412） |
| body_ang_vel | −0.05 | | |
| angular_momentum | −0.02 | | |
| dof_pos_limits | −1.0 | | |
| action_rate_l2 | −0.1 | | |
| action_target_limits_l2 | −1.0 | overshoot 0.3 | |
| self_collisions | −1.0 | | |
| feet_distance | −1000 | min 5 cm | |

**课程**（cfg:387-412）：第 3000 次迭代时把 vx 放宽到 ±0.35、ωz ±0.75、原地转 ±1.5，air_time 3.0，no_stepping −1.0，standing 10 %。**这个 run 只训到 2600，课程一次都没触发**。而且就算触发了，速度范围也改不到采样器：`command.build = lambda env, _cmd=command: ...`（cfg:330）把 deepcopy 之前的 cfg 对象绑进了默认参数，`env.command_manager.get_term_cfg()` 返回的是 manager 里那份拷贝（`site-packages/mjlab/managers/command_manager.py:259-260`）。第一阶段做 parity 时只能直接写 command term，就是这个原因。

**命令采样**（cfg:329-342；env.yaml:1782-1811；`site-packages/mjlab/tasks/velocity/mdp/velocity_command.py:75-138`）：vx U(±0.25)，vy U(±0.15)，ωz U(±0.375)；10 % standing（全零），10 % 原地转向（|ωz| ∈ [0.5, 0.75]），20 % forward-only（vx = 0.3，vy = ωz = 0）；**每 3–8 s 重采样**（所以 episode 中途"走着切到零"本来就有，约占重采样的 10 %）。heading 环境 0 %。

**域随机化**：见第一阶段 `data_collect_sim/NOTES.md` §0.2 的表（脚底摩擦 0.3–1.2、encoder bias ±0.015、CoM ±5 mm、armature ×0.9–1.1、joint friction ×0.5–2.0（BAM 下无效）、推 ±0.5 m/s / 1–3 s、vin 7–8 V、跌落增益 0–0.2、执行器延迟 3–6 物理步、观测噪声和 0–1 步观测延迟）。

### 0.3 头部关节
14 个关节里第 5–8 个（`OBSERVATION_DOF_ORDER`）是头部：`neck_pitch`、`head_pitch`、`head_yaw`、`head_roll`。和腿一样由策略直接输出目标（target = default + action），没有任何头部命令输入；reward 里只有 pose 项约束它们（std 站立 0.08、行走 0.12，cfg:237-263）。默认值 0.3491、0.3491、0、0 rad；限位 neck_pitch / head_pitch [0.10, 0.60]，head_yaw ±0.349，head_roll ±0.25。
- 相机俯角对这两个关节的灵敏度（运动学实测）：**相机俯角变化 = −Δneck_pitch + Δhead_pitch**（各 1°/1°）。
- **在关节限位内，相对默认姿态最多只能再低头 28.6°**（neck 0.10、head 0.60）。所以任务里写的"再往下 15–35°"超过 28.6° 的部分做不到。头部接管训练和测试用 15–28°，平均分给两个关节（25° = neck −12.5°、head +12.5°）。

### 0.4 kick stage 1 配置（`~/microduck_ws/duck_soccer/training/ai_fange_kick/af_kick_cfg.py`；`.../af_kick_stage1/params/env.yaml`）
- 观测：和 walk 相同的 51 维（mounted gyro / 向上重力 / q / q̇ / 上一动作 / twist 命令，命令训练时保持在 ≈0）；**actor 不看球**，critic 有球的位置和速度。
- 奖励：ball_forward_velocity 12.0（球速在"复位时机器人朝向"上的投影，上限 1 m/s）、ball_speed_overshoot −4.0、support_foot_grounded 2.0、pose_stand_legs 2.0（std 0.5）、pose_stand_neck 1.0（std 0.3）、height_stand 1.0（0.115 m）、upright 2.0、body_ang_vel −0.05、angular_momentum −0.02、dof_pos_limits −1、action_rate −0.1 → 课程到 −1.0、self_collisions −1。**球的横向速度不受惩罚**：方向偏 8° 时，主奖励只损失 1 − cos 8° ≈ 1 %，所以 6–8° 的右偏没人纠正。
- 初始状态：只从站立开始（z 0.11–0.12，倾角 ≤ 5°，关节 HOME ± 0.05），xy ±0.5，yaw 任意。球：根节点 yaw 系下 (0.09, −0.042) ± 0.015，半径 0.035，质量 15 g，摩擦 (0.5, 0.005, 0.0001)。右脚踢。
- episode 5 s；终止：超时、倾角 > 70°、NaN；推（课程到 ±0.3 m/s）、CoM、头部 CoM、armature、质量 / 惯量随机化；2048 env × 4000 迭代。

### 0.5 左右不对称的结构性原因：**结构上基本对称**
- MJCF（`robot_walk.xml`，零位运动学）：五对腿部 body 的质量完全相同（踝差 5e-7 kg），惯量镜像误差 ≤ 5e-10，质心镜像误差 ≤ 2e-6 m。
- 关节轴：五对腿关节在镜像变换下都满足 q_R = −q_L（s = −1），限位也互为镜像（hip_yaw 左 [−0.175, 0.524] / 右 [−0.524, 0.175]）。默认姿态同样满足这个映射。
- 头部：neck 关节锚点在 y = +14.5 mm，`yaw_roll_motion` 质心在 y = −5 mm（48.6 g），两者相互抵消。**整机质心（HOME）y = −0.06 mm**，相对躯干。
- BAM：一组参数用于全部 14 个关节，不分左右。命令采样：vy、ωz 范围对称。奖励：没有左右不对称的项。
- 结论：向左偏航和右转弱**不是结构造成的**，是训练出来的。可以用镜像对称（头部做近似）。镜像算子：腿关节左右互换并乘 −1；neck_pitch、head_pitch 不变，head_yaw、head_roll 取反；actor 的 gyro 观测乘 diag(−1, 1, −1)，向上重力乘 diag(1, −1, 1)（安装旋转是绕 y 轴，和镜像可交换）；命令 (vx, −vy, −ωz)。

### 0.6 mjlab / rsl_rl 支持情况
| 需求 | 支持？ | 接口 / 需要自己加的 |
|---|---|---|
| 镜像对称增强 / 对称损失 | rsl_rl 5.0.1 支持（`rsl_rl/algorithms/ppo.py:58-107, 228-250, 310-350`，`symmetry_cfg = {use_data_augmentation, use_mirror_loss, data_augmentation_func, mirror_loss_coeff}`；`_env` 由 `rsl_rl/extensions/symmetry.py:12-28` 注入） | mjlab 的 `RslRlPpoAlgorithmCfg` 没有这个字段 → 在 rl_train 里继承加一个 `symmetry_cfg`。增强函数自己写（actor 和 critic 两组观测都要镜像）。critic 的 `foot_contact_forces` 是世界系向量，精确镜像需要 yaw → v2 的 critic 换成每只脚的竖直力（只影响 critic） |
| 平面坡道地形 | 部分支持：`HfPyramidSlopedTerrainCfg`（`site-packages/mjlab/terrains/heightfield_terrains.py:194-361`）是 x·y 双线性"帐篷"形，**不是平面**，高度量化 5 mm | 在 rl_train 里写一个平面棱锥高度场子地形（四个平面坡面，正 / 倒两种，竖直量化 0.5 mm），复用 mjlab 的 hfield 输出（`TerrainOutput`） |
| BAM 电压随机化 | 支持：`BamActuatorCfg.vin_range`、`vin_drop_gain_range`、`vin_min`（`site-packages/bam/mjlab.py:55-120`） | 无 |
| 执行器延迟随机化 | 支持：`delay_min_lag` / `delay_max_lag`（每物理步重采样） | 无 |
| 训练中外部覆盖部分关节目标 | 不支持 | 在 rl_train 里继承 `JointPositionAction`：零命令环境里按计划用脚本目标覆盖头部 4 个关节（策略输出的这几维被忽略，`last_action` 观测仍是策略原始输出，和部署端 switcher 的行为一致） |
| 命令采样 | 支持，但有 0.2 节的 lambda bug | v2 用 `UniformVelocityCommandWithRotationCfg`（`mdp.py:79-95`，带自己的 `build`）构造，不用 lambda |

### 0.7 GPU 和吞吐（`rl_train/runs/bench/`，Mjlab-Velocity-Microduck，20 次迭代，后 10 次平均）
RTX 4080 Laptop 12 GB，空闲时占用 0.9 GB。

| num_envs | 每次迭代 | 环境步 / s | 显存峰值 |
|---|---|---|---|
| 1024 | 0.50 s | 48.9 k | 2.7 GB |
| 2048 | 0.61 s | 80.3 k | 4.2 GB |
| 4096 | 0.96 s | 102.9 k | 7.4 GB |
| 6144 | 1.43 s | 103.1 k | 11.1 GB |

4096 时 GPU 已经饱和。按 4096 env 估：3000 次迭代约 48 min，5000 次约 80 min，15000 次约 4 h。坡道地形（hfield）和对称增强（每次更新 batch 翻倍）会更慢，第一个 run 时实测。两个 seed 想并行的话只能各用 2048 env（4.2 GB × 2），总吞吐不会更高。

---

## 第 1 步：扫现有 checkpoint（2026-10-24，`data_collect_sim/outputs/candidates/v1_scan/report.md`）

### 1.1 验收测试的定义（`data_collect_sim/configs/acceptance.yaml`，在看任何 v2 结果之前定下）
任务没写死的定义，这里定下并说明理由：
- **停住**：机身水平速度 < 0.02 m/s 且 |ωz| < 0.1 rad/s，并保持 0.5 s。停住时间 = 从给零命令到这 0.5 s 区间开始的时间。漂移 = 停住时刻起 2 s 内机身 xy 的位移。命令集合：vx 0.1 / 0.2 / 0.25，原地 ωz ±0.5 / ±1.0（切换器会按 card 裁到 ±0.75），组合 (0.2, 0, ±0.3)、(0.15, ±0.1, 0)、(0.25, 0, ±0.3)。每个命令 3 个 seed，先走 3 s。**每一次试验都要过**（≤ 0.6 s 且漂移 < 3 cm）。理由：静止站立时机身晃动约 2–8 mm/s，0.02 m/s 能区分"站着"和"在踏步"。
- **直走偏航**：vx = 0.2 走 2 s 后取 10 s 的平均 ωz，5 个 seed，**每个 seed** 都要 |ωz| < 0.03。
- **±ωz 转速比**：两种工况都要在 0.8–1.25：原地 ±0.5，以及边走边转 (0.15, 0, ±0.3)。各 3 个 seed，比较的是平均实际转速。±vy 只记录。
- **坡道**：执行器式航向保持（第一阶段 ramp_test 的变体 B），5° 上下坡各 10/10 为门槛；0° / 3° / 6° 只记录。
- **头部接管**：站稳 1.5 s → 接管低头 25°、保持 3 s → 恢复后站 1 s → 给 vx 0.2 走 3 s。"不失稳" = 不摔，接管期间到恢复后 1 s 内躯干倾角 < 10°、机身位移 < 5 cm。"起步正常" = 不摔，且后 2 s 平均 vx 是同一策略从普通站立起步的 0.7–1.3 倍（基线在同一次测试里测）。20 次。
- **旗垫可见**：站立 2 s 后，2 m 处 40 × 40 cm 旗垫的四个角都在画面里，且有像素。

### 1.2 结果
`walk_stable_5s` 这个 run 每 200 次迭代一个（0–2600）加上游 `walk.onnx`，共 15 个，**没有一个全部通过**：

| checkpoint | 停住（39 次） | 直走 |ωz| 最大 | ±ωz 比（原地，走） | 5° 坡 上 / 下 | 头部接管 |
|---|---|---|---|---|---|
| it1300 附近（1200 / 1400） | 0/39 | 0.18 / 0.10 | 0.40 / 0.25，0.04 / 11.5 | 8/10 0/10，1/10 3/10 | 8/20，20/20 |
| it1800 | 0/39 | 0.067 | 1.35 / 0.87 | 10/10 10/10 | 20/20 |
| it2200 | 9/39 | 0.151 | 0.17 / 0.93 | 10/10 10/10 | 20/20 |
| it2400 | 30/39 | 0.122 | 1.04 / 4.42 | 10/10 9/10 | 20/20 |
| it2600 | **39/39**（最长 0.60 s） | 0.036 | 4.49 / 0.73 | 6/10 10/10 | 20/20 |
| 上游 walk.onnx | 5/39 | 0.042 | 0.84 / 0.51 | 10/10 10/10 | 20/20 |

- 停住能力是训练后期才慢慢出来的（2400 → 2600），**和 no_stepping 无关**（这个 run 里它的权重一直是 0）。
- **对称性所有 checkpoint 都不过**，±ωz 比在 0.04 到 11 之间乱跳：原地转（训练里只占 10 %，|ωz| 0.5–0.75）一直没学匀。
- 站立和行走时相机俯仰差 9–21°。
- 结论：没有现成可用的，进入第 2 步。

### 1.3 工具 / 踩坑
- `data_collect_sim/scripts/eval_candidates.py`：checkpoint → ONNX（`rl_train/export_onnx.py`，加了 `--task`）→ policy card → 全套 walk 测试 → 对比表。
- **内存**：每个评估进程约 1–1.3 GB（EGL 渲染器 + 网格模型 + onnxruntime）。第一次用 10 个进程，把 14 GB 内存吃满，OOM killer 杀掉了**用户的微信进程（WeChatAppEx）**和一个评估进程；进程池卡死；同时跑的冒烟训练也被杀。改为：进程数按可用内存封顶（每个 1.3 GB，留 2 GB），默认 4 个；换成 `ProcessPoolExecutor`，进程死了会记错误而不是卡住；训练放进 `systemd-run --user --scope -p MemoryMax=...`。

## 第 2 步：walk v2

### 2.0 实现（`rl_train/rl_tasks/`，配置在 `rl_train/configs/walk_v2_*.yaml`）
- `walk_v2.py`：在上游 `make_microduck_velocity_env_cfg()` 的基础上，按 YAML 改命令 / 奖励 / 地形 / 执行器 / 头部接管 / 对称性。每个 `configs/walk_*.yaml` 注册成一个任务 `RoboDuck-<文件名>`；`train.py` 先注册这些任务，再调用 mjlab 自己的 `train`。
- `mdp_v2.py`：`stand_still` 奖励、critic 的 `foot_force_z`、训练日志指标（`Metrics/straight_yaw_rate`、`turn_gain_left/right/ratio`、`standing_speed`、`head_override_fraction`）、平面棱锥地形 `HfPlanarPyramidCfg`、头部接管动作项 `HeadOverrideJointPositionActionCfg`、镜像增强 `mirror_augment`。
- 命令改用 `UniformVelocityCommandWithRotationCfg` 实例（有自己的 `build`），绕开上游 lambda 的问题。
- **镜像映射验证**（`tests/check_mirror.py`）：env 1 放在 env 0 的精确镜像状态（位姿、速度、关节、命令、上一动作），两个都用镜像动作走 5 步，比较 env 1 的观测和 mirror(env 0 的观测)。第 1 步最大误差 4.9e-3，5 步内每一项都在自身幅值的 3 % 以内（后面变大是接触和头部那点不对称造成的）。负对照（故意把 head_yaw 的符号弄反）误差大 16 倍。→ 映射正确。
- 冒烟（256 env × 3 次迭代）：r1、r2 都能训练，指标都写进了日志。

### 2.1 第 1 轮（r1）：停住 + 偏航 / 转向对称（`configs/walk_v2_r1.yaml`）
| 改动 | 针对的缺陷 | 假设 | 预期 |
|---|---|---|---|
| `no_stepping` −1.0 从第 0 次迭代起生效（原来到 3000 才打开） | 停不下来 | 零命令下踏步没有代价 | 零命令时脚不离地 |
| 零命令比例 10 % → 20 %（每次重采样时抽，重采样间隔 3–8 s，所以大多是 episode 中途"走着切到零"） | 停不下来 | 停下的练习太少 | 更快停住 |
| 新增 `stand_still` +1.0：零命令时 exp(−(vx² + vy² + ωz²)/0.05²) | 停不下来 / 漂移 | 跟踪核 std 0.32 太宽，0.1 m/s 和 0 几乎没区别 | 停住后漂移 < 3 cm |
| `track_angular_velocity` std 0.707 → 0.316 | 偏航、转得慢 | 0.13 rad/s 的偏航只损失 3 % 的奖励 | 直走偏航 < 0.03，转速跟上 |
| 镜像数据增强（rsl_rl `symmetry_cfg`，actor 和 critic 都镜像） | 左右不对称 | 结构对称（§0.5），不对称是训练出来的 | ±ωz 比接近 1 |
平地，域随机化和上游相同。4096 env × 3000 次迭代，2 个 seed（1、2）先后跑。

### 2.x 旁注：第一阶段的 kick 结果偏乐观（2026-10-24 写 kick 验收时发现）
- stage-1 盲踢**只有从接近 HOME 的姿态出发才会踢**。stand 策略（model_2600）站 ≥ 1 s 之后，neck_pitch 从 0.35 降到 0.14 rad 以下，其他关节也会偏，这时 kick 策略**根本不出脚**，球速 0。站 0.3 / 0.5 / 0.75 s 时还能踢（球速 1.0–1.2 m/s）。
- 第一阶段的 kick_test 和走后交接测试总共只站了 1.0 s（0.5 s + settle 0.5 s），所以才是 10/10。真实执行器在踢之前要低头测球、站得更久，按那个流程 stage-1 kick 会失败。
- → kick v2 的初始状态必须覆盖 walk v2 的实际站姿：起始关节 = HOME + U(0, 1) ×（walk v2 站姿 − HOME），再加 30 % "头部刚被接管过"的状态。新的验收流程（`duckrt/evaluation.py: kick_suite`）站 1 s 后才发 KICK，再 settle 0.5 s，和执行器的做法一致。

### 2.2 第 1 轮进度（seed 1，4096 env，1.14 s / 迭代）
| 迭代 | 停住（39 次，最长 / 漂移） | 直走 ωz（5 个 seed） | 原地 ±0.5 → 实际 | 走着 ±0.3 → 实际 | ±vy 比 |
|---|---|---|---|---|---|
| 500 | 38/39，0.62 s / 2 mm | 0.002…0.032 | −0.02 / +0.07（还不会转） | 0.04 / −0.02 | 1.07 |
| 750 | 38/39，0.64 s | 0.018…0.081 | 0.15 / −0.10 | 0.14 / −0.06 | 1.01 |
| 1000 | **39/39，0.40 s** / 2 mm | −0.036…0.009 | 0.36 / −0.21（比 1.71） | 0.14 / −0.06（比 2.45） | 1.12 |
- 停住问题在第 500 次迭代就基本解决了（原 run 到 2600 才学会）。
- mjwarp 训练日志里镜像损失 0.0006，左右转增益 0.059 / 0.048（最近 300 次迭代平均），说明策略本身基本对称，只是**转向还没学出来**（增益约 0.05）。MuJoCo C 里看到的左右不对称，是一个很弱、很敏感的转向被小差异放大的结果。等后面的 checkpoint。
- 启动时踩的坑：`num_envs` 没从 YAML 传进 scene（上游 SCENE_CFG 是 1），第一次启动用 1 个 env 跑了 265 次迭代，已丢弃（`runs/discarded/`）。

### 2.3 第 1 轮结论（seed 1 训到 2037 次迭代后手动停掉，seed 2 没跑）
| 迭代 | 停住 | 直走 ωz（5 个 seed） | 原地 ±0.5 → 实际 | 走着 ±0.3 → 实际 |
|---|---|---|---|---|
| 1500 | 35/39，最长 0.82 s | 0.027…0.071 | **−0.078** / +0.118（方向反了） | −0.041 / +0.081 |
| 2000 | 39/39，0.58 s | −0.059…0.057 | 0.073 / −0.070（基本不转） | 0.080 / −0.016 |

- **停住：解决了**（no_stepping 从头开 + 20 % 零命令 + stand_still）。
- **转向：被我的改动弄坏了**。mjlab 的 `track_angular_velocity` 是 exp(−((ωz − cmd)² + ωx² + ωy²)/std²)，**连躯干的横滚 / 俯仰角速度一起罚**。步态里这两项的摆动有 0.5–1 rad/s，std 从 0.707 收到 0.316 之后，这一项几乎恒为 0：训练里它每个 episode 只拿到 0.02–0.03，原 run 是 0.14–0.22。转向没有梯度，所以学不出来。机制很清楚，一个 seed 就能下结论，所以停掉了，没跑 seed 2。
- 下一步（第 2 轮，`configs/walk_v2_r2.yaml`）：保留第 1 轮的停住改动和镜像增强；`track_angular_velocity` 的 std 恢复成 0.707；**新增只看偏航的 `track_yaw_rate`**：exp(−(ωz − cmd)²/0.25²)，权重 1.5；原地转向环境 10 % → 15 %。核算：行走时偏航角速度本身有约 0.4 rad/s 的摆动（MuJoCo 实测），σ = 0.25 时，跟踪正确的期望奖励约 0.40，命令 0.5 却不转约 0.21，梯度明显，同时也会奖励减小偏航摆动。
- 第 3 轮（`configs/walk_v2_r3.yaml`）= 第 2 轮 + 鲁棒性一组（坡道、头部接管、电压 / 延迟随机化），只在第 2 轮转向和停住都过了之后再跑。

### 2.4 第 2 轮进度（seed 1）
| 迭代 | 停住（39 次） | 直走 ωz（5 个 seed） | 原地 ±0.5 → 实际（比） | 走着 ±0.3 → 实际（比） |
|---|---|---|---|---|
| 500 | 38/39，0.72 s | 全部 ≈ −0.002 ✔ | 0.002 / −0.006（还不转） | −0.003 / 0.002 |
| 1000 | 33/39，最长 1.28 s | −0.012…0.000 ✔ | **0.294 / −0.297（0.99 ✔）** | **0.367 / −0.380（0.97 ✔）** |
- 转向修好了：训练里 `track_yaw_rate` 每个 episode 0.25–0.36，原 `track_angular_velocity`（std 0.707）0.34–0.56，mjwarp 左右增益 0.80 / 0.83，MuJoCo C 里比值 0.97–0.99，直走漂移 ≤ 0.012 rad/s。
- 新问题：**从原地转 ±0.5 停下要 0.94–1.28 s**（6 次失败都是这一类）。原因是回摆：命令归零之后机器人往反方向转，−0.5 rad/s 持续约 1 s（+0.5 → 0：+0.18、−0.19、−0.34、−0.52、−0.49、−0.41、…、−0.05）。从 ±0.75 停就没有这个问题，0.3 s 内归零。迭代 1000 时策略还在快速变化，等 2000 / 3000 再看；如果到 3000 还这样，第 3 轮针对它改（原地转之后要求零角速度的奖励 / 命令过渡）。
| 1500 | **39/39，0.40 s** / 2 mm | −0.003…0.009 ✔ | 0.630 / −0.588（1.07 ✔） | 0.325 / −0.302（1.08 ✔） |
| 2000 | 37/39，最长 0.80 s（vx 0.1 / 0.2 → 0，seed 2） | −0.009…0.007 ✔ | 0.543 / −0.529（1.03 ✔） | 0.344 / −0.335（1.03 ✔） |
- 迭代 1000 的回摆到 1500 自己消失了，不需要专门改。
- **全套 walk 验收（`data_collect_sim/outputs/candidates/r2_full/`）**：
  - it1500：**全部通过**。随机 20/20，停住 39/39（最长 0.40 s），对称性 ✔，5° 上下坡 10/10、10/10（0° / 3° 也是 10/10；6° 上坡 0/10、下坡 10/10，只记录），头部接管 20/20（最大倾斜 4.4°，位移 < 1 mm，恢复后起步速度比 0.98–1.00），2 m 旗垫可见。站立相机俯角 23.2°，行走 11.6°，差 +11.6°（只记录）。
  - it2000：除停住（37/39）外全部通过；6° 上坡 10/10；站 / 走相机俯仰差只有 +4.4°。
  - 说明停住时间在相邻 checkpoint 之间有波动，定版时要看多个 checkpoint 和两个 seed，不能只看一个点。
- **第 2 轮还不能直接定版**：任务要求 walk v2 训练里包含平面坡道、站立时头部接管和电压 / 延迟随机化（任务 §2.1），第 2 轮没有这些。第 2 轮的平地策略在 MuJoCo C 里能过 5° 坡和头部接管测试，是"顺带"通过的；BAM 7–8 V、15–30 ms 延迟之外的真机条件它没见过。
- 下一步：第 2 轮 seed 2 照常跑完（每种配置至少 2 个 seed）；第 3 轮 = 第 2 轮 + 鲁棒性一组，已排在它后面。定版在第 2 轮和第 3 轮所有 checkpoint 里按验收表选，同等情况下优先第 3 轮（见过坡道 / 接管 / 随机化）。
- 小事故：排队第 3 轮时 `pgrep` 匹配到了多个 PID，导致第 3 轮和第 2 轮同时启动；第 3 轮在 GPU 上分配显存失败（12 GB 卡），在建 run 目录之前就退出了，第 2 轮没受影响（迭代数照常增长），日志移到 `runs/discarded/`。现在用精确的 launcher PID 排队。

### 2.5 第 2 轮 seed 1 全部 checkpoint 的完整验收（`data_collect_sim/outputs/candidates/r2_full/report.md`）
| 迭代 | 随机 | 停住（39） | 直走 max\|ωz\| | ±ωz 比（原地 / 走） | 5° 上 / 下 | 6° 上 | 头部接管 | 站−走相机俯仰 | 全过 |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 20/20 | 33/39（1.28 s） | 0.012 | 0.99 / 0.97 | **0/10** / 10 | 0 | 20/20 | +11.3° | |
| 1250 | 20/20 | 37/39（0.70 s） | 0.015 | 1.08 / 1.05 | **0/10** / 10 | 0 | 20/20 | +11.2° | |
| 1500 | 20/20 | 39/39（0.40 s） | 0.009 | 1.07 / 1.08 | 10 / 10 | 0 | 20/20 | +11.6° | ✔ |
| 1750 | 20/20 | 36/39（0.82 s） | 0.014 | 1.03 / 0.97 | 10 / 10 | 4 | 20/20 | +9.5° | |
| 2000 | 20/20 | 37/39（0.80 s） | 0.009 | 1.03 / 1.03 | 10 / 10 | 10 | 20/20 | +4.4° | |
| 2250 | 20/20 | 37/39（0.82 s） | 0.014 | 0.99 / 0.94 | 10 / 10 | 0 | 20/20 | +17.6° | |
| 2500 | 20/20 | 37/39（0.78 s） | 0.023 | 1.00 / 0.98 | 10 / 10 | 0 | 20/20 | +9.5° | |
| 2750 | 20/20 | 39/39（0.50 s） | 0.018 | 0.98 / 0.96 | 10 / 10 | 0 | 20/20 | +10.6° | ✔ |
| 2999 | 20/20 | 35/39（1.22 s） | 0.014 | 1.01 / 0.97 | **0/10** / 10 | 0 | 20/20 | +12.2° | |

- 随机、对称、头部接管、旗垫：迭代 1000 之后的所有 checkpoint 都过。
- **停住处在门槛边缘**：大部分 checkpoint 有 1–4 次在 0.62–0.86 s 停住，规律是走着突然归零时刚好在迈步中间，要多走半步。
- **5° 上坡是全有或全无**（0/10 或 10/10）：平地训练的策略到了坡上会稳定地卡住，属于确定性行为，不是噪声。第 3 轮的坡道训练就是针对它的。
- **扩展停住测试（10 个 seed × 13 个命令 = 130 次，只作稳健性参考，门槛仍然按 acceptance.yaml 的 3 个 seed）**：it2750 **130/130，最长 0.50 s**；it1500 128/130（最长 0.98 s）；it2000 125/130（0.84 s）。所以 it1500 的 39/39 有运气成分，it2750 才是有余量的。定版时同样会对候选做 10-seed 扩展测试，在报告里写明。
- 第 2 轮目前最好的是 **s1 it2750**：全部门槛通过，停住有余量。
