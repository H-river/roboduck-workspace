# data_collect_sim — RoboDuck 仿真底座（第一阶段）

裸 MuJoCo（CPU）+ BAM XL330 执行器跑现有的 walk / kick ONNX 策略，观测构造和 mjlab 训练环境逐项一致；
头部相机（内外参、像素 ↔ 地面投影）；YAML 模块拼场景；坡道测试；键盘遥控查看器。
调研结论、每步做了什么 / 发现了什么：`NOTES.md`。

上游目录 `../mjlab_microduck/`、`../microduck/` 只读：部署端的 `constants.py` / `imu_reader.py` 是 import 进来的（不写字节码缓存），
MJCF 在加载时用 MjSpec 修改。

## 安装
```bash
scripts/setup_workspace.sh          # 在 workspace 根目录：子仓库 + 两个 venv + 自检（见 ../HANDOFF.md）
# 或者只装这一个：
cd data_collect_sim && uv sync --locked --group dev      # Python 3.13（.python-version），按 uv.lock 装：mujoco 3.16.0、onnxruntime 1.23.2、bam@0007411…
```
这台机器的 shell 里有 `PYTHONPATH=/opt/ros/jazzy/...`，会把 ROS 的包带进来（pytest 会直接崩）。**所有命令前加 `env -u PYTHONPATH`**，例如
`env -u PYTHONPATH uv run python scripts/walk_test.py`。无窗口渲染默认用 `MUJOCO_GL=egl`（脚本里已设）。

在用和候选策略的 ONNX 在 `outputs/policies/`（仓库里只有这些和验收报告；`outputs/` 其余内容不进 git）。从 checkpoint 重新导出的做法（见 NOTES §0.3）：
```bash
cd outputs/policies   # 先 mkdir -p
../../../mjlab_microduck/.venv/bin/python ../../../mjlab_microduck/src/mjlab_microduck/scripts/export_onnx.py \
  --checkpoint ../../../mjlab_microduck/checkpoints/walk_stable_5s/model_1300.pt --device cpu && mv policy.onnx walk_flat_model1300.onnx
../../../mjlab_microduck/.venv/bin/python ../../../mjlab_microduck/src/mjlab_microduck/scripts/export_onnx.py \
  --checkpoint ../../../mjlab_microduck/logs/rsl_rl/mjlab_microduck_velocity/2026-10-16_17-43-09/model_2600.pt --device cpu && mv policy.onnx walk_flat_model2600.onnx
```
kick 的 ONNX 由 `make_policy_cards.py` 从 `~/microduck_ws/runs/ai_fange_kick/...` 复制过来。

## 脚本（都在 `data_collect_sim/` 下运行，`uv run python scripts/<name>.py`）
| 脚本 | 做什么 | 输出 |
|---|---|---|
| **`accept.py --walk <onnx/pt/card> [--stand] [--kick] [--quick]`** | **一键验收**：跑完 `configs/acceptance.yaml` 的全部测试，输出合格 / 不合格（用法见 ../README.md 第 7 节、../HANDOFF.md 第 3 节） | `outputs/acceptance/<name>/` |
| `compare.py <报告...>` | 多个验收报告并排对比，第一个当基线 | stdout / `--out` |
| `make_policy_cards.py` | 从训练配置（`params/env.yaml`、训练源码、ONNX metadata）生成 `configs/policies/*.yaml`，每个值带出处注释 | `configs/policies/` |
| `check_parity.py` | 和 mjlab 训练环境逐步比较观测 / 单步动力学 / 闭环（自动调用 mjlab venv 跑 `parity_mjlab_rollout.py`） | `outputs/parity/report.json` |
| `walk_test.py [--card walk_flat_2600] [--quick]` | 20 seed × 60 s 随机命令、命令跟踪表、零命令、停稳时间、视频、单核实时倍率；更新 card 的 `validated_on` | `outputs/walk_test/<card>/` |
| `kick_test.py` | KICK 流程（stand → kick → stand），踢球窗口扫描、走后热交接、`duration_s` | `outputs/kick_test/` |
| `camera_check.py` | 投影验证（分割图标记）、可见性 + 俯角 × fovy 扫描、步态抖动 | `outputs/camera_check/` |
| `build_scene.py --layout demo_v1 [--set ramp.angle_deg=7] [--all]` | 构建 + 校验布局，站立 5 s 检查，俯视图 + 起点头部相机图 | `outputs/scenes/` |
| `ramp_test.py [--angles 0 3 5 7] [--runs 10]` | 坡道上下坡（固定命令 / 航向保持两个变体），力矩、俯仰曲线、视频 | `outputs/ramp_test/` |
| `teleop.py --layout demo_v1` | 键盘遥控查看器（见下）；`--autopilot` / `--smoke N` 是无窗口自测 | 窗口 |
| 测试 | `env -u PYTHONPATH uv run --group dev pytest -q`（25 个，约 10 s） | |

### teleop 按键
W/S 前后，A/D 左右平移，Q/E 左右转（按住生效、松开归零、0.3 s 到满速）；Shift 低 / 高档（50 % / 100 %）；
1 = WALK_FLAT，0 = STAND（**走过后要按 0 才会停**），J = KICK，R 重置，Esc 退出。在坡道上用低档、边走边转。

## 配置（所有可调量都在这里）
- `configs/sim.yaml`：机器人模型组合、物理参数、BAM 标称值、延迟、看门狗、摔倒阈值、**哪个 card 担任哪个角色**（`runtime.policies`）。
- `configs/camera.yaml`：头部相机（挂载、俯角、fovy、分辨率、帧率，真机值未知的都标了占位）、俯视相机、测试视频相机。
- `configs/palette.yaml`：模块颜色。`configs/tests.yaml`：各测试参数。`configs/teleop.yaml`：遥控参数。
- `configs/policy_sources.yaml`：每个策略的 ONNX / checkpoint / 训练配置路径。`configs/policies/*.yaml`：生成的 policy card。
- `assets/modules/*.yaml`：模块定义；`assets/layouts/*.yaml`：布局；`assets/textures/`：以后放真实地垫照片。

## 代码结构
```
duckrt/
  messages.py     CommandMsg / StateMsg / FrameMsg（机器人系 x 前 y 左 z 上；m、rad、s）
  policy_card.py  读 card；update_card_fields() 给测试回写字段
  obs.py          按 card 的 obs_layout 构造观测（IMU 项用部署端公式）
  policy.py       OnnxPolicy：观测 → 动作 → 关节目标，reset() 清动作历史
  actuation.py    bam.mjlab.BamActuator 的 numpy 移植 + mjlab 延迟线
  backend_sim.py  MuJoCo 后端：步进（mjlab 顺序）、IMU 合成、真值、渲染（只算运动学的独立 MjData）、模块查询
  switcher.py     模式切换 / 看门狗 / 裁剪 / 摔倒检测 / KICK 交接
  runtime.py      统一仿真时钟：上层 10 Hz、策略按自己的频率、相机 15 fps；on_tick 回调
  camera.py       相机模型编辑、内参、外参、pixel_to_ground / ground_to_pixel
  scene/modules.py  模块 schema、校验、几何、足迹
  scene/builder.py  机器人 + 场景 + 相机组装；布局加载、重叠检查
  upstream.py     从 microduck/src import 部署端常量和 IMU 函数
  config.py, testing.py
```
