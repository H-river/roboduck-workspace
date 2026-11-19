# RoboDuck 交接文档（RL 技能训练）

2026-11-20 整理。面向第一次接手的人：照着做能装好环境、启动训练，并且自己判断训练出来的策略合不合格。
过程记录（每一步做了什么、为什么）在 `HANDOFF_NOTES.md`；更早的调研和训练记录在 `data_collect_sim/NOTES.md`（第一阶段）、`rl_train/NOTES.md`（第二阶段）。

## 1. 一句话现状

<!-- STATUS:BEGIN -->
**walk 有一个全部门槛通过的候选（walk v2 r2 s1 it2750），但还没有定版；kick 没有能用的。**

| 策略 | 文件 | 验收（完整模式） | 问题 |
|---|---|---|---|
| walk v1（在用，第一阶段） | `walk_flat_model1300.onnx`（走）+ `walk_flat_model2600.onnx`（站） | 不合格：停住 0/39、直走偏航 0.129 rad/s | 1300 走起来停不下来，只能靠 2600 零命令站住；直走向左偏（vx 0.2 时 +0.13 rad/s）；右转弱于左转 |
| **walk v2 r2 s1 it2750**（第二阶段最好的候选） | `walk_v2_r2_s1_it2750.onnx` | **walk 门槛全部通过**（停住 39/39 最长 0.50 s、偏航 ≤ 0.02、±ωz 比 0.98 / 0.96、5° 坡 10/10、头部接管 20/20、热切换 25/25） | 只在平地训练过；没见过坡道 / 头部接管 / 6–7.5 V 电压 / 40 ms 延迟（这些是 r3 的内容） |
| kick stage 1（AI-FanGe，在用） | `kick_af_stage1_model3999.onnx` | 和 walk v1 的站姿（2600）配：踢球窗口 0/50；**和 walk v2 r2 it2750 配：窗口 52%（26/50）、偏差 +0.6°、交接 20/20** | 盲踢（观测里没有球）；只有站姿接近 HOME 时才出脚；窗口窄 |

下一步要训练（第二阶段的目标）：
- **walk v2**：修停不下来、直走偏航、右转弱，覆盖 5° 坡，容忍站立时的头部接管。r2（平地）已经解决前三项；**r3 = r2 + 坡道 + 头部接管 + 电压 / 延迟随机化**，是定版轮。
- **kick v2**：扩大踢球窗口（x 0.06–0.12、y 脚线 ±0.03 m）、修右偏（横向球速惩罚）、加球位置观测、起始姿态覆盖 walk v2 的站姿。

今晚额外训了 r3 和 kick v2（用户要求"留些能直接用的 ONNX"），结果见下面"今晚的训练"。
<!-- STATUS:END -->

### 今晚的训练

<!-- TRAINING:BEGIN -->
（训练进行中，结果待补）
<!-- TRAINING:END -->

## 2. 10 分钟上手

前提：Linux x86_64，NVIDIA 显卡 + **驱动 ≥ 580**（torch 2.14.1 用的是 CUDA 13 运行时），git，[uv](https://docs.astral.sh/uv/)。
没有 GPU 也能做第 1–3 步（验收只用 CPU），只有训练需要 GPU。

```bash
# 1. clone（含两个子仓库；三个都是私有仓库，需要有权限：gh auth login 后 gh auth setup-git，或者用 token）
git clone --recursive https://github.com/H-river/roboduck-workspace.git
cd roboduck-workspace

# 2. 装环境 + 自检（第一次要下载约 5 GB 的 torch / CUDA 库）
scripts/setup_workspace.sh
```
预期最后几行（数字可能不同）：
```
== self-check
   mjlab: torch 2.14.1+cu130, CUDA available: True (NVIDIA GeForce RTX 4080 Laptop GPU)
   tasks: Mjlab-Velocity-Microduck RoboDuck-kick_v2_r1 RoboDuck-kick_v2_r1_stand2750 RoboDuck-walk_v2_r1 RoboDuck-walk_v2_r2 RoboDuck-walk_v2_r3
   runtime: card walk_v2_r2_s1_it2750, ONNX walk_v2_r2_s1_it2750.onnx loaded
   runtime: stood 2 s and walked 2 s at 0.15 m/s -> x = +0.140 m, upright
== done in ... s.
```

```bash
# 3. 对在用的策略组合跑一次快速验收（约 10–30 s）
cd data_collect_sim
.venv/bin/python scripts/accept.py --walk walk_flat --stand walk_flat_2600 --kick kick --quick --out outputs/acceptance/my_first_quick
```
预期第一行：`# 验收结论：**不合格**（14 项门槛中 5 项不通过：stop, straight_yaw, turn_ratio_inplace, kick_rate, kick_bias）`。
和仓库里的 `outputs/acceptance/baseline_quick/report.md` 应该逐项一致（仿真是确定性的）。完整模式的基线在 `outputs/acceptance/baseline/report.md`。

```bash
# 4. 冒烟训练：2 次迭代 + 导出 ONNX + 快速验收（约 2 分钟，需要 GPU）
cd ..
NUM_ENVS=256 ITERATIONS=2 RUN_NAME=smoke ACCEPT_FINAL=quick rl_train/train.sh walk_v2_r3
```
预期：`training exited with code 0`、`export model_1`、`model_1: # 验收结论：**不合格**…`（两次迭代的策略当然站不住），
ONNX 在 `rl_train/runs/logs/rsl_rl/walk_v2/<日期>_smoke/onnx/model_1.onnx`。

<!-- VERIFY:BEGIN -->
**实测（2026-11-20 01:55，这台机器，从 GitHub 重新 clone 到 `/tmp/roboduck_verify`，shell 里带着 ROS 的 `PYTHONPATH`）**：
完全按上面四步执行（脚本只是把命令逐条跑一遍并计时）：

| 步骤 | 结果 | 耗时 |
|---|---|---|
| `git clone --recursive …` | 主仓库 + 两个子仓库 | 5 s |
| `scripts/setup_workspace.sh` | 两个 venv 装好，自检通过（CUDA available: True，6 个任务，walk 站 2 s + 走 2 s 不倒） | 14 s（uv 缓存是热的；新机器第一次要下载约 5 GB，几分钟到十几分钟） |
| `accept.py … --quick` | 第一行 `不合格（14 项门槛中 5 项不通过…）`，表格和仓库里的 `baseline_quick/report.md` **逐行相同** | 10 s |
| 冒烟训练 2 次迭代 + 导出 + 验收 | `training exited with code 0`，导出 `model_1.onnx`，快速验收出报告（不合格，属于预期） | 105 s |

第一次验证失败过一次：`setup_workspace.sh` 在 git 里没有执行权限（exit 126），修好、推送后重新验证通过。
<!-- VERIFY:END -->

## 3. 怎么训练

### 入口
- 任务都在 `rl_train/`：每个 `rl_train/configs/{walk,kick}_*.yaml` 注册成一个 mjlab 任务 `RoboDuck-<文件名>`（`rl_train/rl_tasks/__init__.py`）。配置可以 `inherit:` 另一个配置，只写差异。
- 改训练内容 = 新建一个 YAML（例如 `walk_v2_r4.yaml`，`inherit: walk_v2_r3.yaml`），**不要改已经训过的配置**（结果就对不上了）。所有可调量都在 YAML 里，`rl_tasks/walk_v2.py`、`kick_v2.py` 只负责接线。
- 现有配置：`walk_v2_r1`（停住 + 对称，转向被改坏，弃用）、`walk_v2_r2`（r1 + 只看偏航的跟踪项，平地）、**`walk_v2_r3`**（r2 + 坡道 + 头部接管 + 电压 6–7.5 V + 延迟 10–40 ms）、`kick_v2_r1`（站姿偏移留空）、**`kick_v2_r1_stand2750`**（站姿偏移 = walk v2 r2 s1 it2750 的实测站姿）。
- 上游任务 `Mjlab-Velocity-Microduck` 只是 walk v2 的基础配置，一般不直接训。

### `rl_train/train.sh`
```bash
rl_train/train.sh <配置名> [其他 mjlab train 参数]
```
训练 + 每 `EXPORT_EVERY` 次迭代导出 ONNX + 对每个导出的 ONNX 跑快速验收 + 训练结束后对最后一个 checkpoint 跑完整验收。参数都是环境变量，都有默认值：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SEED` | 1 | 每个配置至少跑 2 个 seed 再下结论（seed 间差别很大，见"已知的坑"） |
| `NUM_ENVS` | 配置里的 `train.num_envs`（walk 4096，kick 2048） | |
| `ITERATIONS` | 配置里的 `train.max_iterations`（walk 3000，kick 4000） | |
| `RUN_NAME` | `<配置>_s<SEED>` | run 目录 `rl_train/runs/logs/rsl_rl/<experiment>/<日期>_<RUN_NAME>/` |
| `EXPORT_EVERY` | 500 | checkpoint 每 250 次迭代存一个（配置里 `save_interval`） |
| `ACCEPT` / `ACCEPT_FINAL` | quick / full | `none` 关掉（例如没有 EGL 的服务器） |
| `KICK_WALK` | `walk_v2_r2_s1_it2750` | 验收 kick 时负责走和站的 walk card |
| `LOGGER` | tensorboard | `wandb` 需要 `WANDB_API_KEY` |
| `MEM_MAX` | 8G | 训练进程的内存上限（systemd user scope，没有 systemd 时自动跳过） |
| `DETACH` | 0 | 1 = 后台运行，关掉终端也不停 |

输出：`<run>/model_*.pt`、`<run>/onnx/model_*.onnx`、`<run>/acceptance/model_*/report.md`；训练日志 `rl_train/runs/train_<RUN_NAME>.log`；tensorboard：`mjlab_microduck/.venv/bin/tensorboard --logdir rl_train/runs/logs`。

### 本地（RTX 4080 Laptop 12 GB）
```bash
DETACH=1 rl_train/train.sh walk_v2_r3                 # 约 2.2 h
DETACH=1 SEED=2 rl_train/train.sh walk_v2_r3
DETACH=1 rl_train/train.sh kick_v2_r1_stand2750       # 约 1.3 h
tail -f rl_train/runs/walk_v2_r3_s1.driver.log
```
实测吞吐（`RoboDuck-walk_v2_r3`，20 次迭代取后 10 次平均，2026-11-20）：

| num_envs | 每次迭代 | 环境步 / s | 显存峰值（整卡） |
|---:|---:|---:|---:|
| 512 | 1.49 s | 8.3 k | 2.0 GB |
| 1024 | 1.87 s | 13.2 k | 2.9 GB |
| 2048 | 2.14 s | 22.9 k | 4.7 GB |
| 4096 | 2.61 s | 37.6 k | 8.5 GB |

- r3 一次完整训练（4096 env × 3000 次迭代）≈ **2.2 h**；4096 时采样占 2.3 s（坡道的 hfield 碰撞），学习 0.3 s。
- r2（平地）4096 env 1.14 s / 迭代，3000 次 ≈ 1 h。kick v2 2048 env 1.13 s / 迭代，4000 次 ≈ 1.3 h，显存 4.3 GB。
- 12 GB 显存一次只能跑一个 4096 env 的 r3。两个 run 想并行就各用 2048（总吞吐不会更高）。
- 内存：训练约 3–4 GB；每个验收进程约 1.3 GB。这台机器只有 14 GB、没有 swap，第二阶段曾经 OOM 杀掉过桌面程序。`accept.py` 会按可用内存自动减少进程数。

### RunPod（没实测过，按本地经验写的）
- 选 **驱动 ≥ 580（CUDA 13）** 的机器（显卡 ≥ 12 GB：4090 / A5000 / L4 / A6000…），磁盘 ≥ 40 GB，Ubuntu 22.04 / 24.04 模板。驱动不够新时 `torch.cuda.is_available()` 是 False，`setup_workspace.sh` 自检会提示。
- 两个 run 并行：24 GB 卡可以各 4096 env 跑两个 seed（每个 8.5 GB）。
```bash
apt-get update && apt-get install -y git curl libegl1 libgl1 libglib2.0-0 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
# 私有仓库：用一个 fine-grained token（只读这三个仓库）
git config --global url."https://<TOKEN>@github.com/".insteadOf "https://github.com/"
git clone --recursive https://github.com/H-river/roboduck-workspace.git && cd roboduck-workspace
scripts/setup_workspace.sh
DETACH=1 ACCEPT=none ACCEPT_FINAL=none rl_train/train.sh walk_v2_r3     # 验收拿回本地做
# 训完拷回本地（只要 run 目录）：
rsync -av <pod>:roboduck-workspace/rl_train/runs/logs/rsl_rl/walk_v2/<run>/ rl_train/runs/logs/rsl_rl/walk_v2/<run>/
```
- 验收要渲染（相机测试、失败视频），需要 EGL。容器里 EGL 不可用时（`NVIDIA_DRIVER_CAPABILITIES` 不含 graphics），用 `MUJOCO_GL=osmesa`（`apt-get install libosmesa6`，慢）或者把 run 目录拷回本地再验收。
- 用完把 token 删掉（`git config --global --unset url...`）。

## 4. 怎么判断训练结果

**以 `data_collect_sim` 的验收结果为准**（裸 MuJoCo + BAM 执行器，观测和执行器与训练环境逐项对齐，误差 < 1e-6，见 `data_collect_sim/NOTES.md` §1.2）。mjlab 训练日志里的指标只作参考。

```bash
cd data_collect_sim
# 一个 checkpoint（自动导出 ONNX、生成 policy card；任务名从 run 目录推断，推断不了用 --task / --rl-config）
.venv/bin/python scripts/accept.py --walk ../rl_train/runs/logs/rsl_rl/walk_v2/<run>/model_2750.pt
# ONNX（需要训练配置来源：--config <run 目录>；kick 用 --kick-config）
.venv/bin/python scripts/accept.py --walk <x.onnx> --config ../rl_train/runs/logs/rsl_rl/walk_v2/<run>
# walk + 单独的 stand + kick
.venv/bin/python scripts/accept.py --walk walk_v2_r2_s1_it2750 --kick <kick.onnx> --kick-config <kick run 目录>
# 训练中挑 checkpoint：--quick（规模缩小、门槛不变，约 10 s）
# 和基线并排比较
.venv/bin/python scripts/compare.py outputs/acceptance/baseline outputs/acceptance/<name> --out compare.md
```
- 输出 `outputs/acceptance/<name>/`：`report.md`（第一行是总结论，下面每项一行：数值、门槛、✔ / ✘ / 记录）、`report.json`（给脚本读）、`videos/`（只有失败用例，每项最多 3 个，单独重放录的）。退出码 0 合格、1 不合格、2 输入错误。完整模式约 30 s（4–6 个进程）。
- **门槛全部在 `data_collect_sim/configs/acceptance.yaml`**（第二阶段在看任何 v2 结果之前定的，和任务给的门槛表一致，任务没定义的细节在文件注释和 `rl_train/NOTES.md` §1.1）。代码里没有写死的门槛（`tests/test_accept.py` 会检查）。
- **不允许为了通过去改阈值。** 觉得某个指标定义有问题，写进 `rl_train/NOTES.md`，等我确认后再改。
- 定版前：完整模式 + 至少 2 个 seed + 相邻几个 checkpoint 都看（停住时间在相邻 checkpoint 之间会波动）；对候选再跑一次 10 seed 的扩展停住测试（`scripts/stop_test.py --seeds 10`）。新策略每一项都不能比基线差（`compare.py` 会标"⚠ 退步"）。

门槛表（完整模式）：

| 技能 | 指标 | 门槛 |
|---|---|---|
| walk | 20 seed × 60 s 随机命令 | 20/20 不摔 |
| walk | 13 个命令 × 3 seed，切零命令后停住 | 每次 ≤ 0.6 s 停住，之后 2 s 内漂移 < 3 cm |
| walk | vx = 0.2、ωz = 0 时的偏航角速度（5 seed） | 每个 seed \|ωz\| < 0.03 rad/s |
| walk | ±ωz 实际转速之比（原地 ±0.5；边走边转 (0.15, 0, ±0.3)） | 两种都在 0.8–1.25 |
| walk | 命令跟踪误差（12 档）、±vy 比 | 只记录 |
| walk | 5° 坡上下坡（航向保持） | 各 10/10；0° / 3° / 6° 只记录 |
| walk | 站立时头部接管低头 25°、保持 3 s（20 次） | 20/20：倾斜 < 10°、位移 < 5 cm，恢复后起步速度是普通起步的 0.7–1.3 倍 |
| walk | 站立时 2 m 处 40 cm 旗垫在画面内 | 完全可见；站 / 走相机俯角差只记录 |
| walk ↔ stand | 热切换（站 → 走 → 站 → 走，25 次，切换时刻随机） | 25/25 不摔 |
| kick | 球在 x ∈ [0.07, 0.11] m、y 在踢球脚 ±0.03 m（5 × 5 × 2） | 成功率 ≥ 80%（球速 ≥ 0.3 m/s 且方向偏差 ≤ 10°） |
| kick | 踢出方向平均偏差 | \|偏差\| < 3° |
| kick | 低头 25° 测球后再踢 | 成功率 ≥ 直接踢 |
| kick | walk → stand → kick → stand → walk（20 次） | 20/20 不摔 |

## 5. 目录结构

| 路径 | 是什么 | 谁的 |
|---|---|---|
| `HANDOFF.md`、`HANDOFF_NOTES.md`、`README.md` | 交接文档、过程记录 | 我们 |
| `rl_train/` | 第二阶段训练任务：`configs/`（每个 YAML 一个任务）、`rl_tasks/`（walk v2 / kick v2 的接线、新奖励项、平面坡道地形、头部接管动作、镜像增强）、`train.py`（注册任务后调 mjlab 的 train）、`train.sh`、`export_onnx.py`、`tests/`（镜像映射、课程修复检查）、`checks/`（检查结果）、`reference/ai_fange_kick_stage1/`（kick stage 1 的训练配置副本）、`NOTES.md`；`runs/` 不在 git 里 | 我们 |
| `data_collect_sim/` | 验收环境（裸 MuJoCo + BAM，CPU）：`duckrt/`（运行时、观测、执行器、切换器、相机、场景、`evaluation.py` 验收测试、`report.py` 判定和报告）、`scripts/`（`accept.py`、`compare.py`、各单项测试、card 生成、遥控查看器）、`configs/`（`acceptance.yaml` 门槛、`sim.yaml`、`camera.yaml`、`policies/*.yaml` policy card）、`outputs/policies/*.onnx`（在用和候选策略）、`outputs/acceptance/`（验收报告）、`NOTES.md` | 我们 |
| `mjlab_microduck/`（子仓库） | AI-FanGe 的训练包（mjlab 任务 `Mjlab-Velocity-Microduck`、机器人模型 `robot/`、BAM 执行器配置）。私有仓库 `roboduck-workspace-mjlab_microduck` 的 `roboduck-main` 分支 = 上游历史 + 我们的 3 个提交 | 上游（Marc Duclusaud / AI-FanGe），我们有改动 |
| `microduck/`（子仓库） | 真机部署运行时（Pi 上跑 ONNX、IMU、舵机、`constants.py`）。`data_collect_sim` 从这里 import 部署端的常量和 IMU 函数。私有仓库 `roboduck-workspace-microduck` = 上游历史，没有改动 | 上游 |
| `patches/` | 我们对上游的改动：`mjlab_microduck/0001-…sh`（恢复上游删掉的 `robot/`、`README.md`）、`0002-…patch`（`uv.lock`）、`0003-…patch`（今晚的修复）；`check_patches.sh` 从上游干净的 commit 重放补丁并和子仓库逐 tree 比较 | 我们 |
| `scripts/setup_workspace.sh` | 子仓库 + 两个 venv + 自检 | 我们 |

上游：`https://github.com/AI-FanGe/Microduck-build-tutorial` @ `4967821`（2026-09-11，main）。这个 workspace 原本是它的 zip 解压版（没有 git）。根目录的上游文件（教程 README → 改名为 `README.upstream.md`、`docs/`、`LICENSE`、3D 打印文件、IMU 校准脚本）不在我们的仓库里，需要时看上游仓库。

## 6. 已知的坑

**环境**
1. **必须用 uv**（`uv sync --locked`）。pip 装不上：BAM 和 mjlab 对 protobuf 的版本要求冲突，只有 pyproject 里 uv 的 override 能解开。
2. 版本（锁在 `mjlab_microduck/uv.lock`、`data_collect_sim/uv.lock`）：Python 3.13.5，torch 2.14.1（**CUDA 13，驱动 ≥ 580**），mujoco 3.16.0，mujoco-warp 3.8.1，mjlab 1.3.0，warp-lang 1.12.0，rsl-rl-lib 5.0.1，BAM（better-actuator-models 1.0.0）= `rhoban/bam@0007411`（doc 分支）。**和 Pollen `microduck_rl` 的组合（Python 3.12 / torch 2.9.1 / mujoco 3.10.0 / BAM frictionloss 分支）不同**：两个 BAM 分支的执行器模型不一样（摩擦和电流限制的实现不同），所以不要混用两边的 venv / lock / checkpoint。现有策略和 parity 都是在这套版本上训练和验证的。
3. shell 里有 ROS 的 `PYTHONPATH`（`/opt/ros/jazzy/...`）的话，会把 ROS 的包带进 venv，pytest 直接崩。所有脚本已经 `unset` / `env -u PYTHONPATH`；自己跑命令时也加 `env -u PYTHONPATH`。
4. 验收要 EGL 渲染（`MUJOCO_GL=egl`，脚本里已设）。没有 EGL 的机器用 `osmesa`。

**训练**
5. 上游 `num_envs` 默认 1：`rl_train` 的配置里写了 `train.num_envs`；直接训上游任务必须传 `--env.scene.num-envs 4096`。
6. 上游默认用 wandb，没有 `WANDB_API_KEY` 会卡住：今晚已改成 tensorboard（`rl_train` 一直是 tensorboard）。
7. 上游 `microduck_velocity_env_cfg.py` 的 lambda 让后期课程（第 3000 次迭代提高速度和转向范围）永远不生效：今晚已修（`patches/mjlab_microduck/0003`），证据 `rl_train/checks/curriculum_{fixed,buggy}.json`。walk v2 没用课程，不受影响。
8. **坡道配置（r3）在 4096 env 下显存溢出**（2026-10-25 第一次启动 r3 就是这样失败的）：mujoco_warp 遇到 hfield 会把每个 world 的接触 / 约束缓冲放大到 128 / 1024。已在 `walk_v2_r3.yaml` 里设 `sim: {nconmax: 48, njmax: 128}`（实测坡道上每个 world 最多 8 个接触、34 行约束）。新的地形配置要重新量（`HANDOFF_NOTES.md` 第 1 步有测量脚本的做法）。
9. **r3 的摔倒判据**（今晚发现并修好）：上游的 `fell_over` 是躯干**绝对**高度 < 0.10 m；坡道地形的倒金字塔中心在 z < 0，20% 的出生点一出生就被判摔倒（实测 103/512 个环境 25 步内全部终止），r3 第一次训练（00:46–02:00）因此作废（`rl_train/runs/discarded/`）。`walk_v2_r3.yaml` 改成离**正下方地面**的高度（`terminations.fell_over.type: trunk_clearance`，躯干上加了一个测高射线传感器），平地上和原判据等价（差 < 1 mm）。**新的地形配置先跑 `rl_train/tests/check_terrain_termination.py <任务>`**，看出生后有没有成片终止。
10. **seed 之间差别很大**：r2 的 seed 1 有两个 checkpoint 全过，seed 2 一个都没过（停住、5° 上坡）。每个配置至少 2 个 seed。
11. **停住在门槛边缘**：相邻 checkpoint 之间停住时间在 0.4–1.3 s 之间跳（走着突然归零正好在迈步中间，要多走半步）。定版要看多个 checkpoint + 10 seed 扩展测试。
12. **5° 上坡是全有或全无**（0/10 或 10/10）：平地策略到坡上会稳定卡住，r3 的坡道训练就是针对它。
13. 头部在关节限位内最多再低头 28.6°（neck 0.10、head 0.60 rad），所以头部接管训练和测试用 15–28°。
14. 镜像对称增强在 r1 起都开着（`symmetry:`）。镜像算子在 `rl_tasks/mdp_v2.py`，验证 `rl_train/tests/check_mirror.py`；改观测或关节顺序时要同步改。
15. kick v2 的起始站姿偏移（`kick_v2_r1_stand2750.yaml`）是从 walk v2 r2 s1 it2750 实测的；**walk v2 定版后站姿变了，要用 `data_collect_sim/scripts/stand_pose.py` 重新量、训新的一轮。**
16. 第一阶段发现 kick stage 1 的结果偏乐观：之前的测试只站了 1 s，真实执行器要低头测球、站得更久，这时 stage 1 根本不出脚（基线 0/50）。**换成 walk v2 r2 it2750 站的话 stage 1 能踢**（窗口 52%、偏差 +0.6°、交接 20/20），说明问题在站姿离 HOME 太远。
17. **`train.sh` 在跑的时候不要改它**：bash 是边读边执行的，改了文件长度，正在跑的那个会读错位置（今晚差点踩到）。要改就等它跑完，或者复制一份改。

**模型 / 部署**
18. `robot_allcollisions.xml` 把 hip_yaw 限位放宽到 ±0.436，训练用 `robot_walk.xml` 的 ±0.175。验收环境用 allcollisions 的碰撞几何 + walk 的关节限位（`configs/sim.yaml: robot.joint_ranges_from`）。
19. XML 里的 `imu` site 正好是传感器帧的逆；训练和部署都不读它。**不要用 XML 的 imu 传感器模拟 BNO08x**（x、z 会反号；`microduck` 里的 MuJoCoController 就是这么做的）。
20. 部署端 `scale_velocity` 允许的速度是训练范围的 2–4 倍（vx 0.7、ωz 1.5 / 3.0 vs 训练 0.25 / 0.375 / 0.75）。验收环境的切换器按 card 的 `command_range` 裁剪；**上真机前部署端也要裁剪**。
21. ONNX metadata 里的 `default_joint_pos` 只保留 3 位小数，部署端拿它当偏置；card 里是全精度。
22. walk 的头部 4 个关节由策略自己输出，没有头部命令；头部接管是执行器在 STAND 模式下覆盖这 4 个目标（`sim.yaml` 的 `head_override`）。

**验收工具**
23. `--quick` 只用于训练中挑 checkpoint，quick 结论和完整模式可能不同（例如基线的原地转速比 quick 0.80 ✘、完整 0.81 ✔）。
24. 每个验收进程约 1.3 GB 内存（EGL + 模型 + onnxruntime），和训练同时跑时注意内存。
25. "低头后再踢"门槛是"成功率 ≥ 直接踢"：两个都是 0% 时也算通过，要结合踢球窗口那一项看。
26. `eval_candidates.py` 是第二阶段的批量扫描工具（只跑 6 项 walk 测试），新工作用 `accept.py`。

## 7. 待真机确认的参数

| 参数 | 训练里现在用的值 | 真机上怎么测 | 真机值不同时 |
|---|---|---|---|
| **IMU 安装朝向** | `BNO080_MOUNT_QUAT = (0.7071068, 0, 0.7071068, 0)`（绕 y +90°）；观测 = `quat_apply(mount, ·)`，部署端 `imu_quat_to_body(q) = q ⊗ mount`（两边一致，`rl_train/NOTES.md` §0.1） | 机器人放平，跑 `microduck/src/imu_orientation_check.py`：重力观测应是 (1, 0, 0)；前倾 / 左倾时符号要和仿真一致（`data_collect_sim` 里同样姿态的观测可以对照） | 只是安装方向和假设不同：改部署端 `IMU_MOUNT_QUAT` 校准，不用重训。如果传感器帧约定本身不同（轴的对应关系不同）：所有策略（walk、kick）都要重训 |
| **负载下的电压** | BAM `vin_range`：walk v1 / v2 r1–r2 7–8 V，r3 6–7.5 V；验收环境标称 7.5 V（`sim.yaml`）。README 写电池 6 V | 走路和踢球时用万用表或读舵机 `present_input_voltage`（XL330 寄存器 144）记录最低电压 | 低于 7 V：r2 及更早的策略没见过（电流不够会走不动 / 摔）→ 用 r3，或者把 `actuator.vin_range` 改成实测范围再训一轮；验收环境 `sim.yaml` 的标称电压也改成实测值重跑验收 |
| **舵机通信延迟** | 执行器延迟 3–6 物理步（15–30 ms；r3 是 2–8 步 = 10–40 ms）；IMU / 关节速度观测延迟 0–1 步 | 给一个关节阶跃目标，同时 sync_read `present_position`，看响应开始的时间；或者在部署端记录每个 50 Hz 周期的读写耗时 | 超过 30 ms：r2 及更早的没见过 → 用 r3，或者加宽 `actuator.delay_steps` 再训 |
| **头部相机** | `data_collect_sim/configs/camera.yaml`：挂在 `head_camera` site，fovy 45°，分辨率 / 帧率是占位值 | 拍已知距离的棋盘格 / 尺子量视场角和安装俯角 | 不影响策略（策略不看图像）；影响数据采集仿真、2 m 旗垫可见性测试，以及头部接管要低多少头 |

## 8. 授权提醒

- **STL 网格**（`mjlab_microduck/src/mjlab_microduck/robot/microduck/assets/*.stl`，验收环境也加载它们）：来自 Microduck 硬件，按 **CC BY-NC-SA 4.0**（禁止商用、相同方式共享）对待——Pollen 的 `microduck_rl` 在 `LICENSE-MODELS.md` 里明确这样声明，`microduck/LICENSE` 对硬件文件也是 CC BY-NC-SA 4.0。AI-FanGe 仓库没有单独说明这些网格的授权。
- AI-FanGe 教程仓库根目录：**MIT**（Copyright 2026 AI-FanGe）。
- `mjlab_microduck` 代码：**Apache-2.0**（Marc Duclusaud）。
- `microduck` 部署代码：**GPL-3.0**（硬件文件 CC BY-NC-SA 4.0）。GPL 有传染性：分发修改后的部署代码要同样开源。
- 我们的代码（`data_collect_sim/`、`rl_train/`）没有声明授权。
- **仓库保持私有。打包进收费课程之前，要先弄清楚网格（NC）和部署代码（GPL）的授权，必要时联系作者。**

## 9. 今晚改了什么

<!-- CHANGES:BEGIN -->
完整过程和理由在 `HANDOFF_NOTES.md`。

**训练 repo（`mjlab_microduck`，提交在子仓库 `roboduck-main` 分支，补丁在 `patches/mjlab_microduck/`）**
- 恢复上游删掉的 `robot/` 和 `README.md`（10-16 就恢复过；今晚把 README 换回上游原文，占位版不要了）。
- 提交 `uv.lock`（上游 `.gitignore` 忽略了它）；bam 从 `branch = "doc"` 改成按 commit 钉死（lock 内容不变）；加 `.python-version`（3.13）。
- 修课程不生效的 lambda bug；日志默认 tensorboard。

**`rl_train/`**
- `walk_v2.py` 支持配置里的 `sim: {nconmax, njmax}`；`walk_v2_r3.yaml` 设了 48 / 128（修 r3 显存溢出）。
- `walk_v2.py` / `mdp_v2.py` 支持 `terminations.fell_over.type: trunk_clearance`（离正下方地面的高度），`walk_v2_r3.yaml` 用它（修 r3 的倒金字塔一出生就"摔倒"）；`tests/check_terrain_termination.py`。
- 新增 `configs/kick_v2_r1_stand2750.yaml`（kick v2 的起始站姿用 r2 s1 it2750 实测值；原配置没动）。
- 新增 `train.sh`、`tests/check_curriculum.py`、`checks/`、`reference/ai_fange_kick_stage1/`。

**`data_collect_sim/`**
- 新增 `scripts/accept.py`、`scripts/compare.py`、`duckrt/report.py`、`tests/test_accept.py`。
- `duckrt/evaluation.py`：测试可以只跑指定试验（失败视频重放用）、STAND 可以用单独的策略、新增 walk↔stand 热切换和命令跟踪两项；原来写在代码里的测试参数（摔倒高度、kick 流程时长、quick 规模）搬进 `acceptance.yaml`，数值不变。
- `configs/acceptance.yaml`：只**增加**了上面这些参数、`quick:` 段、`videos:`、`runner:`；**已有门槛一个没改**。
- `configs/policy_sources.yaml`：kick 的配置路径指向仓库里的副本；登记 `walk_v2_r2_s1_it2750`。
- `outputs/policies/walk_v2_r2_s1_it2750.onnx` + `configs/policies/walk_v2_r2_s1_it2750.yaml`（从候选里提出来，作为 kick 验收和自检用的 walk）。
- 加 `.python-version`；`.gitignore` 改成只提交报告和命名的 ONNX。

**workspace**
- `mjlab_microduck/`、`microduck/` 原地 `git init`，接上上游真实历史（`git subtree split`），作为子仓库。
- 根目录 `README.md`（上游教程）改名为 `README.upstream.md`，新的 `README.md` 指向本文。
- 新增 `scripts/setup_workspace.sh`、`patches/`、本文、`HANDOFF_NOTES.md`。
- 开工前的备份：`~/backups/Microduck-build-tutorial-main_20261120_pre-handoff.tar.gz`（只在我的电脑上）。

**偏离 prompt 的决定**
- prompt 写"今晚不训练任何正式策略"，但用户后来追加"时间多就多跑几个动作、留些能直接用的 ONNX"，按后者训了 r3 两个 seed 和 kick v2（见"今晚的训练"）。
- 仓库结构：workspace 里原本没有任何 git 仓库，所以没有"嵌套仓库的 origin"可言。按 prompt 的规则，有本地改动 / 没有可单独使用的 remote 的上游目录各建了一个私有仓库：`mjlab_microduck`（有改动）和 `microduck`（没改动，但上游是大仓库里的一个子目录，不能单独做 submodule）。两个私有仓库都保留了上游的真实提交历史。
- 门槛以 `data_collect_sim/configs/acceptance.yaml`（第二阶段定的）为准，和 prompt 的表一致；它多了一项"低头测球后再踢"。
<!-- CHANGES:END -->

## 10. 没提交的文件（需要另外传）

没有 git-lfs，`.pt` 都不在仓库里。下面这些是在用 / 候选策略的 checkpoint（只有重新导出 ONNX、重新生成 card 或者接着训练时才需要；验收只要仓库里的 ONNX 和 card）：

<!-- FILES:BEGIN -->
| 文件（本机路径） | 大小 | sha256 | 用途 |
|---|---:|---|---|
| `mjlab_microduck/checkpoints/walk_stable_5s/model_1300.pt` | 4 720 565 | `99f8139983f175aadb713ab404933456f617c85bd92496bee371799cfe07572d` | walk v1（WALK_FLAT） |
| `mjlab_microduck/logs/rsl_rl/mjlab_microduck_velocity/2026-10-16_17-43-09/model_2600.pt` | 4 720 565 | `4c9b41c844db369e3902c93016d5ac9eae17702efa9284d4869f70f09207cc35` | walk v1（STAND） |
| `rl_train/runs/logs/rsl_rl/walk_v2/2026-10-24_23-32-51_walk_v2_r2_s1/model_2750.pt` | 4 695 797 | `bb288bba9a364c9f03f84c82a7a40d822c0b85ea56e6cbc5214446f369f822b2` | walk v2 候选 |
| `rl_train/runs/logs/rsl_rl/walk_v2/2026-10-24_23-32-51_walk_v2_r2_s1/model_1500.pt` | 4 695 797 | `e80b643a3a4925f25a3ccaef1e5713d1101f2e0deb41c3dda7403ee1a472dc2d` | walk v2 候选 |
| `~/microduck_ws/runs/ai_fange_kick/models/2026-10-06_13-14-13_af_kick_stage1/model_3999.pt` | 4 745 141 | `4c43bcc5b32d0c23a1fda3f7401052d85e1e555cbb1475eae173782b611422a4` | kick stage 1 |
<!-- FILES:END -->

最省事的传法：把整个 run 目录（含 `params/`）打包：`rl_train/runs/logs/rsl_rl/`（约 300 MB，所有 v2 run）、`mjlab_microduck/logs/rsl_rl/mjlab_microduck_velocity/2026-10-16_17-43-09/`、`mjlab_microduck/checkpoints/walk_stable_5s/model_1300.pt`、`~/microduck_ws/runs/ai_fange_kick/models/2026-10-06_13-14-13_af_kick_stage1/`，放回同样的相对路径。

## 11. 仓库

<!-- REPOS:BEGIN -->
| 仓库 | 内容 | 可见性 |
|---|---|---|
| https://github.com/H-river/roboduck-workspace | 主仓库（本文、`rl_train/`、`data_collect_sim/`、`patches/`、`scripts/`），分支 `main` | 私有 |
| https://github.com/H-river/roboduck-workspace-mjlab_microduck | 子仓库 `mjlab_microduck/`，分支 `roboduck-main`（上游历史 + 3 个我们的提交） | 私有 |
| https://github.com/H-river/roboduck-workspace-microduck | 子仓库 `microduck/`，分支 `roboduck-main`（上游历史，没有改动） | 私有 |

- 同学要三个仓库都有权限才能 `clone --recursive`：`gh repo edit` 不能加协作者，用
  `for r in roboduck-workspace roboduck-workspace-mjlab_microduck roboduck-workspace-microduck; do gh api -X PUT repos/H-river/$r/collaborators/<同学的用户名> -f permission=push; done`
  （或者网页上 Settings → Collaborators）。
- 本地两个子仓库的 remote 叫 `roboduck`（不是 `origin`）；从 GitHub clone 下来的子仓库 remote 是 `origin`。都没有指向上游的 remote。
<!-- REPOS:END -->
