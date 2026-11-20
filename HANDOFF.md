# RoboDuck 参考手册

第一次接手请先读 [README.md](README.md)。这份是查资料用的，不用从头读到尾。

| 节 | 内容 |
|---|---|
| [1](#1-各个模型的详细情况) | 各个模型的详细情况 |
| [2](#2-训练参考) | 训练：配置列表、`train.sh` 参数、训练速度、云 GPU |
| [3](#3-验收参考) | 验收：命令、报告格式、完整门槛表 |
| [4](#4-已知的坑) | 已知的坑（29 条） |
| [5](#5-上真机前要测的参数) | 上真机前要测的参数 |
| [6](#6-授权) | 授权 |
| [7](#7-交接那晚改了什么) | 交接那晚改了什么 |
| [8](#8-不在仓库里的文件) | 不在仓库里的文件（`.pt`） |
| [9](#9-仓库和权限) | 仓库和权限 |

各次训练的详细经过见 [TRAINING_LOG.md](TRAINING_LOG.md)。

---

## 1. 各个模型的详细情况

ONNX 都在 `data_collect_sim/outputs/policies/`，对应的 card 在 `data_collect_sim/configs/policies/`，文件名相同。

### 推荐使用

**`walk_v2_r2_s1_it2750`：走路，完整验收走路项全过**
- 成绩：停住 39/39（最长 0.50 s）；直走偏航 0.018 rad/s；左右转速比 0.98 / 0.96；5° 坡上下各 10/10；头部接管 20/20；走停切换 25/25。
- 问题：
  - 只在平地训练过，训练时没见过坡道、头部接管、6–7.5 V 电压、40 ms 延迟。不过在验收环境里把条件改成 6 V + 40 ms，照样能走。
  - **停住对电压和延迟敏感**：改成 6.75 V、延迟 2–8 步时只有 35/39，最长 1.28 s 才停住。
  - 6° 上坡 0/10（只记录，不算门槛；旧的 walk v1 是 10/10）。

**`kick_v2_r1_stand2750_s1_it3999`：踢球，配上面的 walk，完整验收踢球项全过**
- 成绩：踢中 50/50；平均方向偏差 −1.0°；低头看球后再踢 100%；走 → 停 → 踢 → 停 → 走 20/20，20 次都踢中。同一轮的 it3000–3750 也全过。
- 问题：
  - **输入是 53 维**，比走路多了球的位置，部署端还不支持（第 4 节第 19 条）。
  - 球的参数（15 g、半径 3.5 cm、摩擦）是假设的，真机的球还没定。
  - **换种子后方向偏差不稳**：种子 2 只有 2 个 checkpoint 合格（见训练记录）。

推荐组合的验收报告：`data_collect_sim/outputs/acceptance/recommended/`。

### 不合格，留着做对比

| 模型 | 验收 | 问题 |
|---|---|---|
| walk v1：`walk_flat`（第 1300 次迭代，负责走）+ `walk_flat_2600`（第 2600 次迭代，负责站） | 不合格：停住 0/39，直走偏航 0.129 rad/s | 走起来停不下来；直走往左偏；右转比左转弱 |
| `walk_v2_r3warm_s1_it3000_candidate`（r3 在 r2 上接着训） | 不合格：只有停住不过（31/39，最长 1.18 s），其余 13 项全过 | 鲁棒性训练让"走着归零时停住"变差；站立时头更低。只用于以后和真机对比 |
| kick stage 1（`kick`，AI-FanGe 原版） | 配 walk v1 的站姿：踢中 0/50；配 r2 it2750 的站姿：52%，偏差 +0.6° | 盲踢，输入里没有球（51 维）；只有站姿很接近初始姿态时才会出脚 |

### 下一步

- **walk r3**（加坡道、头部接管、电压 / 延迟随机化）还没成功：从零训 3000 次迭代学成了站着不动；在 r2 上接着训能走，但停不住。建议给坡道加地形课程（第 4 节第 10 条），或者接着训时加大停住相关奖励的权重。
- **kick v2**：部署端要支持 53 维输入；确定真机球的参数；种子 2 的方向偏差复现不了（系统性偏 −2° 到 −4.5°），定版前多跑几个种子，或者查偏差从哪来。
- **真机验证**：先测第 5 节的参数。

---

## 2. 训练参考

### 训练任务和配置

- 每个 `rl_train/configs/{walk,kick}_*.yaml` 会注册成一个 mjlab 任务，名字是 `RoboDuck-<文件名>`（注册代码在 `rl_train/rl_tasks/__init__.py`）。
- 配置可以用 `inherit:` 继承另一个配置，只写不同的部分。
- 所有可调的东西都在 YAML 里。`rl_tasks/walk_v2.py`、`kick_v2.py` 只负责把配置接到 mjlab 上。
- **要改训练内容就新建一个 YAML，不要改已经训过的**，否则以前的结果就对不上配置了。

现有配置：

| 配置 | 内容 | 状态 |
|---|---|---|
| `walk_v2_r1` | 加了"停住"和"左右对称" | 转向被改坏了，弃用 |
| `walk_v2_r2` | r1 + 只看偏航的跟踪奖励，平地 | **推荐的 walk 出自这里** |
| `walk_v2_r3` | r2 + 坡道 + 头部接管 + 电压 6–7.5 V + 延迟 10–40 ms | 还没训成功 |
| `kick_v2_r1` | 踢球，起始站姿的偏移留空 | |
| `kick_v2_r1_stand2750` | 同上，起始站姿用 walk r2 s1 it2750 实测的站姿 | **推荐的 kick 出自这里** |

上游任务 `Mjlab-Velocity-Microduck` 只是 walk v2 的底子，一般不直接训。

### `rl_train/train.sh`

```bash
rl_train/train.sh <配置名> [其他 mjlab train 参数]
```

它会依次做：训练 → 每隔 `EXPORT_EVERY` 次迭代导出 ONNX → 对每个导出的 ONNX 跑快速验收 → 训练结束后对最后一个模型跑完整验收。

参数用环境变量传，都有默认值：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SEED` | 1 | 每个配置至少训 2 个种子再下结论 |
| `NUM_ENVS` | 配置里的 `train.num_envs`（walk 4096，kick 2048） | 同时模拟多少只鸭子 |
| `ITERATIONS` | 配置里的 `train.max_iterations`（walk 3000，kick 4000） | |
| `RUN_NAME` | `<配置名>_s<SEED>` | 结果放在 `rl_train/runs/logs/rsl_rl/<walk_v2 或 kick_v2>/<日期>_<RUN_NAME>/` |
| `EXPORT_EVERY` | 500 | 每隔多少次迭代导出 ONNX 并验收。模型文件每 250 次迭代存一个（配置里的 `save_interval`） |
| `ACCEPT` / `ACCEPT_FINAL` | quick / full | 过程中 / 结束时的验收方式；`none` 表示不验收（比如服务器上没有 EGL） |
| `KICK_WALK` | `walk_v2_r2_s1_it2750` | 验收踢球时，负责走和站的 walk 模型 |
| `LOGGER` | tensorboard | 改成 `wandb` 需要 `WANDB_API_KEY` |
| `MEM_MAX` | 8G | 训练进程的内存上限（没有 systemd 时自动跳过） |
| `DETACH` | 0 | 设成 1 就在后台跑，关掉终端也不停 |

输出：
- 模型：`<run 目录>/model_*.pt`
- ONNX：`<run 目录>/onnx/model_*.onnx`
- 验收报告：`<run 目录>/acceptance/model_*/report.md`
- 训练日志：`rl_train/runs/train_<RUN_NAME>.log`
- 看曲线：`mjlab_microduck/.venv/bin/tensorboard --logdir rl_train/runs/logs`

**在已有模型上接着训：**

```bash
RUN_NAME=<名字> ITERATIONS=1500 rl_train/train.sh walk_v2_r3 \
  --agent.resume True --agent.load-run 2026-10-24_23-32-51_walk_v2_r2_s1 --agent.load-checkpoint model_2750.pt
```

被接着训的 run 目录必须在同一个 experiment 下（walk 都在 `rl_train/runs/logs/rsl_rl/walk_v2/`），迭代编号会从 2750 往后接。

### 本机训练（RTX 4080 笔记本，12 GB 显存）

```bash
DETACH=1 rl_train/train.sh walk_v2_r3                 # 约 2.2 小时
DETACH=1 SEED=2 rl_train/train.sh walk_v2_r3
DETACH=1 rl_train/train.sh kick_v2_r1_stand2750       # 约 1.3 小时
tail -f rl_train/runs/walk_v2_r3_s1.driver.log
```

实测速度（`walk_v2_r3`，2026-11-20）：

| 同时模拟的鸭子数 | 每次迭代 | 每秒环境步数 | 显存 |
|---:|---:|---:|---:|
| 512 | 1.49 s | 8.3 k | 2.0 GB |
| 1024 | 1.87 s | 13.2 k | 2.9 GB |
| 2048 | 2.14 s | 22.9 k | 4.7 GB |
| 4096 | 2.61 s | 37.6 k | 8.5 GB |

- r3 完整训一次（4096 × 3000 次迭代）约 **2.2 小时**。慢主要慢在坡道地形的碰撞计算。
- r2（平地）每次迭代 1.14 s，3000 次约 1 小时。kick 每次迭代 1.13 s，4000 次约 1.3 小时，显存 4.3 GB。
- 12 GB 显存一次只能跑一个 4096 的 r3。想同时跑两个就各用 2048，但总速度不会更快。
- 内存：训练约 3–4 GB，每个验收进程约 1.3 GB。这台电脑只有 14 GB 内存、没有 swap，以前内存爆掉时把桌面程序杀掉过。`accept.py` 会按剩余内存自动减少进程数。

### 云 GPU（RunPod，没实际试过，按本机经验写的）

- 选**驱动 ≥ 580**的机器，显卡显存 ≥ 12 GB（4090、A5000、L4、A6000 等），磁盘 ≥ 40 GB，系统 Ubuntu 22.04 或 24.04。驱动不够新时 CUDA 用不了，`setup_workspace.sh` 的自检会提示。
- 24 GB 显卡可以同时跑两个种子，每个 4096。

```bash
apt-get update && apt-get install -y git curl libegl1 libgl1 libglib2.0-0 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
# 私有仓库：用一个只能读这三个仓库的 fine-grained token
git config --global url."https://<TOKEN>@github.com/".insteadOf "https://github.com/"
git clone --recursive https://github.com/H-river/roboduck-workspace.git && cd roboduck-workspace
scripts/setup_workspace.sh
DETACH=1 ACCEPT=none ACCEPT_FINAL=none rl_train/train.sh walk_v2_r3     # 验收拿回本机做
# 训完只把 run 目录拷回本机：
rsync -av <pod>:roboduck-workspace/rl_train/runs/logs/rsl_rl/walk_v2/<run>/ rl_train/runs/logs/rsl_rl/walk_v2/<run>/
```

- 验收要渲染画面，需要 EGL。容器里没有 EGL 时（`NVIDIA_DRIVER_CAPABILITIES` 里没有 graphics），改用 `MUJOCO_GL=osmesa`（先 `apt-get install libosmesa6`，会慢很多），或者把 run 目录拷回本机再验收。
- 用完删掉 token：`git config --global --unset url...`。

---

## 3. 验收参考

**判断模型好坏，以 `data_collect_sim` 的验收结果为准。**它的观测和执行器和训练环境逐项对齐过，误差小于 1e-6（见 `data_collect_sim/NOTES.md` §1.2）。mjlab 训练日志里的数字只作参考。

```bash
cd data_collect_sim

# 验收一个 .pt：自动导出 ONNX、生成 card；任务名从 run 目录推断，推断不出来就加 --task 或 --rl-config
.venv/bin/python scripts/accept.py --walk ../rl_train/runs/logs/rsl_rl/walk_v2/<run>/model_2750.pt

# 验收一个 ONNX：要告诉它训练配置在哪（--config <run 目录>；kick 用 --kick-config）
.venv/bin/python scripts/accept.py --walk <x.onnx> --config ../rl_train/runs/logs/rsl_rl/walk_v2/<run>

# walk + 踢球
.venv/bin/python scripts/accept.py --walk walk_v2_r2_s1_it2750 --kick <kick.onnx> --kick-config <kick 的 run 目录>

# 训练中挑模型：加 --quick（规模缩小、门槛不变，约 10 秒）

# 和基线对比
.venv/bin/python scripts/compare.py outputs/acceptance/baseline outputs/acceptance/<名字> --out compare.md
```

输出在 `outputs/acceptance/<名字>/`：
- `report.md`：第一行是总结论，下面每项一行：实测值、门槛、✔ / ✘ / 只记录；
- `report.json`：给脚本读的；
- `videos/`：只录不合格的用例，每项最多 3 个。

退出码：0 合格，1 不合格，2 输入有误。完整验收约 30 秒（4–6 个进程）。

**规则：**
- 门槛全部在 `data_collect_sim/configs/acceptance.yaml`。它是在看到任何 v2 结果之前定的。代码里没有写死的门槛，`tests/test_accept.py` 会检查这一点。
- **不准为了通过去改门槛。**觉得哪个指标定义有问题，写进 `rl_train/NOTES.md`，商量好了再改。
- 定版前：跑完整验收，至少 2 个种子，相邻几个 checkpoint 都要看（停住时间在相邻 checkpoint 之间会跳）。再对候选跑一次 10 个种子的停住测试：`scripts/stop_test.py --seeds 10`。
- 新模型每一项都不能比基线差，`compare.py` 会把退步的项标成"⚠ 退步"。

### 完整门槛表

| 动作 | 测什么 | 门槛 |
|---|---|---|
| 走路 | 20 个种子 × 60 秒随机速度命令 | 20/20 不摔 |
| 走路 | 13 种速度命令 × 3 个种子，走着切到零命令 | 每次 0.6 s 内停住，之后 2 s 内漂移 < 3 cm |
| 走路 | vx = 0.2、ωz = 0 时的偏航角速度（5 个种子） | 每个种子 \|ωz\| < 0.03 rad/s |
| 走路 | 左右转速之比：原地转 ±0.5；边走边转 (0.15, 0, ±0.3) | 两种都在 0.8–1.25 之间 |
| 走路 | 命令跟踪误差（12 档）、左右侧移速度之比 | 只记录 |
| 走路 | 5° 坡上坡、下坡（开航向保持） | 各 10/10；0°、3°、6° 只记录 |
| 走路 | 站立时把头强制压低 25°，保持 3 秒，做 20 次 | 20/20：倾斜 < 10°、位移 < 5 cm；恢复后起步速度是正常起步的 0.7–1.3 倍 |
| 走路 | 站立时，2 m 外 40 cm 的旗垫在相机画面里 | 完全可见；站和走时相机俯角的差只记录 |
| 走路 ↔ 站立 | 站 → 走 → 站 → 走，25 次，切换时刻随机 | 25/25 不摔 |
| 踢球 | 球在 x ∈ [0.07, 0.11] m、y 在踢球脚 ±0.03 m 内，5 × 5 × 2 个位置 | 成功率 ≥ 80%（球速 ≥ 0.3 m/s 且方向偏差 ≤ 10°） |
| 踢球 | 踢出方向的平均偏差 | \|偏差\| < 3° |
| 踢球 | 低头 25° 看球之后再踢 | 成功率不低于直接踢 |
| 踢球 | 走 → 停 → 踢 → 停 → 走，20 次 | 20/20 不摔 |

---

## 4. 已知的坑

### 环境

1. **必须用 uv**（`uv sync --locked`）。pip 装不上：BAM 和 mjlab 要求的 protobuf 版本冲突，只有 pyproject 里写给 uv 的 override 能解决。
2. **版本是锁死的**（`mjlab_microduck/uv.lock`、`data_collect_sim/uv.lock`）：Python 3.13.5、torch 2.14.1（**CUDA 13，驱动 ≥ 580**）、mujoco 3.16.0、mujoco-warp 3.8.1、mjlab 1.3.0、warp-lang 1.12.0、rsl-rl-lib 5.0.1、BAM 1.0.0（`rhoban/bam@0007411`，doc 分支）。
   网上 Pollen 版 `microduck_rl` 用的是另一套版本（Python 3.12、torch 2.9.1、mujoco 3.10.0、BAM frictionloss 分支），两边的舵机模型不一样，**不要混用两边的环境、lock 文件和模型**。
3. **shell 里有 ROS 的 `PYTHONPATH`**（`/opt/ros/jazzy/...`）时，ROS 的包会混进环境，pytest 直接崩。仓库里的脚本已经处理了；自己跑命令时前面加 `env -u PYTHONPATH`。
4. **验收要用 EGL 渲染**（`MUJOCO_GL=egl`，脚本里已经设好）。没有 EGL 的机器用 `osmesa`。

### 训练

5. **上游任务默认只模拟 1 只鸭子**。`rl_train` 的配置里已经写了数量；直接训上游任务时必须加 `--env.scene.num-envs 4096`。
6. **上游默认用 wandb**，没有 `WANDB_API_KEY` 会卡住。已经改成默认用 tensorboard。
7. **上游的课程学习不生效**：`microduck_velocity_env_cfg.py` 里有个 lambda 的写法问题，导致第 3000 次迭代时提高速度和转向范围这一步从来没执行过。已修（`patches/mjlab_microduck/0003`），证据在 `rl_train/checks/curriculum_{fixed,buggy}.json`。walk v2 本来就没用课程，不受影响。
8. **r3 用 4096 个环境会显存溢出**（2026-10-25 第一次启动 r3 就是这样失败的）。原因是坡道地形会让 mujoco_warp 把每个环境的接触和约束缓冲放大到 128 / 1024。已经在 `walk_v2_r3.yaml` 里设成 `sim: {nconmax: 48, njmax: 128}`，实测坡上每个环境最多 8 个接触、34 行约束。**换新地形要重新量**，做法见 `HANDOFF_NOTES.md` 第 1 步。
9. **r3 的"摔倒"判断出过错**（已修）：上游判断摔倒的条件是"躯干离 z = 0 的高度 < 0.10 m"，可坡道地形有一部分低于 z = 0，20% 的出生点一出生就被判摔倒（实测 512 个环境里 103 个在 25 步内全部终止），r3 第一次训练因此作废（放在 `rl_train/runs/discarded/`）。现在改成"躯干离正下方地面的高度"（`terminations.fell_over.type: trunk_clearance`，躯干上加了一个向下测距的传感器），在平地上和原来的判断等价（误差 < 1 mm）。**换新地形之前先跑 `rl_train/tests/check_terrain_termination.py <任务名>`**，看出生后有没有大批环境马上终止。
10. **r3 从零训练学不会走**：
    - 现象：到第 1250 次迭代还不会走（mjlab 和验收环境里速度都约等于 0，带探索噪声也只有 0.01 m/s），而 r2 在这个阶段已经会走了；接着训到第 2999 次迭代还是不会。
    - 已排除：低电压和长延迟本身不是原因（r2 it2750 在验收环境里用 6 V + 40 ms 照样能走）。
    - 怀疑（没做对照实验，只是推测）：坡道从一开始就是随机 0–6°，没有从易到难的地形课程；加上"零命令时站着不动"的奖励，让"一直站着"成了很容易陷进去的局部最优。
    - 建议：
      - 下一轮打开 mjlab 的地形课程（`TerrainGeneratorCfg(curriculum=True)` 加 `mdp.terrain_levels_vel` 课程项）。注意 `walk_v2.py` 现在会把 `cfg.curriculum` 清空，要改一下接线；
      - 或者像那晚一样，在 r2 上接着训。walk v2 没有课程，接着训时迭代编号继续往上加也没有副作用。
    - **训练日志里的跟踪奖励和转向增益会误导**：零命令的环境站着就能拿满分，而且数值有噪声。要看 `train.sh` 每 500 次迭代的验收。
11. **坡上偶尔会算出 NaN**：rsl_rl 一遇到 NaN 就让整个训练退出（那晚接着训的一轮在第 ~3032 次迭代就是这样）。`walk_v2_r3.yaml` 已经打开 `terminations.nan_state: true`，出 NaN 的环境会被单独重置。**新配置也要加上。**
12. **不同种子差别很大**：r2 种子 1 有两个 checkpoint 全过，种子 2 一个都没过（停住、5° 上坡不过）。每个配置至少训 2 个种子。
13. **停住时间在门槛边缘晃**：
    - 相邻 checkpoint 之间，停住时间在 0.4–1.3 s 之间跳。命令突然归零时如果正好在迈步中间，就要多走半步。定版要看多个 checkpoint，再加一次 10 个种子的停住测试。
    - **停住对舵机条件也很敏感**（13 种命令 × 3 个种子）：r2 it2750 在标称条件下（7.5 V、延迟 3–6 步）是 39/39，换成 6.75 V、延迟 2–8 步就只有 35/39（最长 1.28 s）。r3 接着训的候选在这两种条件下分别是 31/39 和 35/39。
    - 所以真机电压低、延迟长的话，停住很可能不达标。要先测真机的电压和延迟（第 5 节），把 `data_collect_sim/configs/sim.yaml` 里的标称值改成实测值，再重新验收。
14. **5° 上坡要么全过、要么全不过**（0/10 或 10/10）：平地训练的模型到坡上会稳定地卡住。r3 加坡道训练就是为了解决这个问题。
15. **头最多还能再低 28.6°**（受关节限位约束：neck 0.10、head 0.60 rad），所以头部接管的训练和测试都用 15–28°。
16. **左右镜像对称增强**从 r1 起一直开着（配置里的 `symmetry:`）。镜像的映射写在 `rl_tasks/mdp_v2.py`，用 `rl_train/tests/check_mirror.py` 检查。改了观测内容或关节顺序，要同步改映射。
17. **kick v2 的起始站姿**（`kick_v2_r1_stand2750.yaml`）是从 walk r2 s1 it2750 实测来的。**walk 定版后站姿会变，要用 `data_collect_sim/scripts/stand_pose.py` 重新测，再训一轮 kick。**
18. **kick stage 1 以前的结果偏乐观**：之前的测试只站了 1 秒；实际流程要低头看球、站得更久，这时 stage 1 根本不出脚（基线 0/50）。换成用 walk r2 it2750 站立后，stage 1 就能踢了（踢中 52%、偏差 +0.6°、交接 20/20），说明问题出在站姿离初始姿态太远。
19. **kick v2 的 ONNX 输入是 53 维**：51 维和走路一样，再加 2 维球的位置（在机器人朝向坐标系下的 (x, y)，出脚前测一次，踢球过程中不再更新）。
    **部署端 `microduck/src/moves/walk.py` 遇到 53 维的模型，会以为是"带步态相位"的模型，在最后 2 维喂步态相位**。直接加载 kick v2 会把球的位置喂错，而且不报错。上真机前要专门写一个踢球动作，流程照 `data_collect_sim/duckrt/switcher.py` 里的 KICK：站稳 → 测球 → 播放 `duration_s` 秒 → 站稳。
20. **`train.sh` 跑着的时候不要编辑它**：bash 是边读边执行的，文件长度一变，正在跑的那个就会读错位置（那晚差点出事）。要改就等它跑完，或者复制一份再改。

### 模型和部署

21. **两份机器人模型的关节限位不同**：`robot_allcollisions.xml` 把 hip_yaw 限位放宽到 ±0.436，训练用的 `robot_walk.xml` 是 ±0.175。验收环境用前者的碰撞形状、后者的关节限位（`configs/sim.yaml` 里的 `robot.joint_ranges_from`）。
22. **XML 里的 `imu` site 方向正好是传感器坐标系的逆**。训练和部署都不读它。**不要用 XML 里的 imu 传感器来模拟 BNO08x**，否则 x、z 会反号（`microduck` 里的 MuJoCoController 就是这么做的）。
23. **部署端允许的速度太大**：`scale_velocity` 允许的速度是训练范围的 2–4 倍（vx 0.7、ωz 1.5 / 3.0，训练时只有 0.25 / 0.375 / 0.75）。验收环境会按 card 里的 `command_range` 截断；**上真机前部署端也要截断**。
24. **ONNX 文件里存的默认关节角只保留了 3 位小数**，部署端拿它当偏置用；card 里是全精度的。
25. **walk 模型自己控制头部的 4 个关节**，没有"头部命令"这个输入。头部接管是指：在站立模式下，执行器用固定目标覆盖这 4 个关节（`sim.yaml` 里的 `head_override`）。

### 验收工具

26. **快速验收只用来在训练中挑模型**，它和完整验收的结论偶尔不同。例子：基线的原地转速比，快速 0.80 ✘、完整 0.81 ✔；kick v2 种子 2 it3500 的方向偏差，快速 −2.4° ✔、完整 −3.5° ✘。
27. **每个验收进程约占 1.3 GB 内存**，和训练同时跑时注意内存。
28. **"低头后再踢"的门槛是"成功率不低于直接踢"**：两个都是 0% 也算通过，所以要结合踢中率那一项一起看。
29. `eval_candidates.py` 是旧的批量扫描工具，只跑 6 项走路测试。现在一律用 `accept.py`。

---

## 5. 上真机前要测的参数

| 参数 | 训练里现在用的值 | 在真机上怎么测 | 如果真机值不一样 |
|---|---|---|---|
| **IMU 安装方向** | `BNO080_MOUNT_QUAT = (0.7071068, 0, 0.7071068, 0)`，即绕 y 轴转 +90°。训练和部署两边的算法一致（`rl_train/NOTES.md` §0.1） | 鸭子放平，跑 `microduck/src/imu_orientation_check.py`：重力观测应该是 (1, 0, 0)；往前倾、往左倾时，符号要和仿真里同样姿态下一致 | 如果只是装的方向不同：改部署端的 `IMU_MOUNT_QUAT` 校准，不用重训。如果传感器本身的坐标轴约定不同：所有模型都要重训 |
| **走路时舵机的实际电压** | 舵机模型的电压范围：walk v1 和 v2 r1–r2 是 7–8 V，r3 是 6–7.5 V；验收环境标称 7.5 V（`sim.yaml`）。教程 README 写的电池是 6 V | 走路和踢球时，用万用表或者读舵机寄存器 `present_input_voltage`（XL330 第 144 号），记下最低电压 | 低于 7 V 时，r2 及更早的模型没见过，可能电流不够、走不动或者摔倒 → 用 r3，或者把 `actuator.vin_range` 改成实测范围再训一轮；同时把 `sim.yaml` 的标称电压改成实测值，重新验收 |
| **舵机通信延迟** | 执行器延迟 3–6 个物理步（15–30 ms），r3 是 2–8 步（10–40 ms）；IMU 和关节速度的观测延迟 0–1 步 | 给一个关节突然设一个新目标，同时连续读 `present_position`，看多久开始动；或者在部署端记录每个 50 Hz 周期读写用了多久 | 超过 30 ms 时，r2 及更早的模型没见过 → 用 r3，或者把 `actuator.delay_steps` 放宽再训 |
| **头部相机** | `data_collect_sim/configs/camera.yaml`：挂在 `head_camera` 位置，视场角 45°，分辨率和帧率是占位值 | 拍已知距离的棋盘格或尺子，量出视场角和安装俯角 | 不影响走路和踢球模型（它们不看图像）；影响以后的 VLA 数据采集、2 m 旗垫可见性测试，以及头部接管要低多少 |

---

## 6. 授权

| 内容 | 授权 | 注意 |
|---|---|---|
| 鸭子的 3D 模型文件（`mjlab_microduck/src/mjlab_microduck/robot/microduck/assets/*.stl`，验收环境也会加载） | 按 **CC BY-NC-SA 4.0** 处理：**禁止商用**，修改后要用同样的授权分享 | Pollen 的 `microduck_rl` 在 `LICENSE-MODELS.md` 里明确是这个授权，`microduck/LICENSE` 对硬件文件也是这个授权。AI-FanGe 仓库没有单独说明 |
| AI-FanGe 教程仓库根目录 | MIT（Copyright 2026 AI-FanGe） | |
| `mjlab_microduck` 代码 | Apache-2.0（Marc Duclusaud） | |
| `microduck` 部署代码 | **GPL-3.0**（硬件文件是 CC BY-NC-SA 4.0） | GPL 有传染性：分发修改过的部署代码，必须同样开源 |
| 我们自己的代码（`data_collect_sim/`、`rl_train/`） | 还没声明 | |

**仓库保持私有。打包进收费课程之前，要先弄清楚 3D 模型（禁止商用）和部署代码（GPL）的授权，必要时联系作者。**

---

## 7. 交接那晚改了什么

完整过程和理由见 `HANDOFF_NOTES.md`。

### 训练包 `mjlab_microduck`

改动提交在子仓库的 `roboduck-main` 分支，补丁在 `patches/mjlab_microduck/`：
- 恢复了上游误删的 `robot/` 和 `README.md`（10 月 16 日就恢复过；这次把 README 换回上游原文）。
- 提交了 `uv.lock`（上游的 `.gitignore` 会忽略它）；BAM 从"跟着 doc 分支走"改成钉死在一个 commit（lock 内容不变）；加了 `.python-version`（3.13）。
- 修了课程学习不生效的 bug；日志默认改成 tensorboard。

### `rl_train/`

- `walk_v2.py` 支持在配置里写 `sim: {nconmax, njmax}`；`walk_v2_r3.yaml` 设成 48 / 128，修显存溢出。
- `walk_v2.py` 支持 `terminations.nan_state: true`，坡上偶尔出 NaN 不再让整个训练退出；`walk_v2_r3.yaml` 已打开。
- `walk_v2.py` / `mdp_v2.py` 支持 `terminations.fell_over.type: trunk_clearance`（按离正下方地面的高度判断摔倒）；`walk_v2_r3.yaml` 改用它；新增 `tests/check_terrain_termination.py`。
- 新增 `configs/kick_v2_r1_stand2750.yaml`（起始站姿用 r2 s1 it2750 的实测值；原来的配置没动）。
- 新增 `train.sh`、`tests/check_curriculum.py`、`checks/`、`reference/ai_fange_kick_stage1/`（kick stage 1 训练配置的副本）。

### `data_collect_sim/`

- 新增 `scripts/accept.py`、`scripts/compare.py`、`duckrt/report.py`、`tests/test_accept.py`。
- `duckrt/evaluation.py`：可以只重跑指定的测试用例（录失败视频用）；站立可以用单独的模型；新增走停切换、命令跟踪两项测试；原来写在代码里的测试参数（摔倒高度、踢球流程时长、快速验收规模）搬进了 `acceptance.yaml`，数值没变。
- `configs/acceptance.yaml`：只**增加**了上面这些参数和 `quick:`、`videos:`、`runner:` 三段；**原有的门槛一个都没改**。
- `configs/policy_sources.yaml`：kick 的配置路径改成指向仓库里的副本；登记了 `walk_v2_r2_s1_it2750`。
- `outputs/policies/` 新增三个模型（各带 card）：`walk_v2_r2_s1_it2750`、`walk_v2_r3warm_s1_it3000_candidate`、`kick_v2_r1_stand2750_s1_it3999`。
- 验收报告：`outputs/acceptance/` 下的 `recommended/`、`baseline/`、`baseline_quick/`、`r2_s1_it*/`、`scan_r2_s2/`、`scan_kick_v2_s1/`，以及对比表 `compare_*.md`。
- 加了 `.python-version`；`.gitignore` 改成只提交验收报告和指定的 ONNX。

### 整个工作区

- 原来的工作区是 AI-FanGe 教程的 zip 解压版，没有 git。`mjlab_microduck/`、`microduck/` 就地建成 git 仓库，用 `git subtree split` 接上了上游真实的提交历史，作为子仓库。
- 根目录原来的教程 `README.md` 改名为 `README.upstream.md`（不在我们的仓库里），新的 `README.md` 是我们的入口文档。
- 新增 `scripts/setup_workspace.sh`、`patches/`、本文、`HANDOFF_NOTES.md`。
- 开工前做了备份：`~/backups/Microduck-build-tutorial-main_20261120_pre-handoff.tar.gz`（只在原来那台电脑上）。

### 和原计划不一样的地方

- 原计划那晚不训练正式模型，后来改成"时间多就多跑几个、留些能直接用的 ONNX"，所以训了 kick v2（种子 1、2）、r3 从零（种子 1）和 r3 在 r2 上接着训。原计划的 r3 种子 2 没跑：种子 1 从零都没学会，再跑一个意义不大。
- 原计划以为上游目录是别人仓库的 clone，实际上工作区里没有任何 git 仓库。所以给两个上游目录各建了一个私有仓库：`mjlab_microduck`（有改动）和 `microduck`（没改动，但上游是大仓库里的一个子目录，没法单独做 submodule）。两个都保留了上游真实的提交历史。
- 门槛以 `data_collect_sim/configs/acceptance.yaml` 为准，和原计划的门槛表一致，只多了一项"低头看球后再踢"。

---

## 8. 不在仓库里的文件

没有用 git-lfs，所以 `.pt` 模型文件都不在仓库里。**只是验收的话用不到它们**，仓库里的 ONNX 和 card 就够了。只有重新导出 ONNX、重新生成 card，或者在它们上面接着训时才需要。找我要，按原来的相对路径放回去。

| 文件（原来那台电脑上的路径） | 大小（字节） | sha256 | 是什么 |
|---|---:|---|---|
| `mjlab_microduck/checkpoints/walk_stable_5s/model_1300.pt` | 4 720 565 | `99f8139983f175aadb713ab404933456f617c85bd92496bee371799cfe07572d` | walk v1（负责走） |
| `mjlab_microduck/logs/rsl_rl/mjlab_microduck_velocity/2026-10-16_17-43-09/model_2600.pt` | 4 720 565 | `4c9b41c844db369e3902c93016d5ac9eae17702efa9284d4869f70f09207cc35` | walk v1（负责站） |
| `rl_train/runs/logs/rsl_rl/walk_v2/2026-10-24_23-32-51_walk_v2_r2_s1/model_2750.pt` | 4 695 797 | `bb288bba9a364c9f03f84c82a7a40d822c0b85ea56e6cbc5214446f369f822b2` | **walk v2 推荐** |
| `rl_train/runs/logs/rsl_rl/walk_v2/2026-10-24_23-32-51_walk_v2_r2_s1/model_1500.pt` | 4 695 797 | `e80b643a3a4925f25a3ccaef1e5713d1101f2e0deb41c3dda7403ee1a472dc2d` | walk v2 候选 |
| `~/microduck_ws/runs/ai_fange_kick/models/2026-10-06_13-14-13_af_kick_stage1/model_3999.pt` | 4 745 141 | `4c43bcc5b32d0c23a1fda3f7401052d85e1e555cbb1475eae173782b611422a4` | kick stage 1 |
| `rl_train/runs/logs/rsl_rl/kick_v2/2026-11-20_03-12-56_kick_v2_r1_stand2750_s1/model_3999.pt` | 4 769 717 | `39d28d74c80802db5f5b39bf9384b8f97bdf012422e7e9250e4bbd943492ce56` | **kick v2 推荐** |
| `rl_train/runs/logs/rsl_rl/walk_v2/2026-11-20_04-23-23_walk_v2_r3_from_r2s1it2750_b/model_3000.pt` | 4 695 797 | `532eb8995b0ee85d32f6a3b1c584b9ae80eda43a70bab6b73ceab1f694200f95` | walk r3 接着训的候选（不合格） |
| `rl_train/runs/logs/rsl_rl/kick_v2/2026-11-20_06-57-29_kick_v2_r1_stand2750_s2/model_2750.pt` | 4 769 717 | `fd1440389a15d3c8c89bf0fcabeeba476f24ea1e49e5baddbe6fc20888a236a8` | kick v2 种子 2 备选（合格，偏差 −1.7°；仓库里没有它的 ONNX） |

最省事的传法是把整个 run 目录（包括 `params/`）打包，放回同样的相对路径：
- `rl_train/runs/logs/rsl_rl/`（约 300 MB，所有 v2 的训练）
- `mjlab_microduck/logs/rsl_rl/mjlab_microduck_velocity/2026-10-16_17-43-09/`
- `mjlab_microduck/checkpoints/walk_stable_5s/model_1300.pt`
- `~/microduck_ws/runs/ai_fange_kick/models/2026-10-06_13-14-13_af_kick_stage1/`

---

## 9. 仓库和权限

| 仓库 | 内容 | 可见性 |
|---|---|---|
| https://github.com/H-river/roboduck-workspace | 主仓库：文档、`rl_train/`、`data_collect_sim/`、`patches/`、`scripts/`，分支 `main` | 私有 |
| https://github.com/H-river/roboduck-workspace-mjlab_microduck | 子仓库 `mjlab_microduck/`，分支 `roboduck-main`：上游历史 + 我们的 3 个提交 | 私有 |
| https://github.com/H-river/roboduck-workspace-microduck | 子仓库 `microduck/`，分支 `roboduck-main`：上游历史，没改过 | 私有 |

- **三个仓库都要有权限**，才能 `git clone --recursive`。在网页上 Settings → Collaborators 加人，或者用命令（`gh repo edit` 加不了协作者）：
  ```bash
  for r in roboduck-workspace roboduck-workspace-mjlab_microduck roboduck-workspace-microduck; do
    gh api -X PUT repos/H-river/$r/collaborators/<同学的 GitHub 用户名> -f permission=push
  done
  ```
- 原来那台电脑上，两个子仓库的 remote 叫 `roboduck`；从 GitHub clone 下来的，remote 叫 `origin`。都没有指向上游（AI-FanGe）的 remote，不会误推到别人的仓库。
