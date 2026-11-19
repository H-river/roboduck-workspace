# HANDOFF_NOTES — 交接准备的过程记录（2026-11-20 夜，无人值守）

按步骤追加：做了什么、发现了什么、为什么这样做。整理后的结论在 `HANDOFF.md`。
路径相对 workspace 根目录 `~/Microduck-build-tutorial-main/`。

开工前整个 workspace（不含 `.venv`、`__pycache__`）备份到
`~/backups/Microduck-build-tutorial-main_20261120_pre-handoff.tar.gz`。原因：这个目录没有任何版本控制，今晚的改动要能退回。

---

## 第 0 步：盘点（只读）

### 0.1 目录

workspace 是 GitHub 上 `AI-FanGe/Microduck-build-tutorial`（main）的 **zip 解压版**，不是 git clone，**整个目录里没有任何 `.git`**。所以 prompt 假设的"嵌套仓库的 origin 指向 AI-FanGe / Pollen"在这里不存在，第 5 步的仓库方案按实际情况调整（见第 5 步）。

把它和上游 clone（`~/microduck_ws/rl_fange/third_party/fange_tutorial`，HEAD `4967821`，2026-09-11）逐文件比较（`diff -rq`，排除 `.venv`、`logs`、`checkpoints`、`__pycache__`、我们自己的两个目录）：

| 目录 / 文件 | 来源 | 大小 | 相对上游 `4967821` 的本地改动 |
|---|---|---|---|
| `mjlab_microduck/` | 上游（AI-FanGe 训练包） | 7.0 GB（`.venv` 6.8 GB、`logs/` 135 MB、`checkpoints/` 9.9 MB、`src/` 31 MB） | 新增 `README.md`（10-16，修 pyproject 引用不存在的 README）、新增 `uv.lock`（10-16）、恢复 `src/mjlab_microduck/robot/`（上游 `34656ce` 删掉了；和 `34656ce^` 逐文件比较**完全相同**） |
| `microduck/` | 上游（Pi 部署运行时） | 9.5 MB | **无** |
| `docs/`、`scripts/imu_mujoco_orientation_viewer.py`、`README.md`、`LICENSE`、`bno08x_calibrate_ui.py`、`microduck3D打印.3mf`、`.gitignore` | 上游根目录 | 12 MB | 无 |
| `Repo_Structure.md` | 我们（10-01，目录说明） | 18 KB | — |
| `data_collect_sim/` | 我们（第一阶段：裸 MuJoCo 验收环境 + 第二阶段的验收代码） | 1.1 GB（`.venv` 209 MB、`outputs/` 392 MB） | — |
| `rl_train/` | 我们（第二阶段：walk v2 / kick v2 训练任务） | 285 MB（`runs/` 几乎全部） | — |

超过 20 MB 的文件（不含 `.venv`）：只有 `data_collect_sim/outputs/parity/*.mjb` 4 个，各 80 MB（parity 测试的编译模型，可重新生成，不提交）。

### 0.2 第二阶段（`rl_train/`）做到了哪一步

读 `rl_train/NOTES.md`、`rl_train/runs/*.log`、`data_collect_sim/outputs/candidates/*/report.md`（`SKILLS_REPORT.md` 不存在）：

- §0 调研、§1 扫描旧 checkpoint（`walk_stable_5s` 这个 run 的 15 个 checkpoint 没有一个全过验收）都做完了。
- walk v2 第 1 轮（r1，seed 1，2037 次迭代后手动停）：停住修好了，转向被改坏（机制已查明）。
- walk v2 第 2 轮（r2）：seed 1、seed 2 都训满 3000 次迭代。**seed 1 的 it1500 和 it2750 通过全部 walk 门槛**，NOTES 认为 it2750 最好（130 次扩展停住测试全过）。seed 2 训完了，**但没有做验收**。
- walk v2 第 3 轮（r3：坡道 + 头部接管 + 电压 / 延迟随机化）：2026-10-25 01:34 启动，**两个 seed 都在启动时 GPU 显存分配失败退出**（`Failed to allocate 704643072 bytes`），之后没人处理。今晚已复现并修好（第 1 步）。
- kick v2 第 1 轮：配置和任务写好了，只跑过冒烟（`runs/smoke_kick.log`），**没有正式训练**。配置里 `start.joint_offsets` 还空着（要等 walk v2 定版后填站姿）。
- **没有定版策略**。现在在用的还是第一阶段的组合：WALK_FLAT = `walk_stable_5s/model_1300`，STAND = 同一 run 的 `model_2600` 零命令，KICK = AF stage 1（`~/microduck_ws/runs/ai_fange_kick`）。
- IMU 朝向（`rl_train/NOTES.md` §0.1）：**训练和部署一致**。XML 里的 `imu` site 和传感器帧正好相反，但训练和部署都不读它；真机上只剩"BNO08x 实际安装方向是否满足 `imu_quat_to_body` 的假设"这一项硬件测量。

### 0.3 训练入口

**入口 = `rl_train/` 注册的 `RoboDuck-*` 任务**（每个 `rl_train/configs/{walk,kick}_*.yaml` 一个任务）：
- 下一步要训的是 `RoboDuck-walk_v2_r3`（walk v2 定版轮）和 `RoboDuck-kick_v2_r1`。
- `Mjlab-Velocity-Microduck`（上游任务）只作为 `walk_v2` 的基础配置，被 `rl_train/rl_tasks/walk_v2.py` import 后再改。
- 运行环境：`rl_train` 没有自己的 venv，用 `mjlab_microduck/.venv`（`train.py` 先注册 `RoboDuck-*`，再调用 mjlab 自己的 `train`）。

### 0.4 "已知情况"逐条核实

| # | 说法 | 现状 | 证据 |
|---|---|---|---|
| 1 | `pyproject.toml` 声明了不存在的 README | **本地已修**（10-16 加了 `mjlab_microduck/README.md`），上游仍缺 | 和上游 diff |
| 2 | 依赖没锁、没 `uv.lock` | **本地已有 `uv.lock`**（10-16 生成），但 pyproject 里 bam 写的是 `branch = "doc"`（会漂移），lock 锁在 `0007411`；实际版本是 Python 3.13.5、torch 2.14.1、mujoco 3.16.0、mujoco-warp 3.8.1、mjlab 1.3.0、warp-lang 1.12.0、rsl-rl-lib 5.0.1、BAM 1.0.0 @ `rhoban/bam@0007411`（doc 分支）、onnx 1.23.2。**和 prompt 说的"上游组合"（Python 3.12、torch 2.9.1、mujoco 3.10.0，是 Pollen `microduck_rl` 的 lock）不同** | `.venv` 里实际安装的版本；`uv.lock` |
| 3 | `command.build = lambda env, _cmd=command: ...` 让后期课程不生效 | **仍存在**于上游 `microduck_velocity_env_cfg.py:330`。`rl_train` 的 walk v2 已绕开（命令换成 `UniformVelocityCommandWithRotationCfg` 实例，并且去掉了课程） | 源码 |
| 4 | `num_envs` 默认 1 | **仍存在**（上游 `SCENE_CFG`，cfg:93）。`rl_train` 已从 YAML 读取（`train.num_envs`） | 源码 |
| 5 | 默认 wandb，没有 key 会卡住 | 上游 `MicroduckVelocityRlCfg` 仍是 wandb；`rl_train` 已固定 tensorboard | 源码 |
| 6 | 必须用 uv | 仍然成立（pyproject 里 protobuf 的 override 只有 uv 认） | pyproject `[tool.uv] override-dependencies` |
| 7 | AF 训练包缺 `robot/`，是从 `34656ce^` 恢复的 | ✔ 恢复的内容和 `34656ce^` 完全相同，没有额外的本地改动 | `git archive 34656ce^` + `diff -rq` |

prompt 里"策略和验收"那几条和 `data_collect_sim/NOTES.md`、`rl_train/NOTES.md` 的结论一致，具体数字以 `rl_train/NOTES.md` §1.2 为准。

### 0.5 `data_collect_sim/` 已有的测试（全部复用）

- 第一阶段：`walk_test.py`、`camera_check.py`、`ramp_test.py`、`kick_test.py`、`check_parity.py`、`build_scene.py`、`teleop.py`。
- 第二阶段：`eval_candidates.py`（checkpoint → ONNX → card → 全套 walk 验收 → 对比表）、`stop_test.py`、`symmetry_test.py`、`head_override_test.py`、`handoff_test.py`、`stand_pose.py`；测试逻辑都在 `duckrt/evaluation.py`（random / stop / symmetry / ramp / head_override / camera + kick_window / kick_head_down / kick_handoff）。
- 门槛文件已存在：`data_collect_sim/configs/acceptance.yaml`（第二阶段在看任何 v2 结果之前定的，和 prompt 的表一致，并且把 prompt 没定义的细节定死了）。**以它为准。**
- 和 prompt 门槛表相比**缺的**：walk ↔ stand 热切换 25/25（没有实现）；失败用例的视频；`--quick` 模式；统一入口 / 报告 / 对比脚本。
- 单元测试：`tests/test_runtime.py`、`tests/test_scene.py`（19 个）。

### 0.6 密钥

- 没有 `.env`、`.netrc`、wandb 配置目录（`~/.config/wandb` 也不存在）、私钥文件。
- grep（`api_key`、`WANDB_API`、`ghp_`、`gho_`、`sk-`、`hf_`、`AKIA`、`password=`、`token=`）只命中 `bno08x_calibrate_ui.py` 的 `--password` 参数（默认从环境变量 `PASSWORD` 读，没有写死的值）。
- 没有装 gitleaks；提交前再用 grep 扫一次暂存区。

### 0.7 工具

- `nvidia-smi`：RTX 4080 Laptop 12 GB，驱动 580.178.04，空闲时占用约 0.65 GB。内存 14 GiB（可用约 10 GiB），无 swap。
- `gh auth status`：已登录 `H-river`（scopes：repo、workflow、gist、read:org）。
- `git lfs`：**没有安装** → `.pt` 不进仓库，在 HANDOFF 里列路径、大小、sha256。

### 0.8 外部依赖（fresh clone 会缺的东西）

- kick stage 1 的 checkpoint、训练配置（`af_kick_cfg.py`、`ball.xml`、`params/*.yaml`）都在 `~/microduck_ws/` 里（另一个 workspace）。`data_collect_sim/configs/policy_sources.yaml` 用绝对路径引用它们；运行时只需要已经复制过来的 `outputs/policies/kick_af_stage1_model3999.onnx` 和 card。→ 把需要的文件复制进仓库（第 5 步）。
- `data_collect_sim/README.md` 写死了 `uv sync --python /home/jiamo/miniconda3/bin/python3.13`。

---

## 第 1 步：训练 repo 能从零跑通（2026-11-20 00:40–01:45）

### 已完成
- **仓库化（本地，未推送）**：`mjlab_microduck/`、`microduck/` 原地 `git init`（分支 `roboduck-main`，没有任何 remote），用 `git subtree split` 把上游 AI-FanGe `4967821` 里这两个子目录的真实历史接进来（`git reset`，工作区文件没动）。`microduck/` 无本地改动、工作区干净。`mjlab_microduck/` 在上游历史上加了 3 个提交：
  - `7dcee90` 恢复上游删掉的 `robot/`（= `6991771` = `34656ce^`，逐文件相同）和原版 `README.md`（10-16 的 3 行占位版备份在 scratchpad）。
  - `1e0e590` 提交 `uv.lock`（上游 `.gitignore` 忽略了它，已改），以及 `checkpoints/walk_stable_5s/params/*.yaml`、`eval.json`（`.pt` 和视频不进 git）。
  - `70d605f` 今晚的修复：① lambda 课程 bug → `UniformVelocityCommandWithRotationCfg`；② 日志默认 tensorboard；③ bam 按 rev `0007411` 钉死（lock 重新生成，没有版本变化）；④ `.python-version` = 3.13。
- **课程修复的证据**（`rl_train/tests/check_curriculum.py`，结果在 `rl_train/checks/curriculum_{fixed,buggy}.json`）：课程切换点提前到第 100 个 env 步。修复后采样器 cfg 变成 vx ±0.35、原地转 ±1.5，实际采到的 |ωz| 从切换前的 0.749 升到 1.497；原版 lambda（负对照，按上游流程先绑定再 deepcopy）的采样器停在 ±0.25 / ±0.75，课程不生效。
- **r3 显存溢出修好**：mujoco_warp 遇到 hfield 时把每个 world 的 `nconmax` / `njmax` 默认值放大到 128 / 1024，CCD 缓冲在 4096 env 下放不下。实测坡道上每个 world 最多 8 个接触、34 行约束，所以在 `walk_v2_r3.yaml` 里加了 `sim: {nconmax: 48, njmax: 128}`，并改 `walk_v2.py` 读取。修复后 4096 env 显存峰值 8.5 GB。
- **吞吐**（`RoboDuck-walk_v2_r3`，20 次迭代，取后 10 次平均）：512 → 1.49 s/迭代，8.3k 步/s，显存 2.0 GB；1024 → 1.87 s，13.2k，2.9 GB；2048 → 2.14 s，22.9k，4.7 GB；4096 → 2.61 s，37.6k，8.5 GB（采样 2.34 s + 学习 0.28 s）。3000 次迭代约 2.2 h。kick v2 在 2048 env 下 1.13 s/迭代，4000 次迭代约 75 min，显存 4.3 GB。
- **冒烟链路**：512 env 的冒烟 checkpoint（model_19）→ `rl_train/export_onnx.py`（21 s）→ `make_policy_cards.make_walk` → `data_collect_sim` 运行时加载，STAND 跑满 5 s，没有报错（只训了 20 次迭代，0.88 s 时摔倒，属于预期）。
- **kick v2 起始站姿**：新建 `rl_train/configs/kick_v2_r1_stand2750.yaml`（继承 `kick_v2_r1`），用 `stand_pose.py` 实测的 r2 s1 it2750 站姿填进去。原来的 `kick_v2_r1.yaml` 没改。冒烟通过，偏移量已进入解析后的 env.yaml。
- **用户追加要求"多跑几个动作、留可直接用的 ONNX"**：00:50 启动了 `rl_train/runs/queue_20261120.sh`（setsid 脱钩，内存上限 8 GB）：walk_v2_r3 s1 → kick_v2_r1_stand2750 s1 → walk_v2_r3 s2，预计 06:40 前后跑完。日志 `rl_train/runs/train_*.log`，run 目录 `rl_train/runs/logs/rsl_rl/{walk_v2,kick_v2}/`。

- 01:00–01:30 因为用量上限停了半小时，训练队列没受影响。
- **`rl_train/train.sh`**：后台训练 + 每 `EXPORT_EVERY` 次迭代导出 ONNX + quick 验收 + 结束时完整验收；没有 systemd（RunPod）时不加内存上限。walk（256 env × 10 次迭代）和 kick（256 × 6）两条路径都冒烟过。发现 `RUN_NAME` 自定义时 `accept.py` 推断不出配置 → 加了 `--rl-config`，`train.sh` 显式传。

## 第 2 步：一键验收（01:30–01:45）

- `duckrt/evaluation.py`：在原有测试上加 `only=`（只跑指定试验，失败视频重放用）、`stand_card`（STAND 用单独的策略，基线就是 1300 走 + 2600 站）、`recording()`（重放时用跟拍相机录像）；新增 `walk_stand_switch`（prompt 要求的热切换 25/25；每次 站 1 s → 随机命令走 1–3 s → 站 1–2 s → 换一个随机命令再走 1.5 s，切换时刻落在随机的步态相位上）和 `tracking`（命令跟踪表，只记录）。原来写在代码里的测试参数（摔倒高度 0.07、kick 前站 1 s、交接流程时长、kick 卡片缺省值、quick 时的缩小规模、头部接管基线 5 个 seed）全部搬进 `acceptance.yaml`，**数值不变**。
- `duckrt/report.py`：按配置里的门槛从原始测量值重新判定每一项（不信任测试函数自己的 pass），渲染 md / json。
- `scripts/accept.py`：输入可以是 card / card 名 / checkpoint / ONNX（ONNX 必须给训练配置来源，否则退出码 2 并说明怎么办）；每个测试一个进程（坡道按角度拆开），进程数按可用内存封顶；失败用例单独重放录视频（每项最多 3 个），并记录重放有没有复现失败；退出码 0 / 1 / 2。worker 的 stderr 写到 `<out>/logs/`（EGL 渲染器在进程退出时的析构报错会刷屏）。
- `scripts/compare.py`：多个 `report.json` 并排，第一个当基线，标出退步项。
- 门槛：`configs/acceptance.yaml` 第二阶段就有，和 prompt 的表一致，以它为准；只**加了**测试参数、`quick:`、`videos:`、`runner:`。它比 prompt 多一项"低头测球后再踢"。
- 单元测试 `tests/test_accept.py`（6 个，加上原有 19 个共 25 个，7 s）：门槛从配置读（同样的测量值，收紧门槛后结论全部翻转）、quick 只缩规模不动门槛、`evaluation.py` / `report.py` / `accept.py` 里不出现门槛数值（AST 检查）、报告格式、已知会摔的输入（同一个策略把 card 的 `action.scale` × 20）判为不合格 + 原策略同一测试合格作对照、ONNX 没有配置来源时报清楚的错误。
- 速度：完整模式 30 s（4 个进程，训练同时在跑），quick 10 s。CPU 仿真约 40–50 倍实时。

## 第 3 步：基线报告（01:37–01:45）

- **基线**（1300 走 + 2600 站 + kick stage 1，`outputs/acceptance/baseline/`）：不合格，14 项门槛 4 项不过：停住 0/39、直走偏航 0.129 rad/s、踢球窗口 0/50、方向偏差（无成功的踢球）。其余通过：随机 20/20、转速比 0.81 / 1.22、5° 坡 10/10 ×2、头部接管 20/20、旗垫、热切换 25/25、kick 交接 20/20。和 prompt 的已知情况一致（偏航约 +0.13）。kick 0/50 是第二阶段已经发现的问题：stage 1 只有从接近 HOME 出发才出脚，用 2600 站 1 s 之后就不踢了。
- **第二阶段候选**：r2 s1 it2750 和 it1500 的 walk 门槛全部通过（it2750：停住 39/39 最长 0.50 s、偏航 0.018、转速比 0.98 / 0.96、6° 上坡 0/10 只记录）；kick 仍然不过，但**换成 walk v2 的站姿后 stage-1 kick 能踢了**：it2750 下窗口 52%（26/50）、偏差 +0.6°、交接 20/20、低头后 100% vs 直接 90%。
- r2 s2（训完但没验收过）6 个 checkpoint 的 quick 扫描（`outputs/acceptance/scan_r2_s2/`）：没有一个全过，问题在停住和 5° 上坡。
- `walk_v2_r2_s1_it2750` 提成命名策略（`outputs/policies/` + `configs/policies/`），当 `train.sh` 验收 kick 和自检用的 walk。

## 第 5 步：仓库（01:45–）

- workspace 里没有任何 git 仓库，所以 prompt 的"嵌套仓库 origin 指向上游、不能动"这条无从谈起；按它的规则处理两个上游目录：
  - `mjlab_microduck/`：有本地改动 → 私有仓库 `H-river/roboduck-workspace-mjlab_microduck`，本地 remote 名 `roboduck`，分支 `roboduck-main`。
  - `microduck/`：没有本地改动，但上游是 AI-FanGe 大仓库里的一个子目录，没有可以单独做 submodule 的 remote → 同样建私有仓库 `roboduck-workspace-microduck`。
  - 两个都用 `git subtree split` 保留了上游真实的提交历史（作者 AI-FanGe），我们的改动是上面的独立提交；本地没有加任何指向上游的 remote（没有可能推到上游）。
- `patches/`：`0001` 是脚本（恢复 `robot/` 的补丁有 13 MB 二进制，脚本直接从上游历史取文件），`0002`、`0003` 是 `git format-patch`。`patches/check_patches.sh` 从 GitHub 克隆上游 `4967821`、subtree split、重放补丁，和子仓库比较 git tree：两个都 OK（4 s）。
- 主仓库 `roboduck-workspace`（workspace 根目录）：`data_collect_sim/`、`rl_train/`、`patches/`、`scripts/setup_workspace.sh`、`HANDOFF*.md`、`README.md`。根目录的上游文件不进仓库（`.gitignore`）；上游教程的 `README.md` 改名为 `README.upstream.md`，给我们自己的 README 腾位置。
- `data_collect_sim/.gitignore`：`outputs/` 只放行报告（md / json，ramp_test 的 8 MB 轨迹 json 除外）和 `outputs/policies/*.onnx`。暂存区一共 5.8 MB，最大的文件是 4 个 ONNX（各 0.77 MB）。
- 密钥扫描（没有 gitleaks，用 `git grep` 扫暂存区和两个子仓库的 HEAD：api key / token / ghp_ / gho_ / github_pat_ / sk- / hf_ / AKIA / 私钥头 / password=）：没有命中。
- kick stage 1 的训练配置从 `~/microduck_ws` 复制到 `rl_train/reference/ai_fange_kick_stage1/`，`policy_sources.yaml` 指向副本；`.pt` 不进仓库（没有 git-lfs），HANDOFF 里列了路径、大小、sha256。
- 同学的 GitHub 用户名 prompt 里没填 → 没加协作者。

### 第 1 步补记：r3 的摔倒判据（01:54–02:00）
- 01:54 看 r3 s1 的训练日志（第 1500 次迭代）：平均回合长度 1.0 步，每步约 1230 次 `fell_over` 终止，从第 0 次迭代起就是这样（回合长度 4–5 步）。
- 原因（`rl_train/tests/check_terrain_termination.py` 实测，512 env，零动作 25 步）：坡道地形里倒金字塔的出生点 z 最低 −0.37 m，上游 `fell_over` = 躯干**绝对**高度 < 0.10 m，所以出生在倒金字塔里的 103 个环境**全部**在 25 步内终止，平地和正金字塔上的 409 个一个都没有。r2（平地）同样的检查：0 个。也就是说 20% 的环境每一步都在重置，策略也永远学不到走进凹坑。第二阶段的 r3 从来没真正跑起来过（显存溢出），所以这个问题一直没暴露。
- 修法：躯干上加一个向下的地形测高传感器（`TerrainHeightSensorCfg`，4 条射线、3 cm 半径、取平均，和脚上那个同样的设置），`fell_over` 改成"离正下方地面 < 0.10 m"（`mdp_v2.trunk_clearance_below_minimum`，`walk_v2_r3.yaml` 的 `terminations:`）。修好后：三类出生点 25 步内终止数都是 0；平地上测到的离地高度和躯干绝对高度差 ≤ 0.6 mm（射线打到地面，不是自己的腿），即和原判据等价。奖励里的 `foot_clearance` / `foot_swing_height` 本来就用的地形测高传感器，不受影响。
- 处理：停掉队列，r3 s1 的 run 移到 `rl_train/runs/discarded/`（附说明）；01:58 用 `train.sh` 重新排队 `rl_train/runs/queue_20261120b.sh`：r3 s1 → kick v2 s1 → r3 s2。这次直接用 `train.sh`（每 500 次迭代导出 ONNX + quick 验收，结束时完整验收）。
- `walk_v2_r3.yaml` 是原地改的（它在今晚之前从没训成过，算修 bug）。
- 小插曲：队列在跑时我改了 `train.sh`（不导出第 0 次迭代）。bash 边读边执行，插入 10 个字节会让正在跑的实例在循环结束后读错位置。当场在循环之前的注释里删掉 10 个字节补偿：循环起点（5874）和之后的内容逐字节不变。已写进 HANDOFF 的已知坑。

## 第 5 步补记：从 GitHub 重新 clone 验证（01:54–01:58）
- 第一次：`setup_workspace.sh` exit 126，文件在 git 里没有执行权限（用编辑工具新建的文件没 `chmod +x`）。修好推送。
- 第二次（commit `88e5d71`，shell 里带 ROS 的 `PYTHONPATH`，严格照 HANDOFF §2）：clone 5 s；setup 14 s（uv 缓存热）；quick 验收 10 s，表格和 `baseline_quick/report.md` 逐行相同；冒烟训练 2 次迭代 + 导出 + quick 验收 105 s。全部通过。
- r3 的修复是在这之后推的，最后会再完整验证一次。

### 第 1 步补记：r3 从零训练学得慢，改排队（02:20–02:58）
- 修好摔倒判据后，r3 s1 的训练曲线前 30 次迭代和 r2 一样，之后明显落后：第 500 次迭代跟踪奖励 1.04（r2 1.34）、腾空奖励 0.12（r2 0.30）。
- quick 验收（`train.sh` 每 500 次迭代自动跑）：第 500、1000 次迭代的策略在验收环境里**完全不走**（命令 vx 0.25 实际 0.000，原地转 0.5 实际 0.015）。
- 排查是不是 sim2sim 问题：
  - 验收环境里改电压（6.75 V、7.0 V 无压降）、改延迟（2–8 步）：r3 仍然 0.000，r2 在同样条件下 0.12–0.13 m/s → 不是验收环境的执行器设置。
  - 放回 mjlab（`mjlab_walkcheck.py`，固定命令，16 env）：r3 it1000 在 play 配置的坡道地形和平面上、在训练配置（含头部接管、域随机化）上都 ≈ 0（vx ≤ 0.012）；带探索噪声采样动作也只有 vx 0.012、原地转增益约 0.3。r2 it1000 同样测试 vx 0.111。→ **不是 sim2sim，是 r3 从零开始学得慢得多**：坡道 + 6–7.5 V + 10–40 ms 延迟 + 头部接管一起上，到第 1000–1250 次迭代还没学会走。训练日志里的跟踪奖励 / 转向增益是带噪声、所有环境平均的，会误导（零命令的环境站着就拿满分）。
- 处理：为了早上有能用的鲁棒 walk，02:55 停掉队列 b，r3 s1 在存下 `model_1250` 后停（完整验收：4 项不过，停住 / 5° 坡 / 头部接管）。新队列 `queue_20261120c.sh`：
  1. `walk_v2_r3_from_r2s1it2750`：r3 配置，从 r2 s1 it2750 热启动（`--agent.resume True --agent.load-run … --agent.load-checkpoint model_2750.pt`；walk v2 没有课程，恢复迭代计数没有副作用；先用 64 env × 1 次迭代试过能加载），1500 次迭代（2750 → 4249），约 70 min。
  2. kick v2 s1（4000 次迭代，约 75 min）。
  3. r3 s1 从 `model_1250` 续训 1750 次迭代到 3000（从零训练这个实验保留，只是挪到最后）。
- 原计划的 r3 s2（从零）不跑了：s1 还没证明从零能学会，先把热启动和续训做完更有用。

### 第 1 步补记：热启动那一轮因 NaN 中断（03:12），改成队列 d（03:53）
- `walk_v2_r3_from_r2s1it2750` 跑到第 ~3032 次迭代时，rsl_rl 报 "observation group 'actor' contains NaN" 退出：某个环境在坡道上物理状态发散。上游任务没有 NaN 终止项（Pollen 的版本有 `nan_state`）。`njmax` 不太可能是原因：walk 模型只有两只脚和地面有碰撞，最多 8 个接触 ≈ 32 行 + 14 个关节限位，离 128 还远。
- 修法：`walk_v2_r3.yaml` 加 `terminations.nan_state: true`（mjlab 自带的 `nan_detection`：qpos / qvel / qacc / sensordata 有 NaN / Inf 就终止并重置；mjlab 的奖励管理器本来就对奖励做 `nan_to_num`）。64 env × 2 次迭代冒烟通过。
- 第 3000 次迭代（热启动后 250 次）的验收：比起点 it2750 差（完整：停住、边走边转比 1.83、5° 上下坡、头部接管不过）——换了环境之后的性能下降，本来要看后面能不能恢复，被 NaN 打断了。
- 03:12 起队列 c 接着训 kick v2。03:53 停掉队列 c 的父进程（kick 不受影响），新队列 `queue_20261120d.sh`：等 kick 结束 → 热启动重跑（`walk_v2_r3_from_r2s1it2750_b`，带 NaN 兜底）→ r3 s1 从 `model_1250` 续训。
- kick v2 中间结果（quick，walk 用 r2 it2750）：第 1000 / 1500 / 2000 次迭代踢球窗口 9/9、交接 5/5、低头后再踢 100%，只有方向偏差不过（−3.8° / −5.4° / −4.2°，门槛 3°）。
- 另一个实验：r2 it2750 在验收环境里 6.0 V、6.0 V + 40 ms 延迟下照样走（0.117 / 0.129 m/s），所以低电压和长延迟本身不是 r3 学不会走的原因；剩下的嫌疑是坡道地形没有课程（一开始就随机 0–6°），加上 `stand_still` / `no_stepping` 让"站着不动"成了容易陷进去的局部最优。没时间做消融，作为假设写进 HANDOFF。

### 第 1 步补记：热启动结果和停住的敏感性（04:23–05:45）
- 热启动重跑（带 NaN 兜底）没再崩，04:23–05:32 跑完 1500 次迭代。完整验收（配 kick v2 it3999）：it3000 只有停住不过（31/39，最长 1.18 s），it3250 停住 7/39，it3750 停住 + 5° 坡 + 头部接管，it4000 停住 13/39，it4249 停住 28/39 + 头部接管 18/20。没有一个合格。it3000 作为标明"不合格"的候选放进 `outputs/policies/`。
- 停住对执行器条件的敏感性（验收环境里改 `sim.yaml` 的执行器，13 命令 × 3 seed）：

| 策略 | 7.5 V，延迟 3–6（标称） | 6.75 V，延迟 2–8（r3 训练分布中心） | 7.5 V，延迟固定 2 |
|---|---|---|---|
| walk_v2_r2_s1_it2750 | 39/39，0.50 s | 35/39，1.28 s | 39/39，0.40 s |
| walk_v2_r3warm_s1_it3000_candidate | 31/39，1.18 s | 35/39，1.32 s | 24/39，1.08 s |

  → r2 的 39/39 依赖标称条件；真机电压 / 延迟偏一点就可能不达标。写进 HANDOFF 已知坑。
