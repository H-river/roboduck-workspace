# RoboDuck：鸭子机器人的 RL 技能训练

这个仓库用来训练小鸭子机器人的两个动作：**走路（walk）** 和 **踢球（kick）**，并且判断训出来的模型能不能用。

第一次接手，按顺序读完这一页就能开始干活。更细的内容在 [HANDOFF.md](HANDOFF.md)，遇到问题再去查。

---

## 1. 先搞清楚整体流程

```
① 训练（GPU）              ② 导出               ③ 验收（CPU）                   ④ 上真机
mjlab 里几千只鸭子     →    .pt 转成 .onnx   →    在另一个仿真里跑一套固定测试   →   拷到鸭子的树莓派上
同时练习                                          全部达标才算"能用"
rl_train/                                          data_collect_sim/                 microduck/（部署代码）
```

- **训练**在 mjlab 里做。mjlab 是一个用 GPU 并行跑 MuJoCo 物理仿真的 RL 训练框架。
- **验收**在 `data_collect_sim/` 里做。这是一个普通的 CPU MuJoCo 仿真，物理和真机部署代码保持一致，用来公平地判断模型好不好。
- **判断模型好坏只看验收报告**，不要看训练曲线。曲线会骗人：有一次模型学成了站着不动，曲线看起来和会走的模型几乎一样。

### 我们用的是哪只鸭子

同学组装的那只鸭子，照的是 [AI-FanGe/Microduck-build-tutorial](https://github.com/AI-FanGe/Microduck-build-tutorial) 这个复刻教程：14 个 XL330 舵机，主控是树莓派 Pi Zero 2W 加 OpenRB-150 舵机板。仿真里的模型就是这只，已经和教程的 3D 打印件逐个对过尺寸，基本一致。

`mjlab_microduck/README.md` 里那张**黑色、有手臂的机器人图片不是我们的鸭子**。那是原作者更早做的人形机器人 Microban，README 一直没换图，不用管它。

---

## 2. 名字怎么读

模型的文件名都是这个格式，比如 `walk_v2_r2_s1_it2750`：

| 部分 | 意思 |
|---|---|
| `walk` / `kick` | 哪个动作 |
| `v2` | 第几版。v1 是最早从 AI-FanGe 那里拿来的模型，v2 是我们自己重新训练的 |
| `r2` | 训练配置的第几轮修改，对应 `rl_train/configs/walk_v2_r2.yaml` |
| `s1` | 随机种子。同一个配置换个种子，结果可能差很多 |
| `it2750` | 训练到第 2750 次迭代时存下来的那个模型 |

其他常见说法：

- **card（policy card）**：每个 ONNX 配一个 YAML，写着它吃什么输入、输出什么、速度范围多少。验收和运行时都靠它加载模型。放在 `data_collect_sim/configs/policies/`。
- **快速验收 / 完整验收**：快速版（`--quick`）十几秒跑完，训练过程中用来挑模型；完整版约 30 秒，定版前必须跑。两者偶尔结论不同，以完整版为准。
- **热启动**：不从零开始，而是在一个已经训好的模型上接着训。

---

## 3. 现在的状况

**有一套能用的组合，14 项验收全部通过：**

- 走路：`walk_v2_r2_s1_it2750`
- 踢球：`kick_v2_r1_stand2750_s1_it3999`
- 报告：`data_collect_sim/outputs/acceptance/recommended/report.md`

但它还有这些问题：

1. **还没上过真机。**
2. **走路只在平地上训练过。**5° 的坡在验收里能上能下，6° 就上不去了。
3. **"说停就停"对电压和延迟敏感。**真机电压偏低、通信偏慢的话，可能停不及时。
4. **踢球模型多了"球的位置"这个输入**（输入从 51 维变成 53 维），现在树莓派上的部署代码还不支持，直接加载会把数据喂错，而且不报错。
5. **踢球方向不够稳。**换一个随机种子重训，踢中率差不多，但方向会系统性地偏 1.7–4.5°，大部分 checkpoint 因此过不了门槛。

最早从 AI-FanGe 拿来的 walk v1 和 kick stage 1 都验收不合格：走起来停不下来、直走往左偏、右转比左转弱。

---

## 4. 接下来要做的事（按优先级）

1. **把 walk r3 训出来。**r3 在 r2 的基础上加了坡道、头部接管、电压和延迟随机化，目的是让走路模型更适应真机。目前没成功：
   - 从零训练：学成了站着不动；
   - 在 r2 上接着训：能走，但停不住。
   - 建议的改法见 [HANDOFF.md](HANDOFF.md) 第 4 节第 10 条。
2. **踢球多跑几个种子**，查清楚方向偏差是从哪来的，再定版。
3. **上真机之前，先在鸭子上测三个数：**IMU 安装方向、走路时舵机的实际电压、舵机通信延迟。方法见 HANDOFF.md 第 5 节。测完把仿真里的设置改成实测值，重新跑一遍验收。
4. **改部署代码**（`microduck/`）：给踢球写一个专门的动作流程；把速度命令限制在训练范围内（现在允许的速度是训练范围的 2–4 倍）。

---

## 5. 上手：四步，大约 10 分钟

**需要：**Linux 电脑，NVIDIA 显卡，驱动版本 ≥ 580，装好 git 和 [uv](https://docs.astral.sh/uv/)。没有显卡也能做前三步，只是不能训练。

三个仓库都是私有的，先用 `gh auth login` 和 `gh auth setup-git` 登录 GitHub。

```bash
# 第 1 步：下载代码（会一起下载两个子仓库）
git clone --recursive https://github.com/H-river/roboduck-workspace.git
cd roboduck-workspace

# 第 2 步：装环境，最后会自动做一次自检（第一次要下载约 5 GB）
scripts/setup_workspace.sh
```

自检通过时，最后会看到 `CUDA available: True`，以及鸭子"站 2 秒、走 2 秒、没有倒"。

```bash
# 第 3 步：对旧模型跑一次快速验收，确认工具能用（约 10 秒）
cd data_collect_sim
.venv/bin/python scripts/accept.py --walk walk_flat --stand walk_flat_2600 --kick kick --quick --out outputs/acceptance/my_first_quick
```

第一行应该是 `不合格（14 项门槛中 5 项不通过 …）`。这是对的，旧模型本来就不合格。仿真是确定性的，你的结果应该和仓库里的 `outputs/acceptance/baseline_quick/report.md` 一模一样。

```bash
# 第 4 步：试训 2 次迭代，确认训练能跑（约 2 分钟，需要显卡）
cd ..
NUM_ENVS=256 ITERATIONS=2 RUN_NAME=smoke ACCEPT_FINAL=quick rl_train/train.sh walk_v2_r3
```

看到 `training exited with code 0` 就说明训练、导出、验收整条链路是通的。只训了 2 次迭代，验收肯定不合格，这是正常的。

---

## 6. 怎么训练

**改训练内容的规则：新建一个配置文件，不要改已经训过的。**比如想在 r3 上改点东西，就新建 `rl_train/configs/walk_v2_r4.yaml`，第一行写 `inherit: walk_v2_r3.yaml`，下面只写要改的部分。

所有可调的东西（奖励、随机化范围、地形等）都在这些 YAML 里。

```bash
# 在本机后台训练（关掉终端也不会停）
DETACH=1 rl_train/train.sh walk_v2_r3                  # 约 2.2 小时
DETACH=1 SEED=2 rl_train/train.sh walk_v2_r3           # 换个种子再训一次
DETACH=1 rl_train/train.sh kick_v2_r1_stand2750        # 约 1.3 小时

# 看进度
tail -f rl_train/runs/walk_v2_r3_s1.driver.log
```

`train.sh` 会自动完成这几件事：训练，每 500 次迭代导出一次 ONNX 并跑快速验收，训练结束后对最后一个模型跑完整验收。

训练结果在 `rl_train/runs/logs/rsl_rl/<walk_v2 或 kick_v2>/<日期>_<名字>/` 下面，每个导出的模型都有一份验收报告。

**在已有模型上接着训（热启动）**：

```bash
RUN_NAME=我的名字 ITERATIONS=1500 rl_train/train.sh walk_v2_r3 \
  --agent.resume True --agent.load-run 2026-10-24_23-32-51_walk_v2_r2_s1 --agent.load-checkpoint model_2750.pt
```

前提是被接着训的那个模型文件（`.pt`）在你电脑上。`.pt` 文件不在仓库里，找我要，清单在 HANDOFF.md 第 8 节。

用 RunPod 等云 GPU 的做法、`train.sh` 的全部参数、各种显存下的训练速度，见 HANDOFF.md 第 2 节。

---

## 7. 怎么判断模型合不合格

```bash
cd data_collect_sim

# 验收一个训练出来的模型（自动导出 ONNX、生成 card）
.venv/bin/python scripts/accept.py --walk ../rl_train/runs/logs/rsl_rl/walk_v2/<run 目录>/model_2750.pt

# 和旧模型对比，退步的项会标出来
.venv/bin/python scripts/compare.py outputs/acceptance/baseline outputs/acceptance/<你的报告> --out compare.md
```

报告在 `outputs/acceptance/<名字>/report.md`。第一行是总结论，下面每项一行，写着实测值、门槛、过没过；不合格的项会自动录视频。

**三条规则：**

1. **不准为了通过去改门槛。**门槛都在 `data_collect_sim/configs/acceptance.yaml`。觉得哪个指标定得不合理，写进 `rl_train/NOTES.md`，等商量好再改。
2. **定版之前**：跑完整验收，至少训 2 个种子，再看看相邻的几个 checkpoint 是不是也都合格。
3. **新模型不能有任何一项比旧模型差。**

主要门槛：

| 动作 | 测什么 | 要求 |
|---|---|---|
| 走路 | 随机速度命令连续走 60 秒，20 次 | 一次都不摔 |
| 走路 | 走着走着命令变成 0 | 0.6 秒内停住，之后 2 秒内漂移不超过 3 cm |
| 走路 | 命令走直线（vx = 0.2） | 转向速度 < 0.03 rad/s，也就是不跑偏 |
| 走路 | 左转和右转 | 速度之比在 0.8–1.25 之间，左右差不多快 |
| 走路 | 5° 坡上坡、下坡 | 各 10 次全过 |
| 走路 | 站着时把头强制压低 25°，保持 3 秒 | 20 次都站稳，恢复后能正常起步 |
| 走路 ↔ 站立 | 随机时刻来回切换 25 次 | 一次都不摔 |
| 踢球 | 球放在脚前一小块区域里的不同位置 | 踢中率 ≥ 80% |
| 踢球 | 踢出去的方向 | 平均偏差 < 3° |
| 踢球 | 走 → 停 → 踢 → 停 → 走，20 次 | 一次都不摔 |

完整门槛表见 HANDOFF.md 第 3 节。

---

## 8. 最容易踩的坑

1. **必须用 uv 装环境**，pip 会因为依赖冲突装不上。
2. **训练曲线会骗人**，判断模型只看验收报告。
3. **每个配置至少训 2 个种子**，不同种子的结果可能差很多。
4. **`train.sh` 跑着的时候不要编辑它**，bash 是边读边执行的，改了会让正在跑的那个读错位置。要改就复制一份。
5. **加新地形之前先跑 `rl_train/tests/check_terrain_termination.py`。**以前出过鸭子一出生就被判"摔倒"的 bug。
6. **12 GB 显存一次只能跑一个 4096 环境的 r3 训练**。电脑内存只有 14 GB，训练和验收一起跑时注意内存。
7. 如果 shell 里有 ROS 的 `PYTHONPATH`，自己跑命令时前面加 `env -u PYTHONPATH`，否则会崩。

其余的坑（共 29 条）见 HANDOFF.md 第 4 节。

---

## 9. 文件在哪

| 路径 | 是什么 |
|---|---|
| `rl_train/` | **训练代码**（我们写的）。`configs/` 里每个 YAML 是一个训练任务，`train.sh` 是训练入口，`NOTES.md` 是每轮训练的记录 |
| `data_collect_sim/` | **验收环境**（我们写的）。`scripts/accept.py` 是验收入口，`configs/acceptance.yaml` 是门槛，`outputs/policies/` 放能用的 ONNX，`outputs/acceptance/` 放验收报告 |
| `mjlab_microduck/` | AI-FanGe 的训练包（子仓库），有鸭子的仿真模型和基础训练任务。我们只修过 3 处 bug |
| `microduck/` | 树莓派上的部署代码（子仓库），没改过 |
| `patches/` | 我们对上面两个子仓库的修改记录 |
| `scripts/setup_workspace.sh` | 一键装环境 + 自检 |

**文档：**

| 文件 | 什么时候看 |
|---|---|
| 本文 | 第一次接手 |
| [HANDOFF.md](HANDOFF.md) | 查训练参数、坑、真机待测参数、授权 |
| [TRAINING_LOG.md](TRAINING_LOG.md) | 想知道 11 月 20 日那晚各次训练的详细结果 |
| `rl_train/NOTES.md` | 想知道每轮训练改了什么、为什么 |
| `data_collect_sim/NOTES.md` | 想知道验收环境是怎么搭的、怎么和训练环境对齐的 |
| `HANDOFF_NOTES.md` | 交接准备那晚的完整过程记录 |

---

## 10. 授权

仓库保持私有。鸭子的 3D 模型文件是 CC BY-NC-SA 4.0（禁止商用），部署代码是 GPL-3.0。**以后打包进收费课程之前，要先把授权问题弄清楚。**详见 HANDOFF.md 第 6 节。
