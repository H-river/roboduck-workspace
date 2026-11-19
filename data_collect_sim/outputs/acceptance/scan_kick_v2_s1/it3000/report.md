# 验收结论：**合格**（4 项门槛全部通过）

- 名称：`kick_v2_s1_it3000`；时间：2026-11-20T04:24:20+08:00；总耗时 14 s（3 个进程）
- walk：card `configs/policies/walk_v2_r2_s1_it2750.yaml`，ONNX `outputs/policies/walk_v2_r2_s1_it2750.onnx`（sha256 91212aef159dfe16…）
- kick：card `outputs/acceptance/scan_kick_v2_s1/it3000/cards/2026-11-20_03-12-56_kick_v2_r1_stand2750_s1_it3000.yaml`，ONNX `outputs/acceptance/scan_kick_v2_s1/it3000/policies/2026-11-20_03-12-56_kick_v2_r1_stand2750_s1_it3000.onnx`（sha256 296dcae457675cfc…）
- 门槛：`configs/acceptance.yaml`（sha256 b64beff45b650767…）

| 技能 | 项目 | 数值 | 门槛 | 结果 |
|---|---|---|---|---|
| kick | 球在 x ∈ [0.07, 0.11] m、y 在踢球脚 ±0.03 m 内 | 96%（48/50，摔 0） | ≥ 80%（成功 = 球速 ≥ 0.3 m/s 且方向偏差 ≤ 10°） | ✔ |
| kick | 踢出方向平均偏差（成功的那些） | -0.4° | |平均偏差| < 3° | ✔ |
| kick | 头部接管低头 25° 后再踢 | 低头后 100% vs 直接踢 100% | 低头后成功率 ≥ 直接踢 | ✔ |
| kick | walk → stand → kick → stand → walk | 20/20 不摔（踢中 20） | 20/20 不摔 | ✔ |

## 失败用例视频

- `videos/kick_window_0.mp4`：kick_window 0（单独重放复现了失败）
- `videos/kick_window_11.mp4`：kick_window 11（单独重放复现了失败）
