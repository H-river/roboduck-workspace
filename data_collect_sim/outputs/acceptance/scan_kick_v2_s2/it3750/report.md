# 验收结论：**不合格**（4 项门槛中 1 项不通过：kick_bias）

- 名称：`kick_v2_s2_it3750`；时间：2026-11-20T08:10:28+08:00；总耗时 13 s（3 个进程）
- walk：card `configs/policies/walk_v2_r2_s1_it2750.yaml`，ONNX `outputs/policies/walk_v2_r2_s1_it2750.onnx`（sha256 91212aef159dfe16…）
- kick：card `outputs/acceptance/scan_kick_v2_s2/it3750/cards/2026-11-20_06-57-29_kick_v2_r1_stand2750_s2_it3750.yaml`，ONNX `outputs/acceptance/scan_kick_v2_s2/it3750/policies/2026-11-20_06-57-29_kick_v2_r1_stand2750_s2_it3750.onnx`（sha256 5be4f9004768d364…）
- 门槛：`configs/acceptance.yaml`（sha256 b64beff45b650767…）

| 技能 | 项目 | 数值 | 门槛 | 结果 |
|---|---|---|---|---|
| kick | 球在 x ∈ [0.07, 0.11] m、y 在踢球脚 ±0.03 m 内 | 96%（48/50，摔 0） | ≥ 80%（成功 = 球速 ≥ 0.3 m/s 且方向偏差 ≤ 10°） | ✔ |
| kick | 踢出方向平均偏差（成功的那些） | -4.5° | |平均偏差| < 3° | ✘ |
| kick | 头部接管低头 25° 后再踢 | 低头后 90% vs 直接踢 80% | 低头后成功率 ≥ 直接踢 | ✔ |
| kick | walk → stand → kick → stand → walk | 20/20 不摔（踢中 19） | 20/20 不摔 | ✔ |

## 失败用例视频

- `videos/kick_window_5.mp4`：kick_window 5（单独重放复现了失败）
- `videos/kick_window_7.mp4`：kick_window 7（单独重放复现了失败）
