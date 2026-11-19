# 验收结论：**合格**（4 项门槛全部通过）

- 名称：`kick_v2_s1_it3999`；时间：2026-11-20T04:25:15+08:00；总耗时 13 s（3 个进程）
- walk：card `configs/policies/walk_v2_r2_s1_it2750.yaml`，ONNX `outputs/policies/walk_v2_r2_s1_it2750.onnx`（sha256 91212aef159dfe16…）
- kick：card `outputs/acceptance/scan_kick_v2_s1/it3999/cards/2026-11-20_03-12-56_kick_v2_r1_stand2750_s1_it3999.yaml`，ONNX `outputs/acceptance/scan_kick_v2_s1/it3999/policies/2026-11-20_03-12-56_kick_v2_r1_stand2750_s1_it3999.onnx`（sha256 5ffbc7993295be6c…）
- 门槛：`configs/acceptance.yaml`（sha256 b64beff45b650767…）

| 技能 | 项目 | 数值 | 门槛 | 结果 |
|---|---|---|---|---|
| kick | 球在 x ∈ [0.07, 0.11] m、y 在踢球脚 ±0.03 m 内 | 100%（50/50，摔 0） | ≥ 80%（成功 = 球速 ≥ 0.3 m/s 且方向偏差 ≤ 10°） | ✔ |
| kick | 踢出方向平均偏差（成功的那些） | -1.0° | |平均偏差| < 3° | ✔ |
| kick | 头部接管低头 25° 后再踢 | 低头后 100% vs 直接踢 100% | 低头后成功率 ≥ 直接踢 | ✔ |
| kick | walk → stand → kick → stand → walk | 20/20 不摔（踢中 20） | 20/20 不摔 | ✔ |
