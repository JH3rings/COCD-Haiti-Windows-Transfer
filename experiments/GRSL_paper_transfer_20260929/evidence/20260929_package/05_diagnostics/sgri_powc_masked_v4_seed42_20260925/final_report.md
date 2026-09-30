# Haiti COCD: masked geometry双轨训练、单轨部署验证

## 协议与几何

- 采用项目 active Windows v4：train/test=1370/343 locations、无 validation；seed=42、ImageNet ConvNeXt-Tiny 初始化、batch=16、Adam lr=5e-5、weight decay=0、无 scheduler、max epoch=100、test raw Overall AUPRC 监控、patience=2；报告 best monitored checkpoint。
- 本轮所有新 Student 的 segmentation BCE 均逐视角只在 target geometry code `==0` 上计算。S2 KD queries 要求覆盖该 4×4 query cell 的 target geometry 全为0；POWC paired loss 仅在两视角 geometry 都为0计算。geometry 未作为模型输入。
- 几何码已从项目 mask/metadata 核实：0 valid、1 layover、2 shadow、3 layover+shadow。因代码分开，layover/shadow 可分别统计。test 两轨共 11,239,424 个像素样本：BothValid 9,058,636（80.60%），TargetValid 10,098,042（89.84%），OverallValid 11,137,448（99.09%），G10 1,039,406（9.25%，仅诊断）。两轨的 code 1/2/3 合计分别为 1,094,965/44,741/1,676。
- test AUPRC 依当前正式 v4 规则用于每 epoch 监控和 best 选择，因此 test 指标不是完全独立的最终估计；各新 Student 遵循相同规则。
- SG/CG Teacher 不能重训，故复用此前 geometry-masked teacher checkpoint。该 teacher 来自旧 train=1233/validation=137 协议并按 validation 选点；其 geometry loss mask 符合本轮物理约束，但 checkpoint 收敛协议与本轮 Student 不完全匹配，Teacher 行作为冻结参考。S2/SGRI 主结果均用此 masked-CG guide。最初 raw-CG 版 S2/SGRI 与 zero-init-gate POWC 产物保存在独立 reference 子目录，不进入主比较。

## 主结果

| Method | Best epoch | BothValid AUPRC | BothValid IoU | BothValid F1 | TargetValid AUPRC | TargetValid IoU | TargetValid F1 | OverallValid AUPRC | OverallValid IoU | OverallValid F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SG Teacher† | 15 | 0.8676 | 0.6149 | 0.7615 | 0.8742 | 0.6240 | 0.7685 | 0.8695 | 0.6152 | 0.7618 |
| CG Teacher† | 27 | 0.8872 | 0.6371 | 0.7783 | 0.8918 | 0.6445 | 0.7838 | 0.8892 | 0.6388 | 0.7796 |
| CG counter-shuffle† | 27 | 0.8030 | 0.5004 | 0.6670 | 0.8166 | 0.5138 | 0.6788 | 0.8156 | 0.5099 | 0.6754 |
| S0-mask | 18 | 0.8701 | 0.6184 | 0.7642 | 0.8765 | 0.6272 | 0.7709 | 0.8709 | 0.6180 | 0.7639 |
| S2 Search-KD-mask | 9 | 0.8620 | 0.6053 | 0.7541 | 0.8688 | 0.6151 | 0.7617 | 0.8652 | 0.6068 | 0.7553 |
| SGRI-A1 | 10 | 0.8655 | 0.6033 | 0.7526 | 0.8725 | 0.6145 | 0.7612 | 0.8696 | 0.6085 | 0.7566 |
| SGRI-A2 | 16 | 0.8701 | 0.6175 | 0.7635 | 0.8762 | 0.6268 | 0.7706 | 0.8711 | 0.6157 | 0.7621 |
| POWC-B1 | 19 | 0.8701 | 0.6248 | 0.7690 | 0.8756 | 0.6336 | 0.7757 | 0.8693 | 0.6221 | 0.7670 |
| POWC-B2 | 15 | 0.8645 | 0.6130 | 0.7601 | 0.8712 | 0.6227 | 0.7675 | 0.8672 | 0.6148 | 0.7614 |

† Teacher checkpoint is the legacy validation-selected geometry-masked reference described above. CG counter-shuffle permutes counter samples within each test batch (fixed seed 42+1701).

## 机制证据与判断

- **Teacher 双轨互补：** 在 BothValid，CG−SG = +0.01955 AUPRC / +0.02218 IoU / +0.01678 F1；CG−counter-shuffle = +0.08416 / +0.13670 / +0.11131。有效几何区域内仍有明显正确 counter 价值，打乱后优势消失并反转。
- **S2 Search-KD：** 相对 S0 的 BothValid Δ = −0.00809 AUPRC / −0.01306 IoU / −0.01005 F1；best epoch 9。best Student 在 target-valid queries 上 policy JS=0.02169、same-target-memory evidence distance=0.00687；几何有效 query 覆盖率 85.75%，BothValid query 覆盖率 73.81%。
- **SGRI-A1/A2：** A1 best epoch 的训练 α=0.9375，self-path BothValid AP=0.86551，Teacher-policy guided path=0.88012（gap self−guide=−0.01461）；指导路径仍比 Student 自主路径强，且自主指标低于 S0。A2 beneficial-pixel ratio=5.84%，但 guided path BothValid AP=0.32326、IoU/F1=0，best epoch self AP仅比 S0 高 0.00006，IoU/F1反而略低。两者均未显示可靠 internalization。
- **POWC-B1：** single-path BothValid 相对 S0：AUPRC −0.000005、IoU +0.00638、F1 +0.00485；三个阈值指标不是同向改善，且仅一个 seed。训练 best epoch 的 paired/single BCE 为 0.01157/0.01493；paired-guided BothValid AP=0.89008，比同一模型 single path 高 0.02002。配对分支有明显价值，但 AUPRC 优势没有转移到单轨路径。
- **POWC-B2：** 修正 verifier 初始化后 beneficial-mask 平均比例为 7.64%，所以筛选机制确实被触发；然而 best epoch paired/single BCE=0.29031/0.01530，paired-guided BothValid AP=0.47732、IoU/F1近0，single-path三项均低于 S0。该机制实现失败。首次 zero-init pilot 会导致 gate 恒零，已隔离归档，不计入本行。

## 裁决

- **SGRI：FAILED。** A1 的 guided path 明显强于 self path，但 Student 未追上它；A2 的 guided path 退化。
- **POWC：PARTIAL。** B1 单轨 IoU/F1有单 seed正向变化且 paired branch强，但 BothValid AUPRC几乎持平略低于S0，没有满足预定 second-stage 门槛（AUPRC至少+0.005且IoU/F1同向）；B2失败。
- **最佳具体版本：** 若必须选一个进一步复核，选 POWC-B1；它是唯一同时提高 BothValid IoU/F1 的单轨方案，但现有一 seed 证据不足以称为稳定迁移。
- **最终结论：** 有效区域内双视角仍显著增强 Teacher，POWC-B1显示少量阈值指标迁移信号；但在本轮单 seed、预定门槛下，没有证据证明训练期双视角能稳定提升部署期单视角的判别能力。不能据此声称已证明可重复迁移。

## 复现与产物

- [comparison.csv](comparison.csv)：完整模型比较表。
- [final_metrics.json](final_metrics.json)：全精度分区指标、机制指标与 teacher deltas。
- [protocol.json](protocol.json)：训练和 geometry 规则及来源披露。
- [train_sgri_powc_masked_v4.py](../../../cocd/windows_main/train_sgri_powc_masked_v4.py) 与 [evaluate_sgri_powc_masked_v4.py](../../../cocd/windows_main/evaluate_sgri_powc_masked_v4.py)：本轮隔离的训练与评估代码。
- 各方法 progress JSON 和 best checkpoints 在当前目录；`rawCG_guide_reference/`、`zero_init_gate_reference/` 保存排除在主分析外的实现诊断版本。
