# 分析提示词：Haiti SAR 滑坡检测 —— 版本谱系、DIS2 异常、内部方案与高指标溯源

> 用途：把下面全部内容作为一次性输入交给分析方（人或模型），要求其产出结构化诊断报告。
> 项目根目录：`/Users/zhangjiuqi/Desktop/distortion`，核心工程：`Haiti_SAR_GRSL_Audit/`。
> 术语约定：下文 **DIS2** 指「缺失观测补偿蒸馏」这一外部基线方法。本项目中出现过**三个**不同的 DIS2 相关产物——**旧 DIS2-style**、**官方 width-50 复现**、**DIS2-port**——第 2、3 节会拆开说明，三者不可混称。其中「官方 width-50 复现」已按用户指示删除（见 2.3 与 8）。

---

## 0. 你的角色与输出纪律

你是这个项目的代码与实验审计员。请基于本文档给出的全部事实，回答第 5 节的三个核心分析问题，并给出第 6 节要求的结构化报告。

输出纪律（必须遵守）：

1. **不得引入本文档没有给出的数字。** 需要新数字时，写出应当执行的命令，而不是估计。
2. 每条结论必须标注证据等级：`[已证实]`（可从给出的文件/代码逐位核对）、`[推断]`（由多个已证实事实推出）、`[假设]`（待验证，需给命令）。
3. 严禁把「验证集指标」当「测试集指标」比较，严禁把不同分支（self / dual / missing）的输出混在同一列。
4. 每个被讨论的指标都要写明四要素：**来源文件 + partition + region/分支 + checkpoint 选择规则**。
5. 不要因为某个方向「看起来可疑」就下结论为代码错误；先区分「代码错误」与「训练配置差异」两类原因，再分别给证据。

---

## 1. 任务固定的硬约束（不可违反）

- 任务是 **2021 Haiti 地震后 pixel-wise SAR 滑坡分割**，不是几何畸变 mask 分割。
- 部署期只允许：target 轨 `pre` VV/VH、`post` VV/VH、以及已知的 target orbit indicator。每个轨道在工程内是 5 通道 `[pre VV, pre VH, post VV, post VH, orbit]`。
- 训练期 Teacher 额外可用 counter 轨的 pre/post VV/VH。
- `geometry mask` 只能用于 Ours 的训练加权（selective KD 的 gain/protect）和所有方法的 G10 分组评价，**不能进入任何模型的前向输入**。
- G10 = target distorted、counter reliable 的像素分组，计数 1,039,406，绝不能从有效评价像素中删除。它是核对任何新脚本分组口径是否与冻结协议同源的指纹。
- 统一评价协议：空间位置 1,713；1,370 train / 343 test；seed=42 从 train 抽 137 作内部验证，剩 1,233 train；测试集为 686 个 orbit-specific 样本（343 ASC-target + 343 DESC-target）；固定阈值 0.5；主指标 IoU/F1/AUPRC，并按 overall / ASC / DESC × G00 / G01 / G10 / G11 分区报告。
- **不得用测试集选择 checkpoint、阈值、模型名单或超参数。**

---

## 2. 现有模型版本谱系

### 2.1 两代协议

本项目存在两套并存的训练协议，**跨代数字不能直接并列比较**：

| 代次 | 目录 | 优化器 | 训练数据 | 骨干初始化 | 备注 |
|---|---|---|---|---|---|
| V1 协议 | `experiments/rapid_landslide_cocd/` | AdamW 1e-4, wd 1e-4 | `train_modes=True`（含合成模态缺失，数据量约 3×） | ConvNeXt-Tiny ImageNet1K | 生成 S0 / old T / Vanilla KD / 旧 DIS2-style / Ours-V1 |
| V2–V3 协议 | `experiments/ours_v2/` | AdamW 1e-4, wd 1e-4 | `train_modes=True` | ConvNeXt-Tiny ImageNet1K | **Teacher 为 batch 2**（启动命令未传 `--batch`，取默认值；见 `DISTORTION_HANDOFF.md` 第 57 行）；R1/R2/R3/DIS2-port 为 batch 8 |
| 外部席位协议 | `experiments/grsl_external_baselines/` | 各方法自定 | 各方法自定 | 各方法自定 | 见 2.3 |

### 2.2 内部版本

| 编号 | 名称 | 代码位置 | 结构要点 | 参数量 | 状态 |
|---|---|---|---|---|---|
| — | S0 | 脚本 21（V1） | ConvNeXt-Tiny/FPN 单轨，无修正、无蒸馏 | 28.080M | 已训练 |
| — | old T | 脚本 21（V1） | 双轨 Teacher | 28.124M | 已训练 |
| — | Vanilla KD | 脚本 21（V1） | 普通像素级 KD | 28.449M | 已训练（旧协议，不是 Ours-V2 的对照） |
| — | 旧 DIS2-style | 脚本 21，`kind='DIS2'` | **复用 Ours 的 `SingleOrbitStudent(correction=True)` 骨架**，额外加 `0.1·dis2_style_loss + 0.1·MSE(sigmoid(z_s), sigmoid(z_dual))` | ≈28.1M | 已训练（**不是官方 DIS2**） |
| — | Ours-V1 | 脚本 21（V1） | 单轨 Student + 修正头 | ≈28.1M | 已训练 |
| — | Ours-V2/V3 Teacher | `models/landslide_cocd_v2.py` + 脚本 `23_train_ours_v2.py` | 共享 ConvNeXt-Tiny/FPN 编码 target 与 counter，在 P3/P4 产生 counter-dependent correction，并显式减去 zero-counter reference | 28.124M | 已训练（`teacher.pt`，`complete=True`） |
| R0 | 无修正无 KD | `OursV3Student('R0')` | 恒等修正 | 28.0804M | 保留在 `CONFIGS`，未跑 |
| R1 | 自由残差头（= 旧 Ours-V2 Student） | `OursV3Student('R1')` | `ĉ` 来自自身 `q_k`，从零学残差 | 28.1156M | 已训练 |
| R2 | R1 + 继承算子 Ω | `OursV3Student('R2')` | `ĉ = Q_k(a_k)`，修正算子 Ω 从 Teacher 复制 | 28.2715M | 已训练 |
| R3 | R2 + P3 递进驱动 | `OursV3Student('R3')` | P3 的驱动输入由「未修正的 p3」换成「已修正的 a3」 | 28.2715M | 已训练 |
| R4 | R3 去输出 KD | `OursV3Student('R4')` | 结构与 R3 相同，去掉输出 KD | 28.2715M | 保留在 `CONFIGS`，未跑 |
| DIS2-port | 官方 DIS2 蒸馏规则 + 我们的骨架 | `OursV3Student('R1')` + `losses/distill.py::dis2_multilevel_kd` | 四级池化融合特征 L2 + penultimate L2 + T=2 logits KL + 正交项，教师侧 detach | 28.1156M | 已训练（2026-09-17，`DIS2_seed42.pt`，50/50） |

已废弃并删除：A/B/C/D 四个「权重侧重」臂（只跑完 A 的 2 个 epoch）。设计文档：`docs/STUDENT_V3_STRUCTURE_DESIGN.md`（第 4 节消融阶梯、第 7 节预设判定、第 8 节拍板记录）。

**参数计数陷阱（必须在所有表格中按此口径）**：`FPNStateDecoder` 持有同一个 backbone 对象引用，因此 `state_dict()` 里 `decoder.backbone.*` 是 `backbone.*` 的镜像键。从 `state_dict()` 求和会得到 2 倍参数（Teacher 会误报 56.204M）。论文与表格必须按去重后的 `model.parameters()` 计。

### 2.3 外部席位

| 角色 | 名称 | 部署参数量 | 状态 |
|---|---|---:|---|
| 直接 SAR 滑坡分割 | Boehm et al. SAR U-Net++ Haiti-adapted | 26.085M | 已训练 |
| 直接滑坡分割 | CDNetE Early-Fusion adapted | 24.443M | 已训练（无官方 release，必须写 adapted） |
| 直接滑坡分割 | MFEWF-light adapted | 25.945M | 已训练（必须写 adapted） |
| 通用蒸馏 | Vanilla pixel-wise KD | 28.449M | 已训练（旧 V1 协议） |
| 缺失观测补偿蒸馏 | **DIS2-port**（官方蒸馏规则 + 共享骨干，即 2.2 表末行） | 28.116M | **已训练 50/50，IoU 0.6272，当前采用** |
| 缺失观测补偿蒸馏（原席位） | 官方 DIS2 width-50 Haiti-adapted | 完整训练 26.586M / target-only 部署 10.573M | **已删除**（2026-09-17，未收敛，IoU 0.1829，数字仅存记录） |

**关于 DIS2 席位的两次变更**：原计划由「官方 width-50 Haiti-adapted」占据该席位，但它按官方完整配方从零训练在 1,233 行真实 pair 上不收敛（test IoU 0.1829）。改为只搬**蒸馏规则**、接到共享骨干与统一协议上，得到 `DIS2-port`（test IoU 0.6272），该席位由它承担；原实现与其产物、脚本、模型 wrapper 一并删除。

明确排除：`FC-Siam-diff`（不训练、不放主表）；`ProtoKD`（已删除，不恢复）；`rapid_landslide_cocd/DIS2.*`（**不得改名占据 DIS2 席位**，它与 DIS2 的蒸馏规则不同）；S0 / old T / Ours-V1 是内部基准；旧六种畸变分割实验是错误任务。

---

## 3. 结果对比

以下全部为固定阈值 0.5 的统一口径。测试集 686 样本（686 × 128 × 128 = 11,239,424 像素）；正类像素占比实测 **7.2439%**；G10 区域 1,039,406 像素（与冻结协议指纹一致）；desc/G11 区域 50,988 像素。

### 3.1 测试集主表

| 方法 | overall IoU | overall F1 | overall AUPRC | G10 IoU | G10 F1 | G10 AUPRC | desc/G11 IoU | desc/G11 F1 | desc/G11 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0（V1，内部） | 0.5302 | 0.6930 | 0.8391 | — | 0.7009 | 0.8455 | — | — | — |
| old T（V1，内部） | 0.5795 | 0.7338 | 0.8710 | — | 0.7437 | 0.8832 | — | — | — |
| Vanilla KD | 0.5541 | 0.7131 | 0.8545 | — | 0.7193 | 0.8590 | — | — | — |
| 旧 DIS2-style | 0.5662 | 0.7230 | 0.8620 | — | — | — | — | — | — |
| Ours-V1 | 0.5619 | 0.7195 | 0.8589 | — | 0.7333 | 0.8714 | — | — | — |
| Boehm SAR U-Net++ | 0.5926 | 0.7442 | 0.8363 | — | 0.7274 | 0.8368 | — | — | — |
| CDNetE Early-Fusion | 0.4927 | 0.6601 | 0.7715 | — | 0.6550 | 0.7760 | — | — | — |
| MFEWF-light | 0.3232 | 0.4886 | 0.6583 | — | 0.4278 | 0.6442 | — | — | — |
| 官方 DIS2 width-50 † | 0.1829 | 0.3092 | 0.3732 | 0.1298 | 0.2297 | 0.3604 | 0.3047 | 0.4671 | 0.4937 |
| Teacher_self（单轨起点） | 0.6006 | 0.7505 | 0.8820 | 0.6053 | 0.7541 | 0.8865 | 0.6674 | 0.8005 | 0.9131 |
| Teacher_dual（双轨上界） | 0.6250 | 0.7692 | 0.8972 | 0.6320 | 0.7745 | 0.9038 | 0.6861 | 0.8138 | 0.9263 |
| **R1** | 0.6211 | 0.7663 | 0.8924 | 0.6251 | 0.7693 | 0.8978 | 0.6871 | 0.8145 | 0.9304 |
| **R2** | 0.6194 | 0.7650 | 0.8932 | 0.6252 | 0.7694 | 0.8993 | 0.6883 | 0.8154 | 0.9248 |
| **R3** | **0.6298** | **0.7728** | **0.8972** | **0.6372** | **0.7784** | **0.9030** | 0.6946 | 0.8198 | 0.9258 |
| **DIS2-port** | 0.6272 | 0.7709 | 0.8963 | 0.6335 | 0.7756 | 0.9023 | **0.6929** | 0.8186 | **0.9289** |

† 该实现已删除，数字保留为历史记录，**不应再出现在任何正式结果表里**。

注意：旧 DIS2-style / S0 / old T / Vanilla KD / Ours-V1 属 V1 协议、batch=2；R1/R2/R3/DIS2-port 属 V2–V3 协议、batch=8。两代不严格并列。
R1/R2/R3 开启早停（实际停于 ep20 / ep30 / ep50），DIS2-port 关闭早停跑满 50 epoch —— **只有 R3 与 DIS2-port 训练量对等**。

#### batch 2 / 8 的实际代价（步数口径）

训练集 `HaitiPairs(tid, train_modes=True)` 把 1,233 行展开为 3 个模态 → **3,699 样本/epoch**。lr 与 weight decay 在 Teacher（脚本第 198 行）与学生（第 256 行）上完全相同，都是 `AdamW(lr=1e-4, wd=1e-4)`，**未按 batch 做线性缩放**。因此在同为 50 epoch 的名义日程下：

| 对象 | batch | batch/epoch | 优化步数（50 ep） |
|---|---:|---:|---:|
| Teacher（`teacher.pt`，`teacher_progress.json` epoch=50） | 2 | 1,850 | **92,500** |
| R1 / R2 / R3 / DIS2-port | 8 | 463（日志中 `0/463` 与之一致） | **23,150** |

即 batch 8 每 epoch 只走 1/4 的更新、整轮少 4 倍步数。**「batch 8 白送分」的假设方向是反的**：固定 lr 下 2→8 减少的是每 epoch 的优化推进量。

同时，batch 与「起点 / 监督信号」在现有数据里**不可分离**，因为每个 batch-8 臂都是从 batch-2 的 Teacher 权重初始化的（`train_student` 第 248 行 `load_teacher_self`，第 252–253 行断言「修正量为零时 z34 逐位等于 Teacher self path」），所以学生读数 = **teacher 权重 + 更多训练 + KD 软目标**，而不是「batch 8 训出来的学生 vs batch 8 训出来的 teacher」。

### 3.2 三个学生臂的归因

| 对照 | ΔIoU (overall) | ΔAUPRC (overall) | 实际训练 epoch |
|---|---:|---:|---|
| R1 − Teacher_self | +0.0205 | +0.0104 | R1 停于 ep20（best@10） |
| R2 − R1（继承 Ω 的贡献） | −0.0017 | +0.0008 | R2 停于 ep30（best@20） |
| R3 − R2（递进的贡献） | +0.0104 | +0.0040 | R3 跑满 ep50（best@45） |
| R3 − Teacher_dual | +0.0048 | ±0.0001 | — |

**已知口径问题（必须在分析中处理）**：脚本 `23_train_ours_v2.py::train_student` 内置早停 `bad >= 2`（每 5 epoch 评估一次 val AUPRC，连续两次无提升即停），因此三臂实际训练量不同（20 / 30 / 50 epoch）。R3 的训练量是 R1 的 2.5 倍，其 +0.0104 中有多少来自结构、多少来自多训，当前数据无法分离。

### 3.3 三个 DIS2 产物的逐项对照（第 5 节 Q1 的事实基础）

| 维度 | 旧 DIS2-style | 官方 DIS2 width-50 † | DIS2-port |
|---|---|---|---|
| 目录 | `experiments/rapid_landslide_cocd/DIS2.pt` | ~~`experiments/grsl_external_baselines/dis2_width50_haiti/`~~ | `experiments/ours_v2/DIS2_seed42.pt` |
| 网络 | **Ours 的 `SingleOrbitStudent(correction=True)`** | 官方 `models_bank/DLKD_ver4.py`（`third_party` 快照，未改动） | 我们的 `OursV3Student('R1')` |
| 骨干初始化 | ConvNeXt-Tiny **ImageNet1K 预训练** | **无任何预训练加载路径**（快照内无 `pretrained` / `weights=` 调用），从头训练 | ConvNeXt-Tiny ImageNet1K + 教师自轨权重 |
| **蒸馏对象** | **修正量 `r`（残差）** —— 见下 | 四级融合特征图 + penultimate 特征图 + logits（**激活**） | 同官方（激活），逐字对照 |
| 蒸馏项 | `0.1·(1 − cos〈r_s, r_t〉)`（按 GT×教师置信度加权）+ `0.1·MSE(σ(z_s), σ(z_dual))` | 官方 8 项复合损失：`full + rgb + ndsm + missing + full_scale + miss_scale + diversity + (epoch>5 ? 5·KL : 0)`；KL 为 T=2 | `0.5·seg(z4) + 0.5·seg(z34)` + 四级 L2（池化）+ penultimate L2 + T=2 logits KL + 正交项 |
| 训练数据 | `train_modes=True`（含合成模态缺失，约 3×） | `train_modes=False`（**只训真实 pre/post pair**，1,233 行/epoch，约为 Ours 的 1/3） | `train_modes=True`（与其他臂统一） |
| 优化器 | AdamW 1e-4, wd 1e-4 | Adam 5e-4（无 wd） | AdamW 1e-4, wd 1e-4 |
| batch | 2 | 2 | 8 |
| 宽度 | 沿用 Ours 宽度 | `gf_dim=50`（官方默认 32，部署仅 4.450M） | 沿用 Ours 宽度 |
| 早停 | 无 | 有（连续两次无提升） | **关闭**，跑满 50 epoch |
| 验证 AUPRC | **0.8545** | **0.3858**（ep50 时仍在上升） | **0.8932**（ep30 取 best） |
| 测试 IoU / F1 / AUPRC | 0.5662 / 0.7230 / 0.8620 | 0.1829 / 0.3092 / 0.3732 | **0.6272 / 0.7709 / 0.8963** |

† 已删除（2026-09-17，产物 + 脚本 + 模型 wrapper 移入系统废纸篓），数据保留为记录。

**两条需要分析方注意的事实**：

1. **旧 DIS2-style 匹配的是「修正量」而不是「激活」**（`losses/landslide_cocd.py:16`，`dis2_style_loss(student_r, teacher_r, …)` 的两个操作数是学生残差 `rs` 与教师双轨残差 `rt`）。而官方规则匹配激活。这与 `docs/DIS2_DISTILLATION_PORT.md` §3 对官方的批评（「蒸馏激活而非变换，目标不可达」）方向一致。[推断] 旧版的高分可能部分来自这一巧合，但两版协议不同（batch 2 vs 8）、旧版权重仅 0.1，**此推断待验证**。
2. **官方 width-50 的失败与蒸馏思想无关**：把同一套蒸馏规则接到我们的骨干与统一协议上（DIS2-port）即恢复到 0.6272，与同口径 R3（0.6298）只差 0.0026。瓶颈在配方与条件（无预训练、1/3 数据、8 个辅助头分走 77% 梯度），不在蒸馏机制。[已证实]

官方 DIS2 训练日志（`31_run_ours_v3_ablation.sh` 的等待段读出）显示 val AUPRC 单调上升：ep21 0.2708 → ep27 0.2996 → ep32 0.3260 → ep38 0.3260 → ep44 0.3381 → ep49 0.3613 → ep50 0.3858。

### 3.4 官方 DIS2 的损失分解（当时实测；诊断脚本已随实现删除，数字保留）

> 原始诊断脚本 `scripts/34_diagnose_dis2_loss.py` 与它读取的 checkpoint 均已删除，下表数字**不可重新生成**，作为当时的证据保留。

| 项 | 占总损失 | 说明 |
|---|---:|---|
| 8 个多尺度辅助头（`full_scale` 27.0% + `miss_scale` 28.8%） | **55.8%** | 官方 loss 的一部分 |
| rgb / ndsm / full 三个**非部署**头 | 21.4% | 部署时不调用 |
| 5 × KL（warm-up 后） | 16.1% | **不是主因**，纠正了早期「KL 压倒一切」的猜测 |
| `missing_pred`（部署分支自身监督） | **6.7%** | 唯一直接监督部署分支的项 |
| diversity（正交解耦） | 0.0% | 见下 |

分割项内部：Dice 占 **93.7%**（Dice 合计 5.547 vs 加权 NLL 合计 0.373）。NLL 数值小是因为官方对它取了 `mean`（同时除以 batch 与 C·H·W），不代表没在用。

逐项 50 epoch 轨迹：`full_loss` −39.2%（完整分支**确实在学**）、`missing_loss` −11.4%、`kl_loss` −11.6%（**蒸馏本身没收敛**）、`full_scale_loss` −23.5%、`miss_scale_loss` −9.9%、`diversity_loss` −98.4%（**头几个 epoch 就塌到 0.0001，正交约束在此数据尺度上近乎空转**）、总损失 −1.7%。

部署分支的输出尺度（测试集）：正类通道 mean 0.0487、正类像素中位数 0.106、负类像素中位数 0.0009。验证集上正类通道中位数仅 0.0028，>0.5 的像素只占 1.61%。阈值扫描 t=0.3 时 IoU 也只有 0.226 → **不是阈值/校准问题，是排序能力本身弱**。

已确认不是 bug 的两处，不要重复排查：

- `softmax_weighted_loss` 内部是 `log(clamp(output, min=0.005))`，即它期望**概率**；解码器输出的确实是概率（实测 `missing_pred` 沿通道求和恒为 1，`ClassWiseDecoder` 返回 `softmax(final_seg_logits)`）。头与损失自洽，**不存在 logits/概率混用**。
- `DLKD_ver4.py` 第 574 行 `F.kl_div(..., reduction='mean')` 是官方写法（PyTorch 会警告 `mean` 同时除以 batch 与 support，数学上对应 `batchmean`）。我们逐字复制，副作用是 KL 被除以 `B·C·H·W`、量级被压小。属「官方怪癖被忠实复现」，改它必须同时重调那个 5×。

---

## 4. Teacher 与 Student 的任务分工

### 4.1 各自看到什么

| | Teacher | Student / R1 / R2 / R3 |
|---|---|---|
| 输入 | target 轨 5 通道 + counter 轨 5 通道 | **仅** target 轨 5 通道 |
| 输出 | `z0`（self path，零反轨参照）、`z4`（仅 P4 修正）、`z34`（P4+P3 递进修正） | `z4`、`z34`（部署输出） |
| 角色 | 提供双轨上界与监督信号（privileged information） | 部署模型，必须单轨运行 |
| geometry mask | 不进入前向 | 不进入前向；只在 selective KD 里作权重 |

### 4.2 蒸馏机制

- Teacher 的修正头被当作**可继承的算子**：`Ω_k(a, c) = a_k(cat(a, c)) − a_k(cat(a, 0))`，满足 `Ω_k(a, 0) = 0`。R2/R3 把它复制给 Student，Student 只从目标轨状态学一个「修正驱动」`Q_k(a_k)`（3×3，128→32→128，末层零初始化）去喂它。
- Student 初始化：backbone + decoder + Ω3/Ω4 全部复制自 Teacher，`Q3/Q4` 零初始化。实测不变量：`Ω_k(a,0)` 的最大绝对值 0.000e+00；Student 初始输出与 Teacher self path 的最大绝对差 0.000e+00。
- 目标函数：`L = 0.5·(L_seg(z4) + L_seg(z34)) + 0.5·(KD4 + KD34)`，KD 为 selective KL（`gain × protect` 加权，T=1 的 Bernoulli KL，`gain` 含 `(1+g10)`）。
- 不存在显式 change feature、不做特征 MSE / relational KD / 梯度匹配；这些是 DIS2 已做过的。

### 4.3 LUPI 性质（分析时必须考虑）

这是 Learning Using Privileged Information：特权信息（counter 轨）**不可传递到测试期**。Student 的上界是 `E[r | T]`（在只有 target 的条件下能达到的最优判别），而不是 Teacher 的双轨判别本身。因此「Student 能否复现 Teacher 的互补视角」是一个病态问题的表述——单轨在物理上无法复原被相干斑和侧视几何破坏的信息，降维解决不了信息缺失。可检验的主张只能是：Student 在**自身可得信息**的约束下，产生能通过 Ω 改善判别的驱动。

---

## 5. 需要重点分析并解释的三个问题

### Q1 — 为什么之前训练过的 DIS2 版本当时效果特别好，现在却突然出现错误？（**主体已解决，见下**）

**已解决的部分（2026-09-17，不需要再分析）**：

1. 「以前好」的那个**不是官方 DIS2**，是旧 DIS2-style：复用我们的 `SingleOrbitStudent(correction=True)`，蒸馏项是自写的残差 cosine 匹配 + 输出 MSE（V1 协议，test IoU 0.5662）。按既定决定它**没有资格占据 DIS2 席位**，因为蒸馏规则与官方不同。[已证实]
2. val AUPRC 0.47 的差距**不是代码错误**：属配置差异，且已由实验证实——把官方**蒸馏规则**（而非整套配方）接到共享骨干与统一协议上，得到 DIS2-port test IoU 0.6272，与同口径 R3（0.6298）只差 0.0026。瓶颈在「无预训练 + 1/3 数据 + 8 个辅助头分走 77% 梯度」。[已证实]
3. 「把 DIS2 机制搬到共享 ConvNeXt 骨干」这条路线的可行性问题**已实施完成**：需要显式重建的中间量（4 级融合特征、penultimate 特征、logits）由 `models/landslide_cocd_v2.py::forward_taps` 提供，结构与集成断言共 63 + 15 项通过。[已证实]
4. 「补跑 or 如实披露未收敛」的抉择**已作废**：原实现、脚本、产物已按用户指示删除，席位改由 DIS2-port 承担。[已证实]

**仍需你分析的部分**：

1. **旧 DIS2-style 与 DIS2-port 的 0.0610 IoU 差距，该归因于「匹配目标不同（修正量 vs 激活）」还是「协议不同（batch 2 vs 8）」？** 两者同时变化，现有数据无法分离。给出可执行的拆分方案（例如：在同为 batch 8、同 50 epoch 下，把 DIS2-port 的 `feat`/`pen` 两项换成残差匹配 `‖r_s − r_t‖²`，只改这一处）。
2. **正交项 `diversity_loss` 在两个实现上都塌到 ~0**（官方 ep50 = `9.31e-05`；DIS2-port 在 ep3 后 `0.0000` 并保持到 ep50）。请判断：该项在本数据集上应保留、删除，还是重新标定权重？它对论文叙事（「为什么需要补偿而不只是复制」）会造成什么影响？注意这是**跨两个独立实现复现出的同一现象**，不是实现缺陷。[已证实：现象；待判断：处置]
3. **DIS2-port 与 R3 的比较能否支撑结论？** 两者训练量对等（同为 50 epoch、batch 8、无早停），在所有分区上只差 0.0004–0.0047 且交替领先（R3 在 G10 略好、DIS2-port 在 G11 AUPRC 略好）。二者唯一差异是蒸馏规则（R3 = 输出级选择性 KL，DIS2-port = 官方多级蒸馏）。判断这个差异是否足以支撑「多级蒸馏优于输出级 KD」，若不足，给出最小实验集。

### Q2 — 我们自己提出的方案为什么效果很强，甚至比 Teacher 还强？

已知事实：R3 overall IoU 0.6298 > Teacher_dual 0.6250；AUPRC 0.8972 vs 0.8972（相等）；G10 IoU 0.6372 > 0.6320；desc/G11 IoU 0.6946 > 0.6861。要求你回答：

1. 把「Student 超过 Teacher」这一现象拆成候选机制，逐条给证据：
   - Student 从 Teacher self path 初始化（起点 IoU 已有 0.6006），因此它不是在学一个从零的任务；
   - KD 的 soft target 把双轨判断压进单轨映射，可能起正则化作用；
   - Teacher 的 self path 本身是在 dual 监督下拉过的，信息可能已经前置到 self；
   - Student 新增容量仅 147,776 参数且零初始化，容量受限可能降低过拟合；
   - 输出级 KD 的 ensemble 效应。
2. 明确回答：这个「超过」在统计上是否站得住？必须考虑：单种子、差异 0.0048 IoU、R3 训练量是 R1 的 2.5 倍、val 曲线在 0.8896–0.8940 间平坦波动（best@45 与周边差距近噪声）。
3. 主动检查评价侧是否存在以下问题，并给结论：测试集是否被用于任何选择、三个臂与 Teacher 是否共用同一切分与同一 `metric()`、`report()` 修复后是否有分支遗漏。
4. 给出把「超过 Teacher」从观察升级为结论所必需的最小实验集。

### Q3 — 之前某些训练结果中出现极高指标，尤其是 Teacher 达到 0.9，这些版本之间到底是怎么回事？

已知事实：`teacher_progress.json` 的 `best_val_auprc = 0.8962639`（这是**内部验证集** 137 样本的口径）；同一 Teacher 在**测试集**上的 overall AUPRC 是 self 0.8820 / dual 0.8972，而 overall IoU 只有 0.6006 / 0.6250，recall≈0.946、precision≈0.622。V1 协议里最高的 AUPRC 是 old T 的 0.8710，没有任何内部模型在 V1 口径下达到 0.9。要求你回答：

1. **做一张数字溯源表**：把项目中所有出现过的高指标（AUPRC ≥ 0.9 级别）逐个映射到「来源文件 + partition（val/test）+ 分支（self/dual/missing）+ checkpoint 选择规则」。
2. 解释 IoU 0.60 与 AUPRC 0.88–0.90 为什么可以同时成立：说明 IoU 受固定阈值影响、AUPRC 是阈值无关的排序指标，并结合 recall 0.946 / precision 0.622 的过预测倾向与正类像素占比 7.2439% 给出定量说明。
3. 列出会导致「指标虚高」的口径混用清单，至少覆盖：val AUPRC 被当成 test AUPRC；self 分支与 dual 分支混列；V1 的 `binary_metric(F1@0.5)` 与 AUPRC 混列；`report()` 修复前 `teacher_validation.csv` 走 `metrics()` 直调而未暴露 bug 的历史；`all` 阶段复用已完成 Teacher 导致 `teacher_validation.csv` 不被重写（该文件当前不存在，验证指标只存在于日志与 `teacher_progress.json`）。
4. 明确指出哪些数字是可以直接进论文的，哪些必须重算或删除。

---

## 6. 已确认的事实与已排除项（不要重复排查）

### 6.1 已修复的代码错误

- `scripts/23_train_ours_v2.py::report()` 曾把逐像素的 `gt/gc` 掩膜当作逐样本标量使用，导致它**从未被真实数据跑通过**。正确构造是：分区掩膜先 `np.broadcast_to(sel[:, None, None], pred['p'].shape).ravel()`，再与逐像素区域掩膜相与；空区域跳过；区域内无正样本像素时 auprc 记 nan（不是 0.0）。
- 该 bug 曾让官方 DIS2 的收尾评估段崩溃（训练 50/50 完好、权重落盘）。当时用 `scripts/33_eval_dis2_width50.py` 从 checkpoint 补出了测试指标，**未重训**；该脚本与它补出的产物现已一并删除。
- 后续的 `DIS2-port` 那一轮（50/50，2026-09-17）**未再触发该 bug**，评估段正常收尾，这是修复的第二次真实数据验证。
- 回归验收标准：用新实现重算 Teacher 测试预测，与 `teacher_test_verify.csv` 逐位比对 —— 当前为 **30/30 行全等、最大偏差 1.1e-16**。
- 结构验证：`scripts/32_smoke_ours_v3_ablation.py`（CPU，25 项断言）当前 25/25 通过，含 `Ω(a,0)=0`、R2/R3 在驱动权重相同时 `z34` 不同而 `z4` 相同。

**结论：DIS2 测试指标低不是 `report()` 的 bug 造成的**（脚本 33 补出的数字与训练段一致）。这个 bug 只解释「评估段崩溃」，不解释「分数崩」。

### 6.2 环境与运维

- 可用 venv：`/Users/zhangjiuqi/Desktop/联合分类/.venv/bin/python`（Python 3.12.14、torch 2.13.0、MPS 可用）。历史路径 `联合分类论文/.venv` 已失效。
- 本机 Mac17,4，10 核 CPU，24GB 统一内存。数据管道保持 `num_workers=0`。
- 已知无效的加速手段：`torch.compile`（MPS 上更慢）、`channels_last`（更慢）、fp16 autocast（约 6% 收益且引入数值漂移）、teacher 输出预缓存（无收益）。有效手段只有 loader batch：2→8 约 1.5×，8 之后无收益。
- 对比不同配置必须在同一进程内交错测量并重复多轮取中位数。
- 长任务不要用 `nohup ... &`；本机沙箱内 `ps`/`top` 被拒，枚举进程用 libproc（`proc_listpids` + `proc_pidpath`）。

---

## 7. 请你输出的报告结构

```
一、版本谱系勘误
   （确认三个 DIS2 相关产物互不相同；指出「旧 DIS2-style」用的是自写残差匹配损失、
     不是官方规则，因此不占 DIS2 席位）

二、DIS2 已解决项与遗留项（Q1）
   2.1 已解决的四点：逐条复述并复核证据等级
   2.2 遗留一：旧版与 DIS2-port 的 0.0610 IoU 差距 —— 匹配目标（修正量 vs 激活）
       与协议（batch 2 vs 8）同时变化，给出可执行的拆分方案
   2.3 遗留二：正交项在两个独立实现上都塌到 ~0 —— 保留 / 删除 / 重标定，及对论文叙事的影响
   2.4 遗留三：DIS2-port 与 R3 的差异能否支撑「多级蒸馏优于输出级 KD」+ 最小实验集

三、内部方案强度分析（Q2）
   3.1 候选机制表：机制 / 支持证据 / 反证 / 判定
   3.2 统计可靠性判定（能否写进论文）
   3.3 最小补实验集

四、高指标溯源（Q3）
   4.1 数字溯源表：数值 / 来源文件 / partition / 分支 / 选模规则 / 可否进论文
   4.2 IoU 与 AUPRC 的量纲解释（含正类像素占比的实测值）
   4.3 口径混用清单与整改项

五、风险清单
   （按「影响结论」的程度排序，每条给出一句话的处置建议）
```

每条结论后面必须跟证据等级标记 `[已证实]` / `[推断]` / `[假设]`。

---

## 8. 关键文件与命令清单

| 用途 | 路径 |
|---|---|
| 冻结评价协议与数据集 | `Haiti_SAR_GRSL_Audit/scripts/21_train_rapid_landslide_cocd.py` |
| Ours Teacher/Student 训练与 `report()` | `Haiti_SAR_GRSL_Audit/scripts/23_train_ours_v2.py` |
| Ours 模型（含 Ω、Driver、`OursV3Student`、`forward_taps`） | `Haiti_SAR_GRSL_Audit/models/landslide_cocd_v2.py` |
| 五通道骨干（ImageNet 初始化） | `Haiti_SAR_GRSL_Audit/models/landslide_cocd.py` |
| **DIS2 蒸馏规则的迁移实现** | `Haiti_SAR_GRSL_Audit/losses/distill.py` |
| DIS2 迁移的结构 / 集成冒烟（63 + 15 项断言） | `scripts/36_smoke_dis2_distill_ours.py` / `scripts/37_smoke_distill_integration.py` |
| DIS2-port 启动脚本 | `scripts/38_run_dis2_port.sh` |
| **DIS2 三版结果整理与对比（本轮主文档）** | `Haiti_SAR_GRSL_Audit/docs/DIS2_THREE_VERSIONS_RESULT.md` |
| DIS2 蒸馏规则迁移设计（含对官方的四点批评） | `Haiti_SAR_GRSL_Audit/docs/DIS2_DISTILLATION_PORT.md` |
| DIS2 机制与崩因诊断（损失分解证据） | `Haiti_SAR_GRSL_Audit/docs/DIS2_MECHANISM_AND_DIAGNOSIS.md` |
| 官方 DIS2 快照（只读，迁移规则的逐字依据） | `Haiti_SAR_GRSL_Audit/third_party/landslide_baselines/dis2/` |
| 消融排队 / 结构验证 | `scripts/31_run_ours_v3_ablation.sh` / `scripts/32_smoke_ours_v3_ablation.py` |
| 结构设计文档（含第 7 节诚实边界） | `Haiti_SAR_GRSL_Audit/docs/STUDENT_V3_STRUCTURE_DESIGN.md` |
| 消融结果汇总 | `Haiti_SAR_GRSL_Audit/experiments/ours_v2/ABLATION_R1R2R3_SUMMARY.md` |
| 逐位回归基准 | `Haiti_SAR_GRSL_Audit/experiments/ours_v2/teacher_test_verify.csv` |
| ~~官方 DIS2 Haiti wrapper / 训练 / 补评估 / 损失分解~~ | 已删除：`models/dis2_haiti.py`、`scripts/29`、`30`、`33`、`34` |

常用核查命令：

```bash
PY='/Users/zhangjiuqi/Desktop/联合分类/.venv/bin/python'
cd /Users/zhangjiuqi/Desktop/distortion
O=Haiti_SAR_GRSL_Audit/experiments/ours_v2

# Teacher 选模指标与完成状态
cat $O/teacher_progress.json

# 全部方法的结果（含 R1/R2/R3 与 DIS2-port 的 overall 行）
cat $O/results_seed42.csv
ls $O/results_*.csv

# 三个 DIS2 的测试指标
cat Haiti_SAR_GRSL_Audit/experiments/rapid_landslide_cocd/DIS2_metrics.csv   # 旧 DIS2-style
cat $O/results_DIS2_seed42.csv                                              # DIS2-port
# 官方 width-50 的产物已删除，数字只在 DIS2_THREE_VERSIONS_RESULT.md 里

# DIS2-port 的验证轨迹与四项蒸馏分项（feat / pen / logit / div）
grep -a "\[DIS2\]" $O/train_log.txt | tail -60

# 三臂真实的停止 epoch（不是 --epochs）
grep -a "validation epoch=" $O/train_log.txt

# 正类像素占比、G10 计数与测试集几何（用保留的 V1 预测缓存；同一 split）
$PY - <<'EOF'
import numpy as np
z = np.load('Haiti_SAR_GRSL_Audit/experiments/rapid_landslide_cocd/T_test_predictions.npz')
y, gt, gc = z['y'], z['gt'], z['gc']
print('samples', y.shape, 'pixels', y.size)
print('positive fraction %.6f' % y.mean())
print('G10 pixels', int(((gt > 0) & (gc == 0)).sum()))
print('ASC/DESC counts', np.unique(z['orbit'], return_counts=True))
EOF

# 结构不变量的再验证（CPU，不占 MPS）
$PY Haiti_SAR_GRSL_Audit/scripts/32_smoke_ours_v3_ablation.py
$PY Haiti_SAR_GRSL_Audit/scripts/36_smoke_dis2_distill_ours.py
```

---

## 9. 现在请你开始

先回答 Q1（DIS2：复核已解决项、处理三个遗留问题），再回答 Q2（内部方案强度），最后回答 Q3（高指标溯源），按第 7 节的结构输出。若某个结论依赖本文档未给出的数字，不要估算——给出应执行的命令并说明该数字会如何影响结论。
