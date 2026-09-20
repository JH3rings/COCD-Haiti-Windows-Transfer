# DIS2 的核心思路、官方配方崩在哪里、以及怎么修

> **状态（2026-09-17）**：本文档记录官方 DIS2 的机制，以及那套完整配方搬过来为何不收敛。
> 文中引用的 Haiti 适配实现（`models/dis2_haiti.py`、脚本 `29`/`30`/`33`/`34`）与其实验产物
> **已按用户指示删除**；本文档保留为机制说明与诊断记录，其中的实测数字是当时的证据。
> 后续进展见 `docs/DIS2_THREE_VERSIONS_RESULT.md`：把**蒸馏规则**（而非整套配方）接到我们的骨干上，
> 得到 `DIS2-port` test IoU 0.6272，说明瓶颈在配方与条件而不在蒸馏思想本身。

> 依据：`third_party/landslide_baselines/dis2/models_bank/DLKD_ver4.py`（655 行）、
> `models_bank/criterion_rs.py`、`run_experiment_35.py`（逐段核对），
> 以及已删除的 `models/dis2_haiti.py`、`scripts/29_train_dis2_width50_haiti.py` 与
> `scripts/34_diagnose_dis2_loss.py`（在已完成的 checkpoint 上做损失分解，只读）。

---

## 0. 一句话

DIS2 的创新不是「又一个蒸馏损失」，而是两件事连在一起：

1. **把「缺一个模态时该怎么推理」做成一个显式的分支**（missing 分支），而不只是让网络在缺失输入上硬跑；
2. **用「两个模态都在」的完整分支去教这个缺失分支** —— 教的是多尺度融合特征、倒数第二层特征图、以及 T=2 的 logits；
3. 再加一个**正交约束**，逼替补编码器去编码「生存模态自己没有的那部分」，而不是把生存模态的信息重编码一遍。

前两条是「补偿 + 蒸馏」，第三条是「为什么要补偿而不只是复制」。这三件事合起来才是它的创新点。

---

## 1. 官方网络在做什么

### 1.1 两个模态槽位、三种观测状态

官方把两个模态叫 `rgb` 和 `ndsm`（在 Potsdam 上是真实 RGB 与 nDSM）。Haiti 映射：

```text
rgb  := target  [pre VV, pre VH, post VV, post VH, target orbit]
ndsm := counter [pre VV, pre VH, post VV, post VH, counter orbit]
```

`forward()` 按一个 mask 分三种状态：`FULL_MODALITY`（两个都在）、`MISS_NDSM`（ndsm 缺）、`MISS_RGB`（rgb 缺）。训练时一次前向**同时**产出四个分割头：

| 输出键 | 含义 |
|---|---|
| `full_pred` | 两个模态都在的分支 |
| `rgb_pred` | 只用 rgb 槽位的分支 |
| `ndsm_pred` | 只用 ndsm 槽位的分支 |
| `missing_pred` | 缺失状态下的分支 ← **部署时用的就是它** |

### 1.2 每个分支内部

`Encoder`（4 级 e1…e4 的 U 形卷积编码器）→ `HiererchicalFusion`（逐尺度把两个模态的 token 拼起来过 Transformer，带 per-scale fusion token 和跨尺度传递）→ `ClassWiseSpatialAttention`（按类别产出 K 张空间注意力图）→ `ClassWiseDecoder`（在每个尺度注入类别注意力，输出 `final_seg`（**概率**）+ 4 张中间尺度概率图 + `final_seg_logits` + `penultimate_feats`）。

### 1.3 缺模态时的「替补」机制（创新点 ①）

当 `MISS_NDSM` 时：

- `rgb_else`：一个**替代编码器**，输入还是那个唯一在场的模态（rgb），产出 4 级替补特征 `stu_feats`；
- `rgb_dist_pool([rgb_e1..e4])`：把在场模态**自己的**特征另外池化一份，记作 `stu_dist_pooled`（distinct）；
- `miss_ndsm_fuse(rgb_feats, stu_feats)`：把「在场模态特征」与「替补特征」融合成缺失分支的完整金字塔；
- 缺失分支有自己的类别注意力 `miss_ndsm_classwise_attention` 和解码器 `miss_ndsm_classwise_decoder`。

所以缺失分支不是「把零填进去硬算」，而是「用一个专门的替代编码器造出一份补偿表征，再与在场证据融合」。

### 1.4 蒸馏怎么发生（创新点 ②）

三段监督，全部以**完整分支为教师且 detach**：

```text
① 多尺度融合特征   sum_i  l2_kd_loss(miss_fused_pooled[i], full_fusion_pooled[i].detach())     # i = 4 个尺度
② 倒数第二层特征   l2_kd_loss(missing_pen_feats, full_pen_feats.detach(), spatial=True)
③ logits（T=2）    KL(log_softmax(z_miss / T), softmax(z_full / T)) * T^2
```

注意 ①② 是**特征级** L2（对 L2 归一化后的向量做 MSE），只有 ③ 是 logits 级 KL。把 DIS2 简化成「一个 feature MSE」或「一个 KL」都会丢掉它的主体。

### 1.5 为什么要正交（创新点 ③）

```text
orthogonality_loss(a, b) = mean( cos(a, b)^2 )      # a = 替补编码器特征池化, b = 在场模态自身特征池化
```

最小化它 = 让两组表征的余弦相似度趋近 0。没有这一项，替补编码器最省力的解法就是把在场模态的信息重新编码一遍，缺失分支就退化成一个普通单模态网络，补偿是假的。

### 1.6 官方总损失（8 项）

```text
total = full_loss + rgb_loss + ndsm_loss + missing_loss
      + (epoch > 5 ? 5.0 * kl_loss : 0)
      + diversity_loss + full_scale_loss + miss_scale_loss
```

每个分割项 = `softmax_weighted_loss`（逐图逆频率加权的 NLL）+ `dice_loss`；`full_scale_loss` / `miss_scale_loss` 各是 4 个中间尺度头的和。

---

## 2. 我们的复现复现了什么

| 创新点 | 官方位置 | 我们的实现 | 是否完整 |
|---|---|---|---|
| ① 替补分支（`rgb_else` + `miss_*_fuse` + 独立注意力/解码器） | `DLKD_ver4.forward` MISS 段 | 原样调用官方快照，零改动 | ✅ 完整 |
| ② 三段蒸馏（4 尺度融合特征 + penultimate + T=2 logits KL） | 同上，`Begin Distillation Loss Calculation` | 原样，`kl_loss` 由官方 forward 内部计算 | ✅ 完整 |
| ③ distinct/supplement 正交解耦 | `orthogonality_loss` | 原样 | ✅ 完整 |
| 8 项损失 + warm-up（epoch>5 时 KL×5） | `run_experiment_35.cal_train_loss` | 逐项复刻于 `models/dis2_haiti.py::cal_train_loss` | ✅ 完整 |
| 官方快照不被修改 | — | `third_party/` 只读，宽度通过模块级 `gf_dim` 参数化 | ✅ |

**结论：创新点部分没有缺项。** 这一点已由 smoke test 冻结：`--smoke` 断言官方返回字典的 8 个键齐全、各头尺寸正确、loss 有限、warm-up 生效、且部署 wrapper **可证明不读 counter 槽位**。

没复现的是外围条件，都写在 `scripts/29_train_dis2_width50_haiti.py` 的 docstring 里：

| 项 | 官方（Potsdam） | 我们 | 归属 |
|---|---|---|---|
| 类别数 | 5 | 2 | 数据适配 |
| batch 组成 | 每个 batch 采样一个场景 | 两个方向（ASC-target / DESC-target）同 batch | 我们自己的选择 |
| 优化器 | SGD 0.99 + cosine warm restarts | Adam 5e-4 固定，无 wd | 简化 |
| 梯度累积 | 2 步 | 无 | 简化 |
| AMP | 有 | 无（MPS） | 平台 |
| 宽度 | `gf_dim=32`（部署 4.450M） | `gf_dim=50`（部署 10.573M） | 容量对齐要求 |
| 训练数据 | 完整数据集 | `train_modes=False`，只有真实 pre/post pair，1,233 行/epoch | 我们的选择 |
| 骨干初始化 | 无预训练 | **无预训练**（快照内没有任何 `pretrained` / `weights=` 调用） | 继承官方 |

---

## 3. 崩在哪里：实测证据

### 3.1 损失构成（`scripts/34_diagnose_dis2_loss.py`，验证集 4×batch2）

| 项 | 占总损失 |
|---|---:|
| 8 个多尺度辅助头（`full_scale` 27.0% + `miss_scale` 28.8%） | **55.8%** |
| rgb / ndsm / full 三个**非部署**头 | 21.4% |
| 5 × KL（warm-up 后） | 16.1% |
| **`missing_pred`（部署分支自身监督）** | **6.7%** |
| diversity | 0.0% |

分割项内部：**Dice 占 93.7%**（Dice 合计 5.547 vs 加权 NLL 合计 0.373）。NLL 的数值小是因为官方对它取了 `mean`（同时除以 batch 与 C·H·W），不是没在用。

这里要修正一个我早期的猜测：**5×KL 不是主因**（只占 16.1%）。真正吃掉梯度预算的是 8 个多尺度辅助头。

### 3.2 逐项随 epoch 的轨迹（`DIS2_history.csv`，50 epoch）

| 项 | ep1 | ep10 | ep30 | ep50 | 50 epoch 降幅 |
|---|---:|---:|---:|---:|---:|
| `full_loss` | 0.5040 | 0.4479 | 0.4237 | **0.3066** | **−39.2%** |
| `missing_loss` | 0.5273 | 0.5053 | 0.4859 | 0.4673 | −11.4% |
| `rgb_loss` | 0.5382 | 0.5058 | 0.5000 | 0.4973 | −7.6% |
| `ndsm_loss` | 0.5209 | 0.5088 | 0.4941 | 0.4793 | −8.0% |
| `kl_loss`（蒸馏本身） | 0.2613 | 0.2033 | 0.1987 | 0.2309 | −11.6% |
| `full_scale_loss` | 2.0758 | 1.9211 | 1.8663 | 1.5875 | −23.5% |
| `miss_scale_loss` | 2.1507 | 2.0700 | 2.0067 | 1.9373 | −9.9% |
| `diversity_loss` | 0.0058 | 0.0006 | 0.0005 | 0.0001 | −98.4% |
| **总损失** | 6.3227 | 6.9763 | 6.7707 | 6.4299 | **−1.7%** |

三条读数：

1. **完整分支确实在学**（`full_loss` −39%），所以网络没坏、梯度通路没断。
2. **部署分支几乎不动**（`missing_loss` −11%），而**蒸馏本身也没收敛**（`kl_loss` −11.6%）——本应由「完整分支教缺失分支」完成的那一步没有发生。
3. **`diversity_loss` 在头几个 epoch 就塌到 0**（−98.4%）。正交约束被轻松满足，说明在这个数据尺度上它几乎没有起约束作用；「distinct/supplement」在这里是空转的。

### 3.3 部署分支的输出尺度

| 口径 | 数值 |
|---|---|
| 验证集正类通道概率 | mean 0.0391，**中位数 0.0028**，>0.5 的像素占 1.61% |
| 测试集 | mean 0.0487，正类像素中位数 0.106，负类像素中位数 0.0009，分离度 0.224 |
| 测试集阈值扫描 | t=0.3 → IoU 0.226 / F1 0.369；t=0.5 → 0.1829 / 0.3092；t=0.7 → 0.1334 / 0.2354 |
| AUPRC | 0.3732，相对正类占比 7.2439% 只有约 5× 提升 |

即使把阈值降到 0.3，IoU 也只有 0.226 —— **这不是阈值/校准问题，是排序能力本身就弱**。同时正类占比 7.24% 意味着这个任务并不极端不平衡，所以「类别极不平衡导致学不动」也不是主因。

### 3.4 官方快照里两处「可疑但忠实」的地方

- `DLKD_ver4.py` 第 574 行 `F.kl_div(..., reduction='mean')`：PyTorch 会警告 `mean` 同时除以 batch 与 support，数学上应对应 `batchmean`。官方如此，我们逐字复制。**副作用**：KL 被除以 `B·C·H·W`，量级被显著压小（这也是 5×KL 只占 16% 的原因之一）。改它必须同时重调那个 5×。
- `dice_loss` 与 `softmax_weighted_loss` 都按**概率**实现（后者内部是 `log(clamp(output, min=0.005))`），而解码器输出的确实是概率（实测 `missing_pred` 沿通道求和恒为 1）。所以这里自洽，**不是 logits/概率混用**。

### 3.5 已经明确排除的可能（不要再往这些方向查）

- ❌ logits / 概率 混用：见 3.4 第二条。
- ❌ 标签通道顺序错：`expand_target` 是 one-hot，通道 0 = 背景、1 = 滑坡，与 wrapper 取 `out[:, 1]` 一致。
- ❌ 部署分支读到了 counter：smoke test 用「把 counter 槽位换成随机噪声、输出逐位不变」证明过。
- ❌ 训练/验证走了不同分支：两处分别是 `forward_train`（内含官方 MISS_NDSM）与 `forward_target_only`（同一条 MISS_NDSM 推断路径）。
- ❌ `report()` 的逐像素广播 bug：它只影响收尾评估段，修好后补出的指标与训练段一致。
- ❌ 模型没保存好 / 恢复错权重：`DIS2_latest.pt` 内 `complete=True`，测试指标由 `scripts/33_eval_dis2_width50.py` 从 `DIS2_width50.pt` 独立补出，两者一致。

---

## 4. 为什么「以前那个 DIS2 特别好」——差别不在创新点

| | 旧产物 `rapid_landslide_cocd/DIS2.pt` | 新产物 `dis2_width50_haiti/` |
|---|---|---|
| 网络 | **我们的 `SingleOrbitStudent(correction=True)`** | 官方 `DLKD_ver4`（`gf_dim=50`） |
| 骨干 | ConvNeXt-Tiny **ImageNet1K 预训练** | 无预训练，从头训 |
| 数据 | `train_modes=True`，约 3× | 只训真实 pair，1,233 行/epoch |
| 损失 | 基础分割 + `0.1·dis2_style_loss` + `0.1·MSE(sigmoid)` | 官方 8 项复合损失 |
| 优化器 | AdamW 1e-4 + wd 1e-4 + `posweight` | Adam 5e-4，无 wd |
| val AUPRC | 0.8545 | 0.3858（ep50 仍在升） |
| test IoU / AUPRC | 0.5662 / 0.8620 | 0.1829 / 0.3732 |

三句话：

1. 旧的那条根本不是 DIS2，是**我们的骨架加两个附加软损失项**；它 0.5662 的分数主要来自 ConvNeXt-Tiny 的 ImageNet 初始化和 3 倍数据，不是来自 DIS2 的机制。
2. 新的这条才是官方机制，但代价是「从头训 + 数据只有 1/3 + 部署分支只拿 6.7% 的监督」。
3. 所以「以前好、现在崩」既不是模型被训坏，也不是创新点没复现，而是**换了对象**。

---

## 5. 当时评估的三条路（路线 A 已实施）

### 路线 A（推荐做论文主对照）—— 机制忠实 + 同骨架  ✅ **已实施**

> 这条就是后来的 `DIS2-port`：官方三段蒸馏规则（四级融合特征 L2 + penultimate L2 + T=2 logits KL）
> 加正交项，接到我们的 `OursV3Student('R1')` 上，协议全部沿用项目统一策略。
> 结果 test overall **IoU 0.6272 / AUPRC 0.8963**，与同口径的 R3（0.6298）只差 0.0026。
> 说明那套配方崩掉的原因确实在条件而不在蒸馏思想。见 `docs/DIS2_THREE_VERSIONS_RESULT.md`。

把 DIS2 的三件事搬到我们自己的 ConvNeXt-Tiny/FPN 骨架上：

- 保留：替代编码器（`rgb_else` 对应物）、多尺度融合特征的 L2 蒸馏、penultimate 特征蒸馏、T=2 logits KL、正交解耦；
- 需要新写的东西：4 级融合特征金字塔 + penultimate 特征 + 分类 logits 的导出接口（官方解码器输出里本来就有这些，换成我们的 FPN 后要显式重建）；
- 好处：全部方法共用同一骨干、同一初始化、同一数据协议，**唯一的变量就是蒸馏机制**。这是最强的对照，也顺手消掉了「初始化差异」这个审稿人一定会问的点。

### 路线 B —— 保留官方网络，只修训练条件  ✗ 未采用（依赖已删除的实现）

不动机制，只改：把两个模态编码器换成 ImageNet 初始化的 ConvNeXt-Tiny 四阶段特征；`train_modes=True` 补足数据；epochs 50 → 200；加 cosine 调度；前 N 个 epoch 只训主头。
好处是改动可控、容易解释；代价是离「官方复现」更远，必须如实标注为 adapted。

### 路线 C —— 最小探针  ✗ 未采用（依赖已删除的实现）

epochs 50 → 200，同时把 8 个多尺度头的权重降到 0.2，监测 `missing_loss` 与 val AUPRC 是否松动。
它的作用是证伪或证实「只是训练不足 + 梯度预算被辅助头吃光」这个诊断。**这条预测现在由路线 A 间接证实了**：
换到我们的协议（去掉 8 个辅助头、补足数据、换预训练骨干）后立刻恢复正常，说明瓶颈确实在条件侧。

---

## 6. 一个必须写进论文的口径问题

冻结协议对所有方法都用**固定阈值 0.5**。但 DIS2 的部署分支输出中位数只有 0.0028（没有正类加权，概率尺度天然偏低），而 Ours 用 `posweight` 把概率尺度推高。这意味着同一个 0.5 阈值对两边**不等价**，IoU/F1 这一类阈值相关指标对 DIS2 天然不利。

AUPRC 与阈值无关，所以主比较应当同时给出 AUPRC，并在正文说明这件事；不要只报 IoU/F1 然后把 DIS2 当成「方法弱」的证据。

---

## 7. 复现命令

以下三条随实现一并删除，**已不可运行**，保留仅为记录当时的取证方式：

```bash
# 已删除：损失分解诊断（只读）
#   scripts/34_diagnose_dis2_loss.py --batch 2 --batches 4
# 已删除：从 checkpoint 单独重跑评估段
#   scripts/33_eval_dis2_width50.py
# 已删除：完整重训（50 epoch）
#   scripts/29_train_dis2_width50_haiti.py --epochs 50 --batch 2
```

现在起作用的是蒸馏迁移那一条链（官方规则接到我们骨架上）：

```bash
PY='/Users/zhangjiuqi/Desktop/联合分类/.venv/bin/python'
cd /Users/zhangjiuqi/Desktop/distortion

# 结构 + 真实数据冒烟（CPU，几十秒）
$PY Haiti_SAR_GRSL_Audit/scripts/36_smoke_dis2_distill_ours.py
$PY Haiti_SAR_GRSL_Audit/scripts/37_smoke_distill_integration.py

# 训练 / 重跑（50 epoch，batch 8，无早停，约 3 h）
bash Haiti_SAR_GRSL_Audit/scripts/38_run_dis2_port.sh DIS2
```
