# DIS2 蒸馏思路的完整迁移与改进设计

状态：**已实现、已验证、已训练完成**（2026-09-17 13:16）。
产物：`losses/distill.py`、`models/landslide_cocd_v2.py` 的抽头、`scripts/23_train_ours_v2.py` 的 `DIS2` 阶段、
`scripts/36_smoke_dis2_distill_ours.py`、`scripts/37_smoke_distill_integration.py`、`scripts/38_run_dis2_port.sh`。

**结果（50/50 epoch，batch 8，关闭早停，约 3 h）**：测试 overall **IoU 0.6272 / F1 0.7709 / AUPRC 0.8963**，
best val AUPRC 0.8932（ep30）。与同口径臂（同为 50 epoch、batch 8、无早停）比：**R3 0.6298，高 +0.0026**。
三个 DIS2 版本的身份、全部结果与口径红线见 `DIS2_THREE_VERSIONS_RESULT.md`。

**§6 那三条预测的实测结果**：

1. **三项 tap 的下降幅度不齐，且绝对量级都很小**：`feat` 0.0007 → 0.0002（−71%）、`logit` 0.0050 → 0.0028（−44%）、`pen` 0.0003 → 0.0003（持平）。按 §6 第 1 条的判据，这属于「部分下降」的混合情形；但那条的第二个分支（「若明显下降而指标仍差」）不适用——**指标并不差**（0.6272）。合起来的读法是：这些 tap 的绝对量级在 1e-4 ~ 5e-3，远小于分割项，指标恢复更可能来自骨干 / 初始化 / 数据条件的改善，而不是这些蒸馏项传递了多少信息。
2. **`div` 塌了**：ep1 0.0019 → ep2 0.0001 → ep3 起 `0.0000` 并保持到 ep50，与官方（50 epoch 内 `9.31e-05`）**同现象**。这是跨两个独立实现复现出来的，说明该约束在本数据集上就是不活跃、不是实现问题——正好给「用构造而不是惩罚项保证解耦」（`Ω(a,0)=0`）这条改进方向提供了直接证据。
3. **没挂同口径地板**：本轮只跑单臂，所以只能与同为教师自轨初始化的 `Teacher_self`（0.6006）比，差值 +0.0266 里混着 50 epoch 训练本身的贡献。要拆开就在同一条命令后面加 `R1`。

---

## 0. 这一轮要回答什么

用户的问题拆成两半：

1. **把 DIS2 的蒸馏思路完整搬过来**，但不要它那套"水土不服"的训练配方——数据集小，所有方案共用我们自己的训练策略；
2. **DIS2 的蒸馏哪里做得不好，我们怎么改进**。

第 1 条必须"完整"，所以第 1 节逐条对着官方源码抄，并列出替代项。第 2 条必须有实测证据，不能是感觉，
所以第 3 节的所有数字都来自本仓库跑出来的日志。

---

## 1. 官方蒸馏的逐条迁移

官方实现位置：`third_party/landslide_baselines/dis2/models_bank/DLKD_ver4.py`
损失组装位置：`third_party/landslide_baselines/dis2/run_experiment_35.py:55 cal_train_loss`

官方把蒸馏写成三段（`DLKD_ver4.py:554-574`）：

```python
miss_fused_pooled.append(miss_fused_e4_global)
full_fusion_pooled = self.full_fusion_pool([full_fused_e1, full_fused_e2, full_fused_e3])
full_fusion_pooled.append(full_fused_e4_global)
kl_loss  = sum(l2_kd_loss(miss_fused_pooled[i], full_fusion_pooled[i].detach())
               for i in range(len(miss_fused_pooled)))          # 四个尺度
kl_loss += l2_kd_loss(missing_pen_feats, full_pen_feats.detach(), spatial=True)
full_logits, missing_logits = full_logits.detach() / self.T, missing_logits / self.T
kl_loss += F.kl_div(F.log_softmax(missing_logits, 1),
                    F.softmax(full_logits, 1), reduction='mean') * self.T * self.T
diversity_loss = sum(orthogonality_loss(stu_pooled_features[i], stu_dist_pooled[i]) ...)
```

`self.T = 2`。教师侧一律 `.detach()`。

### 迁移对照表

| # | 官方元件 | 官方实现 | 我们的对应物 | 是否逐字一致 |
|---|---|---|---|---|
| 1 | 四级融合特征 L2 | `l2_kd_loss(逐通道归一化后 MSE)`，四级各自 `HierarchicalAttnPool` 池化成 `(B,D)` token | FPN 的四个尺度 `p5/p4c/p3c/p2`，均值池化成 token | 损失函数逐字一致；**池化方式替代**（见 §2.2） |
| 2 | 倒数第二层特征图 L2 | 同一 `l2_kd_loss(..., spatial=True)` | `head[:-1]` 的 64 通道输出（1/4 分辨率） | 逐字一致 |
| 3 | 分类 logits KL，T=2、乘 T² | `kl_div(log_softmax(·/T), softmax(·/T)) * T*T` | 单通道 logits 补一个恒定 0 背景通道成二类，再走同一公式 | 逐字一致（含 `reduction='mean'` 这个官方写法） |
| 4 | 正交/多样性约束 | `orthogonality_loss(a, b)` = 两个 token 集归一化后余弦平方的均值 | 同函数，迁移进 `losses/distill.py` 备用 | 逐字一致 |
| 5 | 教师侧 detach | 所有 full 分支量 `.detach()` | 教师抽头整个在 `torch.no_grad()` 下算 | 一致 |
| 6 | KL 预热 `epoch>5` 时权重 5.0 | `kl_loss_weight = 5.0 if epoch > 5 else 0.0` | **未采用**，见 §2.1 | 有意偏离 |

`l2_kd_loss` 与 `orthogonality_loss` 的实现逐字复制在 `losses/distill.py`，并由
`scripts/36_smoke_dis2_distill_ours.py` 与手写参考实现逐位比对。

### 一个附带发现

官方 `l2_kd_loss` 的 `if spatial:` 两个分支在数学上等价——`(a_norm-b_norm).pow(2).mean(dim=(0,1,2,3))`
与 `F.mse_loss(a_norm, b_norm, reduction='mean')` 都把所有元素求平均。断言
`l2_kd_loss spatial=True == spatial=False` 已通过。

**所以官方那一路"空间匹配"其实也是平铺等权，并没有做任何逐位置的差异化监督。**
这条对 §3 的第 3 点很重要。

---

## 2. 有意没有搬过来的部分

### 2.1 8 个多尺度辅助头与 5×KL 预热

这两项属于官方的**训练配方**，不属于蒸馏思路本身。数据集规模不同（官方 1,233 行真实 pair 对我们的协议
是同一批数据但用法不同），用户的约束是"所有方案统一训练策略"，所以 D 臂一律使用：
AdamW 1e-4 / wd 1e-4、`land_loss`（BCE + 0.2·Dice，含 `posweight`）、每臂 50 epoch、
batch 8、同一初始化（Teacher self 路径）、同一份 train/val/test 划分。

`5×KL` 预热在官方是**与其他 7 项损失之间的相对权重**。在我们这里 KL 是唯一的蒸馏项，
乘 5 与乘 1 只差一个常数系数，因此改为统一的固定权重 `--kd-weight`（默认 1.0），臂间取值完全相同。

### 2.2 注意力池化改为均值池化

官方对前三个尺度用可学习的 `AttnPoolToToken`。我们改用均值池化，
**目的是让蒸馏臂与其对照臂参数量完全相同**——否则 D 臂多出的池化参数会让"结构 vs 目标函数"分不开。
代价是失去了官方的可学习加权；这一点在论文中要写明。

### 2.3 缺失模态分支的网络结构

官方那套 `rgb_else` 替补编码器 + 类别注意力解码器是**结构**而非蒸馏。我们要复现的是蒸馏规则，
而"缺失模态如何被补偿"在我们的设定里由 Student 的修正头（`Correction`/`Ω`）承担。
两边的对应关系是：

```
官方：完整双模态分支  --三段蒸馏-->  缺失模态分支
我们：Teacher 双轨     --三级蒸馏-->  Student 单轨
```

这是同一个 LUPI 结构，因此蒸馏规则可以逐项对应，不需要重建官方的解码器。

---

## 3. DIS2 蒸馏规则的四个实测问题

以下数字全部来自本仓库的运行记录，不是推测。

### 3.1 它蒸馏的是"激活"，而不是"变换"——目标不可达

四级特征与 penultimate 的 L2 直接要求学生特征图等于教师双模态特征图。但学生的编码器只吃目标轨，
那个差量在信息上不可重建（特权信息不可传递是 LUPI 的定义）。这个 L2 的最优解不是"补上"，而是"平均掉"。

**证据**：官方 50 个 epoch 里 `full_loss`（完整分支自身）降了 39.2%，而 `missing_loss`（要部署的那个分支）
只降 11.4%、`kl_loss` 只降 11.6%，总损失只动 1.7%。完整分支确实在学，但"完整分支教缺失分支"这一步没接上。

**我们的改法**：蒸馏教师**算子作用出来的修正量** `r_k = Ω_k(p_k, c_k)`，这是一个可达目标（一个 128 通道的残差），
而不是要求复现不可重建的激活值。

### 3.2 全像素等权，与"特权信息在哪里有用"无关

`l2_kd_loss` 是全域均值，每个像素权重相同。而在我们的设定里，反轨证据只在局部有价值——
测试集正类像素仅占 **7.2439%**，G10 区域 1,039,406 像素。

**证据**：`gain` 门控在真实训练批次上把权重集中在 **0.5%–1.1%** 的像素上
（`w_frac` 实测 0.0051 / 0.0112）。也就是说 99% 的像素上，官方做法仍然在传梯度。

**我们的改法**：所有 tap 都过 `w = gain × protect`，`gain` 是双轨相对自轨的 BCE 改善率、`protect` 在学生已经不输的地方归零。

### 3.3 池化把空间定位抹掉，而"空间匹配"是空头支票

四个尺度里三个被池成 `(B,D)` token，只剩 penultimate 保留空间；而 §1 的附带发现说明
penultimate 那一路的 `spatial=True` 与 `spatial=False` 数学等价，**实际也是平均**。

滑坡的判别信号是局部的（desc/G11 区域仅 50,988 像素）。全局平均正好把要保留的东西平均掉。

**我们的改法**：`feat`/`eff`/`pen` 三项全部在原始分辨率上逐像素比对，不做池化；
用 `sum(w·d)/sum(w)` 归一化，使不同分辨率的尺度之间可直接相加。

### 3.4 正交约束在这批数据上近乎空转

官方 `diversity_loss` 在 50 个 epoch 里很快塌到 **1e-4**，等于没有约束；
它本来要防止"替补编码器把在场模态重编码一遍"这类退化解，
但在 1,233 行的真实数据尺度上根本没形成有效梯度。

**我们的改法**：不靠惩罚项，而靠构造——`Ω(a, 0) = 0` 恒等成立，
所以"零反轨信息 → 零修正"是结构性质，不是需要训出来的约束。
`scripts/36_smoke_dis2_distill_ours.py` 断言了这条性质，并断言了它的一个推论：
**在驱动还没打开之前，继承来的算子受到的梯度恰好为零，是驱动把它打开的。**

### 3.5 顺带排除的两个"像 bug"的地方

- `softmax_weighted_loss` 内部是 `log(clamp(output, min=0.005))`，而解码器返回的就是 `softmax(...)` 概率，
  头和损失自洽，**不存在 logits/概率混用**。
- `reduction='mean'`（数学上应对应 `batchmean`）是官方写法，我们逐字复制，它会**压低** KL 的量级。

这两条在 `docs/DIS2_MECHANISM_AND_DIAGNOSIS.md` 里有完整证据链。

---

## 4. 迁移后的臂与对照

所有臂的学生结构完全相同（`OursV3Student('R1')`：每尺度一个自由残差头），初始化完全相同，
优化器、调度、数据、batch 也完全相同。**唯一的变量是蒸馏规则。** 本轮只跑一个臂：

| 臂 | `kd` | 蒸馏规则 |
|---|---|---|
| `DIS2`（结果表里叫 `DIS2-port`） | `dis2` | **本轮唯一臂。** DIS2 的蒸馏规则逐字复现：四级池化融合特征与 penultimate 图的 `l2_kd_loss`、T=2 的 logits KL、`orthogonality_loss` 正交解耦项，教师侧全部 detach |
| `R1` | `gain` | 可选同口径地板（默认不在队列里）：只有输出级选择性 KL，即已冻结的 R1 规则 |

两个探针保留在代码里、**没有列进 `CONFIGS`**，本轮不跑：
`dis2-logit`（只留 T=2 logits KL，用来判断多级特征项到底有没有贡献）与
`selective`（同样的 tap 改写为 `gain×protect` 加权 + 效应项 `‖r_k^s − r_k^t‖²` + 不池化；
其输出项与 `R1` 逐位相同，集成冒烟实测 `selective.logit = 0.0019` 对 `control = 0.0018`）。

### 实测量级（真实训练批次，单位权重）

| 规则 | 总 KD | 与对照之比 | 分项 |
|---|---:|---:|---|
| `gain`（对照） | 0.0018 | 1.00 | — |
| `dis2` | 0.0015 | 0.85 | feat 0.0006 / pen 0.0002 / logit 0.0007 / div 0.0000 |
| `selective`（未启用） | 0.0187 | 10.4 | feat 0.0004 / eff 0.0156 / pen 0.0009 / logit 0.0019 |

量级都在同一个数量级附近，因此**不需要任何按臂配平的系数**。
（我一开始写过一个"按首批自动配平 KD 预算"的机制，集成冒烟发现它在 `w` 整批为零的批次上会退化到钳位下界，
属于不稳健的隐式调参，已删除。现在 `--kd-weight` 对所有臂取同一个值。）

> **`div` 要盯。** 随机 tap 上它是 0（修正块零初始化，`Ω(a,0)=0` 使修正量恒为零向量），
> 但真实数据跑满 1 个 epoch 后实测 `div = 0.0553`——比另外三项合计（0.0031）大一个数量级。
> 官方也是 1.0 权重直接相加，所以这是忠实复现而非移植失误；但它意味着正交项在这套尺度上
> 可能主导 KD 预算。官方那一路 50 epoch 内塌到 1e-4，我们是否同样塌、塌得多快，必须在训练期读出来。

### 运行

```bash
cd /Users/zhangjiuqi/Desktop/distortion
bash Haiti_SAR_GRSL_Audit/scripts/38_run_dis2_port.sh          # 默认臂 DIS2，~2.5 h
bash Haiti_SAR_GRSL_Audit/scripts/38_run_dis2_port.sh R1 DIS2  # 需要同口径地板时再加 R1
```

`--no-early-stop` 已写进排队脚本：**早停关掉**，臂跑满 50 epoch。
因此它与此前那个早停在 ep20 的 `R1` 不是同一口径，不能直接比。

---

## 5. 这一轮能支持什么、不能支持什么

**能支持**：在同骨干、同初始化、同数据、同优化器、同 epoch 数下，"蒸馏什么"这一变量的对照。
`DIS2` = 官方规则，`R1` = 我们已冻结的规则。这两者的差值只能归因于蒸馏规则本身。

**不能支持**：

- **不能**据此宣称"DIS2 方法差"。官方网络在官方数据上的表现与本轮无关；本轮只比较蒸馏规则。
- **不能**把 `DIS2` 臂与 b=2 训练的 S0/Teacher 严格并列；它是 batch 8，论文需披露。
- **不能**在单种子下宣称超过 Teacher。参考上一轮：R3 − Teacher_dual 只有 +0.0048 IoU，单种子，不能写进论文。
- 只跑 `DIS2` 单臂时，能引用的地板只有 `Teacher_self`（同一初始化的起点 IoU 0.6006），
  而这个差值里混着"训练本身"的贡献。要拆开就必须加跑同口径的 `R1`。
- 冻结协议对所有方法都用固定阈值 0.5，而 DIS2 的原始实现没有正类加权、概率尺度天然偏低
  （验证集正类通道中位数 0.0028）。主比较必须同时给阈值无关的 AUPRC。

---

## 6. 什么结果会证伪本文的诊断

本文的诊断是"DIS2 那套蒸馏在这里不动，是因为它蒸馏的是激活而不是可达的效应"。本次运行能直接读出三条：

1. **`feat` / `pen` / `logit` 三项是否随训练下降。** 官方 50 epoch 里 `kl_loss` 只降 11.6%。
   我们在同骨干、有 ImageNet 初始化、数据量补到 3× 的条件下若同样几乎不降，说明"蒸馏激活"
   这一路本身传递不动，诊断成立，瓶颈在损失侧而不是数据量。若它们明显下降（比如 >50%）
   而指标仍差，诊断就得改：瓶颈不在可学性，而在这些 tap 携带的**信息内容**。
2. **`div` 是否塌。** 官方那一路 50 epoch 内塌到 1e-4，等于正交约束空转。若我们这里 `div`
   稳定在 0.05 上下不降，说明约束在持续施压，与官方不同；若同样塌掉，"用构造而不是惩罚项
   保证解耦"（`Ω(a,0)=0`）这条改进方向就有了直接证据。
3. **测试端与地板的差值。** 加了同口径 `R1` 时，`DIS2 − R1` 是本轮唯一干净的对照读数；
   只跑单臂时只能与 `Teacher_self`（同一初始化，0.6006）比，差值里混着训练本身的贡献。

第 1、2 条运行期就能观察：训练循环每个 epoch 打印 `feat/pen/logit/div`。

---

## 7. 实现说明（给后续维护）

- 抽头全部通过新方法暴露，**没有改动任何既有前向路径**：
  `FPNStateDecoder.pyramid4/taps`、`OursV2Teacher.from_features_taps/forward_pair_taps`、
  `OursV3Student.forward_taps`。冒烟断言 `forward_taps` 的 `z4/z34` 与 `forward` **逐位相同**，
  因此蒸馏臂的部署路径与对照臂完全一致。
- `forward_taps` 一次编码器前向就给出全部 tap，**不会让蒸馏臂变慢一倍**。
- 教师抽头在 `torch.no_grad()` 下计算；断言"没有任何教师参数收到梯度"。
- 训练循环重构后，`gain` 模式的 KD 仍与重构前**逐位相同**（集成冒烟 [A] 段用重写的原表达式比对）。
- `--tag` 同时作用于方法名、结果表和 checkpoint 文件名；`--no-early-stop` 关闭早停但保留验证打印。
