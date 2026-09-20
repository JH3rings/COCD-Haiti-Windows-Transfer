# COCD：Teacher / Student / DIS2 规格说明（供外部核验）

本文件描述三件事：**新 COCD 的 Teacher、新 COCD 的 Student、DIS2 的蒸馏规则**，以及各自的训练逻辑与网络结构。
每一条都标注了 **【已实现】**（仓库里现有代码，可逐行核对）或 **【待实现】**（2026-09-18 定稿的设计，尚无代码）。
标注为「实现注意」的内容**不属于方法**，只影响能否跑起来。

---

## 0. 共同骨架（两个模型共用，【已实现】）

代码：`models/landslide_cocd.py::ConvNeXtTinyFPN`

**输入**：`(B, 5, 128, 128)`，五个通道依序为
`[VV_pre, VH_pre, VV_post, VH_post, orbit]`。
前四通道是同一轨道的灾前/灾后双极化栅格；**第五通道是一个空间常数平面**（ASC 填 0，DESC 填 1），
即轨道信息当前以「常数图」形式喂进网络，没有任何 embedding。

**主干**：ImageNet-1K 预训练的 ConvNeXt-Tiny，首层替换为
`Conv2d(5, 96, kernel=4, stride=4)`。替换时的初始化：
`weight` 先整体置零，再把第 0–3 输入通道设为原 ImageNet 首层权重的通道均值 × 3/4；
**第 4 通道（orbit）的权重保持恰好为零** —— 轨道信息在初始化时对网络输出没有贡献。

**编码器输出**（输入 128×128 时的实际形状）：

| 名称 | 分辨率 | 通道 |
|---|---|---|
| c2 | 32×32 | 96 |
| c3 | 16×16 | 192 |
| c4 | 8×8 | 384 |
| c5 | 4×4 | 768 |

**FPN（自顶向下）**：四条 1×1 侧向投影 `l2: 96→128`、`l3: 192→128`、`l4: 384→128`、`l5: 768→128`，然后

```
p5 = l5(c5)                        4×4,  128ch
p4 = l4(c4) + up(p5)               8×8,  128ch
p3 = l3(c3) + up(p4)               16×16, 128ch
p2 = l2(c2) + up(p3)               32×32, 128ch      <- 多尺度融合后的统一语义特征
```

`up(·)` 是 nearest 上采样。

**解码头**：`head = Conv3×3(128→64) + GELU + Conv1×1(64→1)`。
在 p2 上产生 32×32 的**单通道 logits**，再 `bilinear` 上采样到 128×128 作为输出。
**没有 sigmoid 在模型里**；推理时取 `sigmoid(logits)`，阈值在验证集上选定。

**三个读数共用同一个解码器**（不是三个头）：`z = head(p2)`。因此不同读数之间的差别**只来自喂进去的
`(p3, p4)`**，不掺入头与头之间的差异。

**参数量**（去重后按 `model.parameters()` 计）：骨架约 28.08 M。

---

## 1. 新 COCD — Teacher【待实现】

### 1.1 它要解决的问题

部署时只有一条轨道，训练时两条都能看到。Teacher 是**唯一允许同时看两条轨道的模型**，
它的任务不是「修正目标轨」，而是**显式提取反轨所提供的互补信息**：
target 轨的观测方向是 ASC 时，反轨就是 DESC，反之亦然。两个方向完全对称。

### 1.2 网络结构

```
Target SAR  ──► 共享 Encoder+FPN ──► F_t  ────────────────┐
                                                          │
Counter SAR ──► 同一个 Encoder+FPN ──► F_c                │
                                                          │
        H_tc = DCA(F_t, F_c)        ◄── 唯一的新模块        │
                                                          │
        C_T = Φ(F_t, H_tc) − Φ(F_t, DCA(F_t, 0))          │
                                                          │
        F_T = F_t + C_T  ◄────────────────────────────────┘
                │
            共享 Decoder D
                │
        z_dual = D(F_T)
        z_self = D(F_t)          （同一 D，另一条路）
```

逐步说明：

1. **两轨各自过一次同一个编码器**，得到 `F_t`（目标轨）与 `F_c`（反轨）。
   两者都是上面 §0 里的 **p2：32×32、128 通道** —— 也就是 FPN 多尺度融合之后的统一语义特征。
2. **`DCA`：deformable cross-attention**，唯一的新模块。
   对 `F_t` 的每一个位置，从 `F_c` 的**邻域**里采样若干点并加权求和，采样位置由网络预测的偏移量决定。
   之所以需要它：升降轨的几何并不逐像素对应，强制同坐标对齐会错配；deformable 允许「在附近找最有用的观测」。
   它的输入是 `(F_t, F_c)`，输出 `H_tc` 与 `F_t` 同形状（32×32×128）。
3. **`Φ`：一个普通的卷积投影**，把 `(F_t, H_tc)` 映射成互补表示。
4. **`C_T` 用差分定义**：

   ```
   C_T = Φ(F_t, DCA(F_t, F_c)) − Φ(F_t, DCA(F_t, 0))
   ```

   第二项是「反轨特征置零后再走一遍」。当 `F_c = 0` 时两项**字面上是同一个表达式**，
   所以 `C_T(F_c=0) ≡ 0` 是**代数恒等式**，与 DCA 内部怎么实现无关。
   这条零参照很重要：它让「关掉反轨」有一个精确定义的参照点，也让下面的 counter-drop 诊断成立。
5. **`C_T` 只在唯一一个位置注入**：

   ```
   F_T = F_t + C_T
   ```

   也就是**只在 FPN 融合后的那个统一语义特征上注入一次**，不再在 P3/P4 等多个尺度分别操作。
   理由：FPN 已经负责多尺度建模，COCD 不重复做 multi-scale fusion，只负责学「另一轨额外贡献了什么」。
6. **两个读数**：`z_self = D(F_t)`（不看反轨）、`z_dual = D(F_T)`（看反轨），共用同一个解码器 `D`。

**Teacher 里没有的东西**（明确不存在）：没有 gate、没有 task-relevance head、没有 P3/P4 递进修正、
没有任何额外的正则项或稀疏项。

### 1.3 训练逻辑

只有一个损失，就是两次普通分割 BCE：

```
L_T = 0.5 · L_seg(z_self, y) + 0.5 · L_seg(z_dual, y)
```

- 两项权重各 0.5，总和 1.0。
- **没有额外的任务相关性损失**。`C_T` 获得梯度的唯一途径是：
  `F_t + C_T → 滑坡预测`，梯度直接来自滑坡 GT。这就是它的 task supervision。
- 反轨梯度经共享编码器回传（同一条轨道在另一个方向上是反轨，两次编码是同一个权重）。

**每个训练位置的前向次数**：编码器 2 次（ASC、DESC 各一次，两个方向共用）；
每个方向各 2 次 DCA（一次带真实反轨、一次反轨置零）、2 次 `Φ`、2 次解码（self / dual）。

### 1.4 训完必须做的资格检查（Teacher audit）

在开训 Student **之前**，在验证集上跑三个条件：

| 条件 | 做法 |
|---|---|
| correct counter | 反轨用**同一位置**的另一条轨道 |
| zero counter | 把反轨**特征**置零（等价于 `F_c = 0`） |
| shuffled counter | 反轨换成**另一个位置**同一轨道的栅格 |

看三件事：① `dual > self` 是否成立；② correct 是否明显优于 zero 与 shuffled；
③ `C_T` 在三个条件下的取值是否合理变化。

**如果这一步失败，说明 Teacher 没把互补学出来，不继续蒸 Student。**

---

## 2. 新 COCD — Student【待实现】

### 2.1 它要解决的问题

Student 是最终交付物：**每次前向只看一条轨道**，而且**两条轨道共用同一个模型**
（不训练两个分别对应 ASC / DESC 的模型）。它要学的是：

> 我现在是 ASC，那么「如果 DESC 存在，它会额外给我什么」；我现在是 DESC，则预测 ASC 会额外给什么。

也就是**预测那份当前缺失的互补**，而不是恢复另一轨影像、也不是恢复完整特征。

### 2.2 网络结构

```
Single SAR (5ch) ──► 共享 Encoder+FPN ──► F_t
                                            │
OrbitID ──► Embedding e_o ──► MLP ──► (γ_o, β_o)
                                            │
                          F̃_t = (1+γ_o)⊙F_t + β_o     ← FiLM，只调制 complement 分支
                                            │
                                  C_S = P(F̃_t)          ← 两三个卷积块
                                            │
                                  F_S = F_t + C_S
                                            │
                                      共享 Decoder D
                                            │
                                       z_S = D(F_S)
```

1. **`F_t`**：单轨输入经同一个编码器与 FPN 得到，与 Teacher 的 `F_t` 同定义（p2，32×32×128）。
2. **轨道 embedding**：只有两个可学习向量 `e_ASC, e_DESC`，经一个两层 MLP 产生一对通道级的
   缩放/平移 `(γ_o, β_o)`，对 `F_t` 做 FiLM 调制得到 `F̃_t`。
   它的语义是「**当前缺失的是哪一个反向视角**」，不是「这是 ASC 还是 DESC」。
3. **关键**：**调制只施加在 complement 分支上，主干分割路径仍是未调制的 `F_t`**。
   所以主路径就是一个干净的单轨 baseline，orbit 条件只负责回答「缺的是哪个方向的互补」。
4. **`P`**：complement predictor，两三个普通卷积块，输出 `C_S`（32×32×128）。
   没有 token、没有 experts、没有 router、没有 dynamic convolution、没有独立 spatial gate、没有多尺度 predictor。
5. **注入点与 Teacher 一致**：`F_S = F_t + C_S`，同样只在那个统一语义特征上加一次。

### 2.3 训练逻辑

```
L_S = L_seg(z_S, y) + λ · L_dist(C_S, sg(C_T))

L_seg  在三类 temporal pair 上计算：pre→post（GT = 滑坡）、pre→pre（GT = 0）、post→post（GT = 0）
L_dist = D(C_S, sg(C_T))，只在真实 pre→post 上计算
```

- `sg(·)` = stop-gradient：`C_T` 由**已训练冻结**的 Teacher 产生，梯度只流向 Student。
- **一个 λ，没有别的蒸馏项。** 蒸馏里不出现 `gain×protect`、不出现 logit-level KD、不出现 `L_Δ`。
- 为什么 `L_dist` 只在真实 pre→post 上算：那 2/3 的 pre/pre、post/post 是合成的 no-change 负样本对，
  标签被强制置零，在它们身上要求 Student 去匹配 Teacher 的互补没有意义。这是一条**实验协议**规定。

### 2.4 λ 的标定规则

在固定的 **train real pre→post 子集**上、**不更新任何参数**，测出
`L_seg⁰`（Student 初始状态）与 `L_dist⁰`，然后

```
λ = 0.1 · L_seg⁰ / L_dist⁰
```

使训练开始时 `λ·L_dist ≈ 0.1·L_seg`。λ **永久冻结**，不看 validation，不做 sweep。

取 10% 而不是 1:1 的理由：分割 GT 是真正的任务监督，必须主导；但若只剩 ~1%，
会重演此前「蒸馏项存在却没有足够优化权限」的问题。

### 2.5 消融阶梯

| 臂 | 内容 | 回答什么 |
|---|---|---|
| `N0` | 单轨 baseline：`F_t → D`，无 complement 分支 | 下限 |
| `N1` | `F_t + C_S → D`，**只用 segmentation loss** | 这个结构本身有没有帮助 |
| `N2` | `N1` + `L_dist(C_S, C_T)`（= Full COCD） | 显式学习 Teacher 的互补有没有作用 |
| `N2 w/o Orbit Embedding` | Full 但去掉 orbit embedding 调制 | orbit 条件建模有没有必要 |

**只有三级，没有 N3。**

---

## 3. DIS2 的蒸馏规则【已实现】

代码：`losses/distill.py::dis2_multilevel_kd`（逐字移植自 `third_party/landslide_baselines/dis2/models_bank/DLKD_ver4.py`）

### 3.1 定位

DIS2 是一个**已发表的外部蒸馏方法**（DLKD 家族 / missing-observation 知识蒸馏）。
我们在自己的骨架上做**逐字移植**，所以它回答的是「这个外部规则在我们的结构上能做到什么」，
不是「它的原论文快照能得多少分」。它使用的是**与我们完全相同的那组 tap**。

### 3.2 它匹配什么

四项，都在同一组 tap 上：

| 项 | 匹配对象 | 公式 |
|---|---|---|
| `feat` | 四个尺度的融合特征，池化成 token | `Σ_k MSE(norm(pool(stu.levels[k])), norm(pool(tea.levels[k])))`，k 遍历 4 个尺度 |
| `pen` | penultimate 图（不池化，保持空间） | `MSE(norm(stu.pen), norm(tea.pen))` |
| `logit` | 分类 logits | 见下 |
| `div` | 修正量 vs 它所修正的状态 | `Σ mean(cos²(pool(r_k), pool(p_k)))` |

其中：

- `norm(·)` = 沿**通道维**做 L2 归一化（`F.normalize(p=2, dim=1)`），所以 `feat`/`pen` 是
  **尺度无关的余弦型距离**，对通道增益不敏感。
- `pool(·)` = 对空间维取**均值**得到 `(B, C)` 的 token。
  （官方实现用一个可学习的 `AttnPoolToToken`；我们这里用均值池化，**目的是让被蒸馏的臂与它的对照
  参数量完全一致**，两个臂只差在目标函数。这一点必须披露。）
- `logit`：单通道 logits 先补一个**恒定为 0 的背景 logit** 变成二类，再做
  `KL( log_softmax(s/T) ‖ softmax(t/T) ) × T²`，`T = 2`，`reduction='mean'`。
- `div`：把修正量 `r3/r4` 与它修正的那个状态 `p3/p4` 各自池化成 token，最小化两者**余弦相似度的平方**，
  即要求修正量与自己修正的对象尽量不相关。没有这一项，学生最省事的做法是直接重新编码目标轨，修正量就变得空洞。
- 教师侧一律 stop-gradient（当前实现是在调用处用 `torch.no_grad()` 计算教师 tap）。

### 3.3 与我们规则的三个结构性差别

1. **DIS2 没有逐像素选择权重。** 因此在预/pre、post/post 这两类合成负样本对上，
   DIS2 仍然产生梯度；而旧的选择性臂在这些样本上权重恒为零（实测：0.0011–0.0029 vs 精确 0.0）。
   也就是「那 2/3 的样本只被 DIS2 消费」。这是基线公平性的披露项。
2. **量级不同。** 在初始状态，无选择权重的 KD（`w ≡ 1`，即 vanilla KD）的量级是选择性 KD 的
   6–7 倍（验证集 0.62 vs 0.088），因为选择性权重只在少数像素上活跃。
3. **它的正交项在本数据集上不活跃**：第 1 epoch 0.0019 → 第 2 epoch 0.0001 → 第 3 epoch 起 0.0000
   直到训练结束。这一现象跨两个独立实现复现，不是移植错误。

### 3.4 移植的已知怪癖（保留未改）

- 官方 `DLKD_ver4.py:574` 用 `F.kl_div(reduction='mean')`，该 reduction 会把结果除以 `B·C·H·W`，
  量级被额外压低。我们的移植**保持相同写法**，以便与原论文的权重口径一致；改动它必须同时重调那个 5× 系数。
- 该模块**不读几何 mask**，也**不在推理时运行**。

---

## 4. 旧机制（作为 baseline 与 legacy）【已实现】

新 COCD 落地后，旧机制**不进入新主方法**，但仍作为对照留在仓库里。
两者的分工很清楚：**`R3` = selective prediction KD；`New COCD` = explicit complement distillation。**

### 4.1 旧 Teacher = Additive Cross-Orbit Teacher

代码：`models/landslide_cocd_v2.py::OursV2Teacher`

- 两轨各过一次共享编码器；反轨**只通过加性修正**进入 P3/P4。
- 修正算子 `Correction(256)`：`Conv1×1(256→32) + GELU + Conv3×3(32→32) + GELU + Conv1×1(32→256)`，
  **末层零初始化**。输入是 `cat(目标状态, 反轨特征)`。
- 零参照通过**差分**构造：
  `Ω(a, c) = op(cat(a, c)) − op(cat(a, 0))`，因此 `Ω(a, 0) ≡ 0` 精确成立。
- 三个读数共用同一个解码器：`z0`（不看反轨）、`z4`（只用 P4 修正、并沿自顶向下路径传到 P3）、
  `z34`（P3+P4 都修正）。
- 训练损失：`Σ_d ∈ {asc,desc} Σ_i w_i · L_seg(z_i^{(d)}, y)`，`(w_0, w_4, w_34)` 的取值在两轮之间
  从 `(0.5, 0.25, 0.25)` 变为 `(0.25, 0.25, 0.50)`（对应两个 checkpoint）。
- **与新 Teacher 的关键区别**：注入点在 P3 与 P4 两处（不是一处）；没有注意力、没有可变形算子、
  没有 complement separator；`z4` 这个中间读数在新设计里不存在。

### 4.2 旧 Student 的结构阶梯 `R0`–`R3`

代码：`models/landslide_cocd_v2.py::OursV3Student`

| 臂 | 修正量怎么产生 |
|---|---|
| `R0` | 无修正（恒等），即单轨 baseline |
| `R1` | 每个尺度一个**自由残差头** `q3/q4`，由未修正的 pyramid 状态驱动 |
| `R2` | 把 Teacher 的 `Ω` 算子复制过来，再学一个「驱动器」`d3/d4` 去喂它（驱动器末层零初始化） |
| `R3` | 与 `R2` 相同，但 P3 的修正由**已经被 P4 修正过的**状态驱动（递进驱动），`drive3 = a3` |

- `R2/R3` 的 `Ω` 从训练好的 Teacher 复制（这是方法定义本身：「继承修正算子」），
  与 backbone 初始化是两件独立的事，在配置里是两个独立字段。
- 除 `R2/R3` 的 `Ω` 外，**所有内部臂的 backbone 都来自同一个 ImageNet 初始化**，
  且四种结构的初始修正都精确为零 —— 即所有臂从同一个未修正函数出发。

### 4.3 旧 Student 用的选择性蒸馏规则

代码：`losses/distill.py::gain_protect` + `selective_multilevel_kd`

- 逐像素权重 `w = gain × protect`：
  `gain = clamp((e0 − et) / (e0 + eps), 0, 1)` 是教师双轨读数相对其自轨读数的**相对 BCE 改善**；
  `protect = 1[et < es]`，学生自己已经比双轨教师好时置零。两个因子都 detached，权重不接收梯度。
- 四项 tap：`feat`（四个尺度，空间分辨）、`eff`（匹配修正量 `r` 本身）、`pen`、`logit`
  （T=1 的稳定 Bernoulli KL：`soft_ce − entropy`），全部在 `w` 下按**前景/背景平衡均值**聚合。
- **这套规则不进入新 COCD。**

---

## 5. 统一训练协议（两个模型、所有内部臂共用）【已实现】

- **优化器**：Adam，`lr = 5e-5`，`betas = (0.9, 0.999)`，`eps = 1e-8`，`weight_decay = 0`。
- **调度器**：无。整个训练保持恒定学习率。
- **batch**：physical 16，无梯度累积（`BATCH × ACCUM` 必须等于 16，否则直接报错退出）。
- **分割损失**：**普通 BCE**，不带 `pos_weight`、不带 Dice、不带 Focal。
- **训练轮数**：最多 20 epoch；**每 epoch 验证**；`patience = 3`（连续 3 个 epoch 验证 AUPRC 无提升即停）。
- **选模**：按 **validation AUPRC** 选最佳 checkpoint。
- **阈值**：训练与选模都不用 test 阈值。训练结束后在 **validation** 上按最大 F1 搜索阈值，
  **冻结**，再用于 test 的 IoU / F1 / Precision / Recall；AUPRC 是无阈值指标。
  **绝不在 test 上重新搜索阈值或选模型。**
- **几何 mask**：**不进入训练、不进入任何模型输入**。它**只用于评价分区**：
  `G00`（目标轨好／反轨好）、`G01`（反轨坏）、`G10`（目标轨坏、反轨好）、`G11`（都坏）。
  报告 overall / ASC / DESC × 四个分区。
- **初始化**：所有内部臂一律 ImageNet 冷启动（`init: scratch`），不用 S0 暖启。
- **单一变量原则**：`N0/N1/N2` 与各基线之间只差被声明的那一项。

**数据构造**（`scripts/21` 的 loader，【已实现】）：

- 划分按**空间位置**：1,233 train / 137 内部验证 / 343 test，同一位置的 ASC、DESC、pre、post
  四张栅格**同进同出**，绝不跨 split。
- 每个 train 位置的 spec 是 `(location, mode)`，`mode ∈ {0,1,2}`：
  - `mode 0`：`pre → post`，GT = 滑坡（真实对）
  - `mode 1`：`pre → pre`，GT = 0（合成 no-change）
  - `mode 2`：`post → post`，GT = 0（合成 no-change）
- 每个 spec 项**同时产出 ASC 与 DESC 两个样本**：目标的输入是该轨道的 (pre, post)，
  反轨的输入是另一轨道的 (pre, post)。
- 因此每 epoch 样本共 `1,233 × 3 × 2 = 7,398` 个单轨样本
  = 真实 pre→post 2,466 + pre/pre 2,466 + post/post 2,466。
- **验证/测试不生成合成对**，只评真实 pre→post：val 274 个单轨案例、test 686 个。
- 训练时两个方向拼成一批交给同一组权重（`x = cat((asc_target, desc_target))`）。

---

## 6. 实现注意（不属于方法，只影响能否跑起来）

1. **DCA 只能用 `grid_sample` 手写。** `torchvision.ops.deform_conv2d` 的**反向**在本机 MPS 上
   未实现（`_deform_conv2d_backward` 报 NotImplementedError），前向可用。
   按标准 deformable attention 写法用 `F.grid_sample` 实现时，前向与反向在 MPS 均已实测通过。
2. **Student 的 orbit 索引来源。** 目前轨道信息只存在于**第 5 通道的常数平面**里（ASC=0/DESC=1），
   模型内部没有现成的 orbit index。给 Embedding 喂索引时需要从该通道推出，或让 loader 额外返回轨道标号。
3. **`D` 与聚合方式尚未指定。** `L_dist = D(C_S, sg(C_T))` 里的 `D`（L1 / MSE / SmoothL1 / cosine）
   尚未选定，而 `λ` 的标定值完全由它决定（Student 近零初始化时 `L_dist⁰ ≈ D(0, C_T)`）。
   聚合建议用**普通空间均值**（特征匹配不存在类别不平衡，不需要前景/背景平衡聚合）。
4. **初始化。** complement 分支用**近零**（不是精确零）初始化，使主路径初始几乎等于 baseline，
   同时第一步就能收到梯度。
5. **parity 断言建议做成决策级**：初始时在 0.5 阈值下**没有任何像素翻转**，
   比张量级容差更有意义。
6. **Teacher-T2a 的历史结果**（只调整了旧 Teacher 三个读数的监督权重，未改善）可作为内部先验，
   但**不反向约束新设计**。

---

## 7. 建议外部核验的六个点

1. `C_T = Φ(F_t, DCA(F_t,F_c)) − Φ(F_t, DCA(F_t,0))` 的零参照性质是否确实由定义保证，
   而无需对 DCA 附加任何额外约束？
2. 只有 `L_seg(z_self) + L_seg(z_dual)` 两项、没有任何 relevance 项时，
   `C_T` 是否会退化为「反轨特征新奇度」而非「任务相关互补」？若会，靠什么现象能最早发现？
3. `L_dist` 只在真实 pre→post 上算，是否足以避免合成 no-change 对污染互补蒸馏？
4. 消融 `N0 → N1 → N2` 三级是否足以支撑「互补预测 + 互补蒸馏」两个主张？
5. `λ` 的标定公式 `0.1 · L_seg⁰ / L_dist⁰` 是否合理？「初始 10% 辅助强度」这个口径
   在训练过程中会如何漂移，是否需要如实披露？
6. 与 `R3`（selective prediction KD）的对比是否干净 —— 两者的差别是否真的是
   「显式互补蒸馏」对「选择性预测蒸馏」，而没有被注入点、骨架或协议的其他差异混淆？
