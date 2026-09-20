# Haiti 数据集原作者的训练配置，以及"是不是没收敛"

回答的问题：**「这个数据集原版作者他们是怎么训练的？训练配置（学习率）是多少？
我怀疑是没收敛。」**

结论先说：

1. **原版数据集作者没有发表过训练配方。** Haiti 数据集（Bralet et al., IEEE
   DataPort, DOI `10.21227/4heb-7h07`, 2024-07-14）是一份**纯数据集**发布，
   官方描述里只有数据构成（1713 patches、SAR-optical 配对、layover-shadow mask、
   TODO 标注），**没有 baseline、没有训练超参数、没有推荐 epoch 数**。它被定位为
   "aimed to be used whether for monomodal or multimodal approaches"，即把训练
   配方留给使用者自己定。所以"照抄原作者配置"这条路**物理上不存在**。

2. **能找到的最接近的两个参照**，都在同一个课题组（LISTIC, Université Savoie
   Mont Blanc，同样的四位作者）：
   - **ISSLIDE / SARDINet 的开源代码**：`epochs=30`、`lr=1e-4`（主网络）/
     `1e-5`（判别器）、`batch_size=32`、split `[0.7, 0.2, 0.1]`。这是同一位一作
     在相邻任务上实际使用的配置。
   - **Nava et al., GMD 19:167–2026**（用 Haiti 作为"未见事件"做迁移推理）：
     最多 **500 epochs**、早停判据是**验证损失连续 30 个 epoch 不下降**、
     LR 网格 `{1e-4, 1e-5}`、Haiti 那批模型用 **LR=0.001**、Adam + Focal Loss。

3. **"没收敛"这个假设，用我们自己的数据可以否掉。** 迁移版麦克的
   `rapid_landslide_cocd/metrics.csv` 里，DIS2 跑满 **50 epoch**（
   `Ours_progress.json`: `epoch:50, epochs:50, state:"complete"`），
   拿到 **F1 0.7230 / Recall 0.9437 / Precision 0.5860**。
   Recall 0.94 是典型**收敛到过拟合式高召回**的形态，不是欠训练的形态
   （欠训练会是 P/R 双低）。**那批数字高，是因为它收敛得太"用力"了。**

4. **真正的问题不在收敛，在协议的尺子被换了。** 见 §4。

---

## 0. 一个必须写在前面的边界

我没有拿到原版数据集作者的训练配置，原因有两条，都需要如实记下：

- **数据集本身不含**。IEEE DataPort 与 Zenodo 两个发布页的描述全文都已核对，
  只有数据描述，无实验章节。
- **配套的博士论文正文取不到**。Bralet 的博士论文
  《Deep Learning for Multimodal Detection of Sudden and Slow Moving Slope
  Instabilities on Bitemporal Remote Sensing Images》（Université Savoie Mont
  Blanc, 2024-10-01, HAL `tel-05029007`, theses.fr `2024CHAMA031`）中确有
  Haiti 章节，但 HAL 与 theses.fr 目前都启用了 Anubis 反爬（工作量证明），
  **所有 PDF 直链均返回验证页，公开渠道无法取全文**。

所以下面 §2 给的是"同课题组的相邻任务配置"，**不是** Haiti 任务的原版配置。
这个区别在写方法学对比时不能含糊。

---

## 1. 原版数据集到底给了什么

`Multimodal Remote Sensing Dataset for Landslide Change Detection in Haiti`
（Bralet, Trouvé, Chanussot, Atto；2024-07-14；DOI `10.21227/4heb-7h07`）

| 项目 | 官方描述 |
| --- | --- |
| 事件 | 2021 海地地震后的滑坡早期识别 |
| 数据 | Sentinel-1 / Sentinel-2 事件前后配对（SAR-optical） |
| 切块 | **1713 patches**；因 asc/desc 双过境，SAR 模态 patch 数**翻倍** |
| 几何处理 | layover-shadow mask，按 **Meier et al. 1993** 计算，用于**限制几何畸变影响** |
| 光学处理 | Sentinel-2 附带云/云阴影 mask |
| 标注 | 来自 **NASA TODO database**，栅格化后按同样方式切块 |
| 定位语 | "aimed to be used whether for monomodal or multimodal approaches, and<br>whether for monotemporal or bitemporal approaches" |
| **训练配方** | **无。没有 baseline、没有 epoch、没有 lr、没有 split 建议。** |

**要点：这份数据集是一个"裸"数据集。** 任何在这个数据集上报告的数字，
其训练配置都是使用者自己定的 —— 包括我们自己的。

这也解释了一件事：**"跟原版作者对齐配置"这条路没有终点**，因为原版作者
没有留下配置。我们能做的是（a）引用同课题组相邻任务的配置作为合理性参照，
（b）把**我们自己的**协议写清楚、锁死、可复现。

---

## 2. 能找到的最接近的参照配置

### 2.1 同课题组代码：SARDINet（Bralet 一作，ICIP 2022）

仓库 `github.com/Ant89ne/SARDINet`，`main.py` 里的超参数（原文照录）：

```python
epochs = 30                     # Number of epochs
percentages = [0.7, 0.2, 0.1]   # Percentage of training and evaluation
batch_size = 32                 # Batch size
im_size = 200                   # Size of the crops
lr_main = 1e-4                  # learning rate for SARDINet
lr_discr = 1e-5                 # learning rate for the discriminator
lambdaVal = 0.0005              # Weight for balancing the losses
```

- 任务：SAR→光学**模态翻译**（不是分割），数据集是 SpaceNet6 / SAR-DEM-Optical，
  **不是** Haiti。
- 但它是一作本人**实际跑过的**配置，量级上是最可信的"同门参照"：
  **30 epoch、lr 1e-4、batch 32、7:2:1 三分**。
- 注意这里**保留了独立 val 集**（0.2），和我们 v3 取消 val 的做法相反。

### 2.2 同课题组的 Haiti 相关工作：Nava et al., GMD 19:167–2026

这是目前公开文献里**唯一**用 Haiti 2021 报数字的工作（把 Haiti 当作
"unseen event"做跨事件迁移推理）：

| 项目 | 配置 |
| --- | --- |
| 训练事件 | 11 场地震诱发滑坡，约 73 000 处滑坡 |
| 最大 epoch | **500** |
| 早停判据 | **验证损失连续 30 epoch 无下降即停** |
| LR 搜索范围 | 10⁻⁴, 10⁻⁵ |
| Haiti 推理所用 LR | **0.001** |
| 优化器 / 损失 | **Adam + Focal Loss** |
| 划分 | 每区域 67% train/val、33% test；train 内再切 40% 作 val |
| patch / 采样 | 64×64、非重叠网格；背景:滑坡比 4–6（按事件 8–120） |
| 架构 | 轻量 CNN（3 个卷积块）；对比 ResNet、CBAM |
| 框架 | TensorFlow 2.8 |
| **Haiti 表现** | 摘要称未见事件 **F1 最高 82%**；正文明确 **Haiti 掉 recall** |
| Haiti 掉分归因 | ① 模型注意力偏向**沉积物填充的河床**，产生过预测；<br>② 震后**热带风暴 Grace** 又触发滑坡，使 post 信号混杂 |

**这一条对我们最有用，因为它给出了一个外部锚点：
在 Haiti 事件上，一个用 11 场事件、73 000 处滑坡、Focal Loss、
早停 30 epoch 训练出来的模型，拿到的是"F1 最高 82%、且 recall 明显下降"。**
换句话说，**Haiti 这个任务本身就是难的**，不是只有我们做得低。

### 2.3 我找不到的东西（明确记录）

- Haiti 数据集作者的**原版训练配置**：不存在公开版本。
- 博士论文 Haiti 章节的具体 epoch / lr / F1：**公开渠道取不到**（HAL Anubis 拦截）。
- 若日后需要：走机构订阅（IEEE Xplore / HAL 直连）或直接联系一作
  （`antoine.bralet89@gmail.com`）索取论文 PDF。

---

## 3. 「没收敛」假设的直接检验

你的原话是"我怀疑是他没收敛，因为我原本那个 resnet34 那个结果"。
用我们自己的记录可以直接验：

**证据 A —— 麦克那批跑满了 50 epoch 且标记 complete。**

```
results/rapid_landslide_cocd/Ours_progress.json
{"method": "Ours", "epoch": 50, "epochs": 50, "state": "complete",
 "best_validation_auprc": 0.8547302985562478}
```

`state: "complete"` 意味着没有中途早停，训练计划执行完了。

**证据 B —— 那批数字的形态是"高召回过拟合"，不是"欠训练"。**

`results/rapid_landslide_cocd/metrics.csv`：

| 方法 | partition | IoU | F1 | Precision | Recall | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DIS2（麦克迁移版） | overall | 0.5662 | **0.7230** | 0.5860 | **0.9437** | 0.8620 |
| DIS2 | asc | 0.5650 | 0.7220 | 0.5845 | 0.9444 | 0.8608 |
| DIS2 | desc | 0.5674 | 0.7240 | 0.5876 | 0.9431 | 0.8632 |

**Recall 0.9437 / Precision 0.5860** 是一个诊断性形态：

- **欠训练**的典型形态是 **P 和 R 双低**（模型还没学到东西）。
- **我们这个是 R 极高、P 中等**，即模型**几乎把所有疑似像素都标成滑坡** ——
  这是**收敛过头 / 被正负样本不平衡带偏**的形态。
- IoU 0.5662 与 F1 0.7230 的落差（IoU 明显低于 F1）也印证这点：
  IoU 对"多标"惩罚更重。

**结论：在这个任务上，"没收敛"不成立。** 麦克那批是**收敛到了**一个
高召回、低精度的解。

> 补一条旁证：`results/ours_v2/teacher_v2_progress.json` 显示 v2 teacher
> 是 20 epoch、`best_val_auprc 0.6671`。也就是说**迁移过来的数字里已经混了
> 至少两套 epoch 数（50 和 20）**，这本身就是协议不统一的证据（见 §4）。

---

## 4. 那"感觉差"的真正原因

不是收敛，是**同一份数据被两种尺子量过**：

### 4.1 尺子一：麦克时期（各跑各的）

| 项 | 状态 |
| --- | --- |
| 训练轮数 | **不统一**：`rapid_landslide_cocd` 是 50 epoch，`ours_v2` teacher 是 20 epoch |
| batch | 8（R1/R2/R3 汇总表口径） |
| 阈值 | 验证集搜出来的（`val_report_v2.csv`：0.206–0.215） |
| 骨干 | Boehm 用官方 **resnet50**（48.9922 M） |
| val 划分 | **保留** |

### 4.2 尺子二：v3（全部锁死）

| 项 | 状态 |
| --- | --- |
| 训练轮数 | **20 epoch，无早停，epoch-20 即模型** |
| batch | 16 |
| 阈值 | **固定 0.5，`reselect_on_test: false`** |
| 骨干 | Boehm 降到 **resnet34**（26.0850 M，缩 47%） |
| val 划分 | **取消**（train 1233+137=1370，test 343 不变） |

### 4.3 差异归属（对照 `reports/v3_vs_macos_result_reconciliation.md`）

| 方法 | 麦克 V1 | v3 | Δ F1 |
| --- | ---: | ---: | ---: |
| Boehm SAR U-Net++ | 0.7442 | 0.5810 | **−0.163** |
| CDNetE Early-Fusion | 0.6601 | 0.3877 | **−0.272** |
| MFEWF adapted | 0.4886 | 0.3411 | **−0.148** |

- **测试集没变**：两侧都是 11,239,424 px（686 views，asc/desc 各 5,619,712）。
- **阈值对这批数字不解释落差**：麦克期的外部席位本来就是固定 0.5 报的。
- **主因是训练协议统一**，其中"Boehm 骨干 resnet50→resnet34"是最硬的一条。

---

## 5. 这轮得到的三个可执行结论

### 5.1 不要再往"没收敛"方向找原因

数据形态（R 0.94 / P 0.59）已经排除了欠训练。**继续调 epoch 不会解决
"感觉差"**，因为麦克期的数字本来就是过拟合式高召回撑起来的。

### 5.2 真正待决的两件事（与上一版报告一致，此处不再展开）

1. **阈值口径**：v3 固定 0.5 / 恢复验证集搜阈值 / 固定 0.5 + 附录给 F1(θ) 曲线。
   我仍倾向第三条。
2. **Boehm 骨干**：这是**你已明确否决的点**——你指出 resnet50 版本没问题、
   麦克训练也是这么用的。据此，这条不构成"需要修复的缺陷"，
   而应作为**协议说明**处理：外部基线保留各自公布骨干，
   并在表注里写明"external baselines keep their published backbone"。
   → **本报告据此撤回上一版 §5.2 中"恢复 resnet50"的建议表述。**

### 5.3 关于 `train_modes` 不对称：**已核实，此前的记录有误，现更正**

> **更正（重要）**：上一轮读 `reports/g10_geometry_distillation_alignment_audit.md`
> 时，我记录了一条"`train_modes` 训练协议不对称"，称外部基线用
> `train_modes=False`（只喂真实配对）、我们的臂用 `True`（2/3 标签置零）。
> **这条记录是错的，现在撤回。**

**更正依据 —— 外部席位日志（当前生效的 v3 运行）：**

```
experiments/single_orbit_baselines_v3/logs/boehm.log
  [17:47:44] Boehm SAR U-Net++: 26.0850 M, loss=ce, seat=boehm,
             train=4110 pairs = 8220 views (1370 locations x 3 modes x 2 orbits), test=343
experiments/single_orbit_baselines_v3/logs/cdnette.log   同上，train=4110 pairs = 8220 views
experiments/single_orbit_baselines_v3/logs/mfewf.log     同上，train=4110 pairs = 8220 views
```

`1370 locations × 3 modes × 2 orbits = 8220 views` —— **外部席位和我们的臂
用的是完全相同的训练数据配置（`train_modes=True`）。**

**代码依据：**

```
cocd/scripts/23_train_external_baselines.py:224
    train_set = rapid.HaitiPairs(tid, True)      # ← True，不是 False
cocd/scripts/23_train_external_baselines.py:70
    EXPECTED_TRAIN_VIEWS = 8220                  # 1370 locations x 3 modes x 2 orbits
```

该脚本第 234 行还有一道断言：若 views ≠ 8220 就直接 `SystemExit`。
**训练确实跑通了，说明 8220 成立。**

**g10 报告为何写错：** 它的依据是 `scripts/23_boehm:44`、`scripts/24:18`、
`scripts/26:123` —— 这三个脚本**已经不存在了**（当前 `cocd/scripts/` 下与外部
基线相关的只剩 `23_train_external_baselines.py` 和 `24_external_seat_table.py`）。
g10 记录的是**更早一版的脚本**，那一版外部基线可能确实用 `False`，
但**它不反映当前 v3 的实际配置**。

**这意味着什么：**

| 结论 | 状态 |
| --- | --- |
| "外部基线与我们的臂训练数据配置不同" | ❌ **不成立**，两者都是 `train_modes=True` |
| "2/3 训练样本标签被置零"这一现象本身 | ✅ **成立**（`21:70` 的 `y = land if mode==0 else np.zeros_like(land)`） |
| 该项构成主表的可比性缺陷 | ❌ **不成立** —— 既然两边同配置，它就是**共有的协议特征**，而非不对称 |

**因此：主表在同一训练数据配置下是可比的。** 这条不必再作为待决事项，
也不需要在论文里作为 limitation 单列 —— 但"我们的训练集含 2/3 合成无变化
样本"这件事本身仍应在数据/训练设置里如实描述，因为它是**协议的实际内容**，
不是缺陷。

> 保留一句提醒：这个更正说明**报告里的脚本行号引用会随重构失效**。
> 日后凡引用 `scripts/NN_*.py:行号` 的结论，都应回到当前文件重新核对一次，
> 而不是转引旧报告。

---

## 6. 一句话回答

> **原版数据集作者没有留下训练配方 —— Haiti 是一份纯数据集发布，
> 无 baseline、无 epoch、无 lr。** 同课题组最接近的两个参照是
> SARDINet 的 `30 epoch / lr 1e-4 / batch 32`，以及 Nava et al. 2026 用
> `最多 500 epoch / 早停 30 / lr 1e-3 / Adam + Focal Loss` 训 11 场事件后
> 迁移到 Haiti，拿到 F1 最高 82% 且 **recall 明显下降**。
>
> **"没收敛"可以直接排除**：麦克那批跑满 50 epoch、`state: complete`，
> 且 Recall 0.9437 / Precision 0.5860 是**收敛到高召回过拟合**的形态，
> 与欠训练（P/R 双低）正好相反。
>
> **"感觉差"的真实来源是尺子被换了**：麦克期各跑各的（50 与 20 epoch 混用、
> batch 8、验证集搜阈值、Boehm 用 resnet50），v3 全部锁死
> （20 epoch 无早停、batch 16、阈值固定 0.5、Boehm 降 resnet34）。
> 测试集本身**一个像素都没变**。
>
> **另需注意一处已更正的记录**（§5.3）：此前认为"外部基线与我们的臂训练数据
> 配置不对称"，经日志核实**不成立** —— 外部席位同样是
> `1370 locations × 3 modes × 2 orbits = 8220 views`，两边配置一致。
> 该条撤回，主表在同一训练数据配置下可比。
>
> **结论：不用再往收敛方向找原因。** 下一步该定的是阈值口径（§5.2 的 A/B/C，
> 仍倾向 C），以及是否需要为"外部基线在固定阈值下领先"这一旧结论给出方法学回应。
