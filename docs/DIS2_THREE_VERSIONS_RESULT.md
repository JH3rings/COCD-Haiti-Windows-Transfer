# DIS2 三个版本：结果整理与对比

生成于 2026-09-17 13:20。全部数字为**测试集、固定阈值 0.5**。
冻结协议指纹：686 样本 × 128 × 128 = 11,239,424 像素；asc 343 / desc 343；G10 = 1,039,406 像素。

---

## 0. 先回答你的三个问题

### 「之前效果非常好的那个 DIS2 是哪个版本？」

是**旧 DIS2-style**：

| | |
|---|---|
| 产物 | `experiments/rapid_landslide_cocd/DIS2.pt`（配 `DIS2_metrics.csv`、`DIS2_progress.json`） |
| 训练脚本 | `scripts/21_train_rapid_landslide_cocd.py`，`method=DIS2` |
| 骨架 | 我们自己的 `SingleOrbitStudent(correction=True)`，ConvNeXt-Tiny + ImageNet1K |
| 测试 | **IoU 0.5662 / F1 0.7230 / AUPRC 0.8620**（val AUPRC 0.8545） |

它「非常好」是真的 —— 在它所属的 **V1 协议**里，它是除教师之外最强的一个：

| V1 协议方法 | IoU | F1 | AUPRC |
|---|---:|---:|---:|
| S0（学生，无蒸馏） | 0.5302 | 0.6930 | 0.8391 |
| **T（双轨教师，上界参照）** | **0.5795** | **0.7338** | **0.8710** |
| KD（普通输出级 KD） | 0.5541 | 0.7131 | 0.8545 |
| Ours（残差门控） | 0.5619 | 0.7195 | 0.8589 |
| **DIS2（旧 DIS2-style）** | **0.5662** | **0.7230** | **0.8620** |

比 KD 高 +0.0121，比 Ours 高 +0.0043，距教师只差 0.0133。

**但它不是官方 DIS2**：它是「我们自己的骨架 + 一个手写的残差匹配损失项」。

### 「它跟现在训练的那个比怎么样？」

现在训练的 **DIS2-port** 测试 **IoU 0.6272 / AUPRC 0.8963**，在数值上全面超过旧版（+0.0610 IoU）。
但两者**协议不同**（batch 2 vs 8），这个差值不能直接当作方法改进。
真正的差别在**蒸馏内容**上，见 §3。

### 「DIS2 凭什么能比教师还厉害？以前跑那么多版本怎么一个都超不过？」

**DIS2 没有魔力，而且同口径下它还不如我们自己的 R3（0.6272 vs 0.6298）。** 两个常见误读：

1. 它「超过」的是教师的**较弱读数** —— DIS2-port 0.6272 vs Teacher_self 0.6006。同一教师权重的 z34 读数（= Teacher_dual）是 0.6250，DIS2-port 对它只是打平（IoU +0.0022、AUPRC −0.0009）。
2. 「超过教师」是**每一个**学生臂共有的现象（R1 0.6211 / R2 0.6194 / R3 0.6298 全都超过 Teacher_self），因此它不可能来自 DIS2 的蒸馏规则。

真正成因是**权重接力链 + 额外监督**：ImageNet → S0（50 ep）→ Teacher（50 ep）→ Student（20–50 ep），学生从教师权重起步（地板已是 0.6006）、累计训练量最多、且额外拿到教师的软目标（LUPI）。

V1 时代没有学生能超过教师，是因为当时学生从 ImageNet 从零训、而教师在**测试时用双轨输入**（`scripts/21_train_rapid_landslide_cocd.py` 第 78 行与第 58–63 行），比较本身不对等。

**完整论证与代码依据见 §6。**

---

## 1. 三个叫「DIS2」的东西

| 代号 | 产物路径 | 骨架 / 初始化 | 蒸馏内容 | 协议 | 状态 |
|---|---|---|---|---|---|
| **旧 DIS2-style** | `experiments/rapid_landslide_cocd/DIS2.pt` | 我们的 `SingleOrbitStudent(correction=True)`，ImageNet1K | 残差 cosine 匹配 + 输出 MSE，权重 0.1 / 0.1 | V1，batch 2 | **保留**（IoU 0.5662） |
| **官方 width-50** | ~~`experiments/grsl_external_baselines/dis2_width50_haiti/`~~ | 官方 `DLKD_ver4(gf_dim=50)`，**无预训练加载** | 官方三段 + 正交项 | V2，batch 2 | **已删除**（IoU 0.1829，未收敛） |
| **DIS2-port** | `experiments/ours_v2/DIS2_seed42.pt` | 我们的 `OursV3Student('R1')`，ImageNet1K，教师自轨初始化 | 官方三段 + 正交项（逐字对照） | V3，batch 8，无早停 | **本轮完成**（IoU 0.6272） |

三者的关系：

- 旧 DIS2-style 用的是**我们的骨架**，但蒸馏项是**我们自己写的**（名字里有 DIS2，规则不是 DIS2 的）。
- 官方 width-50 是**官方规则 + 官方骨架**，但条件全换（从头训、1/3 数据、8 个辅助头分走 77% 梯度），没收敛。
- DIS2-port 是**官方规则 + 我们的骨架/协议**——你要的「正确复现」就是这一个。

---

## 2. 测试结果总表（阈值 0.5，overall）

| 方法 | IoU | F1 | AUPRC | 训练量 | 口径 |
|---|---:|---:|---:|---|---|
| 旧 DIS2-style | 0.5662 | 0.7230 | 0.8620 | 50 ep | V1，batch 2 |
| 官方 width-50 † | 0.1829 | 0.3092 | 0.3732 | 50 ep | V2，batch 2 |
| Teacher_self | 0.6006 | 0.7505 | 0.8820 | 50 ep | batch 2 |
| Teacher_dual | 0.6250 | 0.7692 | 0.8972 | — | batch 2 |
| R1 | 0.6211 | 0.7663 | 0.8924 | 20 ep（早停） | V3 |
| R2 | 0.6194 | 0.7650 | 0.8932 | 30 ep（早停） | V3 |
| R3 | **0.6298** | **0.7728** | **0.8972** | 50 ep | V3 |
| **DIS2-port** | **0.6272** | **0.7709** | **0.8963** | 50 ep | V3 |

† 该实现已按你的指示删除（产物 + 脚本 + 模型定义），数字保留在此作为记录。

**分区块对照（DIS2-port vs R3）**：

| 分区 | DIS2-port IoU | DIS2-port AUPRC | R3 IoU | R3 AUPRC |
|---|---:|---:|---:|---:|
| G00 | 0.6181 | 0.8890 | 0.6208 | 0.8905 |
| G01 | 0.6402 | 0.9071 | 0.6414 | 0.9073 |
| G10 | 0.6335 | 0.9023 | **0.6372** | 0.9030 |
| G11 | 0.6910 | **0.9287** | 0.6907 | 0.9270 |
| asc | 0.6261 | 0.8957 | 0.6308 | 0.8969 |
| desc | 0.6283 | 0.8969 | 0.6287 | 0.8976 |

两者在**所有分区上都只差 0.0004–0.0047**，交替领先。G10（target 畸变、counter 可靠）上 R3 略好，G11 上 DIS2-port 的 AUPRC 略好。

---

## 3. 旧 DIS2-style vs DIS2-port：逐项对比 ← 核心

| 维度 | 旧 DIS2-style | DIS2-port |
|---|---|---|
| 协议 | V1（batch 2） | V3（batch 8，无早停） |
| 骨架 | `SingleOrbitStudent(correction=True)` | `OursV3Student('R1')` |
| 初始化 | ImageNet1K | 同 + 教师自轨 |
| **蒸馏对象** | **修正量 `r`（残差）** | 四级特征图 + penultimate 特征 + logits |
| **蒸馏公式** | `0.1·(1 − cos〈r_s, r_t〉)` 按 GT×教师置信度加权<br>`+ 0.1·MSE(σ(z_s), σ(z_t))` | 四级 `L2`（池化后）+ penultimate `L2` + `T=2` logits `KL` + 正交项 |
| 蒸馏位置 | 只 1 个尺度（`f[1]`） | 4 个尺度 + 2 个额外抽头 |
| 是否官方规则 | **否**（我们自己写的近似） | **是**（逐字对照 `DLKD_ver4.py:554-574`） |
| 训练损失总构成 | `land_loss` + 0.1 + 0.1 | `0.5·seg(z4) + 0.5·seg(z34)` + KD 各项 |
| 只在不含合成样本的 pair 上蒸馏 | 是（`real` 掩膜） | 否（全部样本） |
| **测试 IoU** | 0.5662 | **0.6272** |
| **测试 AUPRC** | 0.8620 | **0.8963** |

### 一个值得注意的巧合

旧 DIS2-style 的损失项 `dis2_style_loss` 匹配的是 **`rs`（学生的修正量）对 `rt`（教师的双轨修正量）**，
不是匹配原始激活：

```python
# losses/landslide_cocd.py:16
def dis2_style_loss(student_r, teacher_r, teacher_logits, target):
    a = F.normalize(student_r, dim=1); b = F.normalize(teacher_r.detach(), dim=1)
    d = 1 - (a * b).sum(1)                      # 逐通道 cosine 距离
    fg = torch.sigmoid(teacher_logits[:, 0]).detach()
    w = target * fg + (1 - target) * (1 - fg)   # GT 与教师置信度一致性加权
    ...
```

而**官方 DIS2 匹配的是激活**（四级融合特征、penultimate 特征、logits）。
`docs/DIS2_DISTILLATION_PORT.md` §3 批评官方「蒸馏激活而非变换、目标不可达」，改进方向正是「蒸馏修正量」。

> **也就是说：旧版之所以好，很可能是它无意中做对了这件事。** [推断]
> 而搬到官方规则后，反而把这个性质丢掉了 —— DIS2-port 是「忠实复现官方」，所以它匹配的是激活。
> 这条推断目前**不能当结论**：两版协议不同（batch 2 vs 8），且旧版权重只有 0.1。
> 要验证它，只需跑一次 `38_run_dis2_port.sh R1`（同口径对照），或把 DIS2-port 的 `feat/pen` 换成残差匹配。

---

## 4. DIS2-port 与同口径臂怎么比

**训练量对等的只有 R3**（都是 50 epoch、batch 8、关闭早停）：

> **R3 0.6298 vs DIS2-port 0.6272 → R3 高 +0.0026 IoU**

这是本轮唯一能直接归因的比较。与 R1（早停于 ep20）、R2（ep30）比**不可归因**——它们训练量只有一半左右。

与教师相比：

- DIS2-port 0.6272 > Teacher_dual 0.6250（+0.0022），> Teacher_self 0.6006（+0.0266）。
- 但注意 DIS2-port 的**起点就是 Teacher_self 的权重**（`load_teacher_self`），所以 +0.0266 全部来自 50 epoch 训练，
  且单种子、且 Teacher 是 batch 2 训的。**这个「超过教师」不能写进论文**。

---

## 5. 正交项的负面结果（预期被证实）

DIS2 的第三件事是 `orthogonality_loss`（官方权重 1.0，`DLKD_ver4.py:566`）。
迁移时我把它映射成「学生的修正量 `r_k` vs 它所修正的状态 `p_k`」，并留下一个可证伪的预测：
它会像官方那样塌掉。

**实测（每分钟 epoch 打印）**：

| epoch | feat | pen | logit | div |
|---:|---:|---:|---:|---:|
| 1 | 0.0007 | 0.0003 | 0.0050 | **0.0019** |
| 2 | 0.0004 | 0.0003 | 0.0043 | **0.0001** |
| 3 | 0.0003 | 0.0003 | 0.0046 | **0.0000** |
| 50 | 0.0002 | 0.0003 | 0.0028 | **0.0000** |

**与官方完全一致**：50 epoch 内塌到 0。官方那边是 `9.31e-05`，我们这边是 `0.0000`。

结论：**正交约束在这个数据尺度上就是空转的，这不是我们的实现问题。**
三项蒸馏倒是真的在降（feat 0.0007→0.0002，logit 0.0050→0.0028），说明学生确实在向教师靠。
这条负面结果可以正面写进论文：约束项在本数据集上不活跃，说明「修正退化为重编码」这一失效模式在这里不是主要矛盾。

---

## 6. 为什么「DIS2-port 超过教师」不是 DIS2 的功劳

DIS2-port 的 0.6272 高于 Teacher_self 的 0.6006，容易被读成「DIS2 的蒸馏规则让单轨学生超过了教师」。核对代码后这个读法不成立。

### 6.1 它比的是教师的**较弱读数**，同读数只是打平

- `predict_student` 取 `model(...)[1]` = **z34**（P3 修正后的完整输出）；`predict_teacher` 的 `'self'` 取 `za[0]` = **z0**（完全未修正的目标路径）。两者不是同一个读出分支。
- 同一教师权重的 z34 读数就是表里的 **Teacher_dual = 0.6250**。DIS2-port 0.6272 对它是 IoU +0.0022、**AUPRC −0.0009**，即打平。

### 6.2 「超过教师」是全部 R 臂共有的现象，与 DIS2 的规则无关

| 臂 | 蒸馏规则 | overall IoU | 相对 Teacher_self |
|---|---|---:|---:|
| Teacher_self | —（起点） | 0.6006 | — |
| R1 | 输出级选择性 KL | 0.6211 | +0.0205 |
| R2 | 同上 | 0.6194 | +0.0188 |
| R3 | 同上 | **0.6298** | +0.0292 |
| Teacher_dual | —（双轨，测试时用 counter） | 0.6250 | +0.0244 |
| DIS2-port | 官方多级蒸馏 | 0.6272 | +0.0266 |

**每一个 R 臂都超过 Teacher_self，R3 还超过 Teacher_dual。** 若「超过教师」来自 DIS2 的规则，R 臂不该有这个现象。同口径（50 ep / batch 8 / 无早停）下 **DIS2-port 还低于 R3**。

### 6.3 真正成因：这是一条权重接力链，学生站在链尾

| 环节 | 代码依据 | 起点 | 训练 | 测试 IoU |
|---|---|---|---|---:|
| ImageNet-1K | `models/landslide_cocd.py:11` `convnext_tiny(weights=IMAGENET1K_V1)` | — | — | — |
| S0 | `SingleOrbitStudent(correction=False)` | ImageNet | 50 ep 单轨 | 0.5302 |
| V2 Teacher | `23_train_ours_v2.py:197` `model.load_s0(S0.pt)`（只覆盖 `backbone.*`） | **S0 骨干** | 50 ep 双轨 | 0.6006 / 0.6250 |
| V2/V3 Student | `train_student:248` `model.load_teacher_self(teacher)` | **Teacher 骨干** | 20–50 ep + KD | 0.6194–0.6298 |

学生不是从零训练，而是从教师权重继续训练，累计 120–150 epoch 精调（Teacher 只有 100）。脚本第 252–253 行有断言保证「修正量为零时 z34 逐位等于 Teacher self path」，即**学生的地板就是 0.6006**。所以 +0.0266 是「教师权重之上再训 50 epoch + KD 软目标」的增量。这同时是 LUPI 结构：学生拿到的监督（GT + 教师软目标）严格多于教师（只有 GT），学生超过教师是这一设定的预期行为。

### 6.4 为什么 V1 时代没有一个学生超过教师

`scripts/21_train_rapid_landslide_cocd.py` 里有两处和 V2/V3 不同的设定：

1. **V1 学生不继承任何权重。** 第 78 行 `m=... if kind=='T' else SingleOrbitStudent(...)` 是**新建**，骨干走 ImageNet 初始化。V1 的 S0 / KD / DIS2 / Ours 全都从 ImageNet 从零训 50 epoch。
2. **V1 是单轨学生 vs 双轨教师。** 第 58–63 行：`kind=='T'` 时 `m(a,d)` / `m(d,a)` 用两个轨道（`da`/`bd` 走 counter），学生走 `m(torch.cat((a,d)))` **只用一个轨道**。教师测试时持有特权信息。

V1 的实测数字（`experiments/rapid_landslide_cocd/`）：S0 0.5302（单轨）、old T **0.5795（双轨）**、Vanilla KD 0.5541（单轨）、旧 DIS2-style 0.5662（单轨）、Ours-V1 0.5619（单轨）。**教师 0.5795 是双轨读数，学生全是单轨**，这个比较本身就不对等，学生赢不了属于正常。

V2/V3 同时改掉了这两点（教师从 S0 续训、学生从 Teacher 续训），现象才翻转。**结论：本项目的「学生超过教师」来自权重继承 + 额外训练 + KD，不来自任何一版 DIS2 的蒸馏规则。**

---

## 7. 必须披露的口径问题

1. **batch 不同**：旧 DIS2-style / Teacher 是 batch 2，DIS2-port / R 臂是 batch 8。lr 与 weight decay 两边相同（`AdamW(1e-4, wd=1e-4)`，未按 batch 缩放），因此同为 50 epoch 时**步数差 4 倍**：3,699 样本/epoch 下 Teacher 走 92,500 步，学生走 23,150 步。论文里只陈述事实并声明不可比，**不把该差异归因**；本轮已决定不补跑 batch 2 对照（见 `docs/ANALYSIS_PROMPT_VERSION_AUDIT.md` 的「batch 2 / 8 的实际代价」）。
2. **早停不同**：R1 停在 ep20、R2 停在 ep30、R3 与 DIS2-port 跑满 50。**只有 R3 与 DIS2-port 训练量对等。**
3. **单种子**：全部 seed=42，没有多种子。
4. **共用起点**：R 臂与 DIS2-port 都从 Teacher_self 初始化，所以「涨了多少」是相对该起点的增量。
5. 冻结协议对**所有**方法用固定阈值 0.5；主比较应同时给阈值无关的 AUPRC。

---

## 8. 产物与命令

保留：

| 内容 | 路径 |
|---|---|
| 旧 DIS2-style 权重 / 指标 / 进度 | `experiments/rapid_landslide_cocd/DIS2{.pt,_metrics.csv,_progress.json}` |
| 旧 DIS2-style 损失定义 | `losses/landslide_cocd.py::dis2_style_loss`（V1 脚本 `21` 在用） |
| DIS2-port 权重 / 指标 | `experiments/ours_v2/DIS2_seed42.pt`、`results_DIS2_seed42.csv` |
| DIS2-port 逐 epoch 日志 | `experiments/ours_v2/train_log.txt`（`[DIS2]` 行） |
| 官方蒸馏规则源码（迁移依据） | `third_party/landslide_baselines/dis2/models_bank/DLKD_ver4.py` |
| 机制说明 | `docs/DIS2_MECHANISM_AND_DIAGNOSIS.md` |
| 迁移设计 | `docs/DIS2_DISTILLATION_PORT.md` |

已删除（移入系统废纸篓，可恢复）：

```
experiments/grsl_external_baselines/dis2_width50_haiti/    # 崩掉的官方复现（526 MB）
scripts/29_train_dis2_width50_haiti.py
scripts/30_run_dis2_width50.sh
scripts/33_eval_dis2_width50.py
scripts/34_diagnose_dis2_loss.py
models/dis2_haiti.py
scripts/__pycache__/29_*.pyc  scripts/__pycache__/33_*.pyc  models/__pycache__/dis2_haiti.*.pyc
```

重跑 DIS2-port：

```bash
cd /Users/zhangjiuqi/Desktop/distortion
bash Haiti_SAR_GRSL_Audit/scripts/38_run_dis2_port.sh DIS2      # 50 epoch，batch 8，无早停，约 3 h
bash Haiti_SAR_GRSL_Audit/scripts/38_run_dis2_port.sh R1        # 同口径对照臂（若要拆开训练量）
```

运行期可观察项：每个 epoch 打印 `feat / pen / logit / div`，用来盯正交项是否再次塌零。
