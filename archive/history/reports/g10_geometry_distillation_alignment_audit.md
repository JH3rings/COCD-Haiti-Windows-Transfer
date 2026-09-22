# G10 geometry-conditioned cross-orbit distillation：代码对齐审计

审计时间 2026-09-17。审计对象：`scripts/21`（V1）、`scripts/23`（V2–V3）、`models/landslide_cocd.py`、`models/landslide_cocd_v2.py`、`losses/landslide_cocd.py`、`losses/distill.py`、`losses/corrective_distill.py`。
审计轮**未修改任何网络、未启动任何训练**。所有结论附 `文件:行号`。
**审计后的拍板、最小改动落地、验证与运行见 §8**（§5 与 §6 是审计当时的建议，已被 §8 取代）。

---

## 0. 判定

### **PARTIALLY ALIGNED**

拆成三块看：

| 部分 | 判定 | 依据 |
|---|---|---|
| Teacher 结构（§8 T1–T5） | **ALIGNED** | 五项全部成立，见 §2。 |
| Student / R3 结构（§9） | **ALIGNED** | Ω 继承 + Q 驱动 + P3 递进 + 输出级选择性 KL 四步与 §9 逐条对应，见 §3。 |
| **蒸馏权重是否 geometry-conditioned（§10 Q3）** | **MISALIGNED** | geometry mask 已在训练循环里被**计算出来**，但被送进一个把它丢弃的分支（`scripts/23:107` 是 `'full'` 分支，R 臂走的是 `scripts/23:105` 的 `'gain'` 分支）。当前 R1/R2/R3/DIS2-port 全部是 **Case B：generic cross-orbit distillation**。 |

**关键好消息：几何加权规则不需要新写。** `selective_kl` 里已经有一条完整的、从未被调用的 `'full'` 分支
（`scripts/23:107`：`w = (1. + g10.float()) * gain * protect`），而 V1 的 `Ours` 用过的
`losses/landslide_cocd.py:8-11 gain_gate` 是它的 hard-gate 前身。缺口只在**派发与配置**，不在算法。

---

## 1. 逐问回答（§20）

### Q1. 当前 Teacher 与理论是否一致？

**一致。** 详见 §2 的 T1–T5。两处需要写进论文的限定：

- Teacher 的 z0（self 路径）是**同一套权重**单轨读出，而这套权重是用双轨数据训出来的。它不构成推理期信息泄漏（前向只吃 target），但它是"学生的地板就是 0.6006"这一事实的来源（`scripts/23:248` `load_teacher_self` + `scripts/23:252-253` 断言）。
- 训练数据经 `train_modes=True` 扩成 3 倍，其中 2/3 的样本标签被强制置零（`scripts/21:45`）。这不是 geometry masking，但确实使「Teacher 在完整 landslide GT 上训练」这句话在当前代码下**不字面成立**。见 Q4。

### Q2. 当前 R3 与理论是否一致？

**结构一致、监督不完全一致。** 见 §3 的映射表。R3 已经做到："单轨编码器/解码器从 Teacher self 继承"、"只学驱动"、"修正以加法残差进入"、"P3 的驱动输入是已被 P4 修正过的状态"。
差的一步是 §11 的**加权**：R3 的 KD 权重 `gain × protect` 不含 geometry，也不含"教师改善的绝对量"。

### Q3. 当前 geometry mask 是否真正进入 R3 training？

**没有。是 Case B。** 证据链：

1. `scripts/23:283` 确实在训练循环里构造了 G10 掩膜：
   `g10 = torch.cat(((ga > 0) & (gd == 0), (gd > 0) & (ga == 0)))` —— 语义正确（ASC 半边取 `ga` 为目标、`gd` 为反轨；DESC 半边对称）。
2. `scripts/23:286-287` 把它传给 `selective_kl(..., g10, kd_mode)`。
3. 但 `kd_mode` 来自 `cfg['kd']`，R1/R2/R3 是 `'gain'`（`scripts/23:55-57`），于是 `selective_kl` 走 `scripts/23:105`：
   `elif mode == 'gain': w = gain * protect` —— **`g10` 未被读取**。
4. 唯一使用 `g10` 的 `scripts/23:107`（`'full'` 分支）在 `CONFIGS` 中无任何条目可达。
5. `ga` / `gd` 在 `train_student` 里除此之外**没有第二次使用**；`real` 在 `scripts/23:272` 被解包后从未引用。
6. 几何相关的 smoke 断言（`scripts/36:190`）只检查 forward 不提 geometry，不检查 KD 权重。

**因此：当前 R3 的性能提升与 geometry 无关，geometry 目前只是 post-hoc 的评价分组。** 这一点必须在论文里明确，否则 §13 的 `R3 vs DIS2-port` 叙事（"physics-conditioned"）没有代码支撑。

### Q4. 当前 G10 在代码中到底怎样定义？

四处，语义一致（`>0` 表示该轨在该像素几何失真/不可靠；`read()` 读原始 float，用 `>0` 阈值化）：

| 位置 | 表达式 | 用途 |
|---|---|---|
| `scripts/21:28` | `ga = asc_pre_geom.tif`，`gd = desc_pre_geom.tif` | 数据源头，按轨道各一张静态掩膜 |
| `scripts/23:283` | `(ga>0)&(gd==0)` 与 `(gd>0)&(ga==0)` 拼接 | 训练用的 g10（当前未生效） |
| `scripts/23:179-181` | `gt, gc = pred['gt']>0, pred['gc']>0`；`('G10', gt & ~gc)` | 评价分区；`gt=cat(ga,gd)`、`gc=cat(gd,ga)`（`scripts/23:132-133`） |
| `scripts/22:63` | `region(gt,gc)` 同式 | 旧分析脚本 |
| `losses/landslide_cocd.py:10` | `(target_geom>0)&(counter_geom==0)` | V1 `Ours` 的门控（**真用过**） |
| `losses/corrective_distill.py:4` | `target*(1-counter)` | phase-2 遗留，**不要求 target 失真**，定义与前几处不同——不要混用 |

指纹核对通过：R3 / DIS2-port 的 `results_*_seed42.csv` 里 G10 = **1,039,406** 像素、G01 = 1,039,406、G00 = 9,058,636、G11 = 101,976，与冻结协议一致。

### Q5. 当前 selective KL / correction loss 到底用了哪些 pixel？

- **selective KL（R1/R2/R3，`kd='gain'`）**：`scripts/23:286-287` 对 `z4` 与 `z34` 各算一次、取平均，目标分别是教师的 `z4` / `z34`，参照基线是教师的 `z0`（都是 target-only 读出）。逐像素权重 `w = gain × protect`：
  - `gain = ((e0 - et) / (e0 + EPS)).clamp(0, 1)`（`scripts/23:100`）——**相对**改善，不是 §11 的 `ReLU(e0-et)`；
  - `protect = (et < es)`（`scripts/23:101`）——学生已经比教师好的位置不学；这一项依赖学生自身预测，是移动靶；
  - 聚合是 `balanced_mean`（`scripts/23:113` → `scripts/23:84-91`）：正负类各自取均值再平均，**分母是类内像素数、不是 `sum(w)`**。所以被 `w=0` 的像素仍然进了分母，KD 不是"只在选中像素上算"，而是"权重集中在选中像素"。
  - 数据侧：**全部 3 种 mode 都参与**（`real` 未使用），其中 2/3 的样本 `yy` 全零（`scripts/21:45`）。
- **correction loss（`losses/landslide_cocd.py:12`）在当前 V3 中完全没有被调用。** R3 的修正 `r3/r4` 拿不到任何直接监督 —— 它只通过"继承的 Ω + 零初始化驱动 + 输出级 KL"间接学出来。这是 §9 与当前实现之间**唯一的精神落差**：§9 说 Student 学 `R_S = Q(F_t)`，而 R3 没有任何项去要求 `R_S ≈ R_T`。
- **output-level 分割损失**：`scripts/23:279` 对 `z4` 与 `z34` 各取 0.5，`land_loss` = BCE(pos_weight) + 0.2·Dice（`losses/landslide_cocd.py:4-7`），**全图像素参与、无 mask**。✓ 符合 §4。

### Q6. DIS2-port 是否完全不使用 geometry？

**是。** `scripts/23:289-290` 走 `dis2_multilevel_kd(taps, t)`（`losses/distill.py:82`），签名里没有 geometry；`selective_kl` 在该分支不被调用；`losses/distill.py:19` 的模块级声明与 `scripts/36:190` 的 smoke 断言共同保证 forward 不接触 geometry。骨干、Teacher、初始化、batch、epoch 与 R3 同（`CONFIGS['DIS2'] = {'variant':'R1','kd':'dis2'}`），唯一差异就是蒸馏规则。✓ §13 的要求已满足。

### Q7. 是否已经有 Ours-NoGeometry 等价实验？

**有，而且就是现在这一版 R3。** 因为 geometry 从未进入训练（Q3），Control 1 的两个臂里"关闭几何"那一半已经跑完：

| Control 1 的两臂 | 状态 | overall IoU | AUPRC | G10 IoU | G10 AUPRC |
|---|---|---:|---:|---:|---:|
| **Ours-NoGeometry** = 当前 R3 | **已有结果** | 0.6298 | 0.8972 | 0.6372 | 0.9030 |
| Ours + G10 加权 | 待跑 | — | — | — | — |

含义很重要：**要做的不是补一个"无几何"的臂，而是补一个"有几何"的臂。** 现有 0.6298 可以直接当 Control 1 的基线。

### Q8. 是否已经有 Single+Mask 实验？

**没有。** 最接近的三件事都不是它：

| 已有 | 它实际做了什么 | 为什么不是 Control 2 |
|---|---|---|
| `scripts/11_phase1_b1_b2.py` | 用 SAR 去**预测** geometry mask（`label_definition="binary_bad = geometry_mask > 0"`，`11:379`） | 目标是几何，不是滑坡 |
| `scripts/13_landslide_downstream.py` | geometry 作为**损失掩膜**（`geom` 开关，`13:71-85`），且属已废弃的畸变分割任务 | 既非输入，也非当前任务 |
| `scripts/12_phase2_distill_app.py` | 用**预测出来的**几何当 KD 门控（`1-g`，`12:200`） | phase-2 旧协议，不在当前主表 |

Control 2（`target SAR + M_t → landslide`）需要新写，但代价很小（单通道加进去、复用 S0 的训练循环）。

### Q9. 当前方法中哪些组件可以删除而不破坏核心逻辑？

（**本轮不删**，仅列出。按"删了完全无影响"到"删了要改配置"排序。）

| 组件 | 位置 | 现状 |
|---|---|---|
| `selective_kl` 的 `'all'` 分支 | `scripts/23:102-103` | 无 `CONFIGS` 条目可达 |
| `selective_kl` 的 `'full'` 分支 | `scripts/23:106-107` | **当前不可达——它是本轮的修复目标，别删** |
| `dis2_logit_only` + `'dis2-logit'` 派发 | `losses/distill.py:114`、`scripts/23:291-292` | 无配置可达的探针 |
| `selective_multilevel_kd` + `gain_protect` + `else` 派发 | `losses/distill.py:123-209`、`scripts/23:293-298` | 无配置可达；且它本身是 geometry-free 的 |
| `parts_acc['w_frac']` | `scripts/23:296-298` | 只服务不可达分支 |
| `CONFIGS['R0']` / `CONFIGS['R4']` | `scripts/23:54,58` | 已声明未跑（保留有价值，别删） |
| `OursV2Student` | `models/landslide_cocd_v2.py:176` | 只被 `scripts/32` 的 smoke 引用 |
| `models/student.py`、`models/dual_teacher.py`、`models/correction_module.py`、`losses/corrective_distill.py` | — | 只被 `scripts/17_train_cocd.py` 引用（phase-2 遗留），不在 V1/V2/V3 任何主链路 |
| `losses/landslide_cocd.py` 的 `gain_gate` / `correction_loss` / `dis2_style_loss` | — | 只被 `scripts/21` 的 V1 分支引用 |
| `report()` 里 `if mask.sum()==0: continue` | `scripts/23:182-183` | 空区域跳过是正确行为，**保留**（别当成死代码删） |

### Q10. 最少需要改哪几行/哪几个模块？

**结论：1 处派发 + 1 处配置 = 2 行，几何加权即可生效；若要严格按 §11 的公式，另加 1 行。** 详见 §5。

---

## 2. Teacher 侧逐条核验（§8 T1–T5）

| 编号 | 问题 | 结论 | 代码依据 |
|---|---|---|---|
| **T1** | `z0` 是否真的只依赖 target orbit？ | **是** | `FPNStateDecoder.states`（`models/landslide_cocd_v2.py:101-111`）里 `z0 = decode(c2,p3,p4)`，三者全部来自 `pyramid(features)`，而 `features` 是 `backbone.encode(target)`（`landslide_cocd_v2.py:143`）。`forward_pair` 的两个方向各自把 `fa`/`fd` 当 target（`:147-148`），`predict_teacher` 取 `za[0]`/`zd[0]`（`scripts/23:129`）。counter 与 geometry 都不在 z0 的通路上。 |
| **T2** | `z34` 是否在 target 表示上加 counter-induced correction？ | **是，而且是纯加法** | `p4c = p4 + r4`（`:106`），`p3c = l3(features[1]) + up(p4c)`（`:107`，旁路取自 **target** 自己的 c3，counter 不重入），`z34 = decode(c2, p3c + r3, p4c)`（`:110`）。counter 只通过 `r3`/`r4` 进入。 |
| **T3** | `Ω(a,c)` 是否表示"相对无 counter 的增量"？ | **是** | `models/landslide_cocd_v2.py:54-63`：`return op(cat(a,c),1)) - op(cat(a,zeros),1)`，docstring 明确写着"subtracting the zero-counter forward pass makes `omega(op,a,0)==0` hold exactly"。`Correction` 末层零初始化（`:27-28`），所以初始 Ω 恒为 0。**它是"算子输出的增量"，不是"预测的增量"** —— 预测层面的增量是 `Teacher_dual − Teacher_self`（见 §4），两者不是同一个量。 |
| **T4** | Teacher 训练是否在完整 GT 上、未用 geometry 删 G10？ | **是（无像素级删除），但有一处必须披露** | `scripts/23:116-120 teacher_batch` 只用 `(a,d,y)`（`scripts/23:212` 解包时 `*_` 丢弃 ga/gd），5 个输出全部走 `land_loss` 全图。**但**训练集是 `HaitiPairs(tid, True)`（`scripts/23:337`），每处位置扩成 3 个 mode，`mode!=0` 时 `y = np.zeros_like(land)`（`scripts/21:41-45`）→ **2/3 的训练样本标签被强制置零**。 |
| **T5** | Teacher 是否完全不读 geometry？ | **是** | `OursV2Teacher.forward(target, counter)` 只有两个入参（`:142`）；`from_features` / `from_features_taps` / `forward_pair*` 同样不含 geometry。文件头注释 `models/landslide_cocd_v2.py:5` 明写"Geometry is intentionally absent from every model forward method"。 |

### 关于 T4 的补充：一条尚未披露的协议不对称

> ### ⚠️ 更正（2026-09-19）：本节结论**作废**
>
> 本节据以立论的三个脚本 `scripts/23_boehm:44`、`scripts/24:18`、
> `scripts/26:123` **均已不存在**。当前 `cocd/scripts/` 下与外部基线相关的
> 只剩 `23_train_external_baselines.py` 与 `24_external_seat_table.py`。
>
> 在**当前生效**的脚本里，外部席位用的是：
> ```
> 23_train_external_baselines.py:224   train_set = rapid.HaitiPairs(tid, True)   # ← True
> 23_train_external_baselines.py:70    EXPECTED_TRAIN_VIEWS = 8220
> ```
> 运行日志亦确认（三个席位一致）：
> ```
> experiments/single_orbit_baselines_v3/logs/{boehm,cdnette,mfewf}.log
>   train=4110 pairs = 8220 views (1370 locations x 3 modes x 2 orbits), test=343
> ```
>
> **即：外部席位与我们的臂使用同一配置 `train_modes=True`，
> 不存在本节所说的"协议不对称"。** 见
> `reports/haiti_dataset_origin_and_convergence.md` §5.3。
> 下面这张表与随后的推论**仅供追溯**，不要再据此决策。

| 训练数据 | 方法 | 依据 |
|---|---|---|
| `train_modes=True`（3,699 样本/epoch，2/3 标签全零） | S0 / old T / Vanilla KD / Ours-V1 / 旧 DIS2-style（脚本 21）；Teacher / R1 / R2 / R3 / DIS2-port（脚本 23） | `scripts/21:114`、`scripts/23:337` |
| ~~`train_modes=False`（1,233 真实 pair/epoch，**真实 GT**）~~ | ~~Boehm / CDNetE / MFEWF / FC-Siam~~ | ~~`scripts/23_boehm:44`、`scripts/24:18`、`scripts/26:123`~~（脚本已删除，结论作废） |

这意味着主表上 `R3 0.6298` 与 `Boehm 0.5926` 之间除了 backbone 与预训练之外，**还差一项训练数据配置**。方向上这项对 Ours 不利（2/3 样本在推零），所以不构成"占了便宜"，但成立性必须写清楚。这与 §4「所有方法都在完整 landslide GT 上计算」的字面要求不一致 —— 需要决定是**接受并披露**，还是**给 Ours 也补一个 `train_modes=False` 的读数**。

> **↑ 以上推论已作废**（依据的脚本已删除，外部席位实为 `train_modes=True`）。
> 见本节开头的更正框与 `reports/haiti_dataset_origin_and_convergence.md` §5.3。

---

## 3. R3 与 §9 的逐步映射

| §9 的逻辑步骤 | R3 对应实现 | 是否到位 |
|---|---|---|
| Student 先拥有完整单轨能力；encoder/decoder 从 Teacher self 继承 | `models/landslide_cocd_v2.py:233` `self.backbone.load_state_dict(teacher.backbone.state_dict(), strict=True)`；decoder 与 backbone 同对象（`:71`） | ✓ |
| 不直接拟合 Teacher dual feature | R3 没有任何特征空间 MSE；KD 只在输出层（`scripts/23:286-287`） | ✓ |
| Student 从 target-only 表示预测修正 `R_S = Q(F_t)` | `d4 = Driver(128)`，`r4 = omega(op4, p4, d4(p4))`（`:281`）；`d3` 同理（`:285`）。`Driver` 末层零初始化（`:47-48`） | ✓ 结构完全一致 |
| `F_corr = F_t + R_S` | `p4c = p4 + r4`（`:282`）、`p3c = a3 + r3`（`:286`） | ✓ 加法残差 |
| 继承的 Ω | `self.op3/op4.load_state_dict(teacher.a3/a4.state_dict())`（`:235-236`） | ✓ |
| P3 递进驱动 | `drive3 = p3 if R2 else a3`（`:284`），`a3` 是 P4 修正已并入后的 P3 状态 | ✓ 这是 R3 相对 R2 的唯一变量 |
| selective KL | `scripts/23:284-288`，权重 `gain × protect`，**不含 geometry** | ✗ §11 未落实 |
| （§9 未要求但相关）直接监督修正量 `R_S ≈ R_T` | **不存在**。V1 的 `Ours` 有（`scripts/21:103` `0.2*correction_loss(rs, rt, gate)`），V3 没有 | 缺口 |

---

## 4. 已有的机制证据：Teacher gain by geometry state（§18 第一项，**无需新训练即可给出**）

来源：`experiments/ours_v2/teacher_test_verify.csv`（冻结的 30 行逐位回归基准，测试集、阈值 0.5）。dual − self：

| 分区 | ΔIoU | ΔAUPRC | 像素数 |
|---|---:|---:|---:|
| overall | +0.0244 | +0.0151 | 11,239,424 |
| G00（target 正常 / counter 正常） | +0.0255 | +0.0153 | 9,058,636 |
| **G10（target 失真 / counter 正常）** | **+0.0267** | **+0.0173** | 1,039,406 |
| G01（target 正常 / counter 失真） | +0.0186 | +0.0124 | 1,039,406 |
| G11（两轨都失真） | +0.0208 | +0.0134 | 101,976 |

排序在 IoU 与 AUPRC 上一致：**G10 > G00 > G11 > G01**。G10 的 gain 是 G01 的 **1.44×**（IoU）/ **1.40×**（AUPRC）。

三条读法：

1. **支持物理假设**：互补轨的额外价值在「target 看不清、counter 看得清」处最大，在「target 看得清、counter 看不清」处最小。这是 §6/§18 想要的对比，且它已经成立。
2. **同时不支持 hard-mask（印证 §12）**：G00 的 gain（+0.0255/+0.0153）与 G10（+0.0267/+0.0173）几乎同量级，而 G00 像素数是 G10 的 8.7 倍。**把 KD 限制在 G10 会丢掉大部分有效监督** → `(1+β·G10)` 的软加权方向是对的。
3. **口径声明**：这是**区域级聚合的** ΔIoU/ΔAUPRC，不是 §11 定义的逐像素 `gain(p)=ReLU(e_self−e_dual)`。要得到严格逐像素的版本需要重新推理一次 Teacher（两份权重都在，成本约几分钟），本轮未做。

---

## 5. 最小改动清单（**本轮已执行，见 §8**）

### 必改（让几何真正进入 R3 的蒸馏权重）

| # | 文件:行 | 改动 | 行数 |
|---|---|---|---|
| 1 | `scripts/23_train_ours_v2.py:284` | `if kd_mode == 'gain':` → `if kd_mode in ('gain', 'full'):` | 1 |
| 2 | `scripts/23_train_ours_v2.py:53-66` `CONFIGS` | 增加 `'R3g': {'variant': 'R3', 'kd': 'full', 'label': 'Ours-V3-G'}`（保留 `R3` 原样，作为 Control 1） | 1 |
| 3 | 启动命令 | 必须带 `--no-early-stop`，否则与已跑满 50 ep 的 R3 训练量不等 | 0 |

改完即得 `w = (1 + G10) · gain · protect`。**这与 §11 的 `w = ReLU(e0−et)·(1+β·G10)` 的差别只有两点**：现成分支用相对 gain、且多一个 `protect` 因子。若要求逐字对齐 §11，再改 1 行（把 `:107` 换成 `(e0 - et).clamp_min(0) * (1. + beta * g10.float())`，并把 `beta` 作为参数传进来，默认 1.0）。**两条路线二选一，不要同时上**：路线 A 保持 gain 口径与现有 R3 一致（可比性更好），路线 B 严格对齐定义（代价是与 R3 的差异从"只有几何"变成"几何 + gain 口径"）。

### 建议一并决定（不影响几何生效）

| 事项 | 位置 | 说明 |
|---|---|---|
| 几何加权是否只作用于真实样本 | `scripts/23:272`（`real` 已解包、当前未用） | 合成 mode 1/2 的标签是人为置零的，对其施加"反轨可观测性"权重语义可疑。1 行可改 |
| `train_modes=True` 的披露或补跑 | `scripts/23:337` | 见 §2 的表；与外部席位的训练数据配置不同 |
| 逐像素 `gain` 复核 | 新脚本 | 用已有 `teacher.pt` 重推理一次即可，验证 §4 的区域级结论 |

### 明确不做

- 不改 Teacher 结构（T1–T5 全过）。
- 不改 R3 的 Ω / Driver / 递进驱动（与 §9 一致）。
- 不新建 V4、不重设计骨干、不动 DIS2-port 的任何一行（§13 已满足）。
- 不把 geometry 写进任何 forward（§10 Case C 零容忍）。
- 不做"全模型 × masked/unmasked"大矩阵（§16）。

---

## 6. 需要拍板的三件事（**已于 §8 拍板**）

1. **路线 A 还是路线 B**（见 §5 必改项的口径说明）。
2. **几何加权是否只在 `real` 样本上生效。**
3. ~~**`train_modes=True` 与外部席位 `train_modes=False` 的不对称：披露，还是给 Ours 补一个 `train_modes=False` 读数？**（后者会改动 Teacher 与全部臂，代价远大于几何那一行，建议先披露、后补。）~~

   > **本项已撤销**：外部席位实际同为 `train_modes=True`，不存在不对称。
   > 见本节开头的更正框。第 1、2 项不受影响。

完成这三项决策后，一次 50-epoch 的运行（约 3 h）即可产出 Control 1 的"几何开"臂，与现有 0.6298 直接成对。

---

## 7. 证据文件索引

| 内容 | 路径 |
|---|---|
| Teacher 前向与 Ω | `models/landslide_cocd_v2.py:54-63, 101-173` |
| R3 前向与驱动 | `models/landslide_cocd_v2.py:238-291` |
| 蒸馏损失（含 `'full'` 分支） | `scripts/23_train_ours_v2.py:110-141` |
| 训练循环与 G10 构造 | `scripts/23_train_ours_v2.py:298-335` |
| 评价分区定义 | `scripts/23_train_ours_v2.py:161-186` |
| 数据集与 `train_modes` | `scripts/21_train_rapid_landslide_cocd.py:20-46` |
| V1 几何门控（真用过） | `scripts/21_train_rapid_landslide_cocd.py:102-103`、`losses/landslide_cocd.py:8-15` |
| 冻结 Teacher 分区读数 | `experiments/ours_v2/teacher_test_verify.csv` |
| 当前 R 臂与 DIS2-port 分区读数 | `experiments/ours_v2/results_R{1,2,3}_seed42.csv`、`results_DIS2_seed42.csv` |

---

## 8. 拍板与落地（审计之后，本轮已执行）

### 8.1 三条决定

1. **路线 A，β 固定为 1。** R3 的蒸馏逻辑一行不动（包括现有的 `gain × protect`），只新增几何提升：

   ```
   w_R3g = w_R3 · (1 + 1_real · G10)
   ```

   不改成 `ReLU(e0−et)·(1+β·G10)`：那会同时改动 gain 的定义与几何的使用，两个变量缠在一起无法归因。于是实验天然成对 —— **R3（`gain×protect`）对 R3g（`gain×protect×(1+1_real·G10)`），唯一差别是几何**。

2. **几何提升只作用于 real pre/post pair。** synthetic no-change pair 保留与 R3 完全相同的 KD 权重。理由：pre/pre 与 post/post 是人为构造的 no-change 样本，其"反轨看得更清"没有物理含义；但它们的 KD 本身是有效的 no-change 正则，不删除。

3. ~~**`train_modes` 不对称：披露，不重跑。**~~ **本项已撤销** —— 前提不成立。
   经核实外部席位同样使用 `train_modes=True`，**不存在不对称**，因此无需"披露"，
   也无需在 limitation 中单列。保留一条仍然有效的表述要求：
   "训练集含 2/3 合成无变化样本（`mode` 1/2，标签置零）"是**协议的实际内容**，
   应在数据/训练设置中如实描述，作为 method 说明而非缺陷声明。
   （严格机制结论只来自同协议的 R3 / R3g / DIS2-port 这条**仍然成立**，
   但理由从"数据配置不同"改为"只有这几臂共享全部超参"。）

### 8.2 代码改动（3 个文件，网络与 Teacher 零改动）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `scripts/23_train_ours_v2.py:110` | `selective_kl` 增加可选参数 `real=None` |
| 2 | `scripts/23_train_ours_v2.py:133-135` | `'full'` 分支：`boost = g10 · 1_real`，`w = (1 + boost) · gain · protect`（原为 `(1 + G10)·gain·protect`，未区分 real） |
| 3 | `scripts/23_train_ours_v2.py:81` | `CONFIGS` 新增 `'R3g': {'variant': 'R3', 'kd': 'full'}`；`R3` 原样保留为 Control 1。**未加入 `PLAN`**，避免 `all` 把已完成的 R1/R2/R3 重训覆盖 |
| 4 | `scripts/23_train_ours_v2.py:301, 319-332` | `real` 移到设备；新增 `elif kd_mode == 'full'` 派发，并把 `w_frac` / `geo_frac` / `w_mean` 写进 epoch 日志 |
| 5 | `scripts/23_train_ours_v2.py` 模块 docstring、`scripts/38_run_dis2_port.sh` | 文档化"已完成的臂不会被 `all` 恢复、只会重训"这一陷阱；38 的日志文案改为臂无关 |

**`R3` 的路径逐字未变**：`'gain'` 分支仍不读 `g10`，`gain`、`protect`、`kd_weight=1.0`、`lam_*` 全部原样。

### 8.3 验证

`scripts/36_smoke_dis2_distill_ours.py` 65 条断言、`scripts/37_smoke_distill_integration.py` 37 条断言（本轮为几何新增 [E] 13 条 + [F] 6 条）全部通过。几何部分关键断言：

| 断言 | 结果 |
|---|---|
| real 全真时 `w_full` 恰等于 `(1+G10)·gain·protect`（逐位） | ✓ |
| real 全假时 `w_full` 恰等于 `gain·protect`（逐位） | ✓ |
| G10 全 0 时权重不变（逐位）；G10 全 1 时恰为 2×（逐位） | ✓ |
| 混合批次：synthetic 那一半权重等于基线，real 那一半带提升（逐位） | ✓ |
| 给 `'gain'` 传 `real` 不改变任何结果（逐位） | ✓ |
| 无几何的 `'full'` 与 `'gain'` 评分逐位相同 | ✓ |
| 端到端 1 epoch：日志出现 `[R3g] kd epoch=1 w_frac=0.0043 geo_frac=0.0330 w_mean=0.0041` | ✓ 几何分支确实可达（若不可达会落到 selective 分支，不会打印 `geo_frac`） |

**本轮新发现，已记录未修**：同一套 gain 权重在两个模块里各写了一遍，`selective_kl` 用 `EPS=1e-8`、`gain_protect`（`losses/distill.py:24`）用 `EPS=1e-7`。实测两版逐位可复现各自的 EPS 版本，差异出现 1.09%（713/65536）像素上、最大 `|Δw| = 0.379`；但**实际进入损失的 KD 项只差 0.0004%**（3.435481e-03 vs 3.435493e-03），因为差异集中在教师已近精确、KD 质量本身为零的像素。读 `gain_protect` 的只有 `kd='selective'` 这条无配置可达的分支，**不涉及任何已报数字**。修它是一行，但会改动那条分支的定义，故按最小改动原则留待需要时再统一。

### 8.4 运行与参数对齐

```
bash Haiti_SAR_GRSL_Audit/scripts/38_run_dis2_port.sh R3g
```

| 项 | R3（Control 1，已有） | R3g（本轮） |
|---|---|---|
| Teacher | 冻结 `teacher.pt` | 同一份 |
| 结构 / 初始化 | `OursV3Student('R3')` + `load_teacher_self` | 同 |
| 优化器 | AdamW 1e-4, wd 1e-4 | 同 |
| batch / epochs / seed | 8 / 50 / 42 | 同 |
| 早停 | 关闭未显式传入但**跑满 50 未触发** | 显式 `--no-early-stop` |
| KD 权重 | 1.0 | 同 |
| 权重规则 | `gain × protect` | `gain × protect × (1 + 1_real · G10)` |
| 选模 | best val AUPRC | 同 |
| 评价器 | 同一 `report()`，固定阈值 0.5 | 同 |

产物：`R3g_seed42.pt` / `R3g_seed42_latest.pt`（每 epoch 原子落盘，可续训）、`results_R3g_seed42.csv`、累计表 `results_seed42.csv`（方法名 `OursV3-R3g`）。约 3 h。

### 8.5 读数顺序（成功判据）

1. **`G10`：R3g > R3**，尤其 G10 AUPRC 与 G10 F1（R3 现有 G10 IoU 0.6372 / AUPRC 0.9030）。
2. **overall：R3g ≥ R3**（0.6298），至少不下降。
3. 最理想：R3g > R3 **且** R3g > DIS2-port（0.6272）。
4. 若 overall ≈ R3 而 G10 有提升，仍成立 —— 方法针对的本就是 target-distorted / counter-reliable 的特定物理区域，不是全图平均。

### 8.6 本轮不做

- `Single + Geometry Mask`（Control 2）：优先级低于 R3g，待 R3g 成立后再补，用于回答"直接告诉模型哪里坏是否就够"。
- `Ours-NoGeometry` 补跑：**它已经存在，就是 R3**。
- ~~`train_modes=False` 的 Ours 补跑~~（已撤销：不存在该不对称，无需补跑）、masked/unmasked 大矩阵、任何网络结构改动。

---

## 9. R3g 结果与诊断（2026-09-17 20:17 完成后补）

`R3g` 于 17:39 → 20:17 跑满 50/50 epoch（batch 8、seed 42、`--no-early-stop`、best val AUPRC 0.8931@ep45）。设计与代码改动见 §8，完整分区表见 `experiments/ours_v2/ABLATION_R1R2R3_SUMMARY.md` §1.2。

### 9.1 结果：§8.5 的四条判据全部不成立

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| ① G10 | R3g > R3（0.6372 IoU / 0.9030 AUPRC） | **0.6292 / 0.8989** | ✗ |
| ② overall | R3g ≥ R3（0.6298） | **0.6260** | ✗ |
| ③ vs DIS2-port | R3g > 0.6272 | 0.6260 | ✗ |
| ④ G10 相对 G00 改善 | ΔG10 > ΔG00 | ΔG10 −0.0080 < ΔG00 −0.0033 | ✗ |

分区明细（overall）：`G10 −0.0080`、`G00 −0.0033`、`G01 −0.0017`，唯一上升的是 `G11 +0.0019`。val 曲线同向（R3g best 0.8931 < R3 best 0.8940）。

**训练有效性已核验**：50/50 epoch 跑满；`geo_frac` 全程稳定 0.0303（提升覆盖的像素比例不漂移），`w_frac` 0.0306→0.0142、`w_mean` 0.0224→0.0089。几何分支确实生效 —— 端到端日志里的 `geo_frac` 只在 `'full'` 分支打印，落到 selective 分支不会出现。所以这是**"开关打开了但不划算"**，不是分支不可达或配置写错。

### 9.2 诊断：几何提升指向正确，但它调制的通道只占目标 ~1%

用冻结 Teacher + R3g 最终学生，在 30 个真实训练 batch（7,864,320 px，batch 8，`train_modes=True`）上直接测量，三次独立取样量级一致：

| 量 | 实测 | 含义 |
|---|---:|---|
| `w_frac`（KD 活跃像素比例） | 0.013–0.016 | KD 只在 ~1.5% 像素上非零 |
| `geo_frac`（提升覆盖比例） | 0.023–0.033 | 与训练日志 0.0303 吻合 |
| **活跃集合落在提升区内** | **0.147–0.167**（对照 0.013–0.016） | **富集 10–12×**：提升确实优先落在 KD 选中的像素上 |
| **提升可触及的权重质量** | **0.144–0.167** | 上限即此 —— 不是"够不着" |
| KD 项相对变化 | **+15% ~ +59%** | 对 KD 通道杠杆实质 |
| KD 项 / 分割项量级 | 3.8e-4~9.3e-4 / 4.6e-2~5.0e-2 | — |
| **KD 占目标函数** | **0.8% ~ 1.8%** | 几何提升改动的**总损失**只有 ~0.1–1% |

另用两个量排除"规则选错像素"：

| 量 | 实测 |
|---|---:|
| 教师自轨逐像素 BCE（全体像素） | 2.84e-2 |
| 教师自轨逐像素 BCE（**KD 活跃集合**） | **4.04e-1**（14× 平均） |
| 活跃集合上的绝对改善 `ReLU(e0−et)` | 9.09e-2 |

`gain × protect` 选中的是**教师真正出错、且反轨确实把误差降下来**的像素，不是"教师已做对、相对增益被放大"的伪信号。

**结论：几何规则实现正确、指向正确（富集 10–12×），但蒸馏通道本身只承担 0.8–1.8% 的目标函数。** 该负结果由**通道权限**决定，而非几何规则失效。这同时解释了为什么 R1/R2/R3 三臂彼此也只在 0.4–1.0 IoU 点内 —— 它们的差异同样发生在这条低权重通道之上（R3−R2 是结构改动，量级相同）。

### 9.3 口径

- **单种子、单次读数。** −0.0038 overall 与 −0.0080 G10 的幅度与臂间既有散布同量级，**因此也不能反向断言"几何有害"**。可陈述的最强结论：在 R3 的蒸馏规则与预算下，引入几何优先加权**没有产生可测收益**。
- **该对比依旧是干净的。** R3 与 R3g 共享同一冻结 Teacher、同一初始化、同 batch / 50 epoch / seed / 选模 / 评价器，唯一变量是权重规则；网络结构一行未改。
- **§17 的评价层级未变**：G10 仍是第一优先级，overall 仍是第二。R3g 在两层上都没赢。

### 9.4 后续（未执行，待决策）

1. **先量单种子噪声**（最省）：用同一条命令重跑 R3 的前几个 epoch，看 val AUPRC 是否逐位一致。若 MPS 侧不确定，则 0.5–1.0 IoU 点量级的所有臂间归因都要重新表述。
2. **让几何先验真正有权重**：需改变的不止权重规则 —— 提高 `kd_weight`、放宽把 98.5% 像素归零的 `protect` 门、或把几何先验移到分割项。三者都引入新变量，第三项已偏离"几何条件化蒸馏"的表述。
3. **`Single + Geometry Mask`（Control 2）**：优先级仍低于 1、2，待几何路线确认有效后再补。
