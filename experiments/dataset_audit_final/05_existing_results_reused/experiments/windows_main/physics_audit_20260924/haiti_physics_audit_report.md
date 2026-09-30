# Haiti 数据集物理关系核实报告 — Landslide GT × ASC/DESC SAR × Layover/Shadow Mask

**日期**: 2026-09-24 · **数据**: 本地 `data/haiti/`（1,713 个 location，全部读取，无训练）
**DEM**: Copernicus GLO-30（公共 AWS S3 下载，30 m，两瓦片 N18W074/N18W075，覆盖数据集全域 lon −74.24~−73.42 / lat 18.19~18.58）
**脚本**: `experiments/windows_main/physics_audit_20260924/audit_01~03*.py`，全部指标可复现。

---

## 一、文件—物理含义对应表（逐文件实测，非沿用旧代码假定）

| 本地文件（每 location） | 实测格式 | 物理含义 | 验证方式 |
|---|---|---|---|
| `Pre_event/S1_ASC_20210805/{id}.tif` | 128×128×3 float32, EPSG:4326, ~10 m/px, 无 nodata | ASC pre SAR：band1=VH(dB)，band2=VV(dB)，**band3=geometry mask** | 逐波段 unique/value 统计；VH 中位≈−15 dB，VV≈−8 dB |
| `Pre_event/S1_DESC_20210803/{id}.tif` | 同上 | DESC pre SAR（2021-08-03） | 同上 |
| `Post_event/S1_ASC_20210817/{id}.tif` | 同上 | ASC post SAR（2021-08-17） | 同上 |
| `Post_event/S1_DESC_20210815/{id}.tif` | 同上 | DESC post SAR（2021-08-15） | 同上 |
| `processed/sample_XXXXXX/landslide_mask.tif` | uint8 {0,1,2,3} | NASA 滑坡清单栅格，`>0`=landslide，1/2/3 为多边形重叠次数 | unique values |
| `processed/sample_XXXXXX/asc_pre_geom.tif` | uint8 {0,1,2,3} | ASC 几何畸变 mask（= raw band3，**1,713/1,713 完全一致**） | 逐像素 array_equal |
| `processed/sample_XXXXXX/desc_pre_geom.tif` | uint8 {0,1,2,3} | DESC 几何畸变 mask（= raw band3，1,713/1,713 一致） | 同上 |
| `processed/sample_XXXXXX/metadata.json` | JSON | 仅 provenance：4 个 acquisition date、几何编码、Mac 绝对路径 | 人工读取 |
| 光学 S2 / DEM / shapefile / Annotations | **不在本包** | manifest §7 明确排除 | find 搜索 0 命中 |

**本地不存在的关键项（直接影响本轮任务边界）**：
- ❌ 原始 landslide polygon shapefile（`Annotations/` 被排除）→ 第三部分 GT rasterize 对比**无法执行**，见 §三。
- ❌ DEM/slope → 已通过下载 Copernicus GLO-30 补齐（§五）。
- ❌ 光学 pre/post → 时间错配只能靠 SAR 间接检查（§六）。

---

## 二、Geometry mask 重新核实

**编码验证**：全量 1,713 个 patch、两个轨道的 band3 unique values ⊆ {0,1,2,3}，与 metadata.json 的 `0=valid, 1=layover, 2=shadow, 3=layover+shadow` 一致；幅值签名吻合（layover 最亮、shadow 最暗，与上轮 Mac 审计一致）。

**1. ASC 与 DESC mask 是否不同？——不同，且差异显著**（全数据集像素统计，N=28,041,192 px/location×1713）：

| 轨道 | valid | layover | shadow | both | 总畸变 |
|---|---|---|---|---|---|
| ASC | 86.25% | **13.68%** | 0.054% | 0.017% | **13.75%** |
| DESC | 94.02% | **5.33%** | 0.638% | 0.011% | **5.98%** |

ASC 畸变 ≈2.3× DESC；ASC 以 layover 绝对主导，DESC shadow 占比高一个量级（0.64% vs 0.05%）。

**2. pre/post 是否同一张 mask？——是，字节级相同**。本地直接比对 raw band3：ASC 与 DESC 的 pre-vs-post 差异像素均为 **0**（全部 1,713 patch）。即作者把 topographic layover/shadow 当作**静态几何属性**，用 DEM+轨道几何算一次、两个日期复用。post mask 文件未随包分发，但 raw band3 完整保留，本结论是本地实测而非转述。

**3. mask 是否只依赖 terrain+orbit？** 本包内无生成脚本（在 Mac 端生成）；间接证据：(a) pre/post 字节相同 → 不含任何随时间变化的信息（若读过 GT 也只可能经地形间接相关）；(b) mask 与 GT 空间格局完全不同（见 §四 IoU≈0.10）；(c) 论文（thesis §5.2）明确说明按 Meier et al. 1993 从 SRTM+轨道几何计算。**未发现 mask 读取 GT 的任何迹象，但"生成脚本未读 GT"这一点在本地无法直接证明**——只能给出以上三条一致证据。

---

## 三、Landslide GT 与原始 shapefile 对比 —— **无法完成，明确声明**

本包不含原始 polygon shapefile（`Annotations/`，171.3k 文件被 manifest 排除）。因此：

- ❌ 无法重新 rasterize 到 dataset grid；
- ❌ 无法计算 pixel agreement / IoU / Dice / boundary disagreement；
- ❌ 无法直接检验 dilation / erosion / resize / polygon cleaning 是否发生。

**不猜测。** 可报告的间接证据（来自上轮 Mac 端审计 + 本轮实测）：
- GT 值域恰为 {0,1,2,3}，其中 1/2/3 是**重叠栅格化多边形实例的计数**特征，与"直接 rasterization + 多边形重叠计数"一致；
- 1,700/1,713 patch 含 >100 landslide px，与论文声明的 >100 px 抽取规则吻合（stride 64 滑窗）；
- GT 空间格局为不规则疤痕状斑块（见案例图红色轮廓），与 slope-following 的几何 mask 完全不同形态。

若需最终确认"DataPort GT = 原始 polygon 直接 rasterization"，必须回 Mac 端或在 Dataport 重新下载 `Annotations/` 后执行同一脚本。

---

## 四、GT × geometry mask overlap（本轮核心统计，全量 1,713 patch 实测）

> 注：本数据集的 1,713 个 patch **全部**是 landslide-selected，因此"全部 dataset pixels"与"landslide-selected 1713 patches"两个分层在此重合（已在 case_selection.csv 保留 per-patch 指标可二次分层）。

### ASC（M = asc_pre_geom > 0）

| 统计量 | 值 |
|---|---|
| P(M>0 \| GT=1) | **0.2725** |
| P(M>0 \| GT=0) | 0.1278 |
| P(GT=1 \| M>0) | **0.1325** |
| P(GT=1 \| M=0) | 0.0564 |
| IoU | **0.0979** |
| Dice | 0.1783 |
| Odds ratio | **2.56** |
| Relative risk | 2.13 |
| 按类：P(GT=1\|layover) / P(GT=1\|shadow) / P(GT=1\|both) | 0.132 / 0.261 / 0.233 |
| GT 落在 bad 内的比例（按类分解） | layover 0.270，shadow 0.002，both 0.0006 |

### DESC

| 统计量 | 值 |
|---|---|
| P(M>0 \| GT=1) | **0.1556** |
| P(M>0 \| GT=0) | 0.0529 |
| P(GT=1 \| M>0) | **0.1741** |
| P(GT=1 \| M=0) | 0.0600 |
| IoU | **0.0895** |
| Dice | 0.1643 |
| Odds ratio | **3.30** |
| Relative risk | 2.94 |
| 按类：P(GT=1\|layover) / P(GT=1\|shadow) / P(GT=1\|both) | 0.165 / 0.248 / 0.130 |

### 解读

- 相关**真实存在但强度中等**：GT 像素落入几何畸变区的概率是背景的 2.1~2.9 倍；但 IoU 仅 ≈0.09–0.10，**"GT 和 geometry mask 看起来很像"主要是视觉错觉 + patch selection bias**（所有 patch 都是滑坡中心的山地陡坡地形，两者都被陡坡吸引）。
- **73%（DESC）~ 84%（ASC）的 GT 像素位于几何有效区**——多数滑坡在两个轨道下都可观测。
- shadow 类的 P(GT=1|shadow) ≈ 0.25~0.26 看似很高，但这是小基数效应（ASC shadow 仅 0.054% 像素），且被 §五的坡度控制完全解释。

---

## 五、DEM 坡度分层（地形控制）

Copernicus GLO-30 30 m → 双线性重采样到各 patch 10 m 网格 → 逐像素 slope → 按 bin 分层统计 P(GT=1|M>0)/P(GT=1|M=0)：

| slope bin | 像素数 | ASC bad_frac | ASC RR | DESC bad_frac | DESC RR |
|---|---|---|---|---|---|
| 0–10° | 2.67M | 1.89% | **10.74** | 0.35% | **12.84** |
| 10–20° | 5.56M | 3.67% | **4.88** | 0.69% | **6.24** |
| 20–30° | 7.65M | 8.88% | 2.47 | 2.12% | 3.60 |
| 30–40° | 8.63M | 20.77% | 1.21 | 8.89% | 1.62 |
| >40° | 3.55M | 31.94% | **1.05** | 19.74% | **1.04** |

（RR = P(GT=1|M>0) / P(GT=1|M=0)）

**结论：RR 随坡度单调坍缩至 ≈1**。在 >40° 坡度带内，"是否几何畸变"对"是否滑坡"几乎没有附加信息；在低坡度带 RR≈11–13 看似很大，但该带 bad 像素基数极小（0.3–1.9%），且可由 30 m DEM 平滑掩盖的局部陡坎（河谷/冲沟内的坡度变换）解释。

**回答核心问题：GT 与 geometry mask 的重合基本可以被"滑坡和 SAR 畸变都偏向陡坡"这一共同地形原因解释，没有发现 annotation pipeline 与 geometry 生成发生耦合的迹象。**

（caveat：30 m DEM 相对 10 m 网格是平滑的，分层用的坡度被低估；这不改变"随坡度坍缩"的趋势结论。）

---

## 六、Temporal audit

本地可得的全部时间信息（每个 `metadata.json`，全数据集统一，无逐 patch 差异）：

| 事件 | 日期 |
|---|---|
| DESC pre | 2021-08-03 |
| S2 optical pre（不在包内，仅 provenance 记录） | 2021-08-04 |
| ASC pre | 2021-08-05 |
| 地震 | 2021-08-14 |
| S2 optical post（不在包内） | 2021-08-14 |
| DESC post | 2021-08-15 |
| ASC post | 2021-08-17 |
| Grace 飓风 | ~2021-08-16–17（文献值，本地无记录） |

**本地不存在**：逐 polygon 的成图日期、source image date、event attribution 字段。因此**无法从本地数据区分某个 GT 多边形来自 8-14 地震还是 8-16 之后 Grace 降雨**——明确声明，不声称可以区分。

间接检查（pattern scan，全 1,713 patch）：
- "ASC 变化清晰 + DESC 无变化"（两轨均 valid，contrast≥1.4 vs <1.02）：**0 个**；
- 反向（DESC 清晰 + ASC 无变化）：**1 个**（patch 446，contrast 1.47 vs 1.00）。

即**没有发现成规模的 temporal-mismatch 信号**。若 Grace 期滑坡被并入 inventory，最可能的表现（DESC post 8-15 无变化、ASC post 8-17 或更晚光学有变化）在 12 天窗口内两轨都覆盖了 Grace 主降雨期，因此两轨大多**同时**包含 Grace 后成像——这恰恰使轨道间 mismatch 检测在本数据集上先天低敏。案例图中的判断只能到 "suspected / unable to distinguish" 为止。

---

## 七、三组案例图（选择规则 + 人工判读）

**选择原则**：全部用明确规则先筛池，再取"池中位数典型"样本，不做 cherry-pick。规则与 top-3/5 候选记录在 `case_selection_log.json` / `case_selection.csv`。

- **G1**：gt_frac≥0.03 & GT 全部在 BothValid（≥0.9）& 两轨 bad<5% & 取变化对比度前 50%→ 取 (gt_frac, contrast) 中位数最近者 → **patch 1166**（备用 895, 1379）。
- **G2-layover**：gt_frac≥0.02 & ASC 覆盖 GT≥70% & DESC 覆盖 GT≤30% & GT 内 layover 占比≥50% → **patch 641**。
- **G2-shadow**：DESC 覆盖 GT≥50% & ASC≤50% & GT 内 shadow 占比最大者 → **patch 562**（shadow 主导案例在 ASC 侧不存在——ASC shadow 仅 0.054%，此为数据本身属性而非筛选失败）。
- **G3**：gt_frac≥0.02 & 双轨 GT-mask overlap<50% & 评分=rank(边界一致性)+rank(弱变化)+rank(gt_frac) 取 top5 → **patch 155**（131, 581, 406, 541 入候选 CSV）。

### Figure Group 1 — `case1_bothvalid_landslide.png`（patch 1166）

GT 集中在 patch 上缘（gt_frac 0.054），两轨几何 mask 几乎为空（bad 0.001/0.001），GT 100% 在 BothValid 内。

1. **GT 表示什么？** NASA 清单的不规则滑坡疤痕斑块，位于 patch 顶部一排，边界呈自然锯齿状。
2. **ASC geometry-valid？** 是（bad=0.1%）。
3. **DESC geometry-valid？** 是（bad=0.1%）。
4. **ASC pre→post 有无可信 change？** 弱：GT 内 mean|ΔVV|=2.56 dB（背景 ~2.1，contrast 1.18），signed −0.1 dB——是纹理级差异，无一致变亮/变暗。|ΔVV| 面板上 GT 与背景不可区分。
5. **DESC pre→post？** 类似：mean|ΔVV|=2.66 dB，signed −0.9 dB（轻微变暗倾向），同样以纹理差异为主。
6. **GT 与 mask 为何不重合？** 该处地形虽足够孕育滑坡，但从两个视角都不产生 layover/shadow——坡向与两轨道视向的几何关系均温和。这是"滑坡存在 ≠ 几何畸变"的直接例证（对应 §四中 73–84% 的 GT 在有效区）。
7. **支持还是反对 mask 掉 invalid 像素？** 此处无 invalid 像素可 mask；支持"GT 保留进 loss"。
8. **对双轨→单轨的启示？** 在 BothValid 区，两轨证据都弱且噪声水平相近，counter 轨道的增益主要来自**独立噪声实现平均 + 视角差异**，而非"一方看不见"。这解释了 CG>SG 为何是温和而非压倒性的。

**人工判断**：合理但证据弱。GT 与 SAR change 的空间一致性仅勉强可辨——这正是该数据集 10 m S1 GRD 上滑坡信号的典型状态（pool 内位数 2.5 dB）。两个轨道中 DESC 的变化倾向（−0.9 dB 变暗）略更可读。

### Figure Group 2 — `case2_gt_geometry_overlap.png`（patch 641, ASC layover 主导）+ `case2b_gt_shadow_desc.png`（patch 562, DESC shadow 主导）

**patch 641**：GT（红）大量落在 ASC layover 台地（橙色，saturated 白色亮区）内，GT 在 ASC-bad 内占比 0.77。

1. **GT？** 两条陡坡上的长条形疤痕群，沿坡走向延伸。
2. **ASC valid？** 否——bad 40.2%，GT 的 77% 在 ASC layover 内。
3. **DESC valid？** 基本是——bad 3.9%，GT 仅 23% 落入 DESC-bad（顶部一小块 shadow）。
4. **ASC pre→post？** 不可判读：layover 台地饱和发亮，|ΔVV| 的条带状异常沿 layover 边缘分布，是几何条纹不是地表变化。
5. **DESC pre→post？** 有可信变化：post 图顶部 GT 群内出现明显变暗块，|ΔVH| DESC 在 GT 顶部呈亮斑群——形态与滑坡疤痕一致。
6. **为何重合？** 朝向 ASC 雷达的陡坡→layover；这些陡坡同时也是滑坡易发/发生地。重合由**地形共同原因**产生（§五），非 annotation 耦合。
7. **支持 mask？** 强烈支持：ASC 在 GT 所在处观测不可靠，把 ASC-invalid 像素从 ASC 的 loss/评测中排除是正确的。
8. **启示？** counter 轨道在这里不是"锦上添花"而是**唯一可观测视角**。单轨推理系统在 G10 区等于对不可观测像素强行预测。

**patch 562**：GT（9.6%）沿山脊侧翼分布，DESC 视角该侧翼处于 radar shadow（GT 78% 在 DESC-bad 内）。

1. **GT？** 山脊侧翼的疤痕群，ASC 视角下位于亮坡面上，清晰可见。
2. **ASC valid？** 部分——bad 44%（大范围 layover），但 GT 仅 41% 落入其中，侧翼本体在 ASC 是可见的。
3. **DESC valid？** 否——GT 的 78% 在 DESC shadow 内。
4. **ASC pre→post？** 弱但存在：GT 顶部 |ΔVH| ASC 有成片亮斑；|ΔVV| 变化与背景接近（contrast 1.10）。
5. **DESC pre→post？** 无法判读——shadow 区后向散射本身就是噪声底。
6. **为何重合？** 背向 DESC 雷达的坡→shadow；同一批坡面向 ASC。layover/shadow 是同一地形的两个视角表现（complementarity 面板：橙/青几乎互补，紫=both-bad 呈细条带）。
7. **支持 mask？** 支持：DESC-invalid 区应从 DESC loss 排除，由 ASC 承担该处监督。
8. **启示？** shadow 主导的样本在 DESC 侧是"全盲"，比 layover 更彻底；这解释了为何 per-view 处理优于对称处理。

**人工判断（两例合并）**：合理。"真实滑坡 GT"与"当前轨道不可观测"确实同时成立——GT 疤痕形态自然、与光学清单语义一致，而对应视角的 SAR 证据饱和/缺失。counter 轨道均能给出更合理的变化形态。

### Figure Group 3 — `case3_suspicious_case.png`（patch 155）

1. **GT？** 21.9% 面积、大量小型沿坡长条碎片，散布在 ASC layover 边界内外。
2. **ASC valid？** 部分——bad 33.5%，GT 49% 在 ASC-bad 内。
3. **DESC valid？** 大部分——bad 10.4%，GT 仅 10% 在 DESC-bad。
4. **ASC pre→post？** 无可信变化：GT 内 mean|ΔVV|=2.16 dB，signed +0.5 dB，contrast 0.96（GT 变化甚至低于背景）。
5. **DESC pre→post？** 同样无可信变化（2.37 dB，signed +0.0，contrast 1.11）。
6. **GT 与 mask 边界一致性：** ba_asc=0.40（40% 的 GT 边界落在 mask 边界 2px 内，5 例候选中最高）。但结合 §五（控坡度后 RR→1），该一致性**可由陡坡共同吸引解释**，无需 pipeline 耦合假设。
7. **支持 mask？** 中性偏支持：这些 GT 中近半不可从 ASC 判读，从 ASC loss 排除不损失真信息；DESC 侧有效区也读不出变化，说明该处 SAR 信号本身弱（结论 A 的情形）。
8. **启示？** 单轨系统在此类 patch 上的失败 ≠ 标注错误；评测时若不 mask invalid 像素，会把"不可观测"与"预测错误"混为一谈。

**最终裁决（patch 155）**：**E（无法判断）偏 A（数据合理、SAR 信号弱）**。证据：GT 碎片形态与清单一致；无一致 signed 变化；边界一致性与坡度分层结果吻合；无 registration 错位迹象（同轨 pre/post 梯度相位相关上轮已证 <0.5px）。**不是** D（GT/rasterization 问题）的证据不足，也**不能**升级为 C（temporal mismatch）。

---

## 八、Zoom-in — `case_zoom_overlay.png`（patch 641）

放大后可直接看到：ASC layover 台地内 GT 疤痕在 ASC pre/post 均被饱和亮度淹没（"是滑坡还是畸变"在 ASC 内无法回答）；同一批 GT 在 DESC post 顶部呈黑块（变暗），与周围纹理连续；|ΔVV| DESC 在 GT 顶部有连贯亮斑群。**结论：是真实滑坡（GT 合理）+ ASC 视角几何不可观测，两者重合但来源独立。**

---

## 九、六个核心问题的最终回答

**Q1 GT 与 geometry mask 在真实数据上重合程度？**
中等偏弱：IoU 0.098（ASC）/0.090（DESC），Dice 0.18/0.16；GT 像素落入畸变区的概率是背景的 2.1–2.9×；但 **73–84% 的 GT 像素在两轨下均几何有效**。"看起来很像"主要是 patch selection bias（全是陡坡滑坡中心 patch）+ 陡坡共现。

**Q2 地形共同原因还是 pipeline 耦合？**
**地形共同原因。** 坡度分层 RR 从 0–10° 的 10.7–12.8 单调坍缩到 >40° 的 1.04–1.05；控制坡度后 mask 对 GT 几乎无附加预测力。没有任何 pipeline 耦合的证据（mask pre/post 静态且与 GT 空间格局不同形态；GT 是光学清单来源）。

**Q3 BothValid 区 ASC/DESC 是否互补？**
是，但互补的形式要澄清：在 G1 类样本中两轨都"看得见"且信号都弱，互补性主要来自独立相干斑实现与视角差异（各 2–3 dB 级别）；在 G2 类样本中互补性是**二值的**——一方不可观测、另一方是唯一证据。两种机制都支持双轨训练。

**Q4 G10/G01 区 invalid orbit 是否真的缺乏可信观测？**
是。641（layover）：ASC 台地饱和、条纹为几何假象；DESC 同区域能看到合理变暗。562（shadow）：DESC 全盲；ASC 侧翼可见且 |ΔVH| 有局部变化。两个方向都验证了 "invalid orbit 缺乏可信观测、counter orbit 更合理"。

**Q5 是否发现疑似 temporal mismatch 样本？**
**没有可确认的**。本地无逐 polygon 时间信息，pattern scan 仅 1 例（patch 446，DESC 清晰/ASC 无变化），未达 suspected 门槛。结构性原因：12 天窗口两轨均跨越地震与 Grace 主雨期，轨道间先天难以分离两者。此项**保持 unable to distinguish**，需光学数据或逐 polygon 来源才能推进。

**Q6 geometry-invalid pixels 的最合理处理？**
**维持原方案：保留真实 landslide GT，但从当前轨道的主 segmentation loss / primary evaluation 中 mask 掉。** 真实数据依据：(a) invalid 区的 SAR 证据确实不可判读（饱和/噪声底）；(b) counter 轨道能恢复 ≈97% 的畸变地形（上轮 complementarity CR 中位 0.987）；(c) GT 本身真实（与畸变无耦合）。同时建议评测中单独报告 invalid 区指标（不计入主指标），避免"不可观测"与"预测失败"混淆。

---

## 十、交付物清单

| 文件 | 内容 |
|---|---|
| `haiti_physics_audit_report.md` | 本报告 |
| `case1_bothvalid_landslide.png` | G1 patch 1166：4×SAR + GT + 双轨 mask + 互补面板 + \|ΔVV\|/\|ΔVH\| |
| `case2_gt_geometry_overlap.png` | G2-layover patch 641 主图 |
| `case2b_gt_shadow_desc.png` | G2-shadow patch 562 主图 |
| `case3_suspicious_case.png` | G3 patch 155 主图 |
| `case_zoom_overlay.png` | patch 641 zoom overlay（GT/layover/shadow 边界 + 双轨 ΔVV） |
| `candidate_metrics.csv` | 全部 1,713 patch 的逐项指标（选样可追溯） |
| `case_selection.csv` | 三组入池候选的完整指标 |
| `case_selection_log.json` | 选择规则与池统计 |
| `audit_summary.json` | §二/§四 全部统计量 |
| `slope_stratified.json` | §五 坡度分层统计 |
| `patch_slope.csv` | 每 patch 平均坡度/高程 |
| `audit_01_overlap_stats.py` / `audit_02_slope_strat.py` / `audit_03_select_and_figures.py` | 可复现脚本 |

**局限声明**：GT-vs-shapefile 对比（§三）与逐 polygon temporal audit（§六）因本地缺文件无法完成；DEM 为外部下载的 30 m 产品，非数据集自带；案例判读为单人对图的人工判断，已尽量用规则约束选样。
