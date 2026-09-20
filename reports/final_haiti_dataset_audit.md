# Haiti SAR Dataset — GRSL Feasibility Audit (final)

**Audit date:** 2026-09-12 · **Data:** *Multimodal Remote Sensing Dataset for Landslide Change Detection in Haiti*, Bralet, Trouvé, Chanussot, Atto — IEEE Dataport 2024, doi [10.21227/4heb-7h07](https://dx.doi.org/10.21227/4heb-7h07)
**Source path (read-only):** `/Users/zhangjiuqi/Desktop/distortion/{Pre_event,Post_event,Annotations}`
**Derived workspace:** `Haiti_SAR_GRSL_Audit/` (≈653 MB added; source 3.1 GB untouched)
**Scope:** data/provenance/science validation only — **no deep model was trained**.

> The task expected `*.tar.gz` in `~/Downloads`; Downloads held only an unrelated `sar-ship-dataset-master.zip`.
> The dataset was found **already extracted** in the selected project folder. All findings below come from
> reading the actual rasters; documentation claims (Bralet PhD thesis, HAL tel-05029007, §5.2) are used only to
> corroborate, never to substitute for measurement.

---

## 1. Data scale (requested summary item 1)

| Metric | Value |
|---|---|
| Total samples | **1713** (IDs `0…1712`, contiguous, identical in every folder) |
| Complete samples | **1713 / 1713 = PASS** (no missing ASC/DESC/pre/post/optical/annotation/geometry) |
| SAR files | **6852** (4 per sample: ASC/DESC × pre/post) |
| Optical files | 3426 (S2 pre + post) |
| Annotation files | 1713 |
| Total files | **11991 GeoTIFF**, only `.tif` present (no README/XML/JSON/sidecar shipped) |
| Image size | **128 × 128 px**, uniform; EPSG:4326; ≈ **10 m/px** (9.48 × 9.93 m); float32 SAR / float64 S2 |
| Total extracted size | **3.26 GB** logical (≈3.1 GB on disk): Pre 1.5 GB, Post 1.5 GB, Annotations 114 MB |
| Patch construction (thesis) | 128×128 sliding window, **stride 64 (≈50% overlap, 4×)**, kept if >100 landslide px |

## 2. Actual sample schema (requested item 2)

Each integer `sample_id` maps 1:1 to **7 files** (verified on all 1713, inspected in depth on dozens):

```
sample_id = k
├── Pre_event/S1_DESC_20210803/k.tif   SAR DESC pre  (3×float32)  band1 VH(dB), band2 VV(dB), band3 GEOM mask
├── Pre_event/S1_ASC_20210805/k.tif    SAR ASC  pre  (3×float32)  band1 VH(dB), band2 VV(dB), band3 GEOM mask
├── Pre_event/S2_20210804/k.tif        Optical pre    (4×float64)  band1-3 RGB reflectance, band4 cloud mask
├── Post_event/S1_DESC_20210815/k.tif  SAR DESC post (3×float32)  same band layout
├── Post_event/S1_ASC_20210817/k.tif   SAR ASC  post (3×float32)  same band layout
├── Post_event/S2_20210814/k.tif       Optical post   (4×float64) band1-3 RGB, band4 cloud mask
└── Annotations/k.tif                  Landslide label(1×float32) {0,1,2,3}, NOT georeferenced (identity tfm)
```

- **VV/VH are separate bands, not files.** Order is **band1 = VH, band2 = VV** (assigned from data: VV is 6–7 dB
  stronger with a higher noise floor; the author's thesis also writes the dual-pol pair as "(VH, VV)").
  Values are **σ0 in dB (log scale)**, σ0-calibrated and SRTM-orthorectified (thesis §5.2); no NaN, no zeros,
  no declared nodata. Global medians: VH ≈ −15 dB, VV ≈ −8.3 dB (see `qc/tables/sar_band_stats.csv`).
- **The geometry mask is bundled as SAR band 3** — there is no separate mask file/folder.
- S2 band 4 is a binary **cloud/cloud-shadow mask** (Sen2Cor medium+high cloud ∪ cloud shadow).
- Annotations are the **NASA landslide** label, not geometry (§10).

## 3. ASC/DESC authenticity (requested item 3) — **genuine, not relabelled duplicates**

The patched GeoTIFFs carry **no embedded Sentinel-1 product ID, relative orbit, flight-direction or acquisition
time-of-day** (only `AREA_OR_POINT=Area`; the "log-amplitude" tag is an auto-generated GDAL virtual subdataset).
So orbit direction cannot be read from a metadata tag. It is instead established by four independent, mutually
consistent lines of evidence (**all measured, not assumed from the `ASC`/`DESC` string**):

1. **Different acquisition dates** in the folder names, corroborated by thesis §5.2 (ASC pre 08-05 / DESC pre
   08-03; ASC post 08-17 / DESC post 08-15) — 48 h apart, i.e. distinct satellite passes.
2. **Pixel content is never identical.** Across all 1713 samples, both polarisations, both dates:
   **`array_equal = 0`, identical-MD5 = 0** out of 6852 comparisons (`qc/tables/ad_independence.csv`).
3. **Large radiometric disagreement with *negative* pixel correlation** — median |ASC−DESC| ≈ **6.4–7.1 dB**,
   max ≈ 30–41 dB, Pearson corr **−0.38 (VH) / −0.46 (VV)**. Opposite-view backscatter over steep terrain is
   anti-correlated because radar-facing slopes are bright in one view and dark in the other — the physical
   slope-facing signature, impossible for a duplicated acquisition.
4. **View-specimu׽-�G����ƭy�_mode == 'gain':` → `if kd_mode in ('gain', 'full'):` | 1 |
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
