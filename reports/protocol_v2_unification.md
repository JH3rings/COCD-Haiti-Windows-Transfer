# 统一训练协议 v2：检查清单与落地记录

2026-09-18。本轮**不改网络结构、不启动训练**，只做两件事：按 §10 逐项核查现有实现，
把新协议落成一份 config + 一处实现，并给出差异清单与必须先定的口径。

| 交付物 | 作用 |
|---|---|
| `configs/protocol_v2.json` | 协议的唯一来源（Adam / lr 5e-5 / batch 16 / BCE / 20 ep / patience 3 / no-mask / val 阈值） |
| `scripts/23_train_ours_v2.py` | 协议化：`--protocol v2`（默认）/ `--protocol v1`（已报臂仍可复现，同一份代码） |
| `scripts/47_protocol_batch_feasibility.py` | batch 16 的显存与每 epoch 预算实测 |
| `scripts/48_smoke_protocol_v2.py` | 44 条断言，**全部通过** |
| `scripts/49_run_protocol_v2.sh` | 启动器（并发守卫 + `DRYRUN`），**未运行** |

---

## 1. §10 十项检查

### 1.1 当前 train / val / test location 数

| 划分 | location 数 | 单轨 case 数（ASC+DESC） |
|---|---:|---:|
| train | **1,233** | 2,466 真实 + 4,932 合成（见 1.3/1.4） |
| val | **137** | 274（真实） |
| test | **343** | **686**（真实） |
| 合计 | 1,713 | 1,370 / 343 |

划分单位是 **location**：`manifests/spatial_80_20_split.csv` 的 `split` 列按 `sample_id` 给，
`seed=42` 再从 train 抽 10%（137）作内部验证（`21:19`）。**同一 location 的四张栅格
（asc/desc × pre/post）在 `HaitiPairs.__init__` 里一次性读入**（`21:23-29`），
按 id 分完才做任何配对，所以 asc/desc/pre/post 跨 split 在结构上不可能发生。✅ 符合 §2。

### 1.2 每个 train location 是否真的生成 ASC/DESC × 3 modes

**是，现状与 §3 逐条对应，无需改动。** `spec = [(i, m) for i in range(rows) for m in (0,1,2)]`
（`21:35`）；`__getitem__` 对每个 mode **同时**返回 `one('asc')` 与 `one('desc')`（`21:39-46`）：

| mode | ASC 目标 | DESC 目标 | GT |
|---|---|---|---|
| 0 | `ASC_pre → ASC_post` | `DESC_pre → DESC_post` | landslide |
| 1 | `ASC_pre → ASC_pre`（`post=pre`，`21:41`） | `DESC_pre → DESC_pre` | 0 |
| 2 | `ASC_post → ASC_post`（`pre=post`，`21:42`） | `DESC_post → DESC_post` | 0 |

即 2 orbits × 3 modes = **6 例/location**，与 §3 的六条一一对应。VV/VH 始终同时输入
（`q = cat(pre, post, orbit)`，`21:43`，5 通道 `[VV_pre, VH_pre, VV_post, VH_post, OrbitID]`）。

> 实现细节（本轮**不改**）：ASC 与 DESC 在同一条 spec 条目里成对返回，训练时
> `x = torch.cat((a, d))`（`23:409`）拼成 batch。内容与"拆成 6 条独立样本"等价，
> 额外好处是每批 asc/desc 数量恒等、loss 是 2B 的对称均值。

### 1.3 每 epoch 的真实 pre/post 数

`1,233 × 2 = **2,466** 个单轨真实样本/epoch`（1,233 ASC + 1,233 DESC），
等价地说：每个 location 的每条轨道每 epoch 通过一次真实 pre/post。
每个真实样本每 epoch **恰好出现一次**，所以 spec 长度恒为 3,699。

### 1.4 synthetic pre/pre + post/post 数

| 类型 | 单轨样本/epoch |
|---|---:|
| `pre/pre`（mode 1） | **2,466** |
| `post/post`（mode 2） | **2,466** |
| 真实 `pre/post`（mode 0） | 2,466 |
| **合计** | **7,398 = 1,233 × 6** |

合成样本 `y ≡ 0`（`21:45`）。**上一轮已实测：这些样本在本数据集上的损失与梯度精确为 0**
——因为模型在其上的概率恰好为 0，BCE 与 dice 的值和导数同时归零。这是要披露的训练事实，
不是本轮协议能改掉的（详见 `reports/cocd_stage1_factcheck.md` §2）。

### 1.5 val / test 是否只含真实 pre/post

**是。** `HaitiPairs(vid, False, ...)` / `HaitiPairs(eid, False, ...)` → `train_modes=False`
→ `spec` 只展开 mode 0（`21:35`）→ 只评价 `ASC_pre→ASC_post` 与 `DESC_pre→DESC_post`。
test 343 location → **686 个单轨 case**，与冻结协议的 686（343 ASC-target + 343 DESC-target）一致。
aggregate / ASC / DESC 三档报告已存在于 `report()`。✅ 符合 §4。

### 1.6 Student 是否同时学习 ASC 和 DESC

**是，一个模型。** `x = torch.cat((a, d))`（`23:409`）把 ASC 与 DESC 放进同一次 forward；
Student 的 `forward(target)` 只接收 `(B,5,128,128)`，**每次只有一条轨道**（`v3:238-258`），
部署时哪条灾后轨道先到就用哪条。**没有**分别训练两个学生。✅ 符合 §1。

### 1.7 Teacher 是否双向 target / counter

**是。** `forward_pair(asc, desc) = (from_features(fa, fd), from_features(fd, fa))`
（`v2:145-148`）：ASC 目标配 DESC 反轨，DESC 目标配 ASC 反轨，两个方向各取自己的
target/counter，且总 encoder 调用仍是 2 次。✅ 符合 §8。

### 1.8 新旧协议差异

| 维度 | v1（现有已报臂） | **v2（统一协议）** | 落点 |
|---|---|---|---|
| optimizer | AdamW(lr 1e-4, wd 1e-4) | **Adam(lr 5e-5, betas (0.9,0.999), eps 1e-8, wd 0)** | `make_optimizer()` |
| scheduler | 无（但 lr 固定 1e-4） | **无，constant 5e-5** | 断言 [B] |
| physical batch | Teacher 2 / Student 8 | **16（全部）** | `--batch`，默认取协议 |
| 梯度累积 | 无 | **路径已实现**；`BATCH×ACCUM ≠ 16` 直接报错退出 | `ACCUM` |
| seg loss | BCE(pos_weight) + 0.2·dice | **BCE(pos_weight)，无 dice** | `seg()` + `LOSS_MODE` |
| epochs | 50 | **20** | 协议 |
| 验证频率 | 每 5 epoch | **每 epoch** | `VALIDATE_EVERY` |
| early stop | `bad ≥ 2`（每 5 epoch 判一次） | **patience 3（每 epoch 判一次）** | `PATIENCE` |
| 选模 | best val AUPRC | best val AUPRC（不变） | 已有 |
| threshold | 固定 **0.5** | **val 上按 F1 选 → 冻结 → 用于 test**；AUPRC 无阈值 | `best_f1_threshold()` + `--thr` 逃生舱 |
| geometry mask | 不进训练 | 不进训练，且 **R3g 标记 deprecated、移出 PLAN** | CONFIGS |
| Student 初始化 | 复制 Teacher backbone（隐含） | **显式 `init` 字段**：`scratch` / `s0` / `teacher` | CONFIGS |
| 新增臂 | — | **`SO`（scratch 单轨基线）、`VKD`（vanilla KD = `kd='all'`）、`S0v2`（S0 暖启基线）** | CONFIGS |
| 结果表列 | 9 列 | **+`threshold`、+`iou@0.5`、+`f1@0.5`（连续性列，不用于选择）** | `report()` |

**顺手修掉两个既有 bug：**

1. `all` 分支原来检查 `teacher.pt` 而不是 `teacher{tag}.pt` → 带 tag 的 v2 sweep 会**误用 v1 Teacher**
   训练学生臂。已改为按 tag 检查。
2. 累积结果表原来 `drop_duplicates(subset='method')`，即每个方法只保留最后一行
   （实测 `results_seed42.csv` 只剩 5 行、每方法恰好 1 行、且都是 `desc/G11`）。已改为按
   `(method, partition, region)` 去重，并把累积文件名带上 tag（v2 表不再混进 v1 表）。

### 1.9 Teacher physical batch=16 是否可行

**可行，且不需要梯度累积。** 实测（MPS，`recommended_max_memory = 17.8 GiB`）：

| 配置 | 峰值显存 | batch 步时（随机张量，纯算） |
|---|---:|---:|
| Student `(16,5,128,128)` fwd+bwd | **1.18 GiB** | 0.32–0.39 s |
| Student `(32,…)` | 2.16 GiB | 0.66 s |
| Teacher 16 对（= 32 单轨目标） | **2.50 GiB** | 0.67–0.71 s |

**真实数据实测（batch 16，232 batch/epoch）**：Student **0.716 s/batch → 2.8 min/epoch**；
Teacher **0.752 s/batch → 2.9 min/epoch**；峰值 2.8 GiB。

→ **20 epoch ≈ 1 h/臂**；本轮 sweep（Teacher + SO + VKD + DIS2 + R3）**≈ 5 h**（不含 test）。

### 1.10 需要修改的脚本

| 脚本 | 处理 |
|---|---|
| `scripts/23_train_ours_v2.py` | **已改**：协议取自 config，`--protocol v1|v2`，Adam/BCE/accum/逐 epoch 验证/patience/阈值/新臂 |
| `configs/protocol_v2.json` | **新增**（唯一 config 源） |
| `scripts/47/48/49` | **新增**（可行性 / 断言 / 启动器） |
| `scripts/21_train_rapid_landslide_cocd.py` | **不改**：`HaitiPairs` 已完全符合 §3/§4；`land_loss` 保留给 v1 路径 |
| `models/landslide_cocd_v2.py` | **不改**（§8：结构暂不动） |
| `losses/*.py` | **不改**：BCE 在 `23.seg()` 内切换 |

---

## 2. §7 阈值协议（已落地）

- 选择：**只在 validation** 上按 F1 最大搜阈值，用计数直方图实现
  （`best_f1_threshold`，O(像素+分箱)），语义与 `metrics` 一致（`p ≥ thr` 为正）。
- 冻结：写进 `thresholds_<stage><tag>_seed42.json`，并在日志里打印
  `validation-selected threshold=… -- frozen for test`。
- 应用于 test 的 IoU / F1 / Precision / Recall，**每个分区、每个 G00..G11 区域都用同一个阈值**
  （`report()` 的每一行都带 `threshold` 列）。AUPRC 无阈值照常报告。
- test 上**不重新搜阈值**；`--thr 0.5` 只是复现 v1 表格的逃生舱。
- 0.5 下的 IoU/F1 作为 `iou@0.5` / `f1@0.5` 保留，仅为与前几轮表格对照，**不参与任何选择**。

## 3. 三个口径（当时待定；**2026-09-18 11:58 已拍板并落地，见 §6**）

**D1 —— `pos_weight` 保留还是去掉。** §6 只写 "Loss: BCE"。现协议保留 train split 的反频率权重
（`pos_weight = 14.356`，即正类约占 6.5%）。这**仍然是 BCE**，只是重加权；但**去掉它**会让
6.5% 的正类在"20 epoch + lr 5e-5 + 无 scheduler"下几乎学不到（阈值搜索会把阈值压到很低，
F1 仍会很难看）。config 里是一行开关：`loss.pos_weight: "from_train_split" | null`。
**默认保留**，请裁决。

**D2 —— 各臂初始化。** §9 要求同一训练协议，但没有规定初始化。v1 链条是
S0 → Teacher → Student（学生复制 Teacher backbone）；若"Single-Orbit Baseline"从 ImageNet
起步，两者就差了初始化，不能直接比较。已注册三种，任选：

| 臂 | `init` | 含义 |
|---|---|---|
| `SO` | `scratch` | ImageNet 起步，教科书式单轨基线（绝对底线） |
| `S0v2` | `s0` | 从 S0 暖启，**与 Teacher 同起点** |
| `VKD` / `DIS2` / `R3` | `teacher` | 继承新 Teacher 的 backbone（LUPI 链条） |

**建议**：内部 controlled comparison 用 `S0v2` 作基线（同起点），`SO` 作为另行报告的绝对底线。

**D3 —— "COCD-final" 尚未实现。** 上一轮已确认 `L_corr` / `L_delta` 不在代码里，
现有损失也**没有任何一项要求 `r ≈ r_T`**。所以本轮 sweep 只能覆盖
`SO / VKD / DIS2 / R3 + Dual-Orbit Teacher`，COCD-final 等结构定稿后再进同一协议。

## 4. 已完成的验证

- **`scripts/48_smoke_protocol_v2.py`：44/44 通过。** 关键几条：
  `land_loss` 在 BCE 模式下**永不被调用**（monkeypatch 会直接抛错）；
  lr 连续 20 步不变且全文件无 `lr_scheduler`；
  `physical×accum = 16` 在 (16,1)/(8,2)/(4,4)/(2,8) 四种组合都成立；
  阈值搜索与 999 点暴力扫描的最优值一致（0.14805 @ 0.006）；
  `kd='gain'` 的权重对 geometry mask **逐位不变**，只有 deprecated 的 `'full'` 会被它推动；
  R3g 带 `deprecated` 标记且不在 PLAN 里。
- **真实端到端 1 epoch 验证**（`all --stages SO --epochs 1`，产物与主表已按 md5 还原清理）：
  协议回显 3 行 → `init=scratch` 生效 → **232 batch = 3699/16** → epoch 1 即验证 →
  val 选阈值 **0.752** 冻结 → test 表含 `threshold` 列与 G10 行、`iou@0.5` 连续性列。
- 启动器 `bash -n` 通过；`DRYRUN=1` 只打印不执行；`SETTLE=999999` 时并发守卫正确拒绝启动（exit 1）。

## 5. 下一步（**未启动，等 D1/D2**）

```bash
bash Haiti_SAR_GRSL_Audit/scripts/49_run_protocol_v2.sh train   # TAG=_v2：Teacher → SO,VKD,DIS2,R3，≈5 h
bash Haiti_SAR_GRSL_Audit/scripts/49_run_protocol_v2.sh test    # Teacher-v2 的 test 表
```

产物全部带 `_v2` 标签（`teacher_v2.pt`、`<ARM>_v2_seed42.pt`、`results_all_v2_seed42.csv`、
`thresholds_all_v2_seed42.json`），**v1 的 `teacher.pt`（md5 `b80254a7…`）与已报结果不受影响**。

---

> **§4/§5 是上一次的中间状态，已被 §7/§8/§9 取代。** 下面三个小节记录 11:58 的三条决策如何落地、
> `L_Δ` 的实测量级，以及修订后的下一步。

## 6. 三条决策的落地（2026-09-18 11:58）

### D1 —— `pos_weight` **删除** ✅

- `configs/protocol_v2.json` → `loss.pos_weight: false`，并显式记下 `dice: false` / `focal: false`：
  作者式普通 BCE，不用 14.356、不用 Dice、不用 Focal。
- 断言（[A][C]）：`seg()` **即便被传入 `weight` 也**与无权重 BCE **逐位相同**，且与 14.356 加权版本
  **不同** —— 说明开关真的在起作用，而不是被忽略；`land_loss` 在 BCE 模式下**永不被调用**。
- **风险已实测排除**：正类约占 6.5%，去掉重加权后理论上可能塌成全背景。试点 `SO --epochs 2`：
  训练 loss 0.86 → 0.065，**val AUPRC 0.3412 = 正类占比 0.0685 的 5 倍**，val 选出的阈值 **0.225**
  （不是退化到 0.01 附近），val F1 0.4032 ⇒ **普通 BCE 在 20-epoch 尺度下学得动**。

### D2 —— Backbone **内部统一 ImageNet、外部不强制** ✅

- `init.backbone: "imagenet"`、`init.teacher: "imagenet"`；**PLAN 五臂全部 `init: "scratch"`**。
- 实测断言（[I]）：**R0/R1/R2/R3 四种 variant 的 backbone `state_dict` 逐位相同**（预训练特征在构造时
  加载、随机层在 variant 专属模块之前初始化，RNG 消耗顺序一致），且**初始修正精确为零** ⇒
  所有臂从**同一个未修正函数**出发。这是"内部统一"最硬的一条证据。
- Teacher 不再从 S0 暖启；`--protocol v1` 保留原 S0→Teacher→Student 链条，已报臂可原样复现。
- **唯一保留的继承是 R2/R3 的 Ω 算子**（`op_init: "teacher"`）。它是方法定义本身，现在是**独立字段**，
  不再与 backbone 初始化混在一起；`SO/VKD/DIS2` 无此字段，R3D 属 R3 族故同样继承。
- **必读后果**：去掉 backbone 继承后，学生臂的冷却起点由"教师自轨"变为 ImageNet。同一次前向实测
  `seg` 在初始状态为 **v1 链条 0.011（train）/ 0.128（val）→ scratch 0.742 / 0.742**。
  因此 **v2 表的绝对数值会明显低于 v1 表**（v1 的 S0 本身就训了 50 epoch）：v2 是"内部自洽的新尺度"，
  与 v1 的绝对值**不可比**，两套表必须分开呈现。
- **外部臂**：Boehm / CDNetE / FC-Siam / MFEWF 保留各自架构、不套统一协议，已把该策略写进
  config 的 `external_baselines` 段。本轮未改动它们的任何代码。

### D3 —— 先做 **`R3 + L_Δ`**，不做 `L_corr` ✅

- 新增臂 `R3D`（`variant=R3, kd=gain, op_init=teacher, delta=true`），**`R3` 是它的精确无 Δ 对照**
  （结构、KD 规则、初始化全同；[G] 有断言）。
- 定义：`L_Δ = aggregate( SmoothL1( Δp_S , stopgrad(Δp_T) ) )`，`Δp = σ(z34) − σ(z0)`。
  取**概率空间**而非 logit 空间：两个读数在 logit 上可差到 ~90，学生头又是独立初始化的，
  匹配原始 logit 差没有意义，而"决策"是概率的属性。
- `L_corr` **明确未实现**（[J] 有断言）。Student 的 `forward_taps` 新增 `z0`（一次额外 decode；
  部署 `forward` 路径与 `forward_taps` 的 `z4/z34` 逐位未变，[I] 有断言）。

## 7. `L_Δ` 的实测量级 —— 这决定 R3D 会不会变成"规则正确但通道为空"

**初始状态**逐 batch 实测（教师 = v1 冻结教师，仅作量级探针；学生按 scratch / 继承两种起点）：

| 学生起点 | 划分 | seg | kd(gain) | kd(all) | `L_Δ` 像素均值 | `L_Δ` 平衡均值 | `L_Δ`(G10) |
|---|---|---:|---:|---:|---:|---:|---:|
| scratch（v2） | val | 0.7422 | 0.0883 | 0.6196 | 0.00149（**0.20%**） | **0.00310（0.42%）** | 0.00311 |
| scratch（v2） | train | 0.7443 | 0.1058 | 0.6633 | 0.00015 | 0.00049 | 0.00029 |
| teacher（v1 链） | val | 0.1284 | 0.0092 | 0.0156 | — | — | — |

1. **`L_Δ` 在像素均值下几乎为零（seg 的 0.20%）**。两个后验在 99% 以上的像素上都被压到 ≈0，
   `|Δp_T|` 均值只有 **0.0101**（p99 = 0.242），而 SmoothL1 在 `|x| < 1` 时是平方型 ⇒ 被饱和像素淹没。
   若按像素均值直接用，**R3D 会因为通道权限、而不是因为想法而空转**——这正是 R3g 已经踩过的坑。
2. **改成本项目 KD 项已经在用的 foreground/background 平衡均值后为 0.42%（2.1×）**，并把权重放到
   变化像素上：`L_Δ` 单独在 **G10** 上统计得 0.0031，与平衡均值同量级 ⇒ 信号确实来自失真区。
   默认已设为 `aggregation: "balanced"`（`pixel_mean` 仍可切换，[J] 有断言）。
3. **即便如此，`lam_delta = 1.0` 仍只是 seg 的 0.42%、KD 项的 3.5%。** 要让 `L_Δ` 与 KD 项同权需要
   `lam_delta ≈ 25–30`。**这是 R3D 上唯一还需要你定的数**（不做自动配平：整批残差为零时会退化到
   钳位下界）。**建议先用 `lam_delta = 10`**（约 KD 项的 1/4、seg 的 4%），靠新加的
   `terms epoch=… seg=… kd=… delta=…` 三分解日志确认它是否真的在动。

同一次测量顺带得到两条与 D2 直接相关的量级事实：

- **`kd/seg` 在初始状态是 0.07（v1 继承起点）与 0.12–0.14（scratch 起点）**，同量级；它随训练变大
  （seg 下降而 KL 项有下界）：实测 40 步后 **1.9×**、80 步后 **5.2×**。所有蒸馏臂共享
  `kd_weight = 1.0`，所以**臂间比较仍然受控**；但"目标函数在训练中后期主要由 KD 项驱动"这件事
  需要在论文里如实写出，不能写成"KD 只是个小辅助项"。
- **`kd(all)`（Vanilla KD）在初始状态是 `kd(gain)` 的 6–7 倍**（val 0.62 vs 0.088），因为 `gain` 只在
  约 12% 的像素上活跃。VKD 与 R3 是"同一权重、不同规则"，但两条规则的**量级天然不同**，比较时须说明。

## 8. 验证（已跑完）

- **`scripts/48_smoke_protocol_v2.py`：54/54 通过。** 新增两组：
  - **[I] 初始化族**：四种 variant 的 backbone **逐位相同**；初始修正**精确为零**（`z0 == z4 == z34`）；
    `z0` 已暴露给 `L_Δ` 且部署路径未变；PLAN 五臂全部 `init=scratch`。
  - **[J] `L_Δ`**：可微、教师侧 stop-grad、默认 `pair=z34_minus_z0 / space=prob / beta=1`、
    平衡均值与显式实现**逐位一致**、`pixel_mean` 仍可达、`L_corr` 未实现。
- **端到端试点**（产物已全部删除；`results_seed42.csv` 已按 md5 `acebddfb…` 还原；冻结 `teacher.pt`
  摘要未变）：`SO --epochs 2`（tag `_pilot`）→ val AUPRC 0.3412、阈值 0.225/F1 0.4032、**2:49/epoch**
  （与 `scripts/47` 的预算吻合）；`R3D --epochs 1` → `L_Δ` 训练循环真实走通（当次仍为像素均值，
  日志 `seg=0.1542 delta=0.0007`），val 阈值 0.637 冻结进 test 表。
- 启动器：`bash -n` 通过；`STAGES=SO` 时**不会**先训 Teacher；`DRYRUN=1` 只打印；
  `SETTLE=999999` 时守卫正确拒绝启动（exit 1）。

## 9. 下一步（**未启动**）

PLAN 现为 `SO, VKD, DIS2, R3, R3D`（5 臂）+ Teacher-v2，**合计约 5.8 h**。建议分两段，
先用 2 h 买一个判断依据：

| 顺序 | 命令 | 作用 | 约时 |
|---|---|---|---|
| 1 | `STAGES=SO bash scripts/49_run_protocol_v2.sh train` | 单轨基线（不需要 Teacher），定 v2 尺度的下限 | ~1 h |
| 2 | `bash scripts/49_run_protocol_v2.sh train` | Teacher-v2，随后 VKD / DIS2 / R3 / R3D | ~1 h + 4 h |
| 3 | `bash scripts/49_run_protocol_v2.sh test` | Teacher-v2 的一次 test 读表 | 分钟级 |

**为什么建议先跑第 1 步再看 Teacher**：Teacher 的活更难（两轨 + 修正），却和所有臂一样只有
20 epoch / lr 5e-5 / scratch / 普通 BCE。**如果 Teacher-v2 的 val AUPRC 不高于单轨基线，
四个蒸馏臂的前提（教师带来额外信息）就不成立**，那时应先处理这一点，而不是再花 4 h。

开跑前需要你定的只剩一个数：**`lam_delta`**（§7 第 3 条，建议 10；config 键
`corrective_distillation.weight`）。产物一律带 `_v2` 标签，v1 的 `teacher.pt`（md5 `b80254a7…`）
与已报结果不受影响。
