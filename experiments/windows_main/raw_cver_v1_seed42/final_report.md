# No-Mask / Raw 双轨→单轨正式实验报告（Search-KD 复现 + CVER，v1, seed 42）

**日期**: 2026-09-26 · **协议**: 当前唯一正式 v4 协议（train=1370 / test=343 无 validation，test AUPRC 选点，max 100，patience 2，seed 42，batch 16，Adam 5e-5，wd 0）
**Setting**: 全程 No-Mask / Raw：seg 全像素、search 全 query、KD 全 query、Teacher counter 不做 geometry gating；geometry 仅用于 post-hoc 分区评测。
**脚本**: `cocd/windows_main/train_raw_cver_v1.py` · **产出**: `experiments/windows_main/raw_cver_v1_seed42/`

---

## 1. Teacher provenance（第 3 节，先于一切 Student 实验）

四臂 Teacher（`teacher_target_search_v2.py`）**就是在当前锁定协议下训练的**（脚本内 `assert not vid`，splits_v3 合并 train；同 optimizer 契约；counter 只进 policy context）。因此**直接复用，未重训**：

| Teacher | context | 监控 AUPRC |
|---|---|---:|
| SG0（Raw，gamma=0） | self | 0.8815 |
| CG0（Raw，gamma=0） | counter | **0.8948**（+1.33pp） |
| SG10 | self | 0.8808 |
| CG10 | counter | **0.9012**（+2.04pp） |

**CG_Raw > SG_Raw 成立**（+1.33pp 纯 Raw / +2.04pp 带 G10 加权项）。Student 实验统一使用冻结的 **CG10_seed42.pt**（即原始成功 S2 的配对 Teacher）。

## 2. 主结果（best checkpoint，post-hoc 分区 AUPRC / IoU / F1）

| Method | Overall | BothValid | TargetValid | G10 | G01 | best ep |
|---|---|---|---|---|---|---|
| S0-Raw | 0.8798 / 0.6291 / 0.7724 | 0.8691 / 0.6134 / 0.7604 | 0.8755 / 0.6227 / 0.7675 | 0.8882 / 0.6414 / 0.7815 | 0.8947 / 0.6514 / 0.7889 | 19 |
| **S2-Raw** | **0.8888 / 0.6423 / 0.7822** | **0.8784 / 0.6267 / 0.7705** | **0.8848 / 0.6360 / 0.7775** | **0.8964 / 0.6539 / 0.7908** | **0.9036 / 0.6650 / 0.7988** | 44 |
| CVER-A | 0.8605 / 0.6005 / 0.7504 | 0.8492 / 0.5850 / 0.7381 | 0.8562 / 0.5961 / 0.7470 | 0.8679 / 0.6033 / 0.7526 | 0.8774 / 0.6304 / 0.7733 | 22 |
| CVER-B | 0.8563 / 0.6056 / 0.7544 | 0.8437 / 0.5897 / 0.7419 | 0.8517 / 0.6004 / 0.7503 | 0.8650 / 0.6137 / 0.7606 | 0.8755 / 0.6335 / 0.7756 | 29 |

（S0-Raw 逐位复现历史 S0 = 0.8798@19；S2-Raw 0.8888@44 vs 历史 0.8887@44 —— 复现成功。）

### 回答第 16 节问题 1 与 2

- **Q1 复现**：是。S2-Raw − S0-Raw = **+0.90pp Overall AUPRC**（IoU +1.32pp，F1 +0.98pp），与历史 +0.89pp 几乎一致。原始未 mask 的 Search-KD 正迁移在当前统一协议、CUDA 环境下成立。
- **Q2 收益来源**：**不是 G10/G01 专属**。AUPRC 增益：BothValid +0.93、TargetValid +0.93、G10 +0.82、G01 +0.89 —— 五个分区几乎均匀。原 S2 的 G10/G01 query 加权（w=1+m）只是 KD 采样偏好，正迁移本身来自更一般的 search 行为对齐，BothValid 内真实存在。

## 3. CVER 设计与实现（第 5–10 节逐条对应）

- **Candidate pool**：复用现有 3-level FPN 结构，每级 9 个固定 offset（3×3，特征像素单位）= **27 个 target-side candidates**；counter 永不进 decoder values。
- **Teacher utility**（gradient×activation 近似，第 6 节许可）：Teacher 对 fixed-pool 做均匀聚合 readout → `ℓ=BCE(z,y)` → `g=∂ℓ/∂R`；`u_raw(k)=level-attention(m_l)×(g·e_k)/√C`；`u_T=ReLU(u_raw)`；`p_T=softmax(u_T/τ)`，τ=1。
- **Student**：轻量 scorer（q/k 两个 1×1 conv，唯一新增参数），`p_S=softmax(u_S)`，`R_S=Σp_S·e_k` → 正常 decoder。推理单轨、无 counter。
- **CVER-A**：listwise `KL(p_T‖p_S)` 全 candidate；**CVER-B**：仅 Teacher-positive（u_raw>0）candidates 携带 target mass。λ 一次首 batch 校准冻结（A: 0.1307，B: 0.1052），初始贡献 7.5%。
- 事故记录：B 初版用 `-inf` 掩码 softmax，无 positive 的 query 行产生 NaN 并经反向污染权重；已改为 exp 掩码归一化（无 -inf）后重跑成功。

## 4. CVER 机制量（第 13 节）与失败诊断（第 15 节）

### Teacher 侧

| 量 | 值 | 含义 |
|---|---:|---|
| candidate utility entropy | **3.2958 = ln 27** | p_T 几乎均匀——utility 幅值远小于 τ=1 |
| top-1 / top-3 concentration | 1/27 / 3/27 | 同上 |
| positive candidate 比例 | 44.9% | 符号信息存在，但幅度无排序力 |
| **correct vs shuffled counter ranking agreement（Spearman）** | **0.914** | **换 counter 几乎不改变 utility 排序** |

### Student 侧

| 量 | CVER-A | CVER-B |
|---|---:|---:|
| Teacher/Student ranking Spearman | 0.014–0.02 | 0.05–0.14 |
| top-3 overlap（随机基线 0.111） | 0.089–0.126 | **0.23–0.27** |
| ranking intervention：learned / uniform / random（Overall AUPRC） | 0.8605 / 0.7274 / 0.7164 | 0.8563 / 0.6545 / 0.6278 |
| intervention 差距（learned−uniform） | **+13.3pp** | **+20.2pp** |

### 诊断（第 15 节三分法）

**主判定：情况 A。** Teacher 的 candidate utility 与 shuffled-counter 的 utility 排序一致性高达 0.914——在"uniform fixed-pool readout + grad×activation + level-attention"这个定义下，**utility 主要由 target 自身特征决定，正确 paired counter 并没有实质改变"哪些 target evidence 有用"**。第一版 utility 定义没有捕获 cross-view complementarity（Q3 的答案是否定的）。

次要观察（部分情况 C 特征，但非 C 本身）：Student 确实学出了有任务价值的 ranking——intervention 证明 learned 比 uniform/random 高 13–20pp；B 的 top-3 overlap 0.26 也显示 Teacher-positive 结构可学（排除情况 B）。但这个 ranking 的任务价值来自 seg loss 自身的聚合学习，而 KD target（近均匀/仅含 45% 正例符号信息）几乎不携带额外跨轨知识，因此最终不涨反跌（固定 27-candidate 池 + 均匀化倾向弱于 S0 的 deformable search）。

### Q3–Q5 的直接回答

- **Q3 正确 paired counter 能否稳定改变 target-side evidence utility？** 在本 utility 定义下**不能**（shuffle 一致性 0.914）。要么 utility 定义未触及 complementarity，要么该信息在"逐 candidate 一阶贡献"粒度上本身很弱。
- **Q4 utility ranking 能否仅凭 target 学出来？** 部分能（B 的 top-3 overlap 0.26 vs 随机 0.11），且学出的 ranking 具有真实任务价值（intervention 证明）；但与 Teacher ranking 的全局 Spearman 很低。
- **Q5 Evidence Ranking 是否比 exact Search Policy KD 更容易转化为 segmentation 增益？** 第一版**不是**：S2-Raw +0.90pp，CVER −1.9~−2.4pp。在 utility 定义修复（真正携带 counter 特异性）之前，ranking 蒸馏不构成对 policy KD 的替代。

## 6. 结论

1. 原始 Search-KD 的正迁移在当前协议下**严格复现**（+0.90pp，且增益遍布 BothValid——这是对"收益只来自畸变区"印象的重要修正）。
2. Evidence Ranking 第一版失败，根因是 **Teacher utility 对 counter 不敏感（情况 A）**，而非 Student 学不出 ranking（其 ranking 有任务价值）。
3. 下一步若继续 ranking 路线，优先修复 utility 定义：提高 Δℓ_k 的保真度（真正的 leave-one-out / candidate suppression 而非一阶近似）、按 query 局部校准 utility 尺度（消除 τ=1 下的均匀化），并先单独验证 "correct−shuffled counter utility 差异" 是否显著为正——否则不应再训练 Student。

## 7. 交付物

| 文件 | 内容 |
|---|---|
| `final_report.md` | 本报告 |
| `all_arms_regions.json` | 四臂全分区指标 |
| `ranking_intervention.json` | CVER intervention 评测 |
| `{arm}_progress.json` / `{arm}_summary.json` / `{arm}_seed42_best.pt` | 各臂完整 history、机制量、best checkpoint |
| `protocol.json` | 协议与 Teacher provenance 快照 |

**已知偏差**：CVER-B 初版因 `-inf` softmax NaN 崩溃，修复后重跑（该 arm 为修复后版本）；S2-Raw 的 KD 项在 loss 中的数值占比随 seg 收敛升至 ~67%，这是原始校准设计（绝对尺度固定）的固有行为，未做干预。
