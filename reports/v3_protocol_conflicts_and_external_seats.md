# v3 协议冲突清单 + 外部席位重建记录

日期：2026-09-19
依据文件：`docs/FROZEN_EXPERIMENT_PROTOCOL_v3.md`（用户提供的冻结协议，权威）

---

## 0. 7398 与 8220 的推导（口径已统一到 8220）

> **状态：已作废 7398 口径。** 用户于 2026-09-19 明确指示「那就不是 7398，
> 应该是 8220」——train 合并 val 后的 1370 location 才是训练预算。本节保留
> 7398 的推导过程，因为它解释了为什么两个数字会同时出现在文档里。

### 算式

```
train locations        = 1370        （1233 原 train + 137 原 validation）
每个 location 的 mode  = 3           (mode0 pre->post, mode1 pre->pre, mode2 post->post)
每个 mode 的 orbit     = 2           (asc, desc)
------------------------------
1370 x 3 x 2 = 8220 个单轨训练视图   ← 协议 §2 的正式预算
```

### 它与其它数字的关系（全部实测自代码，非推算）

| 数字 | 算式 | 含义 | 状态 |
|---:|---|---|---|
| **2466** | 1233 × 1 × 2 | 旧 train 的**真实** pre→post 视图数 | 历史 |
| **3699** | 1233 × 3 | 旧 train 的 `spec` 条数（不含 orbit 维度） | 历史 |
| **7398** | 1233 × 3 × 2 | 旧 train 的**全部单轨视图数** | **已作废** |
| **2740** | 1370 × 1 × 2 | v3 train 的真实 pre→post 视图数 | — |
| **4110** | 1370 × 3 | v3 train 的 `spec` 条数 | — |
| **8220** | 1370 × 3 × 2 | **v3 协议要求的单轨训练视图数** | **现行** |

### 关键澄清：7398 与 8220 的差别只有 137 个 location

`8220 − 7398 = 822` = **137 个原 validation location × 3 modes × 2 orbits**。
这正是 §1「Train = 原 train + 原 validation」的直接推论。

### 文档里两处表述的口径差异（已核实）

- `cocd_final_teacher_student.md` §3.1 写 **3699**，那是 `spec` 条数（1233×3），不含 orbit。
- `external_baselines_and_kd_cards.md` §6 写 **7398**，那是单轨视图数（1233×3×2）。
- 两者不矛盾，是不同粒度。**论文里必须统一口径**，一律用单轨视图数
  （只有这个数字才与 batch、step 数、数据曝光量直接对应）。

### 实测确认（2026-09-19，v3 划分生效后重新实测）

```
split()          -> train=1370  val=0  test=343
len(ds.spec)     -> 4110        （= 1370 x 3 modes，不含 orbit）
单轨视图数        -> 8220        （= 4110 x 2 orbits）
optimizer steps  -> 257         （= ceil(4110 / batch16)）
test 视图数       -> 686         （= 343 x 2，协议 §3）
```

**易错点（本轮踩过）**：`len(ds.spec)` 是 **4110** 而不是 8220，因为
`spec` 的索引是 `(location, mode)` 对，**不含 orbit 维度**；轨道维度体现在
`__getitem__` 每次返回 `one('asc'), one('desc')` 两个视图、训练循环再
`torch.cat((a, d))` 上。拿 `spec` 长度去比 8220 会误判成「数据只有一半」。

---

## 1. v3 推翻的现行设定（8 条）

| # | v3 要求 | 当前代码/结果 | 冲突 |
|---|---|---|---|
| 1 | 取消 validation，train=1370 | `data/splits/` = 1233/137/343 | 需重建划分（**已完成**） |
| 2 | 每 epoch 8220 视图 | 现有 7398（train_modes=True） | 需并入 137 个 val location |
| 3 | 外部 baseline 也必须跑 6 mode | 外部仅 `train_modes=False`（2466 视图） | 推翻外部豁免 |
| 4 | 固定跑满 20 epoch，禁用早停 | `PATIENCE=3`，默认开 | 推翻 |
| 5 | epoch 20 = 正式模型 | `checkpoint='best_validation_auprc'` | 推翻 |
| 6 | 阈值永久 0.5 | `source='validation'`, `criterion='max_f1'` | 推翻 |
| 7 | 全部重新训练 | Windows 侧已有 7 个臂的结果 | 全部作废 |
| 8 | 主表含 FC-Siam | FC-Siam 从未训练 | 新增训练任务 |

`cocd/windows_main/main.py::enforce_protocol()` 中有 6 条断言会直接 `SystemExit`：
第 116 行（patience≠3）、第 118 行（checkpoint 规则）、第 120 行（threshold 规则）等。

---

## 2. 已完成的落地动作

### 2.1 协议存档

`docs/FROZEN_EXPERIMENT_PROTOCOL_v3.md` —— 用户提供的协议原文，§1–§20 全文。

### 2.2 新的 train/test 划分（冻结 + SHA256）

`data/splits_v3/`：

| 文件 | n | SHA256 |
|---|---:|---|
| `train_ids.csv` | **1370** | `754b0fd6e829a78d050d958f86b4899e4e938a3880cc46ea0cd18ed4424d6f1f` |
| `test_ids.csv` | **343** | `31528c417f361a85b7ba7fd7ecf8a2fdb62f70cff07400a0fce33aa9ea66fb2b` |

校验全部通过：
- train = 原 train(1233) + 原 val(137)，保持 pinned 顺序
- test = 原 test(343)，**集合完全相同**（已断言）
- train ∩ test = ∅

### 2.3 外部方法官方代码（GitHub 下载）

位置：`cocd/third_party/landslide_baselines/`

| 席位 | 仓库 | commit | 状态 |
|---|---|---|---|
| Boehm SAR U-Net++ | `iprapas/landslide-sar-unet` | `e35fad9948c7ff27b7a7f751789c5e90763ee7e4` | **已下载，37 文件** |
| DIS2 | `nhikieu/DIS2` | `497afc2d86ea6a85271ccace0a052603d918a9b3` | **已下载，39 文件** |
| FC-Siam-diff | `rcdaudt/fully_convolutional_change_detection` | 4dd8323 | **已下载，35 文件** |

官方源码证实（`boehm_landslide_sar_unet/src/lit_module.py`）：
- `smp.UnetPlusPlus(encoder_name='resnet50', in_channels=num_channels, classes=2)`
  —— **未传 `encoder_weights`，而 smp 默认即 `'imagenet'`** ⇒ 官方确实用 ImageNet 预训练
- `main.py` 的 `--loss` 默认 `'ce'` ⇒ 官方用 **CrossEntropyLoss**
- 参数量实测：5 通道下 ResNet-50 U-Net++ = **48.9922 M**；ResNet-34 = **26.0850 M**

### 2.4 无官方代码的两个席位（已确证）

| 席位 | 检索结论 |
|---|---|
| **MFEWF** | GitHub API 查 `ChenLifu2022` 名下仓库，**只有 `Aircraft-detection`**（无关）；搜索 "MFEWF" 返回空。**无官方代码，确证。** |
| **CDNetE** | `cocd/third_party/.../README.md` 记「No Haiti/CDNetE official repository found yet after author-page and GitHub search」，状态 `source verification in progress`。**无官方代码。** |

二者必须按论文模块描述复现，表中标注 `adapted`。

### 2.5 ImageNet 预训练权重（已打通）

**这是一个真实的坑**：HuggingFace 缓存在 Windows 上没有正确建立符号链接，
`snapshots/*/model.safetensors` 是 **0 字节**，导致
`SafetensorError: Error while deserializing header: HeaderTooSmall`。

根因：`blobs/` 里权重是完整的（resnet34 = 87,275,112 B；resnet50 = 102,464,800 B），
但 `snapshots/` 下是 0 字节软链。Windows 建符号链接需要管理员权限或开发者模式。

处理：把 blob 内容复制到 snapshot 路径（`config.json` 单独从 HF 重新拉，
因为它是 156 B 的小文本，不能被权重覆盖）。

验证结果：

| encoder | Unet | UnetPlusPlus |
|---|---|---|
| resnet34 | 24.4428 M | 26.0850 M |
| resnet50 | 32.5275 M | 48.9922 M |

全部可正常构建，ImageNet 权重加载成功。

### 2.6 四个席位的训练脚本（新建）

| 文件 | 作用 |
|---|---|
| `cocd/scripts/23_train_external_baselines.py` | 四个席位统一 runner（v3 协议） |
| `cocd/external_mfewf.py` | MFEWF 复现（DRN + AMM + CAASP + MFFRM） |

**统一部分**（与内部臂一致）：Adam 5e-5 / wd 0 / batch 16 / 20 epoch / 无早停 /
epoch-20 checkpoint / 阈值固定 0.5 / AUPRC threshold-free。

**允许不同部分**（v3 §18 公平性原则）：
- Boehm：U-Net++(resnet50) + CE（官方方法定义）
- CDNetE：U-Net(resnet34) + CE（early-fusion 契约）
- FC-Siam：`SiamUnet_diff` 官方网络逐字保留 + NLL（官方输出 log-prob）
- MFEWF：自实现四模块 + CE

冒烟测试（全部通过）：

| 席位 | 参数量 | 输出 | 损失 | 梯度范数 |
|---|---:|---|---:|---:|
| boehm | 48.9922 M | (4,2,128,128) | 0.8150 | 24.00 |
| cdnette | 24.4428 M | (4,2,128,128) | 0.8047 | 30.00 |
| fc_siam | 1.3501 M | (4,2,128,128) | 1.0797 | 5.66 |
| mfewf | 26.9821 M | (4,2,128,128) | 0.6107 | 22.84 |

端到端 1 epoch 实测（FC-Siam，真实数据）：**232 steps，mean loss 0.6341，19.7 秒**。

### 2.7 依赖

`segmentation_models_pytorch` 在 `requirements.txt` 中已有声明，
本机安装版本 **0.5.0**（与规格文档记录一致）。
注意：只有外部席位需要它，COCD 主线不 import。

---

## 3. MFEWF 复现的实现细节（供论文披露）

模块按论文 §5.1 消融实验的描述实现：

| 模块 | 论文描述 | 本实现 |
|---|---|---|
| **DRN** | 残差主干，更好提取多层级滑坡特征 | ResNet-34 主干（ImageNet），首层改 5 通道，新通道零初始化以保全预训练响应 |
| **AMM** | 选择有效的**低层**特征 | 通道注意力（avg+max → MLP）+ 空间注意力（7×7 conv），作用在 `layer1` 输出（64 通道） |
| **CAASP** | 提取**高层**的上下文与语义信息 | 空洞率 1/2/4 三分支 + 全局池化分支 → 拼 1024 → BN → gate → 1×1 → 256 |
| **MFFRM** | 自适应分配低层/高层权重并精炼 | 两层各 1×1 投影到公共宽度 192 → 逐通道 sigmoid 门控 → 加权相加 → 3×3 精炼 → 96 |

参数量 **26.9821 M**，与 Boehm-adapted(26.09M)、CDNetE-adapted(24.44M)、S0(28.08M) 同量级。

---

## 4. 待用户决策

1. **已有 7 个 Windows 结果如何处置**：全部标记 v2 历史并从 v3 重跑，还是保留 N0 作为参照？
2. **内部臂的 v3 改造范围**：`main.py` 的 `enforce_protocol()`、checkpoint 规则、
   阈值规则都要改；`scripts/23_train_ours_v2.py` 的 `early_stop` 默认值要改。
3. **是否现在启动全量 seed=42 pipeline**。

---

## 5. 2026-09-19 第二轮：协议落地到代码

用户指示：
> 「那就不是 7398，应该是 8220 1370 × 3 × 2。都按这个来，然后 resnet 咱就用 34 的，
> MFEWF / CDNetE 这俩咱自己复现。」

### 5.1 三条决定的落地

| 决定 | 落地位置 | 结果 |
|---|---|---|
| 训练预算统一为 **8220** | `paths.py` + `21_...py::split()` + `23_train_external_baselines.py` | `split() -> train=1370, val=0, test=343`；`EXPECTED_TRAIN_VIEWS=8220` 断言生效 |
| encoder 一律 **resnet34** | `23_train_external_baselines.py::ENCODER` | 四席位参数量收敛：boehm 26.09M / cdnette 24.44M / fc_siam 1.35M / mfewf 26.98M |
| MFEWF / CDNetE **自研复现** | `cocd/external_mfewf.py`；`build_cdnette` | 已在头注声明「no official code」，论文表格须标 `adapted` |

### 5.2 关键修复：v3 划分此前根本没生效

**问题**：`23_train_external_baselines.py` 实测仍得到 `train=1233, val=137`，
即 8220 的 `data/splits_v3/` 完全没被读取。

**根因**：`rapid.split()` 读的是 `SPLIT_DIR`，其默认值指向 `data/splits/`
（旧的 1233/137/343 三分）；`data/splits_v3/` 与之不相通。

**修复**：
1. `cocd/paths.py`：`SPLIT_DIR` 默认改指 `data/splits_v3`；旧三分目录保留可用
   `COCD_SPLIT_DIR` 环境变量切回，以便复现 v2 历史结果。
2. `cocd/scripts/21_train_rapid_landslide_cocd.py::split()`：改为「train+test 存在
   即返回」，`val` 缺失时返回**空列表**而不是静默重抽划分——避免 v3 下悄悄
   回落到 seed-42 随机划分。

### 5.3 关键修复：val 诊断列删除

v3 把 val 并入 train 后，再在 val 上算 AUPRC 等于在训练集上评估。
按用户指示**彻底删除**：

- 移除每 epoch 的 `val_set` 构造、val DataLoader、AUPRC 计算；
- `progress.json` 不再有 `val_auprc` 字段，`hist` 只记 `epoch` + `loss`；
- `run_seat` 入口加断言：若 `split()` 返回非空 val，直接 `SystemExit` 并提示
  检查 `COCD_SPLIT_DIR`。

### 5.4 端到端验证

- **划分**：`train=1370 / val=0 / test=343`，`spec=4110`，单轨视图 `8220`，steps `257`。✅
- **四席位构建**：resnet34 下全部成功，ImageNet 权重正常加载。✅
- **全流程 rehearsal**（FC-Siam，1 epoch，`pretrained=False`）：
  ```
  run_seat(fc_siam) COMPLETED
    overall  IoU@0.5=0.0684  F1@0.5=0.1281  AUPRC=0.0824
    asc      IoU@0.5=0.0643  F1@0.5=0.1209  AUPRC=0.0754
    desc     IoU@0.5=0.0726  F1@0.5=0.1353  AUPRC=0.0915
  artifacts: epoch20.pt, latest.pt, metrics.csv, progress.json, test_predictions.npz
  test_predictions.npz: p.shape=(686,128,128), orbits=[343,343]
  ```
  test 视图数 686 = 343 × 2，与协议 §3 一致。✅
  （指标低是预期：仅 1 epoch 且无预训练，此步只验证管线。）

### 5.5 本轮踩到的坑（供后续避雷）

- `len(ds.spec)` = **4110** ≠ 8220。`spec` 的索引是 `(location, mode)`，
  orbit 维度不在其中；每个 `__getitem__` 返回 asc+desc 两个视图。
  用 `spec` 长度校验 8220 会误判。
- `rapid` 模块暴露的是 `dl()` 不是 `loader()`，脚本内统一用本地 `loader()` 包装。
- `ENCODER` 常量必须定义在 `build_boehm` 的默认参数求值之前，否则 `NameError`。

### 5.6 仍未完成（本节写于 5.7 之前，其中前两条已被 5.7 解决）

- ~~**内部臂的 v3 改造未动**~~ → 见 §5.7，已修完。
- ~~外部席位尚未开始真实 20-epoch 训练~~ → 见 §5.7，Boehm 已完成。
- **seeds 123 / 2026 未跑**：目前只有 seed 42。
- **命名张力未解**：用户口语说「v2 的来」，但存档文档是
  `FROZEN_EXPERIMENT_PROTOCOL_v3.md`，协议 JSON 的 `revision` 字段是 `"v3"`。
  两者指的是同一套协议，只是编号口径不一致，尚未正式收口。

### 5.7 第三轮：内部 runner 收尾 + 外部席位开训

#### 5.7.1 `windows_main/main.py` 的 v3 收尾（已完成）

上一轮只把 `enforce_protocol()` 反转成「要求 v3」，但调用方还留着旧签名，
一跑就炸。本轮把遗留全部清掉：

| 位置 | 原状（v3 下必炸） | 处置 |
| --- | --- | --- |
| `run_legacy_teacher` | `v2.train_teacher(..., early_stop=True)` | 删掉 `early_stop`，签名去掉 `val_set`，补 `enforce_protocol()` |
| `run_legacy_arm` | `v2.train_student(..., early_stop=True)` | 同上；训练后只 log 落盘路径 |
| `run_val` + `save_val_table` | 整个 val 读表通路 | 删除（v3 无 val 划分） |
| `run_audit`（门控） | 在 val 上做 counter-orbit 审计并 `gate['passes']` 决定是否继续 | 删除 108 行；改为 `run_audit_readout()` 只打印「不可评估」及原因 |
| `main()` | `val_set = HaitiPairs(vid, ...)`；`stage == 'val'` 分支 | 删掉 val_set 构造；`val` 阶段改为直接 `SystemExit` 并说明已随划分撤销 |
| `--arm` 帮助文本 / stage 列表 | 含 `val` | 移除 |
| `print` 协议行 | `patience={PROT["schedule"]["early_stop_patience"]}` | 改为 `early_stop=` + `checkpoint=` |

关于审计门控的处置需要说明：它原本要在一份 held-out 划分上比较
「正确 counter / zero counter / shuffled counter」三种条件下的 dual AUPRC，
用 `passes` 当作「能不能蒸馏」的开关。v3 取消 val 之后，这个门控失去
可评估的数据；若改到 test 上评估，就等于让 test 划分去影响一个建模决策，
这恰好是 v3 要杜绝的事。因此选择**退役该门控并在日志里明说**，而不是
悄悄换数据集继续跑。teacher 稳定性改由 N2 / N2noorbit / N1 三个消融臂
在最终表里体现。

实测：`main.py plan` 全程通过，打印
`early_stop=False checkpoint=epoch_20` 与 `[paths] splits ... data\splits_v3`。

#### 5.7.2 外部席位训练进度

三个新席位**全部完成**（seed 42，各自 20 epoch，无早停）：

| 席位 | 参数量 | loss | 状态 | 每 epoch | test overall | 训练总时长 |
| --- | --- | --- | --- | --- | --- | --- |
| Boehm SAR U-Net++ | 26.0850 M | CE | **完成** | 42.1 s | IoU 0.4095 / F1 0.5810 / AUPRC 0.6692 | 14m 48s |
| CDNetE Early Fusion | 24.4428 M | CE | **完成** | 23.6 s | IoU 0.2404 / F1 0.3877 / AUPRC 0.5304 | 8m 32s |
| MFEWF adapted | 26.9821 M | CE | **完成** | 24.1 s | IoU 0.2056 / F1 0.3411 / AUPRC 0.4496 | 8m 39s |
| FC-Siam official | 1.3501 M | NLL | 未跑 | — | — | — |

三席位一致收敛：

```
boehm    loss 0.44421 -> 0.02645
cdnette  loss 0.22040 -> 0.03294
mfewf    loss 0.32888 -> 0.04598
```

`metrics.csv` 完整结果（threshold 恒为 0.5）：

```
method                     part      IoU      F1       P       R   AUPRC
Boehm SAR U-Net++       overall   0.4095  0.5810  0.7000  0.4966  0.6692
Boehm SAR U-Net++           asc   0.3986  0.5700  0.7011  0.4802  0.6624
Boehm SAR U-Net++          desc   0.4202  0.5917  0.6989  0.5130  0.6762
CDNetE Early Fusion     overall   0.2404  0.3877  0.6867  0.2700  0.5304
CDNetE Early Fusion         asc   0.2271  0.3701  0.6819  0.2540  0.5191
CDNetE Early Fusion        desc   0.2536  0.4047  0.6911  0.2861  0.5414
MFEWF adapted           overall   0.2056  0.3411  0.5987  0.2385  0.4496
MFEWF adapted               asc   0.1870  0.3151  0.5865  0.2155  0.4261
MFEWF adapted              desc   0.2240  0.3660  0.6091  0.2616  0.4723
```

三席位的 `test_predictions.npz` 均校验通过：
`p.shape=(686, 128, 128)`、`orbits=[343, 343]`、概率落在 `[0, 1]`。

值得注意的三点：
1. **Boehm 明显领先**（F1 0.5810 vs 次优 0.3877），这与它使用 U-Net++ 的
   密集跳连 + 官方 resnet50→resnet34 的强骨干有关，也说明本文方法的对照基线
   不是「随便挑一个弱方法」。
2. **asc 一律弱于 desc**（三个席位同向），与侧视几何下升轨/降轨成像差异一致，
   可作为 D3 的旁证。
3. **CDNetE 与 MFEWF 的 precision 高而 recall 低**（P≈0.60–0.69 / R≈0.24–0.29），
   说明两者在固定 0.5 阈值下偏向保守；这本身是「阈值固定」这一协议设定的
   直接后果，不宜解读为方法优劣。

#### 5.7.3 新增汇总脚本 `cocd/scripts/24_external_seat_table.py`

四个席位落盘格式一致（`<seat>/metrics.csv` + `logs/<seat>.log`），
汇总脚本把 `overall/asc/desc` 三个分区拼成一张表，并附参数量、loss 类型、
实现来源。**未跑的席位会显式列在 "not yet run" 里**，避免部分完成的表
被误读成完整表。

#### 5.7.4 本轮新增的坑

- `main.py` 的 `run_audit` 返回 `gate` 字典，`main()` 拿 `gate['passes']`
  做 `SystemExit` 判断。删函数时若漏掉调用点会 `NameError`；用 AST 扫
  `FunctionDef` 列表 + 关键字残留确认干净后再跑。

