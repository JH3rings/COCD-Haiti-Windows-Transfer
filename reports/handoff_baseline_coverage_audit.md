# 交接包里的基线覆盖核查

日期：2026-09-19（Windows 侧）
问题：用户问"我交接给你的内容里，有这些方法吗？以及相关的权重结果？"
被问及的方法：Boehm SAR U-Net++ / CDNetE Early-Fusion / MFEWF / DIS2 / Vanilla KD

---

## 直接回答：分三档

| 方法 | 交接文档里有名字吗 | 有结果数字吗 | 有权重吗 | 判定 |
|---|---|---|---|---|
| **Boehm SAR U-Net++** | **没有**（仅在 `reports/` 的叙述里） | 有（只在报告正文，无产物文件） | **没有** | 名字未进交接文档 |
| **CDNetE Early-Fusion** | **没有**（仅在 `reports/` 的叙述里） | 有（同上） | **没有** | 名字未进交接文档 |
| **MFEWF** | **有**（HANDOFF §5.3 清单） | **有，且有产物文件** | **没有** | 唯一有落盘产物的外部基线 |
| **DIS2** | **有**（两处：port + third_party 快照） | **有**（双协议） | **有**（port 权重已发；原版无） | 覆盖最完整 |
| **Vanilla KD** | **有**（HANDOFF §4 表 + 文件地图） | **有**（双协议） | **有**（v2 已训出） | 覆盖完整 |

---

## 1. 交接文档的原文措辞

**`HANDOFF_FOR_WINDOWS_AI.md` 第 147–150 行**（唯一一处把外部基线并列点名的地方）：

> 8. **External baselines are not part of this protocol.** Boehm / CDNetE /
>    FC-Siam / MFEWF keep their own architectures and are not forced onto v2. Their
>    result tables are in `results/grsl_external_baselines/`; their weights were not
>    shipped.

**`TRANSFER_AUDIT.md` 第 49 行**：

> | external baselines | `experiments/grsl_external_baselines/` | Boehm / CDNetE / MFEWF trained; FC-Siam not |

**`TRANSFER_AUDIT.md` 第 98 行**：

> | `baselines/` | 21 MB | external baseline implementations (`unetpp`, `segformer`, `deeplabv3plus`, `upsnet2022`, `wu2021`). Not part of the COCD line; several need separately downloaded encoder weights. |

**`TRANSFER_AUDIT.md` 第 154–156 行**：

> 8. **`baselines/` is absent.** If an external baseline has to be retrained, take it
>    from its upstream repository; the result tables of the three that were run are
>    in `results/grsl_external_baselines/`.

---

## 2. 三处必须点明的落差

### 落差一：承诺"三张结果表"，实际只有一张，且只有 MFEWF

HANDOFF 第 149 行写 "their result tables **are** in `results/grsl_external_baselines/`"，
`TRANSFER_AUDIT` 第 49 行写 "Boehm / CDNetE / MFEWF trained"，第 155 行写 "the result
tables of **the three** that were run"。

实际盘查该目录（递归）只有两个文件：

| 文件 | 大小 | 内容 |
|---|---:|---|
| `metrics.csv` | 792 B | **3 行，全部是 `MFEWF adapted`**（overall / asc / desc） |
| `progress.json` | 101 B | `epoch 50/50, loss 0.0896, best_validation_auprc 0.65247` |

**Boehm 与 CDNetE 在该目录下没有任何文件。** 它们的数字只出现在 `reports/` 的
叙述性表格里，没有可追溯的产物。也就是说 "result tables ... are shipped" 这句话，
对 Boehm / CDNetE 而言是不成立的。

### 落差二：`baselines/` 代码目录被留下了，但 `unetpp` 的实现里没有 Boehm

`TRANSFER_AUDIT` 第 98 行说被丢下的 `baselines/` 里有 `unetpp`、`segformer`、
`deeplabv3plus`、`upsnet2022`、`wu2021` 五个实现。**这五个名字里没有一个是
"Boehm"**，也没有 CDNetE、MFEWF。Boehm 的官方代码是 `iprapas/landslide-sar-unet`
（另一条线记录的），CDNetE 与 MFEWF 的官方代码从未被定位。

所以即便想复现，交接包既没给权重、也没给 Boehm/CDNetE/MFEWF 的实现代码。

### 落差三：唯一随包发的第三方代码是 DIS2 的**参考快照**，且不可直接运行

`cocd/third_party/landslide_baselines/dis2/` 实际有 9 个文件（`models_bank/module/` 下 4 个）。
`TRANSFER_AUDIT` 第 148–150 行明确：

> 6. **`third_party/landslide_baselines/dis2/` is reference code.** It is kept so the
>    port in `losses/distill.py` can be diffed against the original. It is not
>    imported and it brings its own dataset/training utilities that do not run here.

即：DIS2 发的是"用来对拍移植是否正确"的参照物，**不是可运行的官方训练代码**。
真正在跑的 DIS2 是我们自己移植进 `cocd/losses/distill.py::dis2_multilevel_kd` 的版本。

---

## 3. 实际权重盘点（全项目 27 个 .pt）

| 归属 | 文件 | 外部基线？ |
|---|---|---|
| 随包发（7 个，`CHECKPOINTS.md` 台账） | `checkpoints/teacher_v2.pt`；`legacy_checkpoints/v1_protocol/` 下 `teacher / R1 / R2 / R3 / DIS2_port / S0` | **无一个是外部基线** |
| Windows 侧新训（20 个，本地产物） | `experiments/windows_main/` 的 `NewTeacher / N0 / N1 / N2 / N2noorbit / N2_lam0p1`（各 `_latest` + 正式）；`experiments/ours_v2/` 的 `teacher_wm / VKD_wm / DIS2_wm / R3_wm` | **无一个是外部基线** |

**结论：Boehm / CDNetE / MFEWF / 原版 DIS2 的权重，一个都不在交接包里。**
`CHECKPOINTS.md` §3 把这一条写成了明确记录：

> | `experiments/{phase1_b1_b2,grsl_external_baselines,...}/` | 45 | 2.4 GB | separate
> experiment lines (...) Their result tables and logs **are** shipped under `results/`;
> only the weights are left behind. |

---

## 4. 这份核查的净结论

1. **用户点名的五个方法里，交接文档明确点了名的只有三个**：MFEWF、DIS2、Vanilla KD。
   **Boehm 与 CDNetE 不在交接文档里**，它们只存在于 `reports/` 的叙述文本中。
2. **只有 MFEWF 有落盘的结果产物**（3 行 csv + 1 个 progress.json）。
   Boehm 与 CDNetE 的数字**没有产物可追溯**——这与交接文档的承诺不符。
3. **外部基线权重全部未发**（文档已声明）。因此 Boehm / CDNetE / MFEWF
   在当前 Windows 包内**不可能被重新评测**，只能引用旧数字。
4. **DIS2 是唯一发了代码的**，但发的是参考快照；真正在用的是自家移植版，
   而移植版权重（`DIS2_port_seed42.pt` v1 / `DIS2_wm_seed42.pt` v2）都已训出。
5. **Vanilla KD 的 v2 权重是 Windows 侧新训的产物**（`VKD_wm_seed42.pt`），
   交接包内原本是 "configured, **not trained**"（`TRANSFER_AUDIT` 第 40 行）。

### 对论文表述的直接后果

任何"我们与 Boehm SAR U-Net++ / CDNetE 比较"的说法，在当前包内
**无法在本机复算**。若是要写进论文，必须：
- 要么把这两篇的权重补进包并重跑，
- 要么在文中明确标注这些数字**转录自旧报告、未在本协议下复现**（且旧报告用的
  是与 v2 不可比的 V1 协议）。
