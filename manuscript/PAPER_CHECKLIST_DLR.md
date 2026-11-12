# DLR 论文内容清单（main_dlr.tex）

新论文（DLR S1/S2 数据集，光学只在训练时使用）逐项对照旧论文（`main.tex`，Haiti 双轨，5 页），列出每一项需要的图、表、指标和描述，以及数据来源。旧论文保持不变。

所有数字都来自单次训练（seed 42）：用验证集 IoU 选模型，冻结后在测试集上只测一次。

## 编译

```
cd manuscript
tectonic -X compile main_dlr.tex      # 或 pdflatex + bibtex，需要 template/IEEEtran.bst
```

表格行由 `\rows{tables_dlr/*.tex}` 读入，这些文件由脚本生成，**请不要手改**，需要改动时重新运行脚本：

```
cd experiments/dlr_official6_faithful_20261110
python figures/make_tables.py        # 生成 tables_dlr/*.tex 和 figures_dlr/sensitivity.*
python figures/make_figures.py       # 生成 method.pdf（由 figures/method.svg 转换）和 overlap.*
python figures/make_qualitative.py   # 生成 qualitative.*（需要 DLR HDF5 原始数据）
```

DLR 原始数据（Zenodo record 17007637，HDF5 格式）体积较大，不在仓库中，只有 `make_qualitative.py` 和重新训练需要它。所有表格只读 `results*/<arm>/test_metrics.json`，这些文件已经在仓库里。

## 旧论文 → 新论文 对照

| 旧论文中的内容 | 新论文对应内容 | 状态 | 来源 |
|---|---|---|---|
| 标题、摘要、关键词 | 跨事件光学关系蒸馏 | 完成 | `main_dlr.tex` |
| 引言（SAR 应急场景 → 相关工作 → 问题 → 贡献） | 同样的结构；问题是"事件记忆"和"老师背标签"；3 条贡献 | 完成 | 同上 |
| Fig.1 任务场景图 | 没有单独做。光学和 SAR 输入已经在 Fig.2 定性图里展示 | 可选 | — |
| Fig.2 方法结构图（figure*） | Fig.1：(a) 训练与部署，(b) 原型模块 EPM，(c) 跨事件 Wasserstein 损失 | 完成 | `figures/method.svg` → `figures_dlr/method.pdf` |
| 方法公式（问题定义、模块、蒸馏损失、总损失） | 公式 (1)(2) 原型分配与特征修正，(3)(4) L_CEW，(5) 总损失 | 完成 | `final_method.py`、`pr.py` |
| 数据集与设置（数据、划分、训练配方、指标定义） | 官方划分 355/88/120；Ind.44 / Dep.76；6 通道学生、18 通道老师；AdamW 等训练配方；AP、IoU、F1 | 完成 | `common.py` |
| 定性结果图（figure*） | Fig.2：4 个光学无云的测试 patch，列为 S2 前/后、S1 前/后 VV、GT、U-Net、PKDOT、DIS2、Ours、Teacher（TP 绿、FP 红、FN 蓝） | 完成 | `figures/make_qualitative.py` |
| Table I 主对比（含直接分割基线 UNet++、DeepLabv3） | Table I：直接分割（U-Net、UNet++、DeepLabv3）、蒸馏方法（6 个 + Ours）、光学推理参考（2 个） | UNet++ 和 DeepLabv3 在训练中 | `tables_dlr/main_rows.tex` |
| Table II 蒸馏增益（分区域） | Table II：逐事件 F1（DLR 原论文用的就是逐事件 F1），并与原论文报告的依赖事件均值 0.68 比较 | 完成 | `tables_dlr/event_rows.tex` |
| 局部放大的误差图 | 改为 Fig.3 分析图：与老师成对关系的一致度 vs 测试 IoU（r = −0.70） | 完成 | `figures/make_figures.py`、`results_rel/diag_learned.json` |
| Table III 消融（各机制） | Table III：组件、L_CEW 放在哪一层（16/32/64/128 px、三层）、关系目标、关系来源 | 层位置的几组在训练中 | `tables_dlr/ablation_rows.tex` |
| 损失权重敏感性图 | Fig.4：λ ∈ {10, 30, 100}，K ∈ {8, 16, 32} | 在训练中 | `figures_dlr/sensitivity.pdf` |
| Table IV 部署代价 | Table IV：参数量与 MACs（U-Net、UNet++、DeepLabv3、Ours、Teacher） | 完成 | `cost.py` → `results_rel/deployment_cost.json` |
| 结论 | 结论与局限：召回升、精度降；在逐事件平均上不领先；继承了老师在小滑坡上的盲区 | 完成 | — |

## 指标定义（论文中使用的全部指标）

- **AP**：非插值平均精度，在全部有效 SAR 像素上统一计算（pooled）。
- **IoU、F1、Precision、Recall**：阈值 0.5，对 120 个测试 patch 的有效像素统一计数后计算。
- **IoU Ind. / IoU Dep.**：分别在 44 个独立事件 patch 和 76 个依赖事件 patch 上统一计算。
- **逐事件 F1，以及 Ind./Dep. 下 4 个事件的平均**：这是 DLR 原论文的报告方式。
- **参数量、MACs**：一个 128×128 patch 上卷积、转置卷积和线性层的乘加次数，加上原型模块中的 einsum。

## 方法中的数字及其含义

| 量 | 值 | 含义 / 依据 |
|---|---|---|
| K | 16 | 原型个数，训练前设定；Fig.4 给出 8 和 32 的对比 |
| τ | 0.1 | 原型分配的 softmax 温度 |
| 原型所在层 | 解码器第 2–3–4 层（32/64/128 px） | 三层共用同一组原型 |
| L_CEW 所在层 | 第 2 层（32 px） | Table III 比较了 16/32/64/128 px 和三层同时 |
| 区域 | 每个 patch 分成 4×4 个区域，每个区域 320 m × 320 m | 只保留有效像素覆盖率 ≥ 75% 的区域 |
| λ | 30 | 训练前设定，初始时 L_CEW 的梯度约为分割损失梯度的 20 倍；Fig.4 给出 10 和 100 的对比 |

## 结果文件（arm 名 → 论文中的名称）

- `results/`：B0（U-Net），KD、AT、M2U、ACAM、PKDOT、DIS2，T_fusion（老师），T_optical（光学 U-Net）。
- `results_direct/`：UNETPP、DLV3。
- `results_rel/`：
  - F_SSP（Ours），F_SSP0（只有结构），UNET_W（只有 L_CEW），EPRM_W（只在 32 px 放原型），F_MS（分层原型，两层都加损失）；
  - F_SSP_S0/S2/S3/SALL（L_CEW 放在不同层），F_SSP_L10/L100（λ），F_SSP_K8/K32（K）；
  - EPRM_STR、EPRM_CC、EPRM_CEraw、EPRM_RD（关系目标的几种做法），PKDOT_LBL、OursP1（关系来源）。
- 每个 arm 目录下都有：`config.json`、`history.json`、`selection.json`（选中的 epoch、验证 IoU、权重的 SHA）、`test_metrics.json`（总体、逐事件、逐子集的结果）。

## 需要作者决定或核对的事项

1. 逐事件平均 F1 上，Ours（59.1）低于 DIS2（62.8），原因是 IE0001 的小型泥炭滑坡。论文中已经如实写出。
2. EPRM_CEraw（PKDOT 风格，取 30 个最不相似的配对）单次得到 56.68，但同设置重跑只有 56.03；EPRM_RD（32 个分位数）得到 56.55。两者都在消融表中如实列出。选择 Wasserstein 的理由是它有闭式解，而且不需要设定配对数或分位数个数。
3. 所有结果都是单次运行。
