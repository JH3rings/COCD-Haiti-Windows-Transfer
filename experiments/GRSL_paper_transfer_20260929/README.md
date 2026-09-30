# GRSL SAR 滑坡论文迁移包（2026-09-29）

这是一个可独立复制到跑代码项目中的论文文件夹。先把整个文件夹放进目标项目，再把 `NEW_CHAT_PROMPT.md` 的全文发给新 AI 对话。所有论文源码、当前使用的图片和已编译预览都在文件夹内；`manuscript/main.tex` 不引用本机绝对路径或文件夹外的图片。

## 文件位置

- `manuscript/main.tex`：**唯一的正式 LaTeX 入口**，正文在 `manuscript/sections/`，引用在 `manuscript/references.bib`，实际入稿图片在 `manuscript/figures/`。
- `preview/GRSL_revised_20260929.pdf`：当前 5 页预览。把它作为视觉参照，后续以重新编译的 PDF 为准。
- `editable/`：原版详细网络图 PPT 和 TGD 标注草稿。不要把详细结构改成简单概念框图。
- `evidence/20260929_package/`：9 月 29 日主结果、各区域指标、图、定性候选、诊断和来源快照。
- `evidence/earlier_package/`：较早的搜索、损失、DIS2 和定性结果；**是历史参考，不等同于当前 S2-2Stage 主实验**。
- `NEW_CHAT_PROMPT.md`：交给新 AI 的完整工作指令。

## 当前论文的实验逻辑

主 Student 为 **S2-2Stage**，对应几何无掩膜训练的 **CG0 Teacher**。训练使用全部标注像素；部署时 Student 只看目标轨道。TargetValid、BothValid、Overall 和 G10 是对同一次预测结果施加不同评价掩膜，不是不同的推理模型。当前主表以 TargetValid 为主，旁列 Overall；G10 为目标畸变、互补轨道有效的诊断区域。旧 DIS2 的 Overall AUPRC 87.33%、IoU 61.77% 已入表，但缺少可核查的 TargetValid 等区域结果。

论文已包含原详细网络结构图、总览定性图、搜索对齐图、搜索消融表、损失组成消融表和旧版损失权重曲线。旧版搜索、损失与权重结果均已在稿中标明来源。当前主模型的匹配消融与清晰 G10 局部图仍需从目标代码项目补充。

## 编译

在 `manuscript/` 中编译 `main.tex`。本机可用 Tectonic；目标项目也可用 `latexmk -pdf main.tex`。论文依赖 `IEEEtran`、`amsmath`、`graphicx`、`booktabs`、`multirow`、`url`、`xcolor`、`placeins`，并随包携带 `template/IEEEtran.bst`。参考文件和所有图片均为相对路径。

注意：迁移包以论文与来源快照为中心，不包含完整训练影像、所有原始预测或模型权重。新 AI 需要在目标跑代码项目里查找这些材料，并逐一核对后更新论文。

## 证据入口

| 稿件内容 | 先看这些相对路径 | 状态 |
|---|---|---|
| 主表与各区域指标 | `evidence/20260929_package/01_results/paper_main_comparison_all_regions.csv` | 当前主模型与对照的已记录结果 |
| Teacher 引导对照 | `evidence/earlier_package/quantitative/csv/table_teacher_ablation.csv`；`evidence/20260929_package/07_provenance/current_grsl_package/04_tables/table2_ablation.csv` | 核对 SG0/CG0，勿与 CG10 混写 |
| 搜索对齐图 | `evidence/20260929_package/07_provenance/current_grsl_package/03_search_behavior/search_statistics.csv` | 当前搜索行为统计 |
| 旧搜索消融 | `evidence/earlier_package/quantitative/csv/table_search_mechanism.csv` | 历史 CG10 结果 |
| 旧损失组成与权重曲线 | `evidence/earlier_package/quantitative/csv/table_loss_composition.csv`、`table_loss_strength.csv` | 历史 Student 结果 |
| 旧 DIS2 | `evidence/earlier_package/qualitative/csv/table_historical_external_comparison.csv` | 目前仅可核实历史汇总值 |
| 定性图候选 | `evidence/20260929_package/03_qualitative/` | 候选拼图与选择记录，部分畸变区预测被灰化 |

这些来源快照可能含旧机器上的路径；迁移后应以新项目内真实文件位置重新核对。
