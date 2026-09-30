# Haiti 数据集审计唯一入口

**先看 [00_看图入口.md](00_看图入口.md)**：五张图讲清滑坡真值、两轨畸变、掩膜评价域和 S0/S2 的实际差别。以下为详细审计索引。

当前推荐阅读 [final_dataset_audit.md](final_dataset_audit.md)，数值使用 `01_statistics/` 和 `02_tables/dataset_patch_audit.csv`。CSV 的空值表示不可计算/未观测，不能按0解释。

- 完整性：file_integrity.csv（11991 GeoTIFF逐文件）与99_cleanup_manifest/data_hash_verification.json（13704文件SHA256）。
- Geometry/terrain/GT：geometry_summary、joint_geometry_summary、slope_aspect_statistics、gt_geometry_statistics；论文候选图在01_statistics/figures/。
- 变化：sar_change_statistics、matched_background_statistics、cross_orbit_statistics、cross_orbit_support_statistics、g10_g01_statistics、registration_shift_statistics；每patch明细与方法限制见报告。
- 异常：02_tables/anomaly_summary.csv、04_anomaly_cases/type_A…E/，每类20个候选，不是自动判错。
- 人工材料：03_manual_qc/有46张5×4高分辨率联系表；Group1/4盲图随机编号、无GT与异常标签。先盲检，答案在blind_ground_truth_key.csv。manual_qc_form.csv字段留空，只由人工填写；跨组可重复同patch。display_ranges.csv记录每图共享显示范围。

人工看图顺序、误判陷阱与3张练习图见 [03_manual_qc/visual_interpretation_guide.md](03_manual_qc/visual_interpretation_guide.md)。

历史保留内容集中在05_existing_results_reused/：9月24物理审计、9月27扩展审计的不同案例图、代码与来源记录；acquisition_metadata/temporal_pairs/file_inventory、原Mac报告和数据交付说明提供本轮不能恢复的来源历史。旧报告/数字仅为历史，不得与本轮表混用，尤其旧光学存在声明、split和较强配准裁决。旧DEM完整保留。来源与操作逐文件见99_cleanup_manifest/merged_files.csv。

旧9月24 patch_slope.csv 与 slope_stratified.json 已由本轮统计覆盖；旧脚本纬度随行递增，与真实向南行方向不符，且旧采样方法不同。它们已删除（见deleted_files.csv），追溯副本压缩封存在superseded_terrain_outputs.zip，不是可引用的当前科学结果。旧脚本源码保存。

2026-09-28 追加清理：两个旧数据审计运行目录已移除；其中失败运行日志和 Python 缓存无可参考的科学内容。原位重复的 2026-09-12 Mac 报告、目录说明和数据来源分析报告已删除，其核验一致的副本在 `05_existing_results_reused/`。早期 4 份 `cocd/metadata` 采集与文件盘点记录有溯源价值，已移入同一历史区。累计移动 45 个文件、删除 9 个文件；操作清单见 `99_cleanup_manifest/`。模型、训练和协议审计仍按其原有项目归属保存，不属于数据集质量审计结果。

99_cleanup_manifest/all_existing_dataset_analysis.csv为全项目文件名/文本候选inventory（排除.git/node_modules/缓存与本轮目录）；keep为受保护数据/模型/来源，uncertain保留原位并注明原因。删除与移动分别有deletion_manifest.csv和move_manifest.csv，执行检查见cleanup_execution.json。源码、split、checkpoint、训练结果均不删除。

从项目根目录重跑：`python experiments/dataset_audit_final/rerun_quantitative.py`；依次运行 `python experiments/dataset_audit_final/build_final_audit.py supplement`、`python experiments/dataset_audit_final/build_terrain_detail.py`、`python experiments/dataset_audit_final/build_final_audit.py tables` 和 `python experiments/dataset_audit_final/build_final_audit.py qc`（会重生成材料）。DEM路径由`rerun_quantitative.py`中的final来源确定。旧来源审计为历史，原脚本内Mac/旧绝对路径不可盲目运行。
