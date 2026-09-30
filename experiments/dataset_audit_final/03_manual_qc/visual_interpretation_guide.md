# Haiti 数据集人工质检：看图操作卡

本卡用于人工复核候选，不把“看不见变化”自动判为 GT 错误。统计结论见 `../final_dataset_audit.md`。本地没有原始光学影像、原始滑坡矢量和原始轨道几何；因此这套 SAR+DEM 图只能支持“符合/不符合预期、需要复核”，不能独立认证每一个 landslide polygon。

## 顺序

1. 先看 `blind/B*.png`，暂不打开 `blind_ground_truth_key.csv`。看 ASC/DESC 的 pre/post 与带符号变化图：能否先找到沿坡连续的变化？记录位置和哪一轨、哪一极化支持。零散椒盐点不算清晰变化。
2. 打开答案表，定位对应的 `contact_sheets/group*_patch*.png`。SAR 同一轨道、同一极化的 pre/post 使用同一个 2–98% 显示范围；变化图使用同一对称色阶。只在这个前提下比较明暗。带符号变化的红/蓝代表变化方向，不代表滑坡真/假；绝对变化的亮度代表幅度，不代表事件归因。
3. 看第 4 行 GT 红轮廓与 slope/aspect：轮廓是否落在有实际坡面的区域、是否大致顺坡并具有可解释的形态？靠 patch 边缘的截断、多条邻近小斑块、矩形包围盒较满，都只能作为疑点，不能单独否定标注。30 m DEM 比 SAR/GT 粗，不能裁决 1–2 像素尺度的边界。
4. 看第一、二行 geometry。图例：白/灰 valid，橙 layover，蓝 shadow，紫 both。GT 若落在某轨 invalid 区，那一轨 SAR 强度不可靠；再看另一轨是否 valid 并提供局部结构变化。invalid 区不必为黑色或 nodata。若两轨都 invalid，填 `uncertain` 通常比填 `no` 更有依据。
5. 比较 GT 内与附近 **相似坡度/坡向、同样 valid 的背景**，避免被整张图的高变化区域误导。GT 内变化稍强、且与轮廓大致同位，只能记作支持线索；背景同样强或两轨弱，则标记复核，不自动判错。
6. 检查边界/配准：若同一方向挪动 GT 1–3 像素后，多个独立且两轨有效的目标都更贴合结构边缘，记 `possible_registration_shift=yes`；单个 patch 的最佳偏移方向不足以证明全局配准错误。先排除掩码遮挡、陡坡视角差和 speckle。
7. 在 `manual_qc_form.csv` 逐项人工填写。`yes/no` 只针对观察到的证据；模糊、遮挡、传感器间冲突都填 `uncertain` 并在 `notes` 写出原因。勿使用自动 anomaly 排名代替肉眼结论。

## 三张练习图

- `contact_sheets/group1_patch1380.png`：GT 约 4%，两轨 invalid 约 0%/0.7%。适合先练“相同色阶下寻找坡面变化，再看 GT”，但其 GT 内 matched-background |ΔVV| 比值约 ASC 1.23、DESC 1.16，只是弱支持，不能说标注已被证实。
- `contact_sheets/group4_patch0036.png`：GT 约 4.2%，GT 在 ASC/DESC invalid 内的比例约 31%/50%；两轨有效区的 matched |ΔVV| 比值均约 0.86。差异图虽然有大片亮纹，但先检查**GT 所在位置**与 geometry，不能据整张图很亮就判有支持，也不能因 GT 弱而直接判错。
- `contact_sheets/group2_patch0500.png`：GT 约 5.4%，其中约 48% 落在 ASC invalid、约 23% 落在 DESC invalid。练习先辨认 ASC 橙色无效区，再看 DESC 有效区域的 pre/post；同一地貌在两轨中不必同样亮暗。

上述数字来自当前 `02_tables/dataset_patch_audit.csv`，只用于选图和定位。样例不等于人工最终裁决。若能取得与这批数据 **同源、同时间且明确配准** 的原始光学影像与滑坡矢量，应在最后另做逐 polygon 核验；其他公开 Haiti inventory 不能不经来源核对就当作本数据集的原始真值。
