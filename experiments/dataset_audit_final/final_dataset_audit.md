# Haiti COCD 最终数据质量审计

日期：2026-09-27；本轮全量重新读取 1713 个空间 patch，seed=20260927。未修改数据、划分、checkpoint 或训练结果。

## 1. 数据完整性与物理统计

- 数据包 SHA256：13704/13704 匹配；逐 GeoTIFF 完整性表 11991 行。strict_grid 异常 0，非有限像素 0，异常常数 SAR 文件 0。类别常数掩码可能是合法全背景或全有效，不能等同坏数据。
- 本地实际数据：4 个 SAR 时相/轨道目录，每个 1713 个三波段文件，band1 VH、band2 VV、band3 geometry；3 个 processed mask/GT 每位置。原始 optical、polygon、原始 Annotation 目录未在数据根找到。旧元数据 optical 路径是 macOS 来源引用，不能当成本地存在。
- 两块 DEM 已有本地文件：05_existing_results_reused/experiments/windows_main/physics_audit_20260924/dem/N18W075.tif 与 N18W074.tif，EPSG:4326，约30m。坡度按原生 DEM 米制梯度计算、双线性采样，坡向最近邻；无法提供10m独立地形真值。DEM GLO-30 来源依赖旧审计记录，文件 tags 不识别产品版本。
- geometry code 0 valid、1 layover、2 shadow、3 both：证据 data/haiti/processed/sample_*/metadata.json 的 geometry_class_coding；SAR tags 未独立说明代码语义。GT 以原标签>0，不把 GT 原始1/2/3误当geometry。
- ASC valid/layover/shadow/both = 86.247942%/13.681143%/0.054351%/0.016565%；invalid=13.752059%。
- DESC valid/layover/shadow/both = 94.023564%/5.327012%/0.638126%/0.011298%；invalid=5.976436%。
- BothValid=81.048791%；ASC invalid/DESC valid=12.974774%；ASC valid/DESC invalid=5.199151%；BothInvalid=0.777284%。这些是重叠窗口累计像素比例，不是去重土地面积。
- GT slope<3°=0.003731%；GT slope<10°=0.287464%；P(GT|坡度bin)=0.239883%, 0.922402%, 2.655679%, 9.196089%, 25.358782%。没有“大量平地GT”的总体证据，但局部异常仍需看图。
- 两轨 invalid 随坡度总体单调上升。ASC layover 在 W/SW 高，DESC layover 在 E/SE 高；DESC shadow 在 W/NW 高、ASC shadow 更偏 E。支持方向性与互补；缺真实入射角/方位，不能验证每个像素的几何物理。
- GT与geometry的整体关联在陡坡分层后明显减弱：>40° P(GT|invalid)/P(GT|valid) ASC=1.048618、DESC=1.004430；低坡层仍有关联。粗分层不能排除坡向、高程、空间聚集等残余混杂，不能据此宣布异常像素耦合或标注独立性。

BothValid 地形匹配：patch内相同5坡度×9坡向（含平地）×8高程层，几何有效且按同轨比较；以GT像素数加权层均值。没有背景的层不匹配，覆盖率=99.602663%。均值/中位数是patch配对汇总；CI为patch重采样、检验为配对sign-flip/Wilcoxon，均不是独立地块推断。

|轨道|指标|GT−matched BG均值|95% CI|Cohen dz|
|---|---|---:|---|---:|
|asc|dvv_abs|0.095426|[0.07990859883327187, 0.11133622972007856]|0.287792|
|asc|dvh_abs|0.105282|[0.08765067827195604, 0.12270217041449262]|0.288165|
|asc|vv_texture_change|0.026806|[0.021796609878184277, 0.032003246133776746]|0.247626|
|asc|vh_texture_change|0.055538|[0.04814747885770181, 0.06325176113310745]|0.353616|
|desc|dvv_abs|0.119838|[0.10549802208836595, 0.13375273938755972]|0.401895|
|desc|dvh_abs|0.140083|[0.12320076194405313, 0.15639949574646628]|0.409322|
|desc|vv_texture_change|0.037228|[0.03193684055205887, 0.04264814913659269]|0.330313|
|desc|vh_texture_change|0.056472|[0.050077749180299946, 0.06283947888125355]|0.427362|

- 平均事件变化优势存在，但幅度小；局部std变化与texture为同一操作性定义，contrast为5×5局部范围变化。局部邻域可能跨GT/geometry边界；不能把弱散射证据等同于GT错误。
- 有效匹配的 1671 个patch，至少一轨joint change超过matched背景=90.724117%，两轨均超过=52.064632%。GT两轨强度变化rank相关弱；像素级GT与matched背景rank表见 cross_orbit_statistics.csv。matched_background像素相关来自同地形层合并候选背景，非逐GT权重抽样，属于辅助描述。
- G10/G01 的有效轨更强变化概率按patch内像素比例估计，bootstrap CI见 g10_g01_statistics.csv；暗值(<−25dB)/亮值(>0dB)仅描述代理，不是传感器饱和阈值，也不代表“更可解释”。

双轨最佳dx方向相反，且改善patch比例仅约52%/57%，本轮未发现所有样本共同方向系统性改善的证据。单patch与局部边界偏移仍需人工核查。

双轨GT平移 ±1/±2/±3 pixel：49个二维组合、无wrap，固定BothValid，用同一裁边支持域计算平移增益；保留每patch结果。多方向搜索未经独立验证，CI不校正49次搜索与空间重叠。
- ASC 最大平均增益方向 dy=-1, dx=2（图像坐标，dy向下、dx向右），gain=0.017966 dB；CI [0.008770, 0.027200]；patch改善比例=52.223558%。需要人工核查，不自动纠正GT。
- DESC 最大平均增益方向 dy=-1, dx=-2（图像坐标，dy向下、dx向右），gain=0.020091 dB；CI [0.010964, 0.029569]；patch改善比例=56.850962%。需要人工核查，不自动纠正GT。

## 2. 自动异常审计

- Type A–E 各top20，统一标记 needs_manual_review，不是错误率。详见 02_tables/anomaly_summary.csv 和 04_anomaly_cases/type_*/ranking.csv。规则在每行记录，分数不得跨类型直接比较。
- A：GT>=2%、BothValid GT>=100，地形匹配后两轨max VV变化ratio低；B：GT与任一invalid重叠高；C：低坡GT占比高；D：逐patch坡度×坡向层中的invalid比例，相对leave-one-patch-out总体规律偏差最大的代理（相邻重叠地面未剔除，缺真实轨道参数，不能裁决物理错误）；E：同支持域平移增益高。
- 22、23、25、29、36、50、56、72、73、95、96、97、107、108、109 等为复核候选。候选top-K比例是人为选择数量，不能当作发现坏样本比例。每类20/1713=1.1687%，各类可重叠；eligible分母见anomaly_selection_summary.csv。
- GT边缘与rectangle fill统计仅用于提示；标签在patch边缘被正常裁切可能导致border比例高，矩形填充度也不是错误判据。

## 3. 当前能够得出的数据质量判断

|对象|评级|证据与限制|
|---|---|---|
|Landslide GT|PASS WITH LIMITATIONS|完整、标签合法、总体位于坡面且匹配背景后变化略强；缺本地原始矢量/optical与独立事件日期，边界和形态待人工|
|geometry mask|PASS WITH LIMITATIONS|元数据代码、坡度趋势与坡向互补支持；静态pre/post复用、稀少shadow、无原轨道参数，不能完整重推|
|ASC/DESC registration|PASS WITH LIMITATIONS|逐文件严格共网格；共网格不是内容准确配准证据，位移敏感性与散射视角混杂，不能排除边界偏移|
|temporal consistency|PASS WITH LIMITATIONS|pre/post文件存在，geometry相同；日期只能来自文件名和元数据，无时刻与逐polygon事件来源。ASC post标为2021-08-17，与热带风暴Grace影响海地为同一天；无采集时刻，不能判断单个变化发生在风暴前后|
|single-orbit research suitability|PASS WITH LIMITATIONS|有效像素可做单轨事件研究，但变化效应小、样本选择与窗口重叠限制外推；本次不以训练性能证明数据正确|
|geometry-invalid evaluation suitability|PASS WITH LIMITATIONS|两轨互补支持分区描述评价；invalid强度不可作为可靠地面观测，GT覆盖/陡坡混杂需报告，shadow很少|

本轮未发现文件损坏、缺失、非法label、SAR异常常数图等严重结构异常；这不是“所有GT与配准都已通过”的结论。

## 4. 尚待人工判断

必须先盲看03_manual_qc/blind/，再参考答案表及带contour联系表，由你与老师填写 manual_qc_form.csv：滑坡形态与真实坡面、边界自然性、pre/post可见扰动、layover/shadow视觉对应、两轨互补与可疑偏移、最弱变化是否是标注问题。AI没有填写任何人工字段。

## 方法与复核入口

- 主脚本 rerun_quantitative.py、build_final_audit.py；当前结果仅01_statistics与02_tables，旧数据审计05_existing_results_reused仅历史证据。
- 单次事件、patch重叠、DEM分辨率较粗、两轨日期相差2天、几何静态、地形粗匹配、纹理邻域边界混合与多重筛选，均限制因果和统计解释。p值只做探索。
- 事件时序补充：本地文件名给出 DESC post=2021-08-15、ASC post=2021-08-17。[USGS 对地震与热带风暴 Grace 的研究摘要](https://www.usgs.gov/publications/learnings-rapid-response-efforts-remotely-detect-landslides-triggered-august-2021)指出 Grace 于8月17日影响海地并引发更多滑坡。这说明两轨 post 可能不处于相同地表事件状态；由于本地没有采集时刻或逐polygon事件日期，不能把轨道不一致直接归因于GT错误，也不能声称某个patch确受风暴影响。
- 旧 2026-09-12 报告中的本地optical存在与“GT无地理参考”描述属于原Mac包；本地processed GT严格共网格。本轮不重复其“配准YES”的强裁决。旧split与模型结果保留，本任务未重分数据。
