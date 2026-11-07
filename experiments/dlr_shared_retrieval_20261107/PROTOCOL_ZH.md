# DLR 共享SAR检索实验（2026-11-07）

本轮依据用户附件f6f77510。历史论文、M0–M4模型及负结果保留，不覆盖；测试集不加载。

## 原COCD对应关系
- 复用原SearchContext、AnchoredDeformablePolicy、DeformableSearch和稀疏双线性PolicyJS。
- 每尺度固定中心+3个学习点，共三尺度，offset限制原样保留。
- 原Teacher/Student独立Memory改为一套训练中的SAR Encoder、128通道Memory投影、SAR skip及Decoder；双策略输出分别检索同一Value。
- 部署仅导出core：SAR八通道→SAR policy→SAR retrieval→SAR Decoder。没有光学分支。
- 光学交互的Key/Value仅属于查询条件内部，绝不进入SAR检索Value或Decoder。

## 当前执行版本
V0：原OpticalChange共享时相编码+直接条件相加。
V2：相同OpticalChange，SAR query为锚点，3×3局部归一化相关匹配，残差条件更新，然后原deformable policy。
V1：多尺度Siamese变化适配，设计候选，NOT_RUN。
V3：多尺度对应与保留尺度身份的精细搜索，设计候选，NOT_RUN。
V4：任务相关Evidence关系，设计候选，NOT_RUN。不能将当前模型类的版本占位当成V3/V4已完整实现。
后续三个版本须在V2优势与诊断检查后才定稿实现，不能提前宣称有效。

## 公平对照与门槛
V0/V2各先100轮sar_dual与optical no_kd。sar_dual辅助编码器结构、参数量完全相同，输入仅由真实SAR构造的通道表示；这是容量控制，不是伪造光学。
同种子42、同core初值、相同数据顺序/增强、相同batch16和训练预算。辅助分支结构不同的V0/V2不强制新模块逐参数相同；每个版本内两臂完全匹配初值。
双分割监督为0.5*(Lseg_student+Lseg_teacher)，Lseg=BCE+foreground per-image Dice，1:1。
100轮、val5、AdamW1e-4、wd.01排除bias/norm、clip1，50恒定50cosine至1%。验证累计IoU选Student最佳joint snapshot。Teacher指标来自同一snapshot，不能混用Teacher另选最佳权重。
光学noKD Teacher须比同容量SARdual Teacher高≥0.2个百分点，且比自身Student高≥0.2个百分点，才启动该版本KD臂。此门槛为预先固定的筛选规则，不是显著性检验。
KD臂从同一干净初值重训100轮，每次val若Teacher比Student高≥0.2个百分点，才在后5轮启用KD，权重各.05，不提高权重强迫模仿。
这轮没有原来五版本的全测试指标比较；只报告训练/验证，不访问120或44测试影像。

## 梯度
两种seg均训练共享Encoder、投影、Decoder及各自控制器。KD teacher rule与target evidence停止梯度；学生KD控制器使用detached Memory，保持COCD原设计，不允许修改Value来取巧降低匹配损失。KD对Encoder直接为0，seg对Encoder非0；不能宣称新增直接Encoder KD。
共享Encoder/Decoder会通过两种seg受到光学指导，noKD对照正是用于区分这类共同训练收益和额外KD净收益。

## 数据
复用原355/88官方split、PRE1/2/3+POST1 VV/VH、train-only normalization、valid mask、阈值.5；不新增OrbitID通道，遵循现有八通道协议。同步rot90/flip覆盖SAR、光学、标签、valid。
沿用已审计空间配准及观测字段；本轮不重新证明原始首景时序。no_change将POST optical替换PRE仅用于诊断；mismatch为同batch确定性位置错配，仅诊断、不作实际观测。

## 状态
代码与真实TRAIN2前反向检查见results/checks.json。正式运行进度results/*/progress.json。最终validation.json含匹配/错配/无变化/置零证据结果。更细的小滑坡、弱变化、概率与尺度诊断在完成checkpoint后实施，尚未生成不得视为完成。
