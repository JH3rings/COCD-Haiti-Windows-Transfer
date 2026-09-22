# COCD：反轨指导的目标轨证据重分配——实现与训练指令

请在现有 COCD-Haiti-Windows-Transfer 项目中实现并完成下面的一轮训练验证。不是只交付代码或 smoke test，也不要另行提出多个网络方案。已有训练和数据流程不做全面重审，复用可用入口；只修改本方法需要的模型、初始化、辅助损失和结果输出。

仓库：https://github.com/JH3rings/COCD-Haiti-Windows-Transfer

## 1. 唯一研究问题与改动边界

研究问题：训练期，Teacher 利用目标轨和反轨的灾前/灾后 SAR，学习怎样重新使用目标轨已有的变化与上下文证据；部署时，Student 只看目标轨，学习执行其中可预测的证据选择动作。

本轮只有一个核心结构创新：反轨指导的目标轨证据重分配模块，暂名 EAR。
只有一个辅助蒸馏损失：G10 优先、Teacher 实际增益筛选的“证据分配增量一致性”。

不要重建反轨 SAR，不回归旧的高维 C_T，不同时加入 L_corr、L_delta、logit KD、relational KD 或额外几何预测头。不要加第二套完整 encoder，不增加多尺度 correction，不改为新的 Siamese backbone。不要把方案写成“保证恢复不可观测信息”或“保证超过 DIS2”。

主要验收目标：G10 AUPRC、F1 优于 DIS2，并且整体 AUPRC/IoU 不靠明显退化换取。是否达到目标必须由真实结果回答。

## 2. 以当前 N 系列代码为接口，不混用旧 R 系列的叙述

优先读取并复用：
- cocd/models/landslide_cocd.py
- cocd/windows_main/models_complement.py
- cocd/windows_main/main.py
- cocd/configs/protocol_windows_main.json

局部代码比公开快照更新时，以实际执行文件为准，简要记录差异。

保留 ConvNeXtTinyFPN、fused_p2、predict_from_fused 和当前五通道输入：
[VV_pre, VH_pre, VV_post, VH_post, OrbitID]。

主干在当前代码中是五通道早融合。本轮保持不变，不声称实现了独立 pre/post Siamese 编码。
只在融合后的 P2 插入一次 EAR；当前 C=128，128×128 输入对应 32×32 的 P2。

Teacher 与 Student 都必须使用新模块。旧 Teacher 的 Phi 补偿不能直接冒充新 Teacher 的证据分配目标。
为 EAR 增加一个明确的方法配置；只调整与新动作损失/训练期几何权重有关的入口兼容性，不让旧 N2 的 SmoothL1、calib 或 evaluation-only geometry 规则误覆盖新方法，也不要全局解除旧方法的约束。

部署 Student.forward 只接受目标轨输入。counter、GT、geometry 均不得进入 Student.forward。训练函数可以读取这些信息来构造监督。

## 3. 一个共享形式的证据读取/写回模块

固定超参数：C=128，低维 d=32，K=9，P2 上固定 3×3 邻域，dilation=1。
只用一层局部选择，不堆 Transformer block。

令 F 为当前模型自己的目标轨 P2 特征。
令 U=LN_channel(F)，使用无可学习仿射参数的通道 LayerNorm，仅用于 query/key 和 Student driver；value 使用未归一化的 F。

定义：
q0(i) = Wq U(i)
k(i,k) = Wk U(i+offset[k])
v(i,k) = Wv F(i+offset[k])

Wq、Wk、Wv：无 bias 的 1×1 Conv，C→d。
Wo：无 bias 的 1×1 Conv，d→C。

九个候选位置按固定次序排列。Teacher/Student 次序和边界处理必须相同。
用 unfold 等简单实现；超出特征图的候选通过相同的有效邻居 mask 从 softmax 排除，不让 padding 成为可学习的虚假证据。中心位置始终有效。

对任意指导量 b(i)，定义：
q(i;b) = q0(i) + b(i)
a(i,k;b) = softmax_k(q(i;b)^T k(i,k) / sqrt(d))
a0(i,k) = a(i,k;0)
rho(i,k;b) = a(i,k;b) - a0(i,k)
C(i;b) = Wo [sum_k rho(i,k;b) * v(i,k)]
z(b) = Head(F + C(b))
z_self = Head(F)

Head 复用现有分割头及上采样，不另建预测头。

结构限制：所有 key/value 都只能来自目标轨 F。反轨只能影响 b，不能通过 concat、value 或其他旁路直接注入 F+C。

这样 rho 是“相对于当前 self 选择，哪些目标轨证据被增强/减弱”，不是完整反轨 feature。
必须保持 b=0 ⇒ a=a0 ⇒ rho=0 ⇒ C=0 ⇒ z=z_self。

Wq/Wk/Wv/Wo 正常非零初始化。不能同时将 Wo 和指导量的末层全部置零，否则可能使分支没有启动梯度。

## 4. Teacher：真实反轨指导，目标轨证据执行

Teacher 对 ASC、DESC 使用共享 backbone，得到 Ft、Fc。
复用现有 DCA 的实现和默认采样配置：
H = DCA(Ft,Fc)
bT = A LN_channel(H)

A：无 bias 的 1×1 Conv，C→d，权重零初始化。
Teacher 的 EAR 用 Ft 构造全部 query 基准、key 和 value，用 bT 改变选择：
aT = a(bT)
rhoT = aT-a0T
CT = Wo sum_k rhoT_k vT_k
z_dual = Head(Ft+CT)

DCA 只是复用已有跨轨信息读取器，不再添加第二个 cross-attention 模块。九点 EAR 候选数与 DCA 的采样点数不是同一个参数，不要混用。

反轨零参照在特征/指导层：Fc=0 时 H=0、bT=0、CT=0。不要把输入零图像等同于零反轨特征。

初始化：优先复用同一数据协议下已有单轨 N0 的 backbone/head；没有合格 checkpoint 时采用当前统一初始化，不额外训练一个新 S0。EAR 和 DCA 按上述方式初始化。不得直接加载不兼容的旧 Phi 为 EAR。

Teacher 的 GT 损失：
LT = 0.5 * Lseg(z_self,Y) + 0.5 * Lseg(z_dual,Y)

ASC-target、DESC-target 对称训练并平均；尽量只编码两条轨道各一次。
Lseg 沿用当前普通分割损失，不额外加 Dice/类别权重/教师专属正则。
Teacher 在一个正常阶段联合训练；不引入多轮冻结/解冻。

完成后选择并冻结一个 Teacher checkpoint，eval 模式输出蒸馏目标。
比较该 checkpoint 自己的 self/dual overall 和 G10 指标。新 Teacher 不必超过历史最强 Teacher，但必须保留有效的跨轨任务收益，尤其是 G10。
若真实反轨没有带来 G10/任务收益，或新模块不起作用，报告这个结构假设当前未获支持，不继续盲目蒸馏、不放松限制去注入完整反轨 value。

## 5. Student：继承执行操作，只预测目标轨指导量

Student 复制训练后新 Teacher 的：
backbone/FPN/head；Wq、Wk、Wv、Wo。
Student 不保留 DCA 和 A，不读取 counter。

它新增一个轻量 driver：
bS = P(concat(LN_channel(FS), OrbitID_plane))
P = Conv3×3(C+1,d,padding=1) → GELU → Conv1×1(d,d)
P 最后一层 weight/bias 零初始化。

不另加 orbit embedding MLP；使用已有已知轨道标识即可。

Student 用自己的 FS、复制后可微调的读取/写回参数，计算：
a0S、aS、rhoS=aS-a0S、CS、zS=Head(FS+CS)。

初始 P=0，Student 应等于复制过来的 Teacher self 函数，而不是等于 Teacher dual。
复制后所有 Student 参数正常微调；Teacher 始终冻结。不要额外添加复杂的学习率分组或解冻阶段。

注意：Teacher/Student 分别使用自己的目标轨特征。Teacher 的 Ft、key、value 不能偷偷喂给 Student。复制只是初始化，不是推理时共享 Teacher。

## 6. 唯一辅助 loss：匹配证据选择的变化，而不是绝对注意力

不要只匹配 aS≈aT。
因为 Student 的 self 分配也可变，仅匹配绝对 a 可能允许 Student 改变 a0 来降低 loss，却不执行对应修正。
本轮匹配固定候选位置坐标上的增量动作：
rhoT = aT-a0T
rhoS = aS-a0S

在 P2 的每个位置：
D_action(i) = (1/4) * sum_k abs(rhoS(i,k)-stopgrad(rhoT(i,k)))

这是唯一的蒸馏距离，不同时加 KL/JS 或 feature loss。
由于 a、a0 都是概率分布，D_action ∈ [0,1]；不是对通道数求均值，不要再除以 K。
该距离约束空间候选的选择变化，不要求隐藏特征通道一致，也不保证动作相似就一定带来更好预测，后者由 GT 和实验检验。

采用当前真实 pre→post 样本做这一辅助 KD；合成无变化样本继续接受原有分割监督，不改变数据构成。

在原始分辨率、使用冻结 Teacher 计算：
e0 = BCEWithLogits(z_self_T,Y,reduction='none')
e1 = BCEWithLogits(z_dual_T,Y,reduction='none')
g = clamp((e0-e1)/(e0+epsilon),0,1)

G10 = Mt * (1-Mc)，其中 1 表示失真，必须按当前 target/counter 方向交换 mask。
w_geo = 1+G10。

将 D_action 以 nearest 上采样到标签分辨率，得到 Dup；
V 为“真实 pre/post 样本且标签有效”的 mask。

LEAR = sum[V * w_geo * stopgrad(g) * Dup] / (sum[V * w_geo]+epsilon)

含义：
- Teacher 无实际增益：不提供这项 KD。
- 非 G10 但 Teacher 有增益：照样学习。
- G10 且有相同增益：两倍相对优先级。
- 分母不除以 sum(g)，不把极弱/极少的 gain 重新放大。
- 不再叠加 Student protect 或其他门控。
- 本轮 geometry 仅用于训练监督优先级和分区评价，不进入网络前向。

没有真实样本或有效监督时，辅助项返回与 Student 计算图兼容的零，GT 训练照常。
所有 Teacher 输出、g、geometry 权重 detach；Student 的 rho 和 z 保持梯度。

最终：
LS = Lseg(zS,Y) + lambda_EAR * LEAR

## 7. 辅助监督保持小比例：只做一次静态定标

本轮不使用旧 N2 将两个损失调到同一量级的 calib 规则。
完成 Teacher、初始化 Student 后，用固定的最多 8 个训练真实 pre/post batch，eval 前向测量 Lseg0、LEAR0，不更新参数、不读取监控标签选权重。

固定：
lambda_EAR = min(0.05, 0.05 * mean(Lseg0)/(mean(LEAR0)+epsilon))

即初始化时加权辅助项至多约为 GT loss 的 5%，lambda 本身也不超过 0.05。
若动作/增益几乎为零，不能除小数把 lambda 放大；保留上限并报告监督信号很弱。

确定一次后冻结 lambda；不做 lambda 网格搜索、不动态调权、不 warm-up。
这个规则只控制初始损失比例，不保证后期比例或梯度比例恒定。
正常日志同时给出 Lseg、LEAR、lambda*LEAR 及二者比值。若辅助项后期持续显著主导，报告问题，不通过追加 loss 或自动扩大 lambda 解决。

所有同结构蒸馏消融使用相同 lambda，不各自重新定标。

## 8. 执行训练：一套结构，两个必要对照，不展开四条路线

先完成一个新 Teacher，然后在 seed=42 上执行：

A. EAR-GT：新 Student，同样 Teacher 参数继承，只用 Lseg。
B. EAR-Full：相同结构和初始化，加入上述 G10 优先动作蒸馏。

这两项先回答：真正增加收益的是蒸馏，还是初始化/结构/额外训练。
EAR-GT 不是纯单轨训练 S0，因为它继承过双轨训练后的参数，必须准确命名。

若 B 显示有用趋势，补一个必要消融：
C. EAR-noGeo：保持全部不变，只将 w_geo=1。
保留相同 gain 与 lambda，用于判断几何优先级的作用。不要单独给 C 调参。

这是同一模型的 loss 开关，不是三种新网络。

已有 N0/N1、Vanilla KD、DIS2 结果符合当前比较协议时直接复用或统一重评；旧 N2 只作为历史 raw-complement 对照，不重新搜索其权重。
不要混用 Mac/Windows、不同初始化/数据协议的分数。

DIS2 保留实际核心机制，不能通过删分支、减少预算或限制其合理监督把它做弱。结果表标明各方法是否使用训练期反轨、geometry 监督和 Teacher 参数继承；不要把额外监督资源的全部收益都归因于网络结构。
Teacher 能统一时统一；结构不兼容时保留各自 Teacher 并披露差异。Teacher 不同的结果只支持整体方法体系比较，不把差异全部归因于 KD loss；内部 A/B 是更直接的蒸馏增量证据。

完成首轮且值得继续时，EAR-Full、EAR-GT 与 DIS2 做 seed 42/43/44 的成对比较，复用已经合格的同协议运行。新方法可固定同一个 Teacher 来隔离 Student 随机性，但必须说明结果是以该 Teacher 为条件，而不是整个流程多次独立重复。

训练预算、优化器、有效 batch 沿用当地已经统一的现行设置；不因为旧聊天出现过 2/8 就再改 batch。不要同时修改采样、基础分割 loss 和训练日程。

## 9. 只检查新模块的基本正确性，然后真正执行训练

不开展新一轮全项目审计。只检查：
1. b=0 时修正精确/数值容差内为零，并回到 self。
2. Teacher 的 key/value 全部来自 Ft；Student 改动 counter/geometry/GT 不影响前向，因为根本不接收它们。
3. 初始 Student 等于复制的 Teacher self；至少在一个有效真实 batch 上 driver 能得到非零任务或动作梯度，避免双零初始化锁死。
4. 九个动作坐标、空间增强、target/counter 交换与现有 mask 对齐。

检查通过后完成训练与结果分析，不能以“代码写完”“单 batch 跑通”替代实验完成。
不覆盖已有 checkpoint，用一个清楚的新方法名称保存产物即可，不新建阶段管理平台。

保留一个必要的结果解释边界：不得用真正独立 test 选 checkpoint、lambda 或阈值。公开 v4 的 test_auprc 监控入口不能被默默当成独立测试。
若本地仍使用这个已反复监控的分区，按开发监控集执行本轮探索实验并如实标注，不把它重新命名就宣称获得了独立测试；本轮不自动重划数据。确有未参与决策的独立留出集时，只在模型与设置锁定后统一评价。
所有方法沿用同一个预先固定的 checkpoint 选择指标，不因 EAR 的 G10 更好就给它挑另一种 checkpoint。不要按 G10 单独优化阈值。

## 10. 验证“G10 任务优势”，不只验证 loss 下降

同一 checkpoint 报告：
- overall：IoU、F1、AUPRC、precision、recall；
- G00/G01/G10/G11：至少 F1、AUPRC；
- G10：额外报告 precision、recall、有效像素和正像素数量；
- ASC-target、DESC-target 分开结果；
- 完整部署参数、同输入尺寸推理计算量；Teacher/Student 分列。

目标是 EAR-Full 在 G10 上优于 DIS2，同时不牺牲整体性能。
还要比较 EAR-Full vs EAR-GT、EAR-noGeo。
G10 的“优势”是相对相同基线的改进，不是要求 G10 的原始分数超过其他区域，也不能把不同区域 AP 差值称为单位像素效率。

至少给出两项轻量机制读数，复用预测即可：
1. Teacher 在 G10 从 self 到 dual 获得了什么实际收益，FP/FN 如何变化；Student 在其中改进了多少。
2. 同一个训练后 Student 正常推理与强制 rho=0 的区别。关闭模块明显影响任务表现可支持它被使用，但不是单独的因果机制证明。

动作距离下降只是中间读数，不能单独作为成功标准。
如果 G10 召回升高但误检增加、F1/AP 未提升，只能说改变了 precision/recall 权衡。
如果整体提升但 G10 没有优势，只能说整体有收益，不能声称几何互补优势成立。
如果 G10 改善但整体退化，明确报告 trade-off，不称为全面超过 DIS2。
如果 A≈B，新增动作蒸馏贡献未获支持。
如果 B≈C，几何加权的独立贡献未获支持。
单种子优势称初步结果，多种子报告均值、标准差和配对差值；不能把小数点最后一位的胜出写成稳定优越。

## 11. 最终交付

交付实际改动的文件及运行命令、已完成的训练记录、checkpoint 路径、同口径结果表、参数统计和结论。

结论逐一回答：
- 受限为“目标轨 value＋反轨指导”的 Teacher 是否仍有有效互补收益？
- Student 相比同结构同初始化的 EAR-GT，是否因蒸馏变好？
- G10 是否真实优于 DIS2，整体是否保住？
- 失败发生在 Teacher 没有可用动作、Student 没学好动作，还是学到动作但任务没有受益？证据不足的部分保持未知，不强行归因。

若训练无法完成，给出真实已完成阶段和具体错误，不编造成绩。
若结果不支持，本轮如实收尾，不自行叠加新模块、加第二个 KD loss、反复用监控集搜索到赢为止。

论文主张边界：创新候选是“把跨轨内容补偿限制为单轨可执行的证据重分配，并迁移其增量动作”，不是首次使用注意力、残差或 L1。低维且可执行不等于完全可预测，G10 内没有足够目标轨线索时仍可能失败。


## 接口参考（公开代码，不代表本地运行结果）

- https://raw.githubusercontent.com/JH3rings/COCD-Haiti-Windows-Transfer/main/cocd/windows_main/models_complement.py
- https://raw.githubusercontent.com/JH3rings/COCD-Haiti-Windows-Transfer/main/cocd/windows_main/main.py
- https://raw.githubusercontent.com/JH3rings/COCD-Haiti-Windows-Transfer/main/cocd/models/landslide_cocd.py
- https://raw.githubusercontent.com/JH3rings/COCD-Haiti-Windows-Transfer/main/cocd/configs/protocol_windows_main.json
