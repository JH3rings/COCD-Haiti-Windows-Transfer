# COCD / DLR F_SSP 网络结构图解释（源码核验版）

本文件解释图中 A/B/C 的真实含义，依据 main.tex 与 experiments/dlr_official6_faithful_20261110/ 中的 final_method.py、common.py、pr.py、methods.py、run.py。图稿不能代替科学定义；若图与源码冲突，以源码和论文方法为准。

## 一句话
训练时用拥有光学 + SAR + 地形信息的冻结 Teacher，为只能读取 SAR + 地形的 Student 提供跨灾害事件的**相似度分布约束**；推理时仅保留 Student、其三个 EPM 和输出头。不是跨网络特征拼接，也不是让 Student 拷贝 Teacher 分割掩膜。

## A：整体网络
**Teacher:** 前后时相 Sentinel-2 各 6 个波段，共 12；前后时相 Sentinel-1 VV/VH 共 4；DEM + slope 共 2；合计 18 通道。Teacher 是输入 18 通道、输出 1 通道的 U-Net，先用真实 GT 做分割训练，蒸馏阶段冻结。Teacher 的预测输出可以作为展示，但**不进入 Student 的 BCE/Dice 目标**。
**Student:** 仅接收上述 6 个 SAR/地形通道，不接收光学。与 Teacher 一样是 5 层编码器（128→64→32→16→8）和 4 层解码器（16→32→64→128），通道宽度为 64/128/256/512/1024，四条同尺度 skip 通过 Concatenation 接入解码器。最后 1×1 预测单通道滑坡 logit，部署时用 sigmoid 得到概率图。
**EPM 位置:** final_method.py 的 F_SSP 配置 LEVELS=(1,2,3)：Student 解码器索引 1/2/3，特征尺寸 32/64/128；16px 解码阶段没有 EPM。三处都使用一个共享可训练原型库（K=16,d=64），但 Q/V/O 投影分别学习。
**训练损失:** Student 用有效像素 BCE + Dice 对齐 GT，另外只在解码器索引 1（32px）的 EPM 后特征计算 L_CEW，权重 λ=30。直接的 GT→学生损失与教师关系→学生 CEW 是两条不同类型的监督箭头。
**推理:** 移除 Teacher 和 CEW，Student 仍含三个 EPM。标题中的 Optical-guided training and SAR-only inference 说的就是这个区分。

## B：Shared EPM 的一层计算
设第 l 级学生解码特征 f_l ∈ R^(C_l×H_l×W_l)。每个空间位置经本尺度 1×1 Query 投影到 d=64 维，并与归一化共享原型 P_k 计算余弦相似度，温度 τ=0.1，经 Softmax 得到 16 个原型的分配权重 a_lk。然后经该尺度自己的 Value 映射得到 C_l 维原型值，做带权和，再经本尺度的输出 1×1 投影 W_o^l。输出：

f'_l = f_l + W_o^l(Σ_k a_lk W_v^l normalize(P_k))。

W_o^l 初始为全零，所以 EPM 初始是近似恒等映射，残差连接的“+”有真实计算意义。**共享的是原型库，不是三尺度共享 Q/V/O 参数**。图中 Prototype bank 箭头应分别指到 Student 的三个 EPM，不能连接 Teacher 的特征。

## C：Cross-event Wasserstein relation distillation
只从 32px 解码特征提取区域 token：Teacher 从普通 32px 解码输出，Student 从 32px EPM 修正后的输出；在 valid mask 支持下平均池化为每张图最多 4×4=16 个区域，每个区域至少有 75% 有效覆盖，随后按通道 L2 归一化。所有有效区域 token 来自一个 batch，数量**不是恒定 32**。
对每个 anchor i，只考虑同 batch 里灾害事件 ID **不同**的 j：N_i={j:event(j)≠event(i)}。两模型在自己的特征空间分别计算相似度：
T_ij = <t_i,t_j>；S_ij=<s_i,s_j>。
排除 same-event 对（它们的矩阵格子可涂灰），Teacher/Student 各自减去当前 batch 有效跨事件相似度的最小值，随后对每个 anchor 的有效相似度列表**各自升序排序**。对两个排序结果对应顺序位置的值取均方差，即平方一维 Wasserstein。每个 anchor 至少两个跨事件伙伴，保留 anchor 再取平均。这里比较的是分布，而**不是** Student 与 Teacher 必须有完全相同的跨事件邻居排序。
在 C 详细图中，应清楚画出 Teacher 和 Student 两套相似度计算流程。如果为了视觉紧凑只用中央一张大示意矩阵，必须说明它是两套计算的关系示意而非实际共享矩阵。排序后各自输出 T_sorted / S_sorted 再汇入同一个 L_CEW。不能将 same-event excluded 误读成删除事件数据，也不能把灰格伪装成实验测量数值。

## 训练目标和审稿时容易被问到的点
最终 Student 的训练目标：L = L_BCE + L_Dice + 30 L_CEW。这里 Teacher 在蒸馏时冻结；CEW 反向传播到 Student、它在 32px 的 EPM 及共享原型库，但不更新 Teacher。共享原型也接受 64/128px Student 分割损失的梯度。
C 图中的示意矩阵和排序柱如果不是实际数值，必须标为 schematic。绝对不要写固定 S∈R^(32×32) 当成批次真实矩阵尺寸。光学只参与**训练**，输出图有真实 GT、Teacher 与 Student 预测三种不同来源，必须严格标注。

## 对应代码证据
- common.py: UNet, segmentation
- run.py: teacher_fusion
- methods.py: fusion_input
- final_method.py: LEVELS / LOSS_LEVELS / SharedEPM / FinalNet.features / tokens / objective
- pr.py: PROTOS=16, PDIM=64, TAU=.1, cross_event_wasserstein
- manuscript/main.tex: Method / Scale- and Event-Shared Prototypes / Cross-Event Wasserstein Relation Distillation / Learning Objective

## 给绘图人的最短说明
A 画**两条独立的 U-Net 前向路径**，三处 Student EPM 和两种不同监督。B 画**共享原型的加权残差增强**。C 画**独立的 T/S 跨事件相似度分布排序及 Wasserstein 对齐**。禁止凭美观加入任何不存在的连线。
