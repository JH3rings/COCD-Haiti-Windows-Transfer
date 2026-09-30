# 中文图注

图1：锁定的双轨 Teacher 与目标轨 Student 流程。T1 的 policy 同时使用 target 与双轨融合上下文，但采样值始终来自 target memory；S0 只在初始化时继承 T1 参数，随后仅用 target+GT 适应。

图2：相同窗口和显示规则的测试样例。黄色轮廓为 TGD。SAR 面板是两个输入波段的 pre/post 绝对差均值，仅作显示，不代表校准物理单位。

图3：锁定预测上的 TGD 机制证据。(a) S0−SO 的百分点差值；(b) T1 正确 counter、Target-FPN bypass 和 shuffled counter；(c) 固定 SO 错、Teacher 对诊断子集中的 S0 正确比例。
