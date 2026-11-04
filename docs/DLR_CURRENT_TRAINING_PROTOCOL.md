# 唯一有效的 DLR 后续训练策略（2026-11-04）

用户固定当前统一策略。其他策略退出后续实验选择；旧日志、配置、源码和结果只作为历史证据，不能改写或作为替代Baseline。

- 固定现有官方355/88/44+76位置划分、有效像素、训练集归一化和阈值0.5。
- BCEWithLogits + per-image foreground SoftDice，1:1，smooth=1e-6，不额外正类加权。
- 100 epochs，batch16，每5轮验证，无早停。
- AdamW：预训练encoder 2e-5，新模块1e-4；随机初始化U-Net全部1e-4。
- weight decay 0.01，norm/bias/GRN排除；gradient clip1。
- 前50轮恒定，后50轮cosine降到初始1%，无warmup。
- Dropout0，encoder最大DropPath0.05（模型具备该机制时）。
- 验证集累计前景IoU选checkpoint，冻结后测试，测试不得选模型或调参。
- Baseline唯一来源仍为 configs/active_dlr_baseline.json，仅复用，不重训重测。
- KD的辅助项另行记录，不改变统一基础训练策略。

本文不把不同输入和网络的差异归因于DEM；判断DEM净贡献需要同网络有/无DEM对照。
