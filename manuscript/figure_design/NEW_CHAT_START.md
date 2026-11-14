# COCD 图稿新会话启动提示词（v3.0）

请访问 GitHub 仓库 JH3rings/COCD-Haiti-Windows-Transfer，先在分支 figure-design/v3-visual-parity-20261114 的 manuscript/figure_design/ 下阅读 README、REFERENCES、WORKFLOW(v3)、STYLE_CONTRACT(v3)、COCD_SCIENCE_AUDIT、COCD_NETWORK_EXPLAINED_ZH 和 MAC_PPT_SVG_SYNC_QA_ZH，并真实查看 assets 两张参考截图、R2 色板及 R3/R4 两份 PDF 的指定图页。科学依据以 main 分支已锁定的代码及 main.tex 为准，具体包括 final_method.py、common.py、pr.py、methods.py、run.py、原始 draw_method.py 和 method.svg。不要用参考论文中的模块替换我们的科学结构。

**视觉母版已经确定**：用户发来的 GPT 图片的**下半部分**，A 左（上下 Teacher/Student），B 右上（EPM），C 右下（CEW）。以这张原图的区块、布局、文字占比、2.5D 特征块、精确色板、箭头层级和密度作为排版锁定基准。不要另造一种图或把它改成普通机械化 PPT。R2 是唯一色板；不使用浅灰到看不清的正文文字、过长的框内描述或输入白色大卡片。

必须按 v3 流程逐步执行并通过硬门：冻结节点与端口真实语义 → 测量母版 A/B/C 比例 → 科学纠错的母版 → 原生对象 PPTX + 同场景 SVG → **实际渲染 PPTX 和 SVG** → 分别比对主图 A、EPM B、CEW C → 修正文字碰撞、斜向主干箭头、方框比例、字体替换 → 校验 SHA-256 和 Mac 本地确切路径。最后只输出 PPTX 和 SVG，以及内联渲染预览。源码/QA/旧版放私有备份，**Mac 桌面只保留一对正式图稿**。原始论文图不得覆盖。用户评分才算验收，不得自行宣布达到 6–7 分。

已知科学硬性条件：Teacher 18 通道冻结，Student 6 通道；两条 5+4 级 U-Net；三处 Student EPM 在解码 32/64/128；共享 K=16、d=64 原型，但 Q/V/O 各尺度独立；CEW 仅取 32px 的 Teacher 和后 EPM Student 特征；4×4 有效池化、排除 same-event、T/S 各自相似度和独立排序、平方一维 Wasserstein；真实 GT 仅监督 Student BCE+Dice；训练目标 seg+30CEW；推理仅 Student+EPM。不能给 Teacher 加 EPM，也不能让 Teacher 掩膜直接监督 Student。

若已经存在被用户认可的视觉母版，**禁止再重复 GPT 生图或未经确认改变结构布局**；仅进行科学与几何局部修正。任何云端文件“已保存”不代表 Mac App 窗口已刷新：按 MAC_PPT_SVG_SYNC_QA_ZH 核验。
