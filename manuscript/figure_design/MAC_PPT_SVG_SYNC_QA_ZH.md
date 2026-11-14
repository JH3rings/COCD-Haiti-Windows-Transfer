# Mac 桌面 PPT/SVG 同步、旧窗口和版本保护规则

## 事故复盘
2026-11-14 的对话中，用户反馈 Mac 上 SVG 与云端展示一致，但 PPT 看起来是旧版。实际读取 Mac Desktop/COCD_Architecture.pptx 发现文件有新的修改时间和 634 个原生 PowerPoint 对象；其内部构造是真实重建，但这并不能证明 PPT 阅读器当前打开的窗口已刷新，也不能证明它与云端 PPT 的**二进制字节**一致。另在 Desktop/.workbuddy/cocd_recovery_2130/ 存有同名旧候选文件，存在打开错误路径的可能。不能简单说“是缓存”或“已同步成功”就结束。

## 必须区分三个概念
**源场景一致**：PPTX/SVG 从同一个 scene/layout 生成。
**文件字节一致**：云端与 Mac 需比对 SHA-256；仅文件名、对象数或相似截图一致不代表字节相同。
**窗口显示最新**：目标应用必须真正关闭旧文档并从确切路径重新打开，最好有应用导出的新截图。前两者无法自动证明第三者。

## 正式 Mac 位置
~/Desktop/COCD_Architecture.pptx
~/Desktop/COCD_Architecture.svg
Desktop 可以只出现以上一对正式文件，历史版本、脚本、检查预览放在隐藏的 .workbuddy 或独立备份目录。用户不应再按 V8/V9/V10 文件名寻找结果。

## 发布与校验程序
1. 在 staging 生成 PPTX/SVG。将原始数据图片嵌入 pptx 的独立图片对象；其他图形和文字仍为可编辑原生对象。
2. 在 staging 读取 PPTX：核验 1 张幻灯片、必要文本、3 个 EPM、CEW、输入输出缩略图、对象数、无边界越界。核验 SVG 可解析。
3. 渲染 staging PPTX 和 SVG，进行 A/B/C 三处对照。不以独立 SVG 的成功代替 PPT 渲染检查。
4. 若云端文件是正式构建，优先**复制该文件的实际字节**到 Mac，再比较 SHA-256；如果只能在 Mac 重新跑代码，必须明确标注“本地重建而非原件字节同步”，且对渲染结果重新验收。
5. 先保存桌面旧版到备份（含 SHA、时间、视觉母版 ID），再输出到同目录临时文件并以原子替换更新正式 PPTX/SVG。不可对正在使用的文件长期直接写入。
6. 将构建元数据写入 PPTX core properties（不是干扰主图的可见水印）。再次读取最终桌面文件，记录 SHA-256、大小、修改时间、PPTX shape 数、SVG 解析状态和实际渲染路径。
7. 清理 QuickLook 缩略图缓存后重新渲染。请用户彻底关闭 PowerPoint/Keynote/WPS 的旧窗口，从访达中打开~/Desktop/COCD_Architecture.pptx，而不是“最近使用”列表。如果仍旧，检查“文件 → 信息”中的完整路径并与上面比对；保留软件名称和截图继续诊断。
8. 只有本地二进制或重建验证、真实渲染、用户窗口重新打开三项都证实，才能宣称“Mac 的显示已经更新”。否则准确说“磁盘文件更新；应用窗口状态未核实”。

## 诊断输出（请保存在备份，不要污染桌面）
- target_path / file_size / sha256 / mtime / git_commit
- source_scene_sha / source_master_id / embedded_image_hashes
- PPTX shape count, editable text count, independent picture count
- actual_ppt_render.png, svg_render.png, reference_vs_ppt.png
- A/B/C collisions, main-spine slopes, text safe margin and missing-font logs
- user confirmation of open-app rendering

## 禁止事项
不能将“创建了文件”视为“用户下载成功”；不能由 SVG 看似正确推断 PPT 也正确；不能仅因为 PPTX 重新生成就断言用户打开的窗口已更新；不能为了避免文件名冲突而保留十几个可见桌面版本；不能在没有已接受母版快照时覆盖唯一历史版本。
