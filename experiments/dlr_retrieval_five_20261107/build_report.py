"""Build the Chinese study report from recorded results only."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def main():
    status=json.loads((OUT/"status.json").read_text())
    lines=["# DLR 光学引导 SAR 检索蒸馏实验","",f"当前状态为 {status['state']}。","",
           "仅使用真实记录，尚未运行的项目标为 NOT_RUN。独立44张属于全部120张。","",
           "## 结构与梯度检查",""]
    for filename in ("mechanism_AB_checks.json","real_train16_gradients.json"):
        if (OUT/filename).exists():
            lines.append(f"- {filename} 已完成，完整记录见同目录文件。")
    gate=OUT/"M0_teacher_search_gate.json"
    if gate.exists():
        g=json.loads(gate.read_text())
        lines.extend(["","## 教师检索价值验证","",
                      "| 条件 | 验证 IoU |","|---|---:|"])
        for key in ("native_A","native_B","AA","AB","BA","BB"):
            lines.append(f"| {key} | {g[key]['IoU']*100:.3f}% |")
        lines.extend(["",f"固定宿主A与B的规则替换收益分别为 {g['fixed_host_gains'][0]*100:.3f} 和 "
                      f"{g['fixed_host_gains'][1]*100:.3f} 个百分点。",
                      f"预先设定的教师检索验证条件通过情况为 {g['gate_pass']}。"])
    lines.extend(["","## 验证选模","",
                  "| 模型 | 完成轮数 | 选中轮次 | 验证 IoU |","|---|---:|---:|---:|"])
    for method in ("M0","M1","M2","M3","M4"):
        for role in ("sar","teacher","student"):
            if method=="M2" and role!="student":
                continue
            name=f"{method}_{role}_seed42"
            p=OUT/name/"selection.json"
            if p.exists():
                s=json.loads(p.read_text())
                lines.append(f"| {name} | {s['epochs_completed']} | {s['selected_epoch']} | {s['validation_IoU']*100:.3f}% |")
            else:
                progress=OUT/name/"progress.json"
                if progress.exists():
                    p=json.loads(progress.read_text())
                    lines.append(f"| {name} | 运行到第{p['epoch']}轮 | — | — |")
                else:
                    lines.append(f"| {name} | NOT_RUN | — | — |")
    comparison=OUT/"comparison.json"
    if comparison.exists():
        rows=json.loads(comparison.read_text())
        lines.extend(["","## 正式测试","",
                      "| 模型 | 全部120 IoU | 独立44 IoU | 全部120 AP |","|---|---:|---:|---:|"])
        for name,m in rows.items():
            lines.append(f"| {name} | {m['IoU']*100:.3f}% | {m['per_official_subset']['testind']['IoU']*100:.3f}% | {m['AUPRC']*100:.3f}% |")
        lines.extend(["","正式方法筛选使用先前验证集记录。测试结果不用于重新选择损失权重或追加方案。"])
    if status["state"]=="TEACHER_SEARCH_GATE_NOT_PASSED":
        lines.extend(["","## 研究判断","",
                      "M0的教师检索前提未通过。本轮没有进入学生蒸馏阶段，M1–M4学生均为NOT_RUN。"
                      "当前证据不足以将光学引导检索作为已验证贡献。该状态不是学生训练失败。"
                      "后续需要先解决教师规则在固定SAR证据和解码器条件下的功能价值，才能判断蒸馏目标。"])
    elif status["state"]=="COMPLETE":
        lines.extend(["","## 研究判断","",
                      "独立模块的收益须相对各自的同结构无蒸馏对照计算。"
                      "Teacher精度、JS或余弦改善单独不构成学生有效性的证明。"
                      "验证提升触发的seed43为探索性复核，两种子不足以证明普遍稳定性。"])
    else:
        lines.extend(["","最终机制判断仍待教师检索验证和后续实际训练结果。"])
    (HERE/"RESULTS_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
