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
    amendment=OUT/"user_test_amendment.json"
    if amendment.exists():
        lines.extend(["","用户指示停止额外随机种子，直接测试13个已冻结的seed42模型。"])
    if status["state"]=="TEACHER_SEARCH_GATE_NOT_PASSED":
        lines.extend(["","## 研究判断","",
                      "M0的教师检索前提未通过。本轮没有进入学生蒸馏阶段，M1–M4学生均为NOT_RUN。"
                      "当前证据不足以将光学引导检索作为已验证贡献。该状态不是学生训练失败。"
                      "后续需要先解决教师规则在固定SAR证据和解码器条件下的功能价值，才能判断蒸馏目标。"])
    elif status["state"]=="COMPLETE":
        lines.extend(["", "## 同结构蒸馏增益", "",
                      "| 方案 | SAR对照全部120 IoU | Student全部120 IoU | 增益（百分点） | 独立44增益（百分点） |",
                      "|---|---:|---:|---:|---:|"])
        for method in ("M0","M1","M2","M3","M4"):
            arch="M0" if method=="M2" else method
            a=rows[f"{arch}_sar_seed42"]; b=rows[f"{method}_student_seed42"]
            delta=(b['IoU']-a['IoU'])*100
            independent=(b['per_official_subset']['testind']['IoU']-a['per_official_subset']['testind']['IoU'])*100
            lines.append(f"| {method} | {a['IoU']*100:.3f}% | {b['IoU']*100:.3f}% | {delta:+.3f} | {independent:+.3f} |")
        lines.extend(["", "M0和M2相对同结构对照有小幅整体增益，M1、M3、M4没有整体增益。"
                      "五种Student的全部120 IoU均低于原纯SAR B1的60.234%，未达到整体提高2个百分点的目标。"
                      "独立44是全部120的子集，两者不能视为独立重复验证。",
                      "固定宿主规则替换存在明显的宿主依赖：例如M0教师规则在SAR宿主上并未改善验证IoU，"
                      "而在教师宿主上优势很大。因此教师原生精度及原门槛通过不足以单独证明规则可迁移；"
                      "当前结果不支持声称光学检索优势已充分转化为Student收益。"])
        lines.extend(["","## 研究判断","",
                      "独立模块的收益须相对各自的同结构无蒸馏对照计算。"
                      "Teacher精度、JS或余弦改善单独不构成学生有效性的证明。"
                      "本轮按用户要求只报告seed42，额外种子在完成前被停止，结果不用于方法筛选。"])
    else:
        lines.extend(["","最终机制判断仍待教师检索验证和后续实际训练结果。"])
    (HERE/"RESULTS_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
