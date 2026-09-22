"""Assemble only completed, protocol-traceable GRSL experiment artifacts."""
from __future__ import annotations
import csv, hashlib, json, shutil
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiments'/'GRSL_final_experiment_package'
SRC=ROOT/'experiments'/'windows_main'; [ (OUT/x).mkdir(parents=True,exist_ok=True) for x in ('tables','figures','csv','json') ]
def read(p):
 with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def num(x):return float(x) if x not in ('',None) else None
def write(name,rows):
 p=OUT/'csv'/name
 with open(p,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return p
def tex(name,rows,cols):
 s=['\\begin{table}[t]','\\caption{'+name.replace('_',' ')+'}','\\centering','\\resizebox{\\columnwidth}{!}{%','\\begin{tabular}{'+'l'+'c'*(len(cols)-1)+'}','\\toprule',' & '.join(cols)+' \\\\','\\midrule']
 for r in rows:s.append(' & '.join(str(r.get(c,'')) for c in cols)+' \\\\')
 s+=['\\bottomrule','\\end{tabular}}','\\end{table}'];(OUT/'tables'/(name+'.tex')).write_text('\n'.join(s),encoding='utf-8')
def wide(rows,methods,regions=('Overall','TGD','Remaining','OmegaTR')):
 d={(r['method'],r['region']):r for r in rows};out=[]
 for m in methods:
  x={'Method':m}
  for rg in regions:
   z=d.get((m,rg),{});
   for k in ('auprc','iou','f1','accuracy','policy_js','evidence_distance'):x[f'{rg}_{k}']=z.get(k,'')
  out.append(x)
 return out
def main():
 sources={}
 def src(key,p):sources[key]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};return read(p)
 tm=src('teacher_metrics',SRC/'teacher_target_search_v2'/'teacher_mechanism_metrics.csv')
 normal=[r for r in tm if r['method'] in ('SG0','SG10','CG0','CG10') and r['region'] in ('Overall','TGD','Remaining') and r['partition']=='combined']
 teacher=wide(normal,['SG0','SG10','CG0','CG10'],('Overall','TGD','Remaining'));write('table_teacher_ablation.csv',teacher);tex('table_teacher_ablation',teacher,['Method','Overall_auprc','Overall_iou','TGD_auprc','TGD_iou','Remaining_auprc','Remaining_iou'])
 mech=[r for r in tm if r['method'] in ('CG10','CG10-center','CG10-random','CG10-counter-shuffle') and r['region'] in ('Overall','TGD') and r['partition']=='combined']
 md={(r['method'],r['region']):r for r in mech};mrows=[]
 for label,key in [('Normal','CG10'),('Center','CG10-center'),('Random offset','CG10-random'),('Counter shuffle','CG10-counter-shuffle')]:
  o,t=md[key,'Overall'],md[key,'TGD'];mrows.append({'Condition':label,'Overall_AUPRC':o['auprc'],'TGD_AUPRC':t['auprc'],'TGD_IoU':t['iou'],'TGD_F1':t['f1']})
 write('table_search_mechanism.csv',mrows);tex('table_search_mechanism',mrows,list(mrows[0]))
 sm=src('student_metrics',SRC/'student_target_search_v2'/'locked_region_mechanism_metrics.csv');stud=wide(sm,['S0','S1','S2']);write('table_student_distillation.csv',stud);tex('table_student_distillation',stud,['Method','Overall_auprc','Overall_iou','Overall_f1','TGD_auprc','TGD_iou','TGD_f1','OmegaTR_f1','OmegaTR_accuracy'])
 sk=[]
 for r in stud:sk.append({'Method':r['Method'],'Overall_JS':r['Overall_policy_js'],'TGD_JS':r['TGD_policy_js'],'OmegaTR_JS':r['OmegaTR_policy_js'],'Overall_EvidenceDistance':r['Overall_evidence_distance'],'TGD_EvidenceDistance':r['TGD_evidence_distance'],'OmegaTR_EvidenceDistance':r['OmegaTR_evidence_distance']})
 write('table_student_mechanism.csv',sk);tex('table_student_mechanism',sk,list(sk[0]))
 cm=src('composition_metrics',SRC/'student_target_search_v2'/'weight_sweep'/'locked_region_mechanism_metrics.csv');comp=wide(cm,['NoKD','P100E0','P50E50','P25E75','P0E100']);alpha={'NoKD':('0','0'),'P100E0':('1.00','0.00'),'P50E50':('0.50','0.50'),'P25E75':('0.25','0.75'),'P0E100':('0.00','1.00')};cr=[]
 for r in comp:cr.append({'Setting':r['Method'],'Alpha_policy':alpha[r['Method']][0],'Alpha_evidence':alpha[r['Method']][1],'Overall_AUPRC':r['Overall_auprc'],'TGD_IoU':r['TGD_iou'],'OmegaTR_F1':r['OmegaTR_f1'],'Policy_JS':r['Overall_policy_js'],'Evidence_distance':r['Overall_evidence_distance']})
 write('table_loss_composition.csv',cr);tex('table_loss_composition',cr,list(cr[0]))
 stm=src('strength_metrics',SRC/'student_target_search_v2'/'strength_sweep'/'locked_region_mechanism_metrics.csv');st=wide(stm,['NoKD','KD05','KD10','KD20']);sr=[]
 for r,k in zip(st,['0%','5%','10%','20%']):sr.append({'KD_over_Seg':k,'Overall_AUPRC':r['Overall_auprc'],'TGD_IoU':r['TGD_iou'],'TGD_F1':r['TGD_f1'],'OmegaTR_F1':r['OmegaTR_f1'],'OmegaTR_Accuracy':r['OmegaTR_accuracy']})
 write('table_loss_strength.csv',sr);tex('table_loss_strength',sr,list(sr[0]))
 # Only directly re-evaluated, final-protocol methods are included in comparison.
 cg=next(x for x in teacher if x['Method']=='CG10');s0=next(x for x in stud if x['Method']=='S0');s2=next(x for x in stud if x['Method']=='S2')
 compa=[{'Method':'CG10 Teacher','Role':'dual-orbit teacher',**{k:cg[k] for k in cg if k!='Method'}},{'Method':'S0','Role':'target-only search student',**{k:s0[k] for k in s0 if k!='Method'}},{'Method':'S2','Role':'target-only distilled student',**{k:s2[k] for k in s2 if k!='Method'}}]
 co=[{'Method':r['Method'],'Role':r['Role'],'AUPRC':r['Overall_auprc'],'IoU':r['Overall_iou'],'F1':r['Overall_f1']} for r in compa];ct=[{'Method':r['Method'],'Role':r['Role'],'AUPRC':r['TGD_auprc'],'IoU':r['TGD_iou'],'F1':r['TGD_f1']} for r in compa];com=[{'Method':r['Method'],'Role':r['Role'],'F1':r.get('OmegaTR_f1',''),'Accuracy':r.get('OmegaTR_accuracy','')} for r in compa]
 write('table_comparison_overall.csv',co);write('table_comparison_TGD.csv',ct);write('table_comparison_OmegaTR.csv',com);tex('table_comparison_overall',co,list(co[0]));tex('table_comparison_TGD',ct,list(ct[0]));tex('table_comparison_OmegaTR',com,list(com[0]))
 # Figure-ready CSV and a transparent single-panel robustness chart.
 write('figure3_search_mechanism_data.csv',mrows);write('figure4_student_distillation_data.csv',sk);fig=[{'Method':r['Method'],'TGD_IoU':r['TGD_iou'],'TGD_F1':r['TGD_f1'],'OmegaTR_F1':r.get('OmegaTR_f1','')} for r in compa];write('figure5_tgd_robustness_data.csv',fig)
 names=[x['Method'] for x in fig];x=range(len(names));plt.figure(figsize=(7,3));plt.bar([i-.25 for i in x],[num(a['TGD_IoU']) for a in fig],.25,label='TGD IoU');plt.bar(x,[num(a['TGD_F1']) for a in fig],.25,label='TGD F1');plt.bar([i+.25 for i in x],[num(a['OmegaTR_F1']) if a['OmegaTR_F1'] else 0 for a in fig],.25,label='OmegaTR F1');plt.xticks(list(x),names);plt.ylim(0,1);plt.legend(fontsize=8);plt.tight_layout();plt.savefig(OUT/'figures'/'figure5_tgd_robustness.pdf');plt.savefig(OUT/'figures'/'figure5_tgd_robustness.png',dpi=220);plt.close()
 (OUT/'json'/'source_manifest.json').write_text(json.dumps({'sources':sources,'protocol_note':'All values are locked-checkpoint test-set inference or test-monitored selection; test monitoring is not independent blind testing.','excluded':['Legacy DA Student distillation artifacts: decoder raw-feature bypass; excluded from final comparisons.','DIS2/Vanilla KD: no newly re-evaluated checkpoint/prediction under the final CG10/Student-v2 protocol.']},indent=2),encoding='utf-8')
 (OUT/'paper_summary.md').write_text('# GRSL final experiment summary\n\nThe final Teacher is CG10 and the final deployable Student is S2 (policy plus evidence KD, alpha=0.5, initial KD/Seg=10%).\n\n- **Teacher:** CG10 yields the strongest four-arm Teacher performance, and counter shuffle/random offset interventions degrade TGD performance.\n- **Search:** random locations and incorrect counter context degrade TGD IoU, supporting learned, counter-dependent policy use.\n- **Student:** S1 aligns policy JS but has limited task gain; S2 reduces shared-memory evidence distance and improves prediction, with the strongest gains on OmegaTR.\n- **Robustness:** S2 improves TGD IoU/F1 over S1; the AUPRC TGD gain is similar to Remaining, so claims should emphasize IoU/F1 and OmegaTR rather than assert an AUPRC-only TGD concentration.\n- **Loss:** P50E50 is the best completed composition setting; 10% is the best completed total strength.\n\nAll results are test-monitored selection evidence, not an independent held-out blind test.\n',encoding='utf-8')
 (OUT/'experiment_checklist.md').write_text('# Experiment completeness checklist\n\n- [x] Teacher four-arm ablation\n- [x] Search intervention and counter shuffle\n- [x] Student S0/S1/S2 KD study\n- [x] Composition sweep\n- [x] KD strength sweep\n- [x] TGD and fixed OmegaTR analysis\n- [ ] MISSING: unified final-protocol SO baseline table row\n- [ ] MISSING: DIS2/Vanilla-KD/other historical methods re-evaluated under final protocol\n- [ ] MISSING: per-arm Teacher OmegaTR metrics for SG0/SG10/CG0\n- [ ] MISSING: Figure 3 sampling-location qualitative panels (no packaged prediction/location artifact)\n\nLegacy raw-feature-bypass Student KD outputs were deliberately excluded.\n',encoding='utf-8')
if __name__=='__main__':main()
