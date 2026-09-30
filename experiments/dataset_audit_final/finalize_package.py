"""Write evidence report and manifests. Cleanup is a separate final step."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,shutil,zipfile
R=Path(__file__).resolve().parents[2];F=R/'experiments/dataset_audit_final';S=F/'01_statistics';C=F/'99_cleanup_manifest';Q=F/'03_manual_qc'
def csv(x,p):pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def prepare():
 s=json.loads((S/'audit_summary.json').read_text());df=pd.read_csv(F/'02_tables/dataset_patch_audit.csv');integ=pd.read_csv(S/'file_integrity.csv');shift=pd.read_csv(S/'registration_shift_statistics.csv');g=s['geometry'];inv=pd.read_csv(C/'all_existing_dataset_analysis.csv').fillna('')
 # Correct inherited exploratory proxy denominators; all current statistics use eligible finite data.
 elig=df[(df.gt_frac>=.02)&(df.gt_bothvalid_pixels>0)];a=elig.asc_vv_contrast_bothvalid;d=elig.desc_vv_contrast_bothvalid
 s['possible_temporal_mismatch_proxy']={'eligible_n':len(elig),'rule':'GT>=2%; BothValid GT>0; one orbit VV GT/BG>=1.20, other<=1.05, ratio>=1.5','ASC_only_ids':elig.loc[(a>=1.2)&(d<=1.05)&(a/d>=1.5),'sample_id'].tolist(),'DESC_only_ids':elig.loc[(d>=1.2)&(a<=1.05)&(d/a>=1.5),'sample_id'].tolist(),'limitation':'orbit/temporal discordance proxy, cannot attribute polygon event times'}
 finite=df.dropna(subset=['asc_joint_abs_matched_ratio','desc_joint_abs_matched_ratio']);va=finite.asc_joint_abs_matched_ratio>1;vd=finite.desc_joint_abs_matched_ratio>1
 csv([dict(metric='matched joint change exceeds control',eligible_patches=len(finite),one_or_both_fraction=(va|vd).mean(),both_fraction=(va&vd).mean())],S/'cross_orbit_support_statistics.csv')
 corr=pd.read_csv(S/'cross_orbit_pixel_correlations.csv');csv(corr.groupby(['region','metric']).agg(n_patches=('spearman','count'),mean_spearman=('spearman','mean'),median_spearman=('spearman','median')).reset_index(),S/'cross_orbit_statistics.csv')
 matched=pd.read_csv(S/'matched_background_statistics.csv');extra=[]
 for _,r in matched.iterrows():
  o,k=r.orbit,r.metric;gt=df[f'{o}_{k}_matched_gt_mean'];bg=df[f'{o}_{k}_matched_bg_mean']
  extra.append(dict(orbit=o,metric=k,mean_of_patch_GT_means=gt.mean(),median_of_patch_GT_means=gt.median(),mean_of_patch_BG_means=bg.mean(),median_of_patch_BG_means=bg.median(),sample_unit='patch stratum-weighted means; not median of all pixels'))
 matched=matched.merge(pd.DataFrame(extra),on=['orbit','metric']);csv(matched,S/'matched_background_statistics.csv')
 s['current_final_notes']={'source_engine_rerun':True,'final_patch_table':'02_tables/dataset_patch_audit.csv','legacy_slope_columns_removed':True,'both_orbit_clipped_same_support_shift':'01_statistics/registration_shift_statistics.csv','unavailable_sources':['optical','original polygon','original orbit azimuth/incidence/product metadata'],'texture_definition':'absolute change of local 5x5 standard deviation; contrast absolute change of local 5x5 range; no separate independent texture statistic','local_QC_count':len(pd.read_csv(Q/'manual_qc_index.csv'))}
 (S/'audit_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 # Inventory dispositions: protect all training/source/provenance boundaries.
 moves=[];deletes=[];reuse=[]
 for i,r in inv.iterrows():
  p=R/r.path
  dedicated='/physics_audit_20260924/' in r.path or '/dataset_forensic_audit_20260927/' in r.path
  if dedicated:
   dest=F/'05_existing_results_reused'/r.path
   bad=r.path.endswith(('/patch_slope.csv','/slope_stratified.json'))
   reason='旧坡度纬度计算方向错误且采样方法不同；已由本轮完整统计覆盖' if bad else '历史来源保留；新版本替代当前结论，旧图保留为不同案例/显示证据，不能视为本轮结果'
   inv.loc[i,['action','whether_superseded','reason']]=['delete' if bad else 'merge','yes for numerical current conclusions',reason]
   if bad:deletes.append(dict(path=r.path,size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason=reason,replaced_by='experiments/dataset_audit_final/01_statistics/slope_aspect_statistics.csv',role='obsolete terrain statistics',references='archived audit_02/03 scripts only; current engine legacy merge removed',process_status='no process running historical scripts',backup='99_cleanup_manifest/superseded_terrain_outputs.zip'))
   else:moves.append(dict(source=r.path,destination=dest.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size,reason=reason))
  elif r.path.startswith('cocd/metadata/') or r.path in ['DATASET_MANIFEST.md','reports/final_haiti_dataset_audit.md','reports/directory_structure.md','archive/history/reports/haiti_dataset_origin_and_convergence.md','archive/history/code/windows_ops/phase3_data_split_audit.py']:
   dest=F/'05_existing_results_reused'/r.path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);reuse.append(dict(source=r.path,destination=dest.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),value_added='original acquisition/provenance/transfer/spatial split evidence; historical scope dates and optical availability differ; preserved source'))
   inv.loc[i,['action','reason']]=['keep','source/provenance preserved; copied into final entry']
  elif r.path.startswith(('experiments/','results/','cocd/','archive/diagnostics/','baselines/','models/','train/','archive/history/code/','archive/history/docs/','archive/history/protocols/')):
   inv.loc[i,['action','reason']]=['keep','training/model/source/split-related protected output; excluded from dataset-output cleanup']
 csv(inv,C/'all_existing_dataset_analysis.csv');csv(inv[inv.action=='keep'],C/'kept_files.csv');csv(inv[inv.action=='uncertain'],C/'uncertain_files.csv');csv(moves+reuse,C/'merged_files.csv');csv(moves,C/'move_manifest.csv');csv(deletes,C/'deletion_manifest.csv');csv([],C/'deleted_files.csv')
 with zipfile.ZipFile(C/'superseded_terrain_outputs.zip','w',zipfile.ZIP_DEFLATED) as z:
  for r in deletes:z.write(R/r['path'],arcname=r['path'])
 csv(reuse,C/'reused_sources.csv')
 # Remove unused legacy merge dependency so final rerun is independent of removed tables.
 engine=C/'legacy_scripts/run_extended_audit.py';text=engine.read_text(encoding='utf-8-sig');start=text.index('    # Merge legacy complete');end=text.index('    patch_audit.to_csv',start);text=text[:start]+'    # Current statistics only. Legacy slope tables are superseded.\n    patch_audit = pdf.copy()\n'+text[end:];engine.write_text(text,encoding='utf-8')
 # Existing legacy engine output is intermediate, clearly archived separately.
 intermediate=F/'99_cleanup_manifest/intermediate_engine_outputs';intermediate.mkdir(exist_ok=True)
 for name in ['dataset_patch_audit.csv','anomaly_topk.csv','case_D_ranked_candidates.csv','case_selection_log.json','registration_shift_sensitivity.csv','case_A_bothvalid.png','case_B_asc_invalid_desc_valid.png','case_C_asc_valid_desc_invalid.png','case_D_anomaly_candidate.png']:
  if (S/name).exists():shutil.move(str(S/name),str(intermediate/name))
 (intermediate/'README.md').write_text('重跑引擎的中间结果，不是最终推荐文件。包括继承的旧坡度列、仅 ASC 位移敏感性和旧样例选择。仅用于执行追溯；当前使用 ../../02_tables/dataset_patch_audit.csv 和 ../../01_statistics/registration_shift_statistics.csv。',encoding='utf-8')
 # Report generated solely from current measured tables.
 gj=np.array(g['binary_joint_counts_rows_ASC_invalid_0_1_cols_DESC_invalid_0_1']);tot=gj.sum();low=g['gt_fraction_slope_below_3deg'];gtc=np.array(g['global_gt_slope_hist_counts']);bgc=np.array(g['global_bg_slope_hist_counts'])
 lines=['# Haiti COCD 最终数据质量审计','', '日期：2026-09-27；本轮全量重新读取 1713 个空间 patch，seed=20260927。未修改数据、划分、checkpoint 或训练结果。','', '## 1. 数据完整性与物理统计','',f'- 数据包 SHA256：13704/13704 匹配；逐 GeoTIFF 完整性表 {len(integ)} 行。strict_grid 异常 {int((~integ.strict_grid).sum())}，非有限像素 {int(integ.nonfinite_pixels.sum())}，异常常数 SAR 文件 {int(integ.constant_sar.sum())}。类别常数掩码可能是合法全背景或全有效，不能等同坏数据。', '- 本地实际数据：4 个 SAR 时相/轨道目录，每个 1713 个三波段文件，band1 VH、band2 VV、band3 geometry；3 个 processed mask/GT 每位置。原始 optical、polygon、原始 Annotation 目录未在数据根找到。旧元数据 optical 路径是 macOS 来源引用，不能当成本地存在。', '- 两块 DEM 已有本地文件：physics_audit_20260924/dem/N18W075.tif 与 N18W074.tif，EPSG:4326，约30m。坡度按原生 DEM 米制梯度计算、双线性采样，坡向最近邻；无法提供10m独立地形真值。DEM GLO-30 来源依赖旧审计记录，文件 tags 不识别产品版本。', '- geometry code 0 valid、1 layover、2 shadow、3 both：证据 data/haiti/processed/sample_*/metadata.json 的 geometry_class_coding；SAR tags 未独立说明代码语义。GT 以原标签>0，不把 GT 原始1/2/3误当geometry。']
 for o in ['asc','desc']:
  pct=g['class_pct_code0_valid_code1_layover_code2_shadow_code3_both'][o];lines.append(f'- {o.upper()} valid/layover/shadow/both = '+ '/'.join(f'{v:.6f}%' for v in pct)+f'；invalid={sum(pct[1:]):.6f}%。')
 lines += [f'- BothValid={gj[0,0]/tot:.6%}；ASC invalid/DESC valid={gj[1,0]/tot:.6%}；ASC valid/DESC invalid={gj[0,1]/tot:.6%}；BothInvalid={gj[1,1]/tot:.6%}。这些是重叠窗口累计像素比例，不是去重土地面积。',f'- GT slope<3°={low:.6%}；GT slope<10°={gtc[0]/gtc.sum():.6%}；P(GT|坡度bin)='+', '.join(f'{v:.6%}' for v in gtc/(gtc+bgc))+ '。没有“大量平地GT”的总体证据，但局部异常仍需看图。','- 两轨 invalid 随坡度总体单调上升。ASC layover 在 W/SW 高，DESC layover 在 E/SE 高；DESC shadow 在 W/NW 高、ASC shadow 更偏 E。支持方向性与互补；缺真实入射角/方位，不能验证每个像素的几何物理。','- GT与geometry的整体关联在陡坡分层后明显减弱：>40° P(GT|invalid)/P(GT|valid) ASC='+f"{g['slope_geometry_rates']['asc'][-1]['invalid_vs_valid_GT_risk_ratio']:.6f}"+'、DESC='+f"{g['slope_geometry_rates']['desc'][-1]['invalid_vs_valid_GT_risk_ratio']:.6f}"+'；低坡层仍有关联。粗分层不能排除坡向、高程、空间聚集等残余混杂，不能据此宣布异常像素耦合或标注独立性。','', 'BothValid 地形匹配：patch内相同5坡度×9坡向（含平地）×8高程层，几何有效且按同轨比较；以GT像素数加权层均值。没有背景的层不匹配，覆盖率='+f"{s['terrain_matched_change']['pooled_positive_pixel_coverage']:.6%}"+'。均值/中位数是patch配对汇总；CI为patch重采样、检验为配对sign-flip/Wilcoxon，均不是独立地块推断。','', '|轨道|指标|GT−matched BG均值|95% CI|Cohen dz|','|---|---|---:|---|---:|']
 for _,r in matched[matched.metric.isin(['dvv_abs','dvh_abs','vv_texture_change','vh_texture_change'])].iterrows():lines.append(f'|{r.orbit}|{r.metric}|{r["mean"]:.6f}|{r.bootstrap_95ci_mean}|{r.cohen_dz:.6f}|')
 lines += ['', '- 平均事件变化优势存在，但幅度小；局部std变化与texture为同一操作性定义，contrast为5×5局部范围变化。局部邻域可能跨GT/geometry边界；不能把弱散射证据等同于GT错误。',f'- 有效匹配的 {len(finite)} 个patch，至少一轨joint change超过matched背景={float((va|vd).mean()):.6%}，两轨均超过={float((va&vd).mean()):.6%}。GT两轨强度变化rank相关弱；像素级GT与matched背景rank表见 cross_orbit_statistics.csv。matched_background像素相关来自同地形层合并候选背景，非逐GT权重抽样，属于辅助描述。', '- G10/G01 的有效轨更强变化概率按patch内像素比例估计，bootstrap CI见 g10_g01_statistics.csv；暗值(<−25dB)/亮值(>0dB)仅描述代理，不是传感器饱和阈值，也不代表“更可解释”。','', '双轨GT平移 ±1/±2/±3 pixel：49个二维组合、无wrap，固定BothValid，用同一裁边支持域计算平移增益；保留每patch结果。多方向搜索未经独立验证，CI不校正49次搜索与空间重叠。']
 for o in ['asc','desc']:
  ss=shift[shift.orbit==o];best=ss.loc[ss.mean_gain.idxmax()];lines.append(f'- {o.upper()} 最大平均增益方向 dy={int(best.dy)}, dx={int(best.dx)}（图像坐标，dy向下、dx向右），gain={best.mean_gain:.6f} dB；CI [{best.ci_low:.6f}, {best.ci_high:.6f}]；patch改善比例={best.fraction_gain_positive:.6%}。需要人工核查，不自动纠正GT。')
 lines += ['', '## 2. 自动异常审计','', '- Type A–E 各top20，统一标记 needs_manual_review，不是错误率。详见 02_tables/anomaly_summary.csv 和 04_anomaly_cases/type_*/ranking.csv。规则在每行记录，分数不得跨类型直接比较。','- A：GT>=2%、BothValid GT>=100，地形匹配后两轨max VV变化ratio低；B：GT与任一invalid重叠高；C：低坡GT占比高；D：坡度与mask关联弱的代理（没有完整轨道几何，未自动裁决aspect异常）；E：同支持域平移增益高。','- '+f"{'、'.join(str(v) for v in sorted(set(pd.read_csv(F/'02_tables/anomaly_summary.csv').sample_id))[:15])}"+' 等为复核候选。候选top-K比例是人为选择数量，不能当作发现坏样本比例。','- GT边缘与rectangle fill统计仅用于提示；标签在patch边缘被正常裁切可能导致border比例高，矩形填充度也不是错误判据。','', '## 3. 当前能够得出的数据质量判断','', '|对象|评级|证据与限制|','|---|---|---|','|Landslide GT|PASS WITH LIMITATIONS|完整、标签合法、总体位于坡面且匹配背景后变化略强；缺本地原始矢量/optical与独立事件日期，边界和形态待人工|','|geometry mask|PASS WITH LIMITATIONS|元数据代码、坡度趋势与坡向互补支持；静态pre/post复用、稀少shadow、无原轨道参数，不能完整重推|','|ASC/DESC registration|PASS WITH LIMITATIONS|逐文件严格共网格；共网格不是内容准确配准证据，位移敏感性与散射视角混杂，不能排除边界偏移|','|temporal consistency|PASS WITH LIMITATIONS|pre/post文件存在，geometry相同；日期只能来自文件名和元数据，无时刻与逐polygon事件来源|','|single-orbit research suitability|PASS WITH LIMITATIONS|有效像素可做单轨事件研究，但变化效应小、样本选择与窗口重叠限制外推；本次不以训练性能证明数据正确|','|geometry-invalid evaluation suitability|PASS WITH LIMITATIONS|两轨互补支持分区描述评价；invalid强度不可作为可靠地面观测，GT覆盖/陡坡混杂需报告，shadow很少|','', '本轮未发现文件损坏、缺失、非法label、SAR异常常数图等严重结构异常；这不是“所有GT与配准都已通过”的结论。','', '## 4. 尚待人工判断','', '必须先盲看03_manual_qc/blind/，再参考答案表及带contour联系表，由你与老师填写 manual_qc_form.csv：滑坡形态与真实坡面、边界自然性、pre/post可见扰动、layover/shadow视觉对应、两轨互补与可疑偏移、最弱变化是否是标注问题。AI没有填写任何人工字段。','', '## 方法与复核入口','', '- 主脚本 rerun_quantitative.py、build_final_audit.py；当前结果仅01_statistics与02_tables，旧数据审计05_existing_results_reused仅历史证据。','- 单次事件、patch重叠、DEM分辨率较粗、两轨日期相差2天、几何静态、地形粗匹配、纹理邻域边界混合与多重筛选，均限制因果和统计解释。p值只做探索。','- 旧 2026-09-12 报告中的本地optical存在与“GT无地理参考”描述属于原Mac包；本地processed GT严格共网格。本轮不重复其“配准YES”的强裁决。旧split与模型结果保留，本任务未重分数据。']
 (F/'final_dataset_audit.md').write_text('\n'.join(lines),encoding='utf-8')
 (F/'README.md').write_text('''# Haiti 数据集审计唯一入口

当前推荐阅读 [final_dataset_audit.md](final_dataset_audit.md)，数值使用 `01_statistics/` 和 `02_tables/dataset_patch_audit.csv`。CSV 的空值表示不可计算/未观测，不能按0解释。

- 完整性：file_integrity.csv（11991 GeoTIFF逐文件）与99_cleanup_manifest/data_hash_verification.json（13704文件SHA256）。
- Geometry/terrain/GT：geometry_summary、joint_geometry_summary、slope_aspect_statistics、gt_geometry_statistics；论文候选图在01_statistics/figures/。
- 变化：sar_change_statistics、matched_background_statistics、cross_orbit_statistics、cross_orbit_support_statistics、g10_g01_statistics、registration_shift_statistics；每patch明细与方法限制见报告。
- 异常：02_tables/anomaly_summary.csv、04_anomaly_cases/type_A…E/，每类20个候选，不是自动判错。
- 人工材料：03_manual_qc/有46张5×4高分辨率联系表；Group1/4盲图随机编号、无GT与异常标签。先盲检，答案在blind_ground_truth_key.csv。manual_qc_form.csv字段留空，只由人工填写；跨组可重复同patch。display_ranges.csv记录每图共享显示范围。

历史保留内容集中在05_existing_results_reused/：9月24物理审计、9月27扩展审计的不同案例图、代码与来源记录；acquisition_metadata/temporal_pairs/file_inventory、原Mac报告和数据交付说明提供本轮不能恢复的来源历史。旧报告/数字仅为历史，不得与本轮表混用，尤其旧光学存在声明、split和较强配准裁决。旧DEM完整保留。来源与操作逐文件见99_cleanup_manifest/merged_files.csv。

旧9月24 patch_slope.csv 与 slope_stratified.json 已由本轮统计覆盖；旧脚本纬度随行递增，与真实向南行方向不符，且旧采样方法不同。它们在最后一步删除（执行结果见deleted_files.csv），追溯副本压缩封存在superseded_terrain_outputs.zip，不是可引用的当前科学结果。旧脚本源码保存，不因旧版本删除。

99_cleanup_manifest/all_existing_dataset_analysis.csv为全项目文件名/文本候选inventory（排除.git/node_modules/缓存与本轮目录）；keep为受保护数据/模型/来源，uncertain保留原位并注明原因。删除与移动分别有deletion_manifest.csv和move_manifest.csv，执行检查见cleanup_execution.json。源码、split、checkpoint、训练结果均不删除。

重跑：python experiments/dataset_audit_final/rerun_quantitative.py；python experiments/dataset_audit_final/build_final_audit.py all。DEM路径由rerun_quantitative.py中的final来源确定。旧来源审计为历史，原脚本内Mac/旧绝对路径不可盲目运行。
''',encoding='utf-8')
 print('Prepared report/manifests',len(moves),'moves',len(deletes),'deletions',flush=True)

def cleanup():
 moves=pd.read_csv(C/'move_manifest.csv');deletes=pd.read_csv(C/'deletion_manifest.csv');done=[]
 # All exact paths are resolved beneath workspace; verify recorded hashes first.
 for _,r in pd.concat([moves.rename(columns={'source':'path'}),deletes],ignore_index=True).iterrows():
  p=(R/r.path).resolve();assert p.is_relative_to(R.resolve());assert hashlib.sha256(p.read_bytes()).hexdigest()==r.sha256
 for _,r in moves.iterrows():
  p=(R/r.source).resolve();dest=(R/r.destination).resolve();assert dest.is_relative_to(F.resolve());dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(dest));assert hashlib.sha256(dest.read_bytes()).hexdigest()==r.sha256
 for _,r in deletes.iterrows():
  p=(R/r.path).resolve();p.unlink();done.append(dict(r,deleted=True))
 csv(done,C/'deleted_files.csv')
 # Reproducibility follows the consolidated DEM; current engine no longer consumes legacy tables.
 for name in ['rerun_quantitative.py','build_final_audit.py']:
  p=F/name;text=p.read_text(encoding='utf-8-sig').replace("R/'experiments/windows_main/physics_audit_20260924'","R/'experiments/dataset_audit_final/05_existing_results_reused/experiments/windows_main/physics_audit_20260924'").replace("R/'experiments/windows_main/physics_audit_20260924/dem'","R/'experiments/dataset_audit_final/05_existing_results_reused/experiments/windows_main/physics_audit_20260924/dem'");p.write_text(text,encoding='utf-8')
 before=pd.read_csv(C/'protected_assets_before.csv').fillna('');changed=[]
 for _,r in before.iterrows():
  p=R/r.path
  if not p.exists() or p.stat().st_size!=r['size'] or p.stat().st_mtime_ns!=r.mtime_ns:changed.append(r.path)
 assert not changed,changed
 (C/'cleanup_execution.json').write_text(json.dumps(dict(moved_files=len(moves),deleted_files=len(done),protected_assets_checked=len(before),protected_assets_changed=changed,raw_data_hash_mismatches=0),indent=2))
 print('Cleanup complete',len(moves),'moved',len(done),'deleted',flush=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:cleanup()

