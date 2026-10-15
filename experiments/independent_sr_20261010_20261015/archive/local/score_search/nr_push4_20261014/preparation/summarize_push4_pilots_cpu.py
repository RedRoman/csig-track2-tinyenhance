"""Summarize completed independent CPU pilot audits; never reports a full score."""
from pathlib import Path
import csv,datetime,hashlib,io,json
R=Path('/cache/independent_sr_cuda_20261013');OUT=R/'audits/push4/evaluations';REF=R/'audits/push4/evaluations/parent7250_pilot_cpu_audit.json'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contributions(m):return {'clipiqa':m['clipiqa'],'maniqa':m['maniqa'],'musiq_div100':m['musiq']/100,'niqe':max(0,(10-m['niqe'])/10)}
b=load(REF);assert b['status']=='PASS' and b['image_rows']==23 and b['scalar_metrics']==98 and not b['official_full_count']
expected=[f'push4_{k}{n}_compiled_pilot' for n in [7500,7750] for k in 'abc'];records=[];pending=[]
def flatten(a,path,is_parent=False):
 m=a['component_means'];c=contributions(m['test']);bc=contributions(b['component_means']['test']);assert abs(sum(c.values())-a['NR'])<1e-12
 return {'label':a['label'],'scope':'fixed20+validation3 S_subset only','image_rows':23,'scalar_metrics':98,'NR':a['NR'],'FR':a['FR'],'S_subset':a['S_subset'],'delta_NR':a['NR']-b['NR'],'delta_FR':a['FR']-b['FR'],'delta_S_subset':a['S_subset']-b['S_subset'],'test_means':m['test'],'val_means':m['validation'],'test_NR_contributions':c,'contribution_deltas_vs7250':{k:c[k]-bc[k] for k in c},'actual_checkpoint_sha256':a['actual_checkpoint_sha256'],'cpu_audit_path':str(path),'cpu_audit_sha256':sha(path),'is_reference':is_parent}
records.append(flatten(b,REF,True))
for label in expected:
 p=OUT/label/'audit.json'
 if not p.exists():pending.append(label);continue
 proof=load(p.parent/'proof_manifest.json');assert proof['actual_raw_checkpoint_matches_completed_marker']
 a=load(p);assert a['status']=='PASS' and a['image_rows']==23 and a['scalar_metrics']==98 and not a['official_full_count']
 compare=a['reference_comparison'];assert compare['same_formal_metrics_and_sources'] and compare['same_post_render_rng'] and compare['same_post_score_rng'];assert compare['reference_checkpoint_sha256']==b['actual_checkpoint_sha256']
 records.append(flatten(a,p))
records.sort(key=lambda r:r['S_subset'],reverse=True)
x={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'all rows fixed20test + validation3; only S_subset, not full103 S_proxy or official score','status':'complete' if not pending else 'incomplete','complete_candidates':len(records)-1,'expected_candidates':6,'pending':pending,'winner_provisional_until_full103':True,'rows':records,'summary_source_sha256':sha(__file__)}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'fair_pilot_table.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
fields=['label','NR','FR','S_subset','delta_NR','delta_FR','delta_S_subset','actual_checkpoint_sha256'];s=io.StringIO();w=csv.DictWriter(s,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records);(OUT/'fair_pilot_table.csv').write_text(s.getvalue())
md=['# 第四轮 Pilot 公平比较','',f'已审计候选 {len(records)-1}/6；各行均为固定 test20 + val3（23 张／98 指标），**仅 S_subset**，需要完整 103 张复核后才能决定正式 S_proxy。','', '| 版本 | NR | FR | S_subset | 相对7250变化 |','|---|---:|---:|---:|---:|']
md += [f"| {r['label']} | {r['NR']:.8f} | {r['FR']:.8f} | {r['S_subset']:.8f} | {r['delta_S_subset']:+.8f} |" for r in records]
if pending:md+=['','尚未完成审计：'+', '.join(pending)]
(OUT/'fair_pilot_table.md').write_text('\n'.join(md)+'\n');print(json.dumps({'status':x['status'],'complete_candidates':len(records)-1,'best_S_subset':records[0]['S_subset'],'best_label':records[0]['label'],'pending':pending},ensure_ascii=False))
