"""Summarize root-planned completed Push4 full103 audits; no latency prediction."""
from pathlib import Path
import argparse,csv,datetime,hashlib,io,json,re
R=Path('/cache/independent_sr_cuda_20261013');OUT=R/'audits/push4/evaluations';REF=R/'audits/push4/evaluations/parent7250_full_cpu_audit.json'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contributions(m):return {'clipiqa':m['clipiqa'],'maniqa':m['maniqa'],'musiq_div100':m['musiq']/100,'niqe':max(0,(10-m['niqe'])/10)}
b=load(REF);assert b['status']=='PASS' and b['image_rows']==103 and b['scalar_metrics']==418 and b['official_full_count']
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',action='append',required=True);args=parser.parse_args();expected=args.label;assert 1<=len(expected)<=2 and len(set(expected))==len(expected);assert all(re.fullmatch(r'push4_[abc](7500|7750)_compiled_all',x) for x in expected);records=[];pending=[]
def flatten(a,path,is_parent=False):
 m=a['component_means'];c=contributions(m['test']);bc=contributions(b['component_means']['test']);assert abs(sum(c.values())-a['NR'])<1e-12
 return {'label':a['label'],'scope':'full100test+validation3 S_proxy; no latency/official score','image_rows':103,'scalar_metrics':418,'NR':a['NR'],'FR':a['FR'],'S_proxy':a['S_proxy'],'delta_NR':a['NR']-b['NR'],'delta_FR':a['FR']-b['FR'],'delta_S_proxy':a['S_proxy']-b['S_proxy'],'test_means':m['test'],'val_means':m['validation'],'test_NR_contributions':c,'contribution_deltas_vs7250':{k:c[k]-bc[k] for k in c},'actual_checkpoint_sha256':a['actual_checkpoint_sha256'],'cpu_audit_path':str(path),'cpu_audit_sha256':sha(path),'is_reference':is_parent}
records.append(flatten(b,REF,True))
for label in expected:
 p=OUT/label/'audit.json'
 if not p.exists():pending.append(label);continue
 proof=load(p.parent/'proof_manifest.json');assert proof['actual_raw_checkpoint_matches_completed_marker']
 a=load(p);assert a['status']=='PASS' and a['image_rows']==103 and a['scalar_metrics']==418 and a['official_full_count']
 compare=a['reference_comparison'];assert compare['same_formal_metrics_and_sources'] and compare['same_post_render_rng'] and compare['same_post_score_rng'];assert compare['reference_checkpoint_sha256']==b['actual_checkpoint_sha256']
 records.append(flatten(a,p))
records.sort(key=lambda r:r['S_proxy'],reverse=True)
x={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'all rows full100test+val3 S_proxy; no latency or official leaderboard score','status':'complete' if not pending else 'incomplete','complete_candidates':len(records)-1,'expected_candidates':len(expected),'pending':pending,'winner_provisional_until_all_planned_full_complete':bool(pending),'rows':records,'summary_source_sha256':sha(__file__)}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'fair_full_table.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
fields=['label','NR','FR','S_proxy','delta_NR','delta_FR','delta_S_proxy','actual_checkpoint_sha256'];s=io.StringIO();w=csv.DictWriter(s,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records);(OUT/'fair_full_table.csv').write_text(s.getvalue())
md=['# 第四轮完整 103 张公平比较','',f'已审计候选 {len(records)-1}/{len(expected)}；各行均为完整 test100 + val3（103 张／418 指标），S_proxy = NR(test100)+FR(val3)，不包含延迟或官方榜单分数。','', '| 版本 | NR | FR | S_proxy | 相对7250变化 |','|---|---:|---:|---:|---:|']
md += [f"| {r['label']} | {r['NR']:.8f} | {r['FR']:.8f} | {r['S_proxy']:.8f} | {r['delta_S_proxy']:+.8f} |" for r in records]
if pending:md+=['','尚未完成审计：'+', '.join(pending)]
(OUT/'fair_full_table.md').write_text('\n'.join(md)+'\n');print(json.dumps({'status':x['status'],'complete_candidates':len(records)-1,'best_S_proxy':records[0]['S_proxy'],'best_label':records[0]['label'],'pending':pending},ensure_ascii=False))
