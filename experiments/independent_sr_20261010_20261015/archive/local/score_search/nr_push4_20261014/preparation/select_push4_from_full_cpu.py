"""CPU-only audited full103 winner selection; immutable final selection file."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,re
R=Path('/cache/independent_sr_cuda_20261013');D=R/'audits/push4/evaluations'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
records=[]
for label in ['push4_c7750_compiled_all','push4_a7750_compiled_all']:
 step=7750;path=D/label/'audit.json';audit=read(path);assert audit['status']=='PASS' and audit['image_rows']==103 and audit['scalar_metrics']==418 and audit['official_full_count']
 assert audit['label']==label and audit['actual_png_manifest_scored_sha']=='PASS103' and audit['actual_source_code_sha']=='PASS'
 comp=audit['reference_comparison'];assert comp['same_formal_metrics_and_sources'] and comp['same_post_render_rng'] and comp['same_post_score_rng']
 q=R/'nr_push4_full_evaluations'/label;r=read(q/'scores/result.json');assert r['status']=='complete' and r['completed']==r['expected']==103 and r['official_full_count']
 assert read(q/'render_status.json')['status']=='complete'
 for key,actual in [('NR',r['test']['nr']),('FR',r['validation']['fr']),('S_proxy',r['S_proxy'])]:assert abs(audit[key]-actual)<1e-12
 records.append({'label':label,'global_step':step,'NR':audit['NR'],'FR':audit['FR'],'S_proxy':audit['S_proxy'],'checkpoint_sha256':audit['actual_checkpoint_sha256'],'full_audit':str(path),'full_audit_sha256':sha(path)})
records.sort(key=lambda x:x['S_proxy'],reverse=True);win=records[0];
parent_check=read(D/'parent7250_full_cpu_audit.json');assert parent_check['status']=='PASS' and parent_check['image_rows']==103 and parent_check['scalar_metrics']==418 and parent_check['actual_checkpoint_sha256']=='fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e'
if win['label']!='push4_c7750_compiled_all' or win['S_proxy']<=parent_check['S_proxy']:
 print(json.dumps({'status':'needs_root_action_no_selection_written','best':win,'parent_S_proxy':parent_check['S_proxy'],'reason':'C recipe cannot be applied if A wins or no new candidate beats parent'},ensure_ascii=False));raise SystemExit(3)
table_path=D/'fair_full_table.json';table=read(table_path);assert table['status']=='complete' and table['complete_candidates']==table['expected_candidates']==2;assert set(x['label'] for x in table['rows'] if not x['is_reference'])=={x['label'] for x in records}
global_step=win['global_step'];local=global_step-7250;q=R/'nr_push4_full_evaluations'/win['label'];rp=read(q/'render_protocol.json')['protocol'];checkpoint=Path(rp['runner']['weight_path']);run=checkpoint.parent;resume=run/f'resume_step{global_step}.pt';config=run/'config.json';training=D.parent/'training_stage5'/f'milestone{local}_cpu_audit.json';train=read(training)
assert train['status']=='PASS' and train['CPU_only'] is True and train['CUDA_context_initialized'] is False
rows=[x for x in train['branches'] if x['run']==run.name and x['audited_global_step']==global_step];assert len(rows)==1;row=rows[0]
assert row['status']=='PASS' and row['audited_stage_local_step']==local and row['audited_Adam_step']==1250+local
assert checkpoint.name==f'step{global_step}.pth' and sha(checkpoint)==win['checkpoint_sha256']
for path in [checkpoint,resume,config]:assert sha(path)==row['file_records'][path.name]['sha256']
marker=read(run/f'checkpoint_step{global_step}.json');st=resume.stat();assert (marker['local_step'],marker['global_step'],marker['optimizer_step'])==(local,global_step,1250+local)
assert (marker['resume_bytes'],marker['resume_mtime_ns'],marker['resume_inode'])==(st.st_size,st.st_mtime_ns,st.st_ino)
assert marker['raw_checkpoint_sha256']==win['checkpoint_sha256'] and marker['resume_checkpoint_sha256']==row['file_records'][resume.name]['sha256']
parent=read(D/'parent7250_full_cpu_audit.json');assert win['NR']>parent['NR'] and win['S_proxy']>parent['S_proxy']
selection={'schema':'nr_push4_selection_v1','status':'selected_after_independent_full_audit','selected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':win['label'],'NR':win['NR'],'FR':win['FR'],'S_proxy':win['S_proxy'],'local_step':local,'global_step':global_step,'optimizer_step':1250+local,'checkpoint':str(checkpoint),'checkpoint_sha256':win['checkpoint_sha256'],'resume':str(resume),'resume_sha256':row['file_records'][resume.name]['sha256'],'config':str(config),'config_sha256':sha(config),'model_dir':rp['model_dir'],'quality_dir':str(q),'full_audit':win['full_audit'],'full_audit_sha256':win['full_audit_sha256'],'training_audit':str(training),'training_audit_sha256':sha(training),'renderer_source':str(R/'code/render_cuda_submission.py'),'renderer_source_sha256':rp['source_sha256'],'runner_source_sha256':rp['runner']['runner_sha256'],'all_required_full_candidates_audited':records,'selection_basis':'maximum original-tool S_proxy among both root-planned independently audited full100test+val3 candidates; unchanged formal scoring; no latency or official prediction','parent_full_audit':str(D/'parent7250_full_cpu_audit.json'),'parent_full_audit_sha256':sha(D/'parent7250_full_cpu_audit.json'),'delta_NR_vs7250_full':win['NR']-parent['NR'],'delta_FR_vs7250_full':win['FR']-parent['FR'],'delta_S_proxy_vs7250_full':win['S_proxy']-parent['S_proxy'],'full_comparison_table':str(table_path),'full_comparison_table_sha256':sha(table_path),'raw_recipe_dir':str(D.parent/'raw_recovery'/win['label']),'raw_recipe_restore_proof':str(D.parent/'raw_recovery'/win['label']/'probe_result.json'),'raw_recipe_restore_proof_sha256':sha(D.parent/'raw_recovery'/win['label']/'probe_result.json'),'selector_source_sha256':sha(__file__),'CPU_only':True,'CUDA_context_initialized':False}
# Use the backup consumer's exact independent audit gate before publishing.
spec=importlib.util.spec_from_file_location('backup_gate',R/'code/push4/backup_selected_push4.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.audit_gates(selection);recipe_proof,_=mod.exact_raw_recipe_proof(selection,resume);assert recipe_proof['status']=='PASS' and recipe_proof['raw_sha256']==selection['checkpoint_sha256']
target=R/'results/nr_push4_selection.json'
if a.execute:
 if target.exists():
  old=read(target);assert all(old.get(k)==v for k,v in selection.items() if k!='selected_utc'),'Immutable existing final selection differs'
 else:
  target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,target)
 print(json.dumps({'status':'selected_after_independent_full_audit','label':selection['label'],'NR':selection['NR'],'FR':selection['FR'],'S_proxy':selection['S_proxy'],'selection_path':str(target),'selection_sha256':sha(target),'checkpoint_sha256':selection['checkpoint_sha256'],'resume_sha256':selection['resume_sha256'],'full_audit_sha256':selection['full_audit_sha256'],'training_audit_sha256':selection['training_audit_sha256'],'raw_recipe_restore_proof_sha256':selection['raw_recipe_restore_proof_sha256']},ensure_ascii=False))
else:print(json.dumps(selection,ensure_ascii=False,indent=2))
