"""Read-only stage3 parent6250-anchor metadata/log/source audit; no Torch import or GPU calls."""
from pathlib import Path
import datetime, hashlib, json, math

ROOT=Path('/cache/independent_sr_cuda_20261013')
RUNS=['cuda_nr_stage3_s6250_anchor6250_m1_250','cuda_nr_stage3_s6250_anchor6250_m15_250']
V1_SHA='0998967aa6052ec9a5090219e17cb73c05468b06d7acc1eae27108c1c412fb4b'
V2_SHA='9ac86a8d78ba0fc085b60278a460ecaf1ffef74d0dd4e21c4b5c5d84063c4651'
V3_SHA='539844657210938295aa55eb71a2620bad40b974a9ae48ad7238ae7d70fa7c2c'
HELPER_SHA='13e029954c386a54e36685954066c16715b0e0f820109d55f6591da0dd723812'
PARENT_CFG_SHA='2a459e95e248d1b502b651753b9598ce7adfb92c24e48863458265458ea1ca5a'
PARENT_ID='fa5ea19179561aac206b26f7dd3596406f1ad376dccd38f49821146623b98d52'
PARENT_RESUME_SHA='3dcf896028f41c7c4c6e15074cc69c4cdb2ff7679e6756f5ee57a9a657f61c1f'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def normalized(protocol):
 p=json.loads(json.dumps(protocol))
 p.pop('musiq_weight')
 p['parent_lineage'].pop('musiq_weight')
 return p

def audit():
 results=[];protocols=[];sequences=[]
 parent=ROOT/'adaptation/runs/cuda_nr_s6000_ct175_conservative_pilot100'
 assert sha(parent/'config.json')==PARENT_CFG_SHA
 assert sha(ROOT/'code/cuda_training/train_cuda_nr.py')==V1_SHA
 assert sha(ROOT/'code/cuda_training/train_cuda_nr_stage_v2.py')==V2_SHA
 assert sha(ROOT/'code/cuda_training/train_cuda_nr_stage_v3.py')==V3_SHA
 assert sha(ROOT/'code/cuda_training/png_quantization_surrogate.py')==HELPER_SHA
 for idx,name in enumerate(RUNS):
  d=ROOT/'adaptation/runs'/name
  cfg=json.loads((d/'config.json').read_text());p=cfg['protocol'];protocols.append(p)
  assert hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()==cfg['protocol_id']
  assert cfg['target_stage_steps']==250 and cfg['target_global_step']==6500 and cfg['target_optimizer_step']==500
  assert p['nr_round_ste'] is False and p['quantization_helper_sha256']==HELPER_SHA
  assert p['musiq_weight']==[1.0,1.5][idx]
  assert p['schema']=='cuda_raw_hypir_nr_stage_v3_parentanchor'
  assert p['anchor_sha256']==PARENT_RESUME_SHA and p['teacher_FP32_exact_parent6250'] and p['teacher_frozen']
  assert p['stage_start_global_step']==6250 and p['parent_optimizer_step']==250
  lineage=p['parent_lineage']
  assert lineage['parent_resume_sha256']==PARENT_RESUME_SHA
  assert lineage['parent_config_sha256']==PARENT_CFG_SHA and lineage['parent_protocol_id']==PARENT_ID
  for path,value in p['source_sha256'].items():assert sha(path)==value
  rows=[json.loads(x) for x in (d/'train.jsonl').read_text().splitlines() if x.strip()]
  assert len(rows)==250
  assert [x['local_step'] for x in rows]==list(range(1,251))
  assert [x['global_step'] for x in rows]==list(range(6251,6501))
  assert [x['optimizer_step'] for x in rows]==list(range(251,501))
  assert all(x['nonzero_lora_gradients']==514 for x in rows)
  assert all(math.isfinite(x[k]) for x in rows for k in ['loss','reward','fidelity','gradient_norm'])
  sequence=[(x['case'],x['crop_xy']) for x in rows];sequences.append(sequence)
  status=json.loads((d/'status.json').read_text())
  assert status['stage']=='complete' and status['completed_local_steps']==250
  assert status['global_step']==6500 and status['optimizer_step']==500
  marker=d/'training_complete_step6500';assert marker.is_file()
  grad=json.loads((d/'gradient_preflight.json').read_text())
  assert grad['status']=='passed' and grad['no_optimizer_update_yet'] is True
  assert grad['frozen_metrics'] and grad['frozen_same_latent_anchor']
  assert grad['teacher_FP32_exact_parent6250'] and grad['initial_fidelity_expected_zero']
  assert grad['row']['fidelity']<=1e-6 and grad['musiq_weight']==[1.0,1.5][idx]
  assert set(grad['components'])=={'musiq','maniqa','clipiqa'}
  assert all(v['input_gradient_finite'] and v['input_gradient_norm']>0 for v in grad['components'].values())
  assert grad['row']['global_step']==6251 and grad['row']['optimizer_step']==251
  migration=json.loads((d/'parent_migration.json').read_text())
  assert migration['protocol_id']==cfg['protocol_id'] and migration['parent_lineage']==lineage
  assert migration['restored_stage_local_step']==0 and migration['restored_global_step']==6250
  assert migration['restored_optimizer_step']==250 and migration['FP32_LoRA_exact_equal_before_update'] is True
  assert migration['FP32_LoRA_tensors']==514
  assert migration['teacher_FP32_exact_parent6250'] and migration['teacher_FP32_tensors']==514 and migration['teacher_frozen']
  assert migration['anchor_resume_sha256']==PARENT_RESUME_SHA and migration['musiq_weight']==[1.0,1.5][idx]
  files={f:{'bytes':(d/f).stat().st_size,'sha256':sha(d/f)} for f in ['config.json','status.json','train.jsonl','gradient_preflight.json','parent_migration.json','maniqa_initial_state_proof.json','training_complete_step6500']}
  results.append({'run':name,'status':'PASS','protocol_id':cfg['protocol_id'],'nr_round_ste':False,'musiq_weight':[1.0,1.5][idx],
    'image_training_rows':250,'local_steps':[1,250],'global_steps':[6251,6500],'Adam_update_labels':[251,500],
    'all250_logged_LoRA_gradients_nonzero_514':True,'first_three_real_NR_input_gradients':grad['components'],
    'teacher_metrics_gradient_guards_completed':True,'FP32_parent_restore_proof':migration,
    'case_crop_sequence_sha256':hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
    'elapsed_training_seconds':rows[-1]['elapsed_seconds'],'median_step_seconds':sorted(x['step_seconds'] for x in rows)[125],
    'peak_allocated_bytes':max(x['gpu_peak_allocated_bytes'] for x in rows),
    'first_row':rows[0],'last_row':rows[-1],'small_files':files})
 assert sequences[0]==sequences[1]
 assert normalized(protocols[0])==normalized(protocols[1])
 return {'status':'PASS','audited_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  'CPU_only':True,'Torch_imported':False,'GPU_calls':False,
  'parent_config_v1_v2_v3_helper_source_actual_SHA_unchanged':True,
  'all250_cases_and_crops_equal':True,'protocols_differ_only_MUSIQ_reward_weight':True,
  'parent_resume_actual_SHA_checked_by_parameter_audit':PARENT_RESUME_SHA,
  'results':results}

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
