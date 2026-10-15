"""Record the completed HYPIR teacher NR iteration and its matched comparison."""
from pathlib import Path
import datetime, hashlib, json, os

R=Path('/cache/independent_sr_cuda_20261013')
W=Path('/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push7_20261015')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def atomic(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)

join_path=R/'audits/push7/speed/recovered_lastres_formal_cpu.json'
assert sha(join_path)=='484174d65832f41abe719d7f5b697cc07fc661e7319321275201074433b94909'
j=json.loads(join_path.read_text())
assert j['status']=='PASS_FRESH_RECOVERED_A_B_FULL_AND_4_SERIAL_TIMINGS'
a,b=j['predictions']['reference'],j['predictions']['candidate']
assert b['NR']>a['NR'] and b['FR']>a['FR'] and j['total_prediction_gain']>0
assert sha(j['quality']['path'])==j['quality']['sha256']
label='push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all'
md=R/'submissions/hypir_push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae/model_dir'
cm=md.parent/'candidate_manifest.json';assert sha(cm)=='d7e03918355aa9cb459c1d37f8ffafcee89856bb8b23cdf0346b5e440b3046cb'
c=json.loads(cm.read_text());now=datetime.datetime.now(datetime.timezone.utc).isoformat()
run=R/'adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100'
out=R/'nr_push7_recovered_full_evaluations'/label
selection_path=R/'results/nr_push7_lastres_lr5e6_selection.json'
oldp=R/'results/current_model_preference.json';oldbytes=oldp.read_bytes();old=json.loads(oldbytes)
oldcopy=R/'results/current_preference_before_lastres_lr5e6.json'
assert not oldcopy.exists()
with oldcopy.open('xb') as f:f.write(oldbytes)
record=dict(schema='hypir_teacher_lastres_nr_selection_v1',status='BEST_VALIDATED_CURRENT_MATCHED_HYPIR_NR_CANDIDATE',
    selected_label=label,selected_name='HYPIR教师末端LoRA NR微调100步',model_dir=str(md),
    checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],resume=str(run/'resume_step7850.pt'),
    resume_sha256=c['training_provenance']['resume']['sha256'],teacher_weights_changed=True,
    stage_steps=100,lr=5e-6,optimizer_tensor_count=6,autograd_tensor_count=514,unselected_exact_A_count=508,
    distillation=False,lightweight_model=False,validation_GT_training=False,teacher_output_loss=False,
    NR=b['NR'],FR=b['FR'],S_proxy=b['S_full'],event_p10_ms=b['event_p10_ms'],tool_prediction=b['original_tool_prediction'],
    same_batch_baseline=a,score_gain=j['total_prediction_gain'],quality_only_score_gain=j['quality_only_gain'],
    score_context='Fresh recovered instance; both models full103/418, same GPU0 200/30/231; preselected common min of bracketing G4 p10s',
    normalization=j['normalization'],score_target=65,score_target_achieved=b['reaches65'],official_score_claim=False,
    old_batch_score_not_directly_comparable=True,quality_reused=False,PNG_exact=False,quality_dir=str(out),render_dir=str(out/'render'),
    candidate_manifest=dict(path=str(cm),sha256=sha(cm)),full_quality_audit=j['quality'],
    formal_quality_timing_join=dict(path=str(join_path),sha256=sha(join_path)),
    visual=dict(gallery=str(W/'visuals/hypir_lastres_lr5e6_trt_full_final/index.html'),
        proof_sha256='fb090f77e7012034a359aed3cbdf0427fad46fcad390971be97dfd92c81f9105',
        root_actually_reviewed_cases=[7,16,41,83],independent_reviewer_actually_reviewed_cases=[7,16,41,83],
        finding='In four fixed native1:1 ROIs, no obvious new large geometric deformation or broad smearing; small texture, edge and colour changes remain. Not an all-image visual guarantee.'),
    backup=dict(local_engine_receipt=str(W/'backup_lastres_lr5e6_trtunetvae_verified/backup_receipt.json'),
        local_engine_receipt_sha256='76c33d62660d1bc3b7ff5eaaa8a661792ecbd6b84ec2a39fcc529872080fd7a5',
        local_checkpoint_capsule=str(W/'backup_lastres_lr5e6_nr_verified/capsule'),
        checkpoint_capsule_manifest_sha256='c58a717884a964f5146044a29aa30e481661f2a9fa2789424106ddb541b522c5',
        cache_checkpoint_capsule='/cache/push7_lastres_lr5e6_nr_capsule_20261015',
        capsule_depends_on_persistent_A_resume_sha256='e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b',SFS_new_write=False),
    historical_previous_selection=dict(path=str(oldcopy),sha256=sha(oldcopy),historical_tool_prediction=old['tool_prediction'],
        score_scope='Previous instance and previous G4 reference; preserved, not used in this comparison'),
    updated_utc=now)
write_new(selection_path,record)
pointer=dict(old)
for key in ['model_dir','checkpoint','checkpoint_sha256','resume','resume_sha256','NR','FR','S_proxy','tool_prediction','event_p10_ms',
            'quality_dir','render_dir','quality_reused','PNG_exact','distillation','lightweight_model','teacher_weights_changed','score_target','score_target_achieved']:
    pointer[key]=record[key]
pointer.update(schema='current_model_visual_preference_v3',status=record['status'],preferred_label=label,preferred_name=record['selected_name'],
    source_selection=str(selection_path),source_selection_local=str(W/'results'/selection_path.name),source_selection_sha256=sha(selection_path),
    score_context=record['score_context'],same_batch_baseline=a,score_gain=j['total_prediction_gain'],
    previous_preference=record['historical_previous_selection'],updated_utc=now)
atomic(oldp,pointer)
sp=R/'results/status_push7_20261015.json';status=json.loads(sp.read_text())
status.update(updated_utc=now,current_selected_candidate=record,active_training=False,active_GPU_jobs=False,
              iteration_status='COMPLETE_TRAIN_FULL103_VISUAL4_AND_SINGLE_GPU_TIMING',target65_reached=b['reaches65'])
status['current_iteration']['status']='complete_full103_418_visual4_and_four_fresh_serial_timings'
status['current_iteration']['full103_evaluation']='complete'
status['current_iteration']['score_estimate_claim']=True
status['preserved_best']['score_scope']='Historical previous-instance speed/reference; its full quality was reproduced exactly in the fresh current baseline. Historical score is not compared across references.'
status['fresh_full_evaluation']={}
for lab in ['push7_recovered_a7750_trtunetvae_all',label]:
    e=R/'nr_push7_recovered_full_evaluations'/lab
    status['fresh_full_evaluation'][lab]={name:json.loads((e/sub).read_text()) for name,sub in [('render','render_status.json'),('score','scores/status.json'),('result','scores/result.json')]}
atomic(sp,status)
print(json.dumps(dict(status=record['status'],selection=str(selection_path),sha256=sha(selection_path),pointer_sha256=sha(oldp),predictions=j['predictions']),ensure_ascii=False))
