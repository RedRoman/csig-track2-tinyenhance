"""CPU-only fresh pixG5i full103/418 audit; original baseline and real distilled stages.
No GPU/scoring/latency prediction; unchanged original CM/MANIQA100/mean NIQE clamp.
"""
import argparse,hashlib,json,math,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013')
E=R/'experiments/pixg5i_hypir_distill_20261015'
ROOT=R/'nr_pixg5i_distill_full_evaluations'
ORDER=[('validation',x) for x in range(1,4)]+[('test',x) for x in range(1,101)]
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
TEACHER='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'
ADAPTER='70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
REFERENCE='6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'
PIX='2f1cd3cd8311381e432138f489b01b982e08ebd5b09ecbc05ac87f7bfb577923'
CM_SHA='1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
SCORE_SHA='20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074'
RENDER_SHA='1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
RESTORE_SHA='dbab8b00ca223d101f61a5154d1a161256afe9d367ba18da74de6c1360cd5947'
SEEN = {}

def need(ok, message):
    if not ok: raise RuntimeError(message)

def stamp(p):
    s = Path(p).stat(); return (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)

def sha(p):
    p = Path(p).resolve(); old = stamp(p)
    if p in SEEN:
        need(SEEN[p][0] == old, 'File changed: '+str(p)); return SEEN[p][1]
    h = hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda:f.read(8*1024**2), b''): h.update(v)
    need(stamp(p) == old, 'File changed during read: '+str(p)); SEEN[p] = (old,h.hexdigest()); return h.hexdigest()

def read(p):
    sha(p); return json.loads(Path(p).read_text())

def ref(p):
    return dict(path=str(Path(p).resolve()),sha256=sha(p))

def close(a,b):
    return abs(float(a)-float(b)) <= 1e-12

def nr(m):
    return m['clipiqa']+m['maniqa']+m['musiq']/100+max(0.,(10-m['niqe'])/10)

def fr(m):
    return (1-m['lpips'])+(1-m['dists'])

def normalize(x):
    if torch.is_tensor(x):
        a=x.detach().cpu().contiguous().numpy(); return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(x,np.ndarray):
        a=np.ascontiguousarray(x);return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [normalize(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x

def digest(x):
    return hashlib.sha256(json.dumps(normalize(x),sort_keys=True).encode()).hexdigest()

def score_invariant(p, matched=False):
    d=json.loads(json.dumps(p))
    for row in d['maniqa_offline_init_proof']['before_init_records']:row.pop('before_original_checkpoint_load',None)
    if matched:
        for k in ['label','render_dir']:d.pop(k)
        d['cuda'].pop('physical_visible_devices',None)
    return d

def pix_binding(md,rr,evaluation_label):
    cm_path=md.parent/'candidate_manifest.json';cm=read(cm_path);b=read(md/'pixg5i_binding.json')
    need(cm['schema']=='pixg5i_registered_candidate_v1' and cm['model_type']=='supplied_pixG5i_pixel_UNet' and
         Path(cm['model_dir']).resolve()==md.resolve() and cm['label']==evaluation_label.removesuffix('_all')==b['label'] and
         cm['stage']==b['stage'] and cm['initial_package_checkpoint_sha256']==b['initial_package_checkpoint_sha256']==INITIAL and
         cm['used_as_G4_reference'] is False and b['used_as_G4_reference'] is False and cm['quality_reuse_claim'] is False,
         'Actual original pixG5i family/stage binding differs')
    need(sha(md/'runner.py')==cm['runner_sha256']==rr['runner_sha256']==ADAPTER and
         sha(md/'reference_runner.py')==REFERENCE and sha(md/'pix_model.py')==PIX and
         sha(md/'config.yaml')=='e13bab32185ac95f07ac36133eff89133a33ed90cc5ca534b7d8496bc457250d',
         'Actual original inference implementation changed')
    need(cm['checkpoint_sha256']==b['checkpoint_sha256']==rr['checkpoint_sha256']==sha(md/'model.pth')==sha(cm['checkpoint']) and
         Path(rr['weight_path']).resolve()==(md/'model.pth').resolve() and
         cm['binding']['sha256']==sha(md/'pixg5i_binding.json'),'Actual current weights/binding differ')
    need({p.name for p in md.iterdir() if p.is_file()}==set(cm['files']) and
         all(sha(md/n)==x['sha256'] and Path(x['path']).resolve()==(md/n).resolve() for n,x in cm['files'].items()),
         'Registered model files changed')
    for name,h in rr['source_sha256'].items():need(sha(md/name)==h,'Rendered pix source/binding changed')
    need(rr['schema']=='pixg5i_original_inference_metadata_adapter_v1' and rr['stage']==b['stage'] and
         rr['used_as_G4_reference'] is False and rr['quality_reuse_claim'] is False and
         rr['seed']=='231 before initialization; continuous inference RNG' and
         rr['training_provenance']==b['training_provenance']==cm['training_provenance'],'Actual inference metadata differs')
    raw=torch.load(md/'model.pth',map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(R/'source_payload/pixg5i_original/model_dir/model.pth',map_location='cpu',weights_only=False,mmap=True)
    need(sha(R/'source_payload/pixg5i_original/model_dir/model.pth')==INITIAL and raw['cfg']==initial['cfg'] and
         list(raw['state'])==list(initial['state']) and len(raw['state'])==112 and sum(x.numel() for x in raw['state'].values())==3786764,
         'Original pixG5i cfg/state architecture differs')
    for key,x in raw['state'].items():
        need(x.dtype==torch.float32 and x.shape==initial['state'][key].shape and bool(torch.isfinite(x).all()),'Invalid raw FP32 state')
    changed=sum(not torch.equal(v,initial['state'][k]) for k,v in raw['state'].items())
    if b['stage']=='original_baseline':
        need(cm['checkpoint_sha256']==INITIAL and b['training_provenance'] is None and changed==0 and
             b['teacher']['applied_to_this_checkpoint'] is False and
             sha(cm_path)=='f47a1b5318be533c221d612a2baf99d90ee49267d4728e50bfb8e42b395bfbff','Baseline is not the unchanged original')
    else:
        need(b['stage']=='distilled_checkpoint' and cm['checkpoint_sha256']!=INITIAL and changed>0 and
             b['teacher']['raw_sha256']==TEACHER and b['teacher']['applied_to_this_checkpoint'] is True,
             'Real distillation/current checkpoint/fixed teacher required')
        t=b['training_provenance']
        for key in ['config','marker','input_catalog','resume','training_source']:
            need(sha(t[key]['path'])==t[key]['sha256'],'Actual training reference changed: '+key)
        marker=read(t['marker']['path'])
        need(marker['raw_sha256']==cm['checkpoint_sha256'] and marker['resume_sha256']==t['resume']['sha256'] and
             sha(marker['raw_checkpoint'])==cm['checkpoint_sha256'] and
             Path(marker['resume_checkpoint']).resolve()==Path(t['resume']['path']).resolve() and
             marker['local_step']==marker['optimizer_step']==raw['step'] and marker['reference_step']==30000 and
             marker['state_tensors']==112 and marker['parameter_elements']==3786764 and
             marker['changed_FP32_state_tensors']==changed and marker['no_val_GT_training'] is True and
             marker['teacher_png_scope']=='test100' and marker['protocol_id']==raw['protocol_id'],
             'Actual marker/raw/resume-file identity/counters/noGT/test100 binding differs')
        origin=raw['origin'];need(origin['model_family']=='PixG5i' and origin['initial_package_checkpoint_sha256']==INITIAL and
             origin['teacher_checkpoint_sha256']==TEACHER and origin['new_training'] is True,'Actual raw origin differs')
    return dict(candidate_manifest=ref(cm_path),binding=ref(md/'pixg5i_binding.json'),stage=b['stage'],
        checkpoint_sha256=cm['checkpoint_sha256'],initial_package_checkpoint_sha256=INITIAL,
        raw_tensor_count=112,parameter_elements=3786764,actual_changed_FP32_state_tensors=changed,
        actual_raw_FP32_finite_and_architecture_verified=True,training_provenance=cm['training_provenance'],teacher=b['teacher'],
        resume_model_state_or_Adam_examined_here=False,teacher_training_PNG_bytes_rehashed_here=False,
        training_lineage_scope='Actual raw tensor architecture/dtype/finite/changed count, immutable file refs and marker counters/origin only; separate training audit is required for optimizer/sampler/teacher-data claims')

def audit(e,restore,deploy,env):
    label=e.name
    result=read(e/'scores/result.json'); rs=read(e/'render_status.json')
    need(result['status']=='complete' and result['completed']==result['expected']==103 and
         result['official_full_count'] is True and rs['status']=='complete' and rs['completed']==rs['total']==103 and
         (e/'render_complete').exists(), 'Actual renderer/scorer not fully complete: '+label)
    mf=read(e/'render_manifest.json'); rp=read(e/'render_protocol.json'); p=mf['protocol']; rr=p['runner']
    need(mf['protocol_id']==rp['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() and rp['protocol']==p,
         'Render protocol/manifest identity mismatch')
    md=Path(p['model_dir']);lineage=pix_binding(md,rr,label)
    need(p['source_sha256']==RENDER_SHA and p['cases']=='all' and p['warmup'] is False and
         p['inference_rng']=='seed231 before model initialization; continuous ordered val3 then tests',
         'Original full renderer/seed/domain differs')
    need(p['model_dir_sha256']=={str(x.relative_to(md)):sha(x) for x in sorted(md.rglob('*')) if x.is_file()},
         'Rendered model directory bytes changed')
    sp=read(e/'scores/protocol.json');mp=sp['maniqa_offline_init_proof']
    need(sp['label']==label and Path(sp['render_dir'])==e/'render' and sp['sequence']==[list(x) for x in ORDER] and
         sp['seed']==1234 and sp['script_sha256']==SCORE_SHA and sp['original_tool_deployment_sha256']==sha(R/'scoring_tool/deployment_manifest.json') and
         sp['environment_protocol_sha256']==sha(R/'calibration/environment_protocol.json') and sp['environment_versions']==env['versions'], 'Actual score protocol/config differs')
    need(sp['metric_order']==dict(validation=['lpips','dists','niqe','musiq','maniqa','clipiqa'],test=['niqe','musiq','maniqa','clipiqa']) and
         sp['cuda']['name']=='NVIDIA A100-SXM4-80GB' and sp['cuda']['logical_device']==0 and
         mp['test_sample']==100 and mp['net_training'] is False and mp['all_final_parameters_and_buffers_equal_checkpoint'] is True and
         not mp['missing_keys'] and not mp['unexpected_keys'] and not mp['different_values_or_dtype_or_shape'],'Original MANIQA100/metrics contract differs')
    for row in sp['original_sources']:need(sha(R/'scoring_tool'/row['path'])==row['sha256'],'Actual original tool source differs')
    need(sp['original_sources']==[x for x in deploy['files'] if x['path'].endswith('.py')],'Original source list differs')
    for path,h in [(mp['adapter_path'],mp['adapter_sha256']),(sp['numpy_legacy_compat']['path'],sp['numpy_legacy_compat']['sha256'])]:need(sha(path)==h,'Actual metric adapter changed')
    rows_path=e/'scores/measurements.jsonl';sha(rows_path);rows=[json.loads(x) for x in rows_path.read_text().splitlines() if x]
    need(len(rows)==len(p['tasks'])==len(mf['records'])==103,'Actual row/task/render count differs')
    expected_png=set(); records=[]; scalar_count=0
    for i,(row,task,render,order) in enumerate(zip(rows,p['tasks'],mf['records'],ORDER)):
        split,case=order;rsplit='val' if split=='validation' else 'test'
        need((row['split'],row['case'])==order and row['label']==label and
             (task['split'],task['case'])==(rsplit,case)==(render['split'],render['case']) and render['task_index']==i,'Wrong image order')
        src=Path(task['input']); png=e/f'render/{rsplit}/case{case}.png';expected_png.add(png)
        need(Path(row['sr_path'])==png and render['output']==str(png.relative_to(e)) and
             sha(png)==row['sr_sha256']==render['sha256'] and sha(src)==task['input_sha256'],'Actual PNG/input SHA mismatch')
        with Image.open(png) as im:need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==task['size']==render['size'],'Native canonical PNG format/shape differs')
        need(row['size_hw']==list(reversed(task['size'])) and row['seed_scope']=='split_seed1234_continuous_rng','Image geometry/RNG row differs')
        gt=None
        if split=='validation':
            gt=R/'source_payload/scoring_assets/data/验证集'/f'case{case}_gt.{"jpg" if case==3 else "png"}'
            need(sha(gt)==row['gt_sha256'],'Actual validation GT SHA differs')
        else:need(row['gt_sha256'] is None,'Test must not use GT')
        keys=['lpips','dists','niqe','musiq','maniqa','clipiqa'] if split=='validation' else ['niqe','musiq','maniqa','clipiqa']
        need(list(row['metrics'])==keys and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row['metrics'].values()),'Wrong/nonfinite metrics')
        scalar_count+=len(keys);need(close(row['nr_per_image'],nr(row['metrics'])),'Per-image NR bookkeeping differs')
        records.append(dict(split=split,case=case,input=str(src),input_sha256=task['input_sha256'],PNG_sha256=row['sr_sha256'],
            GT_sha256=row['gt_sha256'],size=task['size'],metrics=row['metrics']))
    need(scalar_count==418 and set((e/'render').rglob('*.png'))==expected_png,'Wrong418 scalar count/extra or missing PNG')
    val={k:float(np.mean([x['metrics'][k] for x in rows[:3]])) for k in rows[0]['metrics']}
    test={k:float(np.mean([x['metrics'][k] for x in rows[3:]])) for k in rows[3]['metrics']}
    N=nr(test);F=fr(val);S=N+F
    for split,means in [('validation',val),('test',test)]:
        need(result[split]['n']==(3 if split=='validation' else 100) and all(close(v,result[split][k]) for k,v in means.items()),'Published means differ')
    need(close(result['validation']['fr'],F) and close(result['validation']['nr_val'],nr(val)) and
         close(result['validation']['score'],F+nr(val)) and close(result['test']['nr'],N) and close(result['S_proxy'],S),'Published original-formula aggregation differs')
    rstate=torch.load(e/'render_progress.pt',map_location='cpu',weights_only=False);sstate=torch.load(e/'scores/rng_checkpoint.pt',map_location='cpu',weights_only=False)
    need(rstate['protocol_id']==mf['protocol_id'] and rstate['records']==mf['records'] and rstate['checkpoint_sha256']==lineage['checkpoint_sha256'] and
         rstate['input_sha256']==[t['input_sha256'] for t in p['tasks']],'Final render state differs')
    need(sstate['rows']==rows and sstate['active_split']=='test' and
         sstate['fingerprint']==hashlib.sha256(json.dumps(score_invariant(sp),sort_keys=True).encode()).hexdigest(),'Final score state differs')
    return dict(label=label,scope='fresh full100test_NR_plus3validation_FR; S_full only',NR=N,FR=F,S_full=S,
        test_means=test,validation_means=val,NR_contributions=dict(CLIPIQA=test['clipiqa'],MANIQA=test['maniqa'],MUSIQ_over100=test['musiq']/100,NIQE_clamped=max(0.,(10-test['niqe'])/10)),
        records=records,registration_lineage=lineage,render_manifest=ref(e/'render_manifest.json'),score_result=ref(e/'scores/result.json'),measurements=ref(rows_path),
        pipeline=rr,render_GPU_name=p['gpu'],render_physical_GPU_not_recorded_by_original_renderer=True,score_CUDA=sp['cuda'],
        metric_environment=sp['environment_versions'],metric_environment_protocol_sha256=sp['environment_protocol_sha256'],
        render_seed231_continuous=True,score_seed1234_once_per_split=True,score_fingerprint=sstate['fingerprint'],
        terminal_render_fourRNG_sha256=digest({k:rstate[k] for k in ['python_rng','numpy_rng','torch_rng','device_rng']}),
        terminal_score_fourRNG_sha256=digest(sstate['rng']),score_comparison_invariant_sha256=digest(score_invariant(sp,True)),model_dir=str(md))

def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--evaluation-dir',type=Path,action='append')
    q.add_argument('--out',type=Path,required=True);q.add_argument('--execute',action='store_true');a=q.parse_args()
    evaluations=[x.resolve() for x in (a.evaluation_dir or [ROOT/'pixg5i_original_20261015_all'])]
    need(evaluations[0]==ROOT/'pixg5i_original_20261015_all' and len(evaluations) in [1,2,3] and
         len(set(evaluations))==len(evaluations) and all(x.parent==ROOT for x in evaluations),
         'Original baseline first, then at most two actual distilled stages in this evaluation root')
    out=a.out.resolve();need(out.is_relative_to(R/'audits/pixg5i_distill') and not out.exists(),'Use a new immutable pixG5i proof')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',evaluations=[str(x) for x in evaluations],out=str(out),
             actual_images_per_model=103,actual_scalars_per_model=418,MANIQA_crops=100,render_seed=231,score_seed=1234,GPU=False)));return
    global torch,np,Image
    import torch,numpy as np
    from PIL import Image
    torch.set_num_threads(4);Image.MAX_IMAGE_PIXELS=None;need(not torch.cuda.is_initialized(),'CPU auditor initialized CUDA')
    proof=R/'restoration_20261015/tool_assets_restore_receipt.json';need(sha(proof)==RESTORE_SHA,'Restored original metric assets proof changed')
    restore=read(proof);deploy=read(R/'scoring_tool/deployment_manifest.json');env=read(R/'calibration/environment_protocol.json')
    need(restore['status']=='PASS_ORIGINAL_TOOL_ASSETS_RESTORED' and restore['original_metric_weight_files']==8 and
         sha(R/'scoring_tool/deployment_manifest.json')==restore['deployment_manifest_sha256'],'Wrong original metric provenance')
    for x in restore['actual_deployment_files']:
        if '/weights/' in x['path']:
            p=Path(x['path']);need(p.stat().st_size==x['bytes'] and str(p.resolve())==x['resolved_path'],'Restored metric weight path/bytes changed')
    need(sha(R/'scoring_tool/csig_metrics.py')==CM_SHA and sha(R/'code/score_cuda_stream.py')==SCORE_SHA and
         sha(R/'code/render_cuda_submission.py')==RENDER_SHA,'Original metric/render/score source changed')
    results=[audit(e,restore,deploy,env) for e in evaluations];parent=results[0]
    need(parent['registration_lineage']['stage']=='original_baseline','Original baseline required')
    for x in results[1:]:
        need(x['registration_lineage']['stage']=='distilled_checkpoint','Candidate must be actual distilled checkpoint')
        need([(r['split'],r['case'],r['input_sha256'],r['GT_sha256'],r['size']) for r in x['records']]==
             [(r['split'],r['case'],r['input_sha256'],r['GT_sha256'],r['size']) for r in parent['records']],'Matched data/order differ')
        need(x['score_comparison_invariant_sha256']==parent['score_comparison_invariant_sha256'] and
             x['terminal_score_fourRNG_sha256']==parent['terminal_score_fourRNG_sha256'],'Original metric protocol/final score RNG differ')
        x['deltas_vs_original_pixg5i']={k:x[k]-parent[k] for k in ['NR','FR','S_full']}
        x['delta_NR_contributions']={k:v-parent['NR_contributions'][k] for k,v in x['NR_contributions'].items()}
        x['delta_FR_contributions']={k:parent['validation_means'][k]-x['validation_means'][k] for k in ['lpips','dists']}
        need(close(sum(x['delta_NR_contributions'].values()),x['deltas_vs_original_pixg5i']['NR']) and
             close(sum(x['delta_FR_contributions'].values()),x['deltas_vs_original_pixg5i']['FR']),'Delta contributions do not sum')
    for p,(st,h) in SEEN.items():need(stamp(p)==st,'Audited file changed before proof commit')
    payload=dict(status='PASS_FRESH_PIXG5I_FULL103_418',auditor_source=ref(__file__),models=results,
        baseline=results[0]['label'],restored_metric_weights_proof=ref(proof),large_metric_weights_rehashed_here=False,
        fresh_quality_measured=True,quality_reuse_claim=False,used_as_G4_reference=False,
        render_fourRNG_equality_required=False,terminal_render_fourRNG_same_observed=len({x['terminal_render_fourRNG_sha256'] for x in results})==1,
        score_fourRNG_same=True,RNG_scope='Final saved endpoints only, no per-call RNG claim',
        no_tool_total_or_latency_prediction=True,no_visual_quality_claim=True,CPU_only=True,CUDA_initialized=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(payload,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=payload['status'],out=str(out),sha256=sha(out),
        table=[{k:x[k] for k in ['label','NR','FR','S_full']} for x in results])))
if __name__=='__main__':main()
