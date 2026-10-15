"""CPU audit of three fresh recovered native HYPIR pilots, exactly23 images/98 metrics.

Uses the recovered parent only. No scoring/GPU/checkpoint registration or large
weight rehash; existing restore/registration proofs retain their distinct scope.
"""
import argparse, hashlib, json, math, os, sys, time, re
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''; sys.dont_write_bytecode = True
R = Path('/cache/independent_sr_cuda_20261013')
B = R/'nr_push7_lastconv_pilot_evaluations'
LABELS = ['push7_recovered_a7750_native_pilot', 'push7_a7750_lastconv_lr1e6_nr_s7800', 'push7_a7750_lastconv_lr1e6_nr_s7850']
TEST = [1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100]
ORDER = [('validation', x) for x in range(1,4)]+[('test', x) for x in TEST]
A_SHA = '589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
RAW_SHA = [A_SHA, None, None] # Children are bound to actual completed registrations, never predicted.
RUN = R/'adaptation/runs/hypir_push7_A7750_lastconv_lr1e6_nr100'
TRAINER_SHA = '8e8d5e1fcac90504d029c6df436172d9380d46ddc5b888dda01fd6b6528a25f7'
DFD = 'dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
BASE_CFG = '8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
SCORE_SHA = '20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074'
RENDER_SHA = '1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
RESTORE_SHA = 'dbab8b00ca223d101f61a5154d1a161256afe9d367ba18da74de6c1360cd5947'
REG_SHA = 'c7bb01c0d9b0cc28fd972e1b4bfd60858116baef1337b0f9a936f680ee986a32'
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

def audit(label,index,restore,deploy,env):
    e=B/label; result=read(e/'scores/result.json'); rs=read(e/'render_status.json')
    need(result['status']=='complete' and result['completed']==result['expected']==23 and
         result['official_full_count'] is False and rs['status']=='complete' and rs['completed']==rs['total']==23 and
         (e/'render_complete').exists(), 'Actual renderer/scorer not fully complete: '+label)
    mf=read(e/'render_manifest.json'); rp=read(e/'render_protocol.json'); p=mf['protocol']; rr=p['runner']
    need(mf['protocol_id']==rp['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() and rp['protocol']==p,
         'Render protocol/manifest identity mismatch')
    if index:
        actual=read(Path(p['model_dir']).parent/'candidate_manifest.json')
        RAW_SHA[index]=actual['checkpoint_sha256']
        need(re.fullmatch(r'[0-9a-f]{64}',RAW_SHA[index]) and RAW_SHA[index]!=A_SHA,'Actual new checkpoint identity absent')
    need(p['source_sha256']==RENDER_SHA and p['cases']=='pilot' and p['warmup'] is False and rr['runner_sha256']==DFD and
         rr['checkpoint_sha256']==RAW_SHA[index] and rr['model_t']==200 and rr['coeff_t']==175 and rr['lora_rank']==256 and
         rr['patch']==512 and rr['stride']==256 and rr['scale']==1 and rr['caption']=='' and rr['internal_precision']=='BF16' and
         rr['seed']=='231 before initialization; continuous inference RNG','Original native HYPIR pipeline differs')
    md=Path(p['model_dir']); cfg=read(md/'config.json');need(sha(md/'runner.py')==DFD,'Actual runner source differs')
    need(p['model_dir_sha256']['runner.py']==DFD and p['model_dir_sha256']['config.json']==sha(md/'config.json'),'Rendered model-dir sources differ')
    for rel,h in rr['source_sha256'].items():need(sha(Path(rr['repo_path'])/rel)==h,'Actual HYPIR source changed')
    lineage=None
    if index==0:need(sha(md/'config.json')==BASE_CFG,'Recovered parent config differs')
    else:
        cm_path=md.parent/'candidate_manifest.json'; cm=read(cm_path)
        need(cm['schema']=='hypir_lastconv_nr_candidate_v1' and cm['label']==label and Path(cm['model_dir'])==md and
             cm['checkpoint_sha256']==RAW_SHA[index] and cm['checkpoint']==rr['weight_path'] and
             cm['initial_checkpoint_sha256']==A_SHA and cm['runner_sha256']==DFD and cm['config_sha256']==sha(md/'config.json') and
             cm['registrar_source_sha256']==REG_SHA and cm['optimizer_tensor_count']==2 and cm['autograd_LoRA_tensor_count']==514 and
             cm['actual_unselected512_equal_A'] is True and cm['actual_raw_and_resume_lora_equal'] is True and
             cm['quality_reuse_claim'] is False and cm['local_step']==cm['optimizer_step']==[0,50,100][index], 'Actual registered lastconv lineage differs')
        for key in ['config','marker','input_catalog','parity_proof']:
            x=cm['training_provenance'][key];need(sha(x['path'])==x['sha256'],'Registration provenance changed: '+key)
        marker=read(cm['training_provenance']['marker']['path'])
        training=read(cm['training_provenance']['config']['path']);tp=training['protocol']
        need(cm['training_run']==str(RUN) and Path(cm['checkpoint'])==RUN/f'step{cm["global_step"]}.pth' and
             Path(cm['training_provenance']['config']['path'])==RUN/'config.json' and
             Path(cm['training_provenance']['marker']['path'])==RUN/f'checkpoint_step{cm["global_step"]}.json' and
             training['protocol_id']==marker['protocol_id']==hashlib.sha256(json.dumps(tp,sort_keys=True).encode()).hexdigest() and
             tp['objective']['lr']==1e-6 and tp['optimizer_tensor_count']==2 and tp['parent_raw_sha256']==A_SHA and
             TRAINER_SHA in marker['source_sha256'].values() and marker['source_sha256']==tp['source_sha256'],
             'HigherLR actual run/config/source/marker identity differs')
        for path,h in marker['source_sha256'].items():need(sha(path)==h,'Actual higherLR source changed')
        need(marker['raw_checkpoint_sha256']==RAW_SHA[index] and marker['optimizer_tensor_count']==2 and
             marker['unselected512_exact_A'] is True and marker['unselected512_exact_A_count']==512 and
             marker['changed_FP32_LoRA_tensors']==2,'Actual checkpoint marker differs')
        lineage=dict(candidate_manifest=ref(cm_path),checkpoint_sha256=RAW_SHA[index],initial_A_sha256=A_SHA,
            local_step=cm['local_step'],optimizer_step=cm['optimizer_step'],optimizer_tensor_count=2,raw514_frozen512_exactA_registration_claim=True,
            training_provenance=cm['training_provenance'],raw_resume_rehashed_here=False)
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
    need(len(rows)==len(p['tasks'])==len(mf['records'])==23,'Actual row/task/render count differs')
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
    need(scalar_count==98 and set((e/'render').rglob('*.png'))==expected_png,'Wrong98 scalar count/extra or missing PNG')
    val={k:float(np.mean([x['metrics'][k] for x in rows[:3]])) for k in rows[0]['metrics']}
    test={k:float(np.mean([x['metrics'][k] for x in rows[3:]])) for k in rows[3]['metrics']}
    N=nr(test);F=fr(val);S=N+F
    for split,means in [('validation',val),('test',test)]:
        need(result[split]['n']==(3 if split=='validation' else 20) and all(close(v,result[split][k]) for k,v in means.items()),'Published means differ')
    need(close(result['validation']['fr'],F) and close(result['validation']['nr_val'],nr(val)) and
         close(result['validation']['score'],F+nr(val)) and close(result['test']['nr'],N) and close(result['S_proxy'],S),'Published original-formula aggregation differs')
    rstate=torch.load(e/'render_progress.pt',map_location='cpu',weights_only=False);sstate=torch.load(e/'scores/rng_checkpoint.pt',map_location='cpu',weights_only=False)
    need(rstate['protocol_id']==mf['protocol_id'] and rstate['records']==mf['records'] and rstate['checkpoint_sha256']==RAW_SHA[index] and
         rstate['input_sha256']==[t['input_sha256'] for t in p['tasks']],'Final render state differs')
    need(sstate['rows']==rows and sstate['active_split']=='test' and
         sstate['fingerprint']==hashlib.sha256(json.dumps(score_invariant(sp),sort_keys=True).encode()).hexdigest(),'Final score state differs')
    return dict(label=label,scope='fresh fixed20test_NR_plus3validation_FR; S_subset only',NR=N,FR=F,S_subset=S,
        test_means=test,validation_means=val,NR_contributions=dict(CLIPIQA=test['clipiqa'],MANIQA=test['maniqa'],MUSIQ_over100=test['musiq']/100,NIQE_clamped=max(0.,(10-test['niqe'])/10)),
        records=records,registration_lineage=lineage,render_manifest=ref(e/'render_manifest.json'),score_result=ref(e/'scores/result.json'),measurements=ref(rows_path),
        pipeline=rr,render_GPU_name=p['gpu'],render_physical_GPU_not_recorded_by_original_renderer=True,score_CUDA=sp['cuda'],
        metric_environment=sp['environment_versions'],metric_environment_protocol_sha256=sp['environment_protocol_sha256'],
        render_seed231_continuous=True,score_seed1234_once_per_split=True,score_fingerprint=sstate['fingerprint'],
        terminal_render_fourRNG_sha256=digest({k:rstate[k] for k in ['python_rng','numpy_rng','torch_rng','device_rng']}),
        terminal_score_fourRNG_sha256=digest(sstate['rng']),score_comparison_invariant_sha256=digest(score_invariant(sp,True)),config=cfg)

def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--execute',action='store_true');q.add_argument('--wait-seconds',type=int,default=0);a=q.parse_args()
    need(0<=a.wait_seconds<=1800,'CPU wait must be bounded to at most30min')
    out=R/'audits/push7/hypir_pilot/lastconv_lr1e6_50_100_recovered_matched.json'
    need(not out.exists(),'Final audit already exists; immutable')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',labels=LABELS,evaluation_root=str(B),out=str(out),requires_all_actual_COMPLETE=True,GPU=False)));return
    if a.wait_seconds:
        deadline=time.monotonic()+a.wait_seconds
        while True:
            status=[]
            for label in LABELS:
                e=B/label
                try:
                    r=json.loads((e/'render_status.json').read_text());s=json.loads((e/'scores/result.json').read_text())
                    done=r['status']==s['status']=='complete' and r['completed']==s['completed']==23 and (e/'render_manifest.json').exists()
                    status.append(dict(label=label,render_completed=r['completed'],score_completed=s['completed'],done=done))
                except (FileNotFoundError,json.JSONDecodeError,KeyError):status.append(dict(label=label,done=False))
            if all(x['done'] for x in status):break
            need(time.monotonic()<deadline,'Bounded wait expired; no final audit written')
            print(json.dumps(dict(status='WAITING_ACTUAL_COMPLETE_COUNTS_ONLY',models=status)),flush=True)
            time.sleep(min(30,max(0,deadline-time.monotonic())))
    global torch,np,Image
    import torch,numpy as np
    from PIL import Image
    torch.set_num_threads(4);Image.MAX_IMAGE_PIXELS=None;need(not torch.cuda.is_initialized(),'CPU auditor initialized CUDA')
    proof=R/'restoration_20261015/tool_assets_restore_receipt.json';need(sha(proof)==RESTORE_SHA,'Restored original weights/data proof changed')
    restore=read(proof);deploy=read(R/'scoring_tool/deployment_manifest.json');env=read(R/'calibration/environment_protocol.json')
    need(restore['status']=='PASS_ORIGINAL_TOOL_ASSETS_RESTORED' and restore['original_metric_weight_files']==8 and
         sha(R/'scoring_tool/deployment_manifest.json')==restore['deployment_manifest_sha256'],'Wrong recovered metric provenance')
    for x in restore['actual_deployment_files']:
        if '/weights/' in x['path']:
            p=Path(x['path']);need(p.stat().st_size==x['bytes'] and str(p.resolve())==x['resolved_path'],'Restored weight path/bytes changed')
    need(sha(R/'scoring_tool/csig_metrics.py')==CM_SHA and sha(R/'code/score_cuda_stream.py')==SCORE_SHA and
         sha(R/'code/render_cuda_submission.py')==RENDER_SHA,'Original metric/render/score source changed')
    results=[audit(label,i,restore,deploy,env) for i,label in enumerate(LABELS)];parent=results[0]
    for x in results[1:]:
        need([(r['split'],r['case'],r['input_sha256'],r['GT_sha256'],r['size']) for r in x['records']]==
             [(r['split'],r['case'],r['input_sha256'],r['GT_sha256'],r['size']) for r in parent['records']],'Matched data/order differ')
        need(x['score_comparison_invariant_sha256']==parent['score_comparison_invariant_sha256'] and
             x['terminal_score_fourRNG_sha256']==parent['terminal_score_fourRNG_sha256'],'Metric configuration/final four RNG differ')
        need({k:v for k,v in x['config'].items() if k not in ['label','weight_path']}==
             {k:v for k,v in parent['config'].items() if k not in ['label','weight_path']},'Runtime config differs beyond label/raw')
        x['deltas_vs_fresh_recovered_parent']={k:x[k]-parent[k] for k in ['NR','FR','S_subset']}
        x['delta_NR_contributions']={k:v-parent['NR_contributions'][k] for k,v in x['NR_contributions'].items()}
        x['delta_FR_contributions']={k:parent['validation_means'][k]-x['validation_means'][k] for k in ['lpips','dists']}
        need(close(sum(x['delta_NR_contributions'].values()),x['deltas_vs_fresh_recovered_parent']['NR']) and
             close(sum(x['delta_FR_contributions'].values()),x['deltas_vs_fresh_recovered_parent']['FR']),'Delta contributions do not sum')
    for p,(s,_) in SEEN.items():need(stamp(p)==s,'Audited file changed before commit: '+str(p))
    payload=dict(status='PASS_LASTCONV_LR1E6_FRESH_RECOVERED_MATCHED23_98',auditor_source_sha256=sha(__file__),parent_label=LABELS[0],
        models=results,restored_weights_proof=ref(proof),large_metric_weights_rehashed_here=False,
        original_A_raw_resume_rehashed_here=False,checkpoint_lineage_scope='Actual candidate manifests/markers from separately verified optimizer2/frozen512 registrar',
        old_pre_reset_parent_score_not_used=True,no_full_score_prediction=True,no_65_claim=True,no_visual_review_here=True,
        terminal_render_fourRNG_same_observed=len({x['terminal_render_fourRNG_sha256'] for x in results})==1,
        terminal_score_fourRNG_same=True,RNG_scope='Final checkpoint endpoints only; not per-image RNG snapshots',
        physical_score_GPU_differences=[x['score_CUDA'] for x in results],quality_only_no_latency_claim=True,CPU_only=True,CUDA_initialized=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=payload['status'],out=str(out),sha256=sha(out),table=[{k:x[k] for k in ['label','NR','FR','S_subset']} for x in results])))

if __name__=='__main__':main()
