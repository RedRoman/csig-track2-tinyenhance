"""CPU audit of two fresh recovered HYPIR TRT-UNet-VAE fulls, exactly103 images/418 metrics.

Uses the recovered parent only. No scoring/GPU/checkpoint registration or large metric-
weight rehash; existing restore/registration proofs retain their distinct scope.
"""
import argparse, hashlib, json, math, os, sys, time, re
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''; sys.dont_write_bytecode = True
R = Path('/cache/independent_sr_cuda_20261013')
B = R/'nr_push7_recovered_full_evaluations'
LABELS = ['push7_recovered_a7750_trtunetvae_all', 'push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all']
TEST = list(range(1,101))
ORDER = [('validation', x) for x in range(1,4)]+[('test', x) for x in TEST]
A_SHA = '589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
RAW_SHA = [A_SHA, None] # Children are bound to actual completed registrations, never predicted.
RUN = R/'adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100'
TRAINER_SHA = '5f99fcebc6cb50ed5c3ae18ae8df0cdf10cb2ab3b62e124c5097b0c81cf5cd9b'
DFD = 'dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
BASE_CFG = '63e227319f2c2fb327257a810f6709b4f1d738732571f6c100e22b75174694e5'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
SCORE_SHA = '20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074'
RENDER_SHA = '1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
RESTORE_SHA = 'dbab8b00ca223d101f61a5154d1a161256afe9d367ba18da74de6c1360cd5947'
REG_SHA = '216c5faa11450a7e77f7c42f89e24d5a5d0a98e29338a9002d3721494cee8e14'
SELECTED = ['up_blocks.3.resnets.2.conv1.lora_A.default.weight','up_blocks.3.resnets.2.conv1.lora_B.default.weight',
            'up_blocks.3.resnets.2.conv2.lora_A.default.weight','up_blocks.3.resnets.2.conv2.lora_B.default.weight',
            'conv_out.lora_A.default.weight','conv_out.lora_B.default.weight']
CLI = R/'code/push7/hypir_direct_nr/config_hypir_lastres_lr5e6_nr100_v4.json'
CLI_SHA = '3ba6800813879ba4fbcf8bcf57dc9af41268a0b367359a370dd7f5738b7ab8fc'
SCHEMA = 'hypir_push7_lastres_nr_niqe768_optimizer6_full_gaussian_fused_bf16_v3'
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

def backend_binding(md,cfg,rr,index,label):
    cm_path=md.parent/'candidate_manifest.json';cm=read(cm_path)
    need(Path(cm['model_dir']).resolve()==md.resolve() and cm['config_sha256']==sha(md/'config.json') and
         cm['runner_sha256']==sha(md/'runner.py')==rr['runner_sha256'] and
         cm['checkpoint_sha256']==rr['checkpoint_sha256']==sha(cfg['weight_path']) and
         cfg['checkpoint_sha256']==cm['checkpoint_sha256'] and cm['checkpoint']==cfg['weight_path'] and
         rr['model_dir_config_sha256']==cm['config_sha256'] and cm['quality_reuse_claim'] is False,'Actual CM/runtime/current raw binding differs')
    if index==0:
        need(sha(cm_path)=='3e422d0e8474532a7fc20838504c743c32e19b03770e440eaada6eee003d38f3' and
             md==R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16/model_dir' and
             cm['schema']=='hypir_trt_unet_vae_hybrid_candidate_v1' and cm['checkpoint_sha256']==A_SHA and
             cm['runner_sha256']=='270d2434179d17658e9f1df228c8ef99124a2695141c92f0b5fecfa73d8d3080' and
             sha(md/'config.json')==BASE_CFG and rr['schema']=='hypir_trt_unet_vae_hybrid_runner_v1' and
             cm['no_training'] is True and cm['no_distillation'] is True,'Not actual restored original-A backend')
        parent_sha='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9'
        builder=R/'code/push7/trt_unet/build_hypir_unet_trt.py'
        builder_sha='4281d9be015142919b51b84d946ad8d23b8bd49451ad54adb9ada8f1d42cdcca'
        native=None
    else:
        need(cm['schema']=='hypir_finetuned_trt_unet_vae_candidate_v1' and cm['label']==label.removesuffix('_all') and
             cm['model_type']=='HYPIR_SD2_rank256_finetuned_TRT_UNet_VAE_native512' and
             cm['runner_sha256']=='c98c406884cf07bc5541e315393b98e11351165b35c94d33f926d5443f9563c3' and
             rr['schema']=='hypir_finetuned_trt_unet_vae_hybrid_runner_v1' and
             cm['checkpoint_finetuned'] is True and cm['backend_build_has_no_training'] is True and
             cm['no_teacher_outputs'] is True and cm['no_val_GT_training'] is True and cm['no_distillation'] is True and
             cm['requires_actual_fresh_full103_418'] is True and cm['initial_A_checkpoint_sha256']==A_SHA and
             cm['registrar_source']['sha256']=='5c3cb0094da5ba12106e9b80f7c9055512f8fcb34ec4fef7f656b9dec255a5f7' and
             sha(cm['registrar_source']['path'])==cm['registrar_source']['sha256'],'Not the reviewed trained backend registration')
        source=R/'code/push7/hypir_direct_nr/trained_backend_v1/trained_native_binding.py'
        need(sha(source)=='db2052014f941f517dee730dcb44e4d8e3d9939e0de15991f663bb098e6c451d','Frozen native binding changed')
        import importlib.util
        spec=importlib.util.spec_from_file_location('_full_registered_native_binding',source)
        b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
        refn=cm['native_candidate_binding']['native_candidate_manifest']
        need(refn['sha256']=='0477903b0bc478064ad2f03a5d429eb37459097ac6a0014ec76d92efef7fa4c0','Wrong selected native MF')
        native=b.bind_native(refn['path'],refn['sha256'],hash_raw=False)
        need(cm['native_candidate_binding']==native and native['training_run']==str(RUN) and
             native['local_step']==native['optimizer_step']==100 and native['global_step']==7850 and
             native['optimizer_tensor_count']==6 and native['unselected_exact_A_count']==508 and
             native['optimizer_parameter_keys']==SELECTED and native['optimizer_parameter_elements']==3113984 and
             cm['training_provenance']==rr['training_provenance']==native['training_provenance'] and
             native['checkpoint_sha256']==cm['checkpoint_sha256'] and rr['checkpoint_finetuned'] is True and
             rr['backend_build_has_no_training'] is True and rr['quality_reuse_claim'] is False,'Actual trained native lineage differs')
        for name,x in cm['files'].items():need(sha(md/name)==x['sha256'] and Path(x['path']).resolve()==(md/name).resolve(),'Registered source changed')
        parent_sha='4b225fbe82c556abf4b409927d16709e771a69fb811d7c6803f7281792875862'
        builder=R/'code/push7/hypir_direct_nr/trained_backend_v1/build_finetuned_unet_trt.py'
        builder_sha='b7dcc1f4fa986506eda23651b94c3f65b463fb46a9fe45e464833e6017fda34d'
    need(sha(md/'reference_runner.py')==DFD and sha(md/'vae_hybrid_runner.py')==parent_sha and
         sha(md/'trt_vae_runtime_v2.py')=='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4','Frozen math/runtime source differs')
    for path,h in rr['implementation_sources'].items():need(sha(path)==h,'Actual implementation source differs')
    need(set(cm['actual_engines'])=={'encoder','decoder','UNet'},'Require actual three-engine identity')
    for name,record in cm['actual_engines'].items():
        if name=='UNet':
            c=cfg['TRT_UNet_engine'];u=rr['UNet_TRT']
            actual=dict(path=u['engine_path'],sha256=u['engine_sha256'],manifest=u['manifest'],manifest_sha256=u['manifest_sha256'])
        else:c=cfg['TRT_engines'][name];actual=rr['TRT']['engines'][name]
        for k in ['path','sha256','manifest','manifest_sha256']:need(c[k]==record[k]==actual[k],'Engine record/runtime/config differs')
        need(sha(record['path'])==record['sha256'] and sha(record['manifest'])==record['manifest_sha256'],'Actual engine bytes or manifest differ')
    u=rr['UNet_TRT'];um=read(cm['actual_engines']['UNet']['manifest'])
    need(um['status']=='BUILT_COMPONENT_PROBED' and um['raw_sha256']==cm['checkpoint_sha256'] and
         um['engine_sha256']==cm['actual_engines']['UNet']['sha256'] and
         um['engine_bytes']==Path(cm['actual_engines']['UNet']['path']).stat().st_size and
         sha(builder)==u['producer_sha256']==um['source_sha256'][str(builder)]==builder_sha and
         um['compiled_fallback'] is False and um['model_t']==200 and um['caption']=='' and
         um['input_shape']==um['output_shape']==[1,4,64,64],'UNet producer/current raw/fullcompile/fixed64 binding differs')
    for path,h in um['source_sha256'].items():need(sha(path)==h,'Actual engine producer source differs')
    if index:
        need(um['schema']=='hypir_finetuned_unet_fp16_trt_component_v1' and um['native_candidate_binding']==native and
             um['checkpoint_finetuned'] is True and um['backend_build_has_no_training'] is True and
             um['require_full_compilation'] is True and um['actual_trained_raw_SHA_verified_by_builder'] is True and
             Path(um['engine_path']).resolve()!=R/'backends/push7_trt/unet_fp16_A7750/component.engine' and
             um['engine_sha256']!='08fca16f286c604614e9322ded93f848b0c1d3dd40de7ca793f10b98f286aec8','Not a fresh fullcompiled trained UNet')
    need(cm['BF16_original_fused_G_state']==um['BF16_original_fused_G_state']==u['BF16_fused_G_digest_from_builder'] and
         cm['fixed_conditioning']==um['fixed_conditioning'] and
         um['BF16_original_fused_G_state']['tensors']==686 and um['BF16_original_fused_G_state']['elements']==865910724 and
         u['fixed_original_BF16_conditioning_sha256']==um['fixed_conditioning']['BF16_sha256']['sha256']=='f65bb78e0c2376a4df09f17d8671ba7dfe163375d3349a1db49f12c5bbd9d0a9' and
         um['fixed_conditioning']['FP16_sha256']['sha256']=='0008da3ec7f91844e17610a5929d68b151938ac5d895bacbb088f9c098d5003b','Fused current G/fixed conditioning differs')
    need(u['construction_all_four_RNG_restored'] is True and u['forward_RNG_reset'] is False and
         u['scheduler_DDPM_original_BF16_outside_engine'] is True,'Original RNG/scheduler policy differs')
    need(sha(Path(cfg['base_model_path'])/'unet/diffusion_pytorch_model.safetensors')==cm['actual_base_UNet_sha256']==um['actual_base_UNet_sha256'] and
         sha(cfg['base_vae_path'])==rr['TRT']['original_VAE_sha256']=='a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815','Actual original base assets differ')
    return dict(candidate_manifest=ref(cm_path),checkpoint_sha256=cm['checkpoint_sha256'],initial_A_sha256=A_SHA,
        actual_three_engine_SHA_rehashed_here=True,actual_raw_SHA_rehashed_here=True,raw_resume_tensors_reopened_here=False,
        native_candidate_binding=native,component_manifest=ref(cm['actual_engines']['UNet']['manifest']))

def audit(label,index,restore,deploy,env):
    e=B/label; result=read(e/'scores/result.json'); rs=read(e/'render_status.json')
    need(result['status']=='complete' and result['completed']==result['expected']==103 and
         result['official_full_count'] is True and rs['status']=='complete' and rs['completed']==rs['total']==103 and
         (e/'render_complete').exists(), 'Actual renderer/scorer not fully complete: '+label)
    mf=read(e/'render_manifest.json'); rp=read(e/'render_protocol.json'); p=mf['protocol']; rr=p['runner']
    need(mf['protocol_id']==rp['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() and rp['protocol']==p,
         'Render protocol/manifest identity mismatch')
    md=Path(p['model_dir']);cfg=read(md/'config.json')
    lineage=backend_binding(md,cfg,rr,index,label)
    RAW_SHA[index]=lineage['checkpoint_sha256']
    need(p['source_sha256']==RENDER_SHA and p['cases']=='all' and p['warmup'] is False and
         rr['checkpoint_sha256']==RAW_SHA[index] and rr['model_t']==200 and rr['coeff_t']==175 and rr['lora_rank']==256 and
         rr['patch']==512 and rr['stride']==256 and rr['scale']==1 and rr['caption']=='' and
         rr['internal_precision']=='FP16 TensorRT UNet+VAE, BF16 eps/latent/DiagonalGaussianDistribution/scheduler/tiling, FP32 wavelet/output' and
         rr['seed']=='231 before initialization; continuous inference RNG' and
         p['inference_rng']=='seed231 before model initialization; continuous ordered val3 then tests','Original full native HYPIR pipeline differs')
    need(p['model_dir_sha256']['runner.py']==sha(md/'runner.py') and p['model_dir_sha256']['config.json']==sha(md/'config.json'),'Rendered model-dir sources differ')
    for rel,h in rr['source_sha256'].items():need(sha(Path(rr['repo_path'])/rel)==h,'Actual HYPIR source changed')
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
    need(rstate['protocol_id']==mf['protocol_id'] and rstate['records']==mf['records'] and rstate['checkpoint_sha256']==RAW_SHA[index] and
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
        terminal_score_fourRNG_sha256=digest(sstate['rng']),score_comparison_invariant_sha256=digest(score_invariant(sp,True)),config=cfg)

def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--execute',action='store_true');q.add_argument('--wait-seconds',type=int,default=0);a=q.parse_args()
    need(0<=a.wait_seconds<=1800,'CPU wait must be bounded to at most30min')
    out=R/'audits/push7/hypir_full/lastres_lr5e6_s7850_trtunetvae_recovered_matched.json'
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
                    done=r['status']==s['status']=='complete' and r['completed']==s['completed']==103 and (e/'render_manifest.json').exists()
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
    need(sha(CLI)==CLI_SHA,'Actual frozen LR5e-6 CLI differs')
    cli=read(CLI);need(cli['run']==RUN.name and cli['lr']==5e-6,'CLI run/LR differs')
    results=[audit(label,i,restore,deploy,env) for i,label in enumerate(LABELS)];parent=results[0]
    for x in results[1:]:
        need([(r['split'],r['case'],r['input_sha256'],r['GT_sha256'],r['size']) for r in x['records']]==
             [(r['split'],r['case'],r['input_sha256'],r['GT_sha256'],r['size']) for r in parent['records']],'Matched data/order differ')
        need(x['score_comparison_invariant_sha256']==parent['score_comparison_invariant_sha256'] and
             x['terminal_score_fourRNG_sha256']==parent['terminal_score_fourRNG_sha256'],'Metric configuration/final four RNG differ')
        x['deltas_vs_fresh_recovered_parent']={k:x[k]-parent[k] for k in ['NR','FR','S_full']}
        x['delta_NR_contributions']={k:v-parent['NR_contributions'][k] for k,v in x['NR_contributions'].items()}
        x['delta_FR_contributions']={k:parent['validation_means'][k]-x['validation_means'][k] for k in ['lpips','dists']}
        need(close(sum(x['delta_NR_contributions'].values()),x['deltas_vs_fresh_recovered_parent']['NR']) and
             close(sum(x['delta_FR_contributions'].values()),x['deltas_vs_fresh_recovered_parent']['FR']),'Delta contributions do not sum')
    for p,(s,_) in SEEN.items():need(stamp(p)==s,'Audited file changed before commit: '+str(p))
    payload=dict(status='PASS_LASTRES_LR5E6_TRTUNETVAE_FRESH_RECOVERED_FULL103_418',auditor_source_sha256=sha(__file__),parent_label=LABELS[0],
        models=results,training_CLI=ref(CLI),restored_weights_proof=ref(proof),large_metric_weights_rehashed_here=False,
        raw_resume_tensor_comparison_not_repeated_here=True,checkpoint_lineage_scope='Actual registered current trained raw/native marker/optimizer6 frozen508; independent new current-weight UNet; fresh recovered A backend baseline',
        old_pre_reset_parent_score_not_used=True,full_quality_measured_here=True,no_tool_total_or_latency_prediction=True,no_65_claim=True,no_visual_review_here=True,
        terminal_render_fourRNG_same_observed=len({x['terminal_render_fourRNG_sha256'] for x in results})==1,
        terminal_score_fourRNG_same=True,RNG_scope='Final checkpoint endpoints only; not per-image RNG snapshots',
        physical_score_GPU_differences=[x['score_CUDA'] for x in results],quality_only_no_latency_claim=True,CPU_only=True,CUDA_initialized=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=payload['status'],out=str(out),sha256=sha(out),table=[{k:x[k] for k in ['label','NR','FR','S_full']} for x in results])))

if __name__=='__main__':main()
