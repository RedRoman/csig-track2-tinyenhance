"""A7750 HYPIR direct NR optimization; no teacher-output distillation.

Exact FP32 A7750 LoRA initialization; fresh Adam and fresh four RNG streams.
Deployment-aligned fused BF16 forward is installed from a separately sealed module.
Only test100 native768 crops, origins on256 grid; original512 inference tiles.
Input content and FP32 parameter trust anchor, PNG STE NR; same NR-dominant objective.
"""
from pathlib import Path
import argparse, fcntl, hashlib, importlib.util, inspect, json, math, os, random, re, sys, time

R=Path('/cache/independent_sr_cuda_20261013')
A_RAW_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
A_RESUME_SHA='e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b'
A_CONFIG_SHA='86015a361f5fa1429f1d797f175d27450cb6f798e1df4483badf01d876b2ab5b'
HELPER_SHA='0998967aa6052ec9a5090219e17cb73c05468b06d7acc1eae27108c1c412fb4b'
START=7750

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def parse():
    q=argparse.ArgumentParser(add_help=False);q.add_argument('--config',type=Path);k,_=q.parse_known_args()
    cfg=json.loads(k.config.read_text()) if k.config else {}
    p=argparse.ArgumentParser(description=__doc__,parents=[q])
    p.add_argument('--root',type=Path,default=R);p.add_argument('--run')
    p.add_argument('--fused-module',type=Path);p.add_argument('--fused-module-sha256')
    p.add_argument('--parity-proof',type=Path);p.add_argument('--parity-proof-sha256')
    p.add_argument('--stage-steps',type=int,default=100);p.add_argument('--stop-at',type=int,default=100)
    p.add_argument('--save-every',type=int,default=50);p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--crop-size',type=int,choices=[768],default=768)
    p.add_argument('--crop-origin-alignment',type=int,choices=[256],default=256)
    p.add_argument('--lr',type=float,default=1e-6);p.add_argument('--reward-weight',type=float,default=.05)
    p.add_argument('--musiq-weight',type=float,default=1.5);p.add_argument('--clip-weight',type=float,default=.1)
    p.add_argument('--input-l1',type=float,default=.1);p.add_argument('--input-lpips',type=float,default=.1)
    p.add_argument('--input-gradient',type=float,default=.2);p.add_argument('--parameter-anchor',type=float,default=1.)
    p.add_argument('--resume',type=Path);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--validate-config-only',action='store_true')
    need(not(set(cfg)-{x.dest for x in p._actions}),'Unknown config fields')
    p.set_defaults(**cfg);a=p.parse_args()
    need(a.run and re.fullmatch(r'hypir_push7_[A-Za-z0-9_-]+',a.run),'New hypir_push7_ run required')
    need(a.stage_steps==100 and 1<=a.stop_at<=100 and a.save_every==50,'Only100 update bounded stage, save50/final')
    need(a.crop_size==768 and a.crop_origin_alignment==256,'Only native768 crops on256 grid')
    need(a.run=='hypir_push7_A7750_nrdominant768_100','Use the isolated NR-dominant768 run')
    for n,lo,hi in [('lr',1e-8,1e-6),('reward_weight',.001,.1),('musiq_weight',.5,2.5),('clip_weight',0.,.1),
                    ('input_l1',0.,1.),('input_lpips',.01,.5),('input_gradient',.01,.6),('parameter_anchor',.01,100.)]:
        v=getattr(a,n);need(isinstance(v,(int,float)) and math.isfinite(v) and lo<=v<=hi,n+' outside bounds')
    need(isinstance(a.seed,int) and 0<=a.seed<2**32,'Integer32 seed required')
    for n in ['fused_module','parity_proof']:
        need(getattr(a,n) is not None and re.fullmatch(r'[0-9a-f]{64}',getattr(a,n+'_sha256') or ''),'Explicit sealed '+n+'+SHA required')
    return a

def main():
    a=parse()
    if a.validate_config_only:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},sort_keys=True));return
    root=Path(a.root).resolve();out=root/'adaptation/runs'/a.run
    parent=root/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500'
    raw=parent/'step7750.pth';resume=parent/'resume_step7750.pt';pc=parent/'config.json'
    for p,h in [(raw,A_RAW_SHA),(resume,A_RESUME_SHA),(pc,A_CONFIG_SHA),(a.fused_module,a.fused_module_sha256),(a.parity_proof,a.parity_proof_sha256)]:
        need(sha(p)==h,'Sealed source/parent/parity changed: '+str(p))
    parity=json.loads(a.parity_proof.read_text())
    need(parity.get('status')=='PASS_DEPLOYMENT_FUSED_PARITY_AND_514_GRADIENTS' and parity.get('module_sha256')==a.fused_module_sha256
         and parity.get('raw_sha256')==A_RAW_SHA and parity.get('script_sha256')=='1153ebabb7047c60b6fa510ab81599800dba064b14201a85a6fb7b19920acf73' and parity.get('merged_weight_exact_count')==257
         and parity.get('eager_fused_UNet_forward',{}).get('exact') is True
         and parity.get('gradient_FP32_finite_nonzero_count')==514 and parity.get('all_fourRNG_unchanged') is True
         and parity.get('master_weights_unchanged') is True,'Fused-forward parity/514grad proof not PASS')
    ref=json.loads(pc.read_text())['protocol'];helper=root/'code/cuda_training/train_cuda_nr.py'
    need(sha(helper)==HELPER_SHA,'Frozen helpers changed');helpers=load('_push7_hypir_helpers',helper)
    atomic_json,atomic_torch=helpers.atomic_json,helpers.atomic_torch
    inputs=ref['inputs'];need([x['case'] for x in inputs]==list(range(1,101)),'Exactly test100 required')
    for r in inputs:
        need(Path(r['path']).resolve().parent==root/'source_payload/assets/input' and sha(r['path'])==r['sha256'],'Test input path/SHA differs')
    sources=dict(ref['source_sha256'])
    for path,h in sources.items():need(sha(path)==h,'Frozen source changed: '+path)
    sources.update({str(Path(__file__).resolve()):sha(__file__),str(a.fused_module.resolve()):a.fused_module_sha256})
    base=Path('/temp/zihaohe_stage/baselines/sd21b')
    for records in [ref['base_provenance']['initial_core_asset_hashes'],ref['base_provenance']['all_config_tokenizer_scheduler_metadata_hashes']]:
        for rel,r in records.items():need(sha(base/rel)==r['sha256'],'Base changed: '+rel)
    init=Path(ref['init_checkpoint']);need(sha(init)==ref['init_sha256'],'Raw6000 initialization changed')
    import torch
    import numpy as np
    from PIL import Image
    from torch.nn import functional as F
    sys.path.insert(0,str(root/'repos/HYPIR_cuda'))
    from HYPIR.enhancer.sd2 import SD2Enhancer
    from HYPIR.utils.common import wavelet_reconstruction
    need(torch.cuda.is_available() and 'torch_npu' not in sys.modules,'Native CUDA required')
    torch.cuda.set_device(0);device=torch.device('cuda:0');torch.set_num_threads(4);Image.MAX_IMAGE_PIXELS=None
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    parent_raw=torch.load(raw,map_location='cpu',weights_only=True,mmap=True)
    parent_saved=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
    need(len(parent_raw)==514 and sum(v.numel() for v in parent_raw.values())==259474432,'Parent514 schema differs')
    need(list(parent_raw)==list(parent_saved['lora']) and all(v.dtype==torch.float32 and torch.equal(v,parent_saved['lora'][k]) and torch.isfinite(v).all() for k,v in parent_raw.items()),'Parent exact FP32 raw/resume proof failed')
    del parent_saved # Deliberately no parent optimizer or RNG inheritance.
    out.mkdir(parents=True,exist_ok=True)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not (out/'config.json').exists() or a.resume is not None,'Existing new run requires explicit immutable resume')
        helpers.seed_all(a.seed,torch,np)
        m=SD2Enhancer(base_model_path=str(base),weight_path=str(init),lora_modules=helpers.MODULES.split(','),lora_rank=256,model_t=200,coeff_t=175,device=str(device))
        m.init_models();m.G.eval().requires_grad_(False);m.vae.eval().requires_grad_(False);m.text_encoder.eval().requires_grad_(False);m.prepare_inputs(1,'')
        # Match the actual A deployment flags AFTER init_models.
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
        enhance_grad=getattr(m.enhance,'__wrapped__',None)
        need(callable(enhance_grad),'Original enhance no_grad wrapper must expose its actual underlying function')
        need(Path(inspect.getfile(enhance_grad)).resolve()==root/'repos/HYPIR_cuda/HYPIR/enhancer/base.py','Unexpected enhance implementation')
        named=[(n,p) for n,p in m.G.named_parameters() if 'lora' in n];need([n for n,_ in named]==list(parent_raw),'LoRA order differs')
        with torch.no_grad():
            for n,p in named:p.data=p.data.float();p.copy_(parent_raw[n].to(p));p.requires_grad_(True)
        params=[p for _,p in named];anchors=[p.detach().clone() for p in params]
        raw_anchor_norms=[p.square().sum().detach() for p in anchors]
        anchor_norms=[v.clamp_min(1e-12) for v in raw_anchor_norms]
        need(all(torch.equal(p.detach().cpu(),parent_raw[n]) for n,p in named),'A7750 FP32 copy not exact')
        del parent_raw
        fused=load('_push7_deployment_fused_module',a.fused_module)
        fused_install_report=fused.install_deployment_fused_forward(m.G)
        m.G.enable_gradient_checkpointing()
        metrics,clip,maniqa_proof,metric_sha=helpers.create_metrics(root,device,20)
        need(metric_sha==ref['metric_weights_sha256'],'Original metric weights differ')
        objective={n:getattr(a,n) for n in ['lr','reward_weight','musiq_weight','clip_weight','input_l1','input_lpips','input_gradient','parameter_anchor']}
        protocol=dict(schema='hypir_push7_direct_nr_crop768_full_gaussian_fused_bf16_v3',parent_raw=str(raw),parent_raw_sha256=A_RAW_SHA,
            parent_resume=str(resume),parent_resume_sha256=A_RESUME_SHA,parent_config_sha256=A_CONFIG_SHA,
            stage_start_global_step=START,stage_steps=100,new_Adam_initial_step=0,seed=a.seed,
            optimizer='fresh Adam FP32 lr configured, betas(.9,.999),eps1e-8,weight_decay0,foreachFalse; no old momentum/RNG inherited',
            objective=objective,loss='input_l1*L1(rgb,input)+input_lpips*LPIPS(rgb,input)+input_gradient*gradient_L1(rgb,input)+parameter_anchor*mean_tensor_relative_squared_FP32_parameter_change - reward_weight*(musiq_weight*MUSIQ/100+MANIQA20+clip_weight*CLIPIQA)',
            parameter_anchor='mean over514: sum((p-pA)^2)/max(sum(pA^2),1e-12); references are detached FP32 initial LoRA parameters, not teacher outputs',
            teacher_output_loss=False,distillation=False,validation_GT_training=False,inputs=inputs,
            input_alignment='JPEG FP32[0,1] -> half()*2-1 -> FP32[0,1]; native768 crop on256 origin grid; no resize',
            training_crop_size=a.crop_size,training_crop_origin_alignment=a.crop_origin_alignment,
            deployment_patch_size=512,deployment_stride=256,
            derivation_source_sha256='8a90192c200efa934e3234984855575c1b25feffdc1279cd64e3db10619f1c4c',
            deployment_aligned_module=str(a.fused_module),deployment_aligned_module_sha256=a.fused_module_sha256,
            deployment_parity_proof=str(a.parity_proof),deployment_parity_proof_sha256=a.parity_proof_sha256,
            model_precision='BF16 base+deployment fused forward/STE into exact FP32 LoRA masters; BF16 VAE encode/decode',
            metrics='original frozen FP32 MUSIQ/100 MANIQA20 CLIPIQA; LPIPS(input); NIQE formal evaluation only',
            numeric_flags=dict(matmul_allow_tf32=True,cudnn_allow_tf32=True,cudnn_benchmark=False),
            PNG_STE='forward original output RGB*2-1 then clamp[-1,1],*.5+.5,round255/255; backward identity wrt clamped RGB; NR only',
            latent='original full enhance underlying function, native768 crop with unchanged patch512/stride256; original three Gaussian BF16/out-FP32count merges, resize/ref/sample/DDPM/wavelet/clamp; no teacher model',metric_weights_sha256=metric_sha,source_sha256=sources)
        identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        opt=torch.optim.Adam(params,lr=a.lr,weight_decay=0.,foreach=False)
        helpers.seed_all(a.seed,torch,np);done=0
        if a.resume:
            need(a.resume.resolve().parent==out.resolve() and a.resume.name.startswith('resume_step'),'Use this run immutable resume only')
            saved=torch.load(a.resume,map_location='cpu',weights_only=False,mmap=True)
            need(saved['protocol_id']==identity and saved['optimizer_step']==saved['local_step'] and saved['global_step']==START+saved['local_step'],'Resume identity/counters differ')
            done=saved['local_step'];need(0<=done<a.stop_at<=100,'Resume/stop bounds differ')
            with torch.no_grad():
                for n,p in named:
                    need(saved['lora'][n].dtype==torch.float32,'Resume LoRA not FP32');p.copy_(saved['lora'][n].to(p))
            opt.load_state_dict(saved['optimizer'])
            need((not opt.state if done==0 else len(opt.state)==514 and {int(v['step']) for v in opt.state.values()}=={done}),'Fresh-stage Adam counters differ')
            need(all(g['lr']==a.lr for g in opt.param_groups),'Resume LR differs')
            need(all(v[k].dtype==torch.float32 and torch.isfinite(v[k]).all() for v in opt.state.values() for k in ['exp_avg','exp_avg_sq']),'Resume Adam moments not finite FP32')
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0)
            del saved
        atomic_json(out/'config.json',dict(protocol_id=identity,protocol=protocol,target_stage_steps=100,target_global_step=START+100,target_optimizer_step=100,save_every=50))
        atomic_json(out/'input_catalog.json',inputs)
        atomic_json(out/'initialization_proof.json',dict(status='PASS',A7750_all514_exact_FP32=True,initialization_phase='before optional new-stage resume',fresh_Adam0=True,old_A_Adam_and_RNG_not_inherited=True,restored_new_stage_local_step=done,actual_optimizer_step=done,teacher_output_loss=False,distillation=False,fused_install_report=fused_install_report,parity_proof_sha256=a.parity_proof_sha256,
            parameter_anchor_unclamped_denominator_min=float(torch.stack(raw_anchor_norms).min()),
            parameter_anchor_unclamped_denominator_max=float(torch.stack(raw_anchor_norms).max()),
            parameter_anchor_denominators_below_floor=sum(float(v)<1e-12 for v in raw_anchor_norms),parameter_anchor_floor=1e-12))
        atomic_json(out/'maniqa_initial_state_proof.json',maniqa_proof)
        def save(step):
            global_step=START+step;milestone=out/f'resume_step{global_step}.pt';marker=out/f'checkpoint_step{global_step}.json'
            need(not milestone.exists() and not marker.exists(),'Never overwrite immutable milestone')
            lora={n:p.detach().cpu().clone() for n,p in named}
            need(all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in lora.values()),'Saved FP32 LoRA nonfinite')
            if step:
                need(len(opt.state)==514 and {int(v['step']) for v in opt.state.values()}=={step},'Adam counter differs')
                need(all(v[k].dtype==torch.float32 and torch.isfinite(v[k]).all() for v in opt.state.values() for k in ['exp_avg','exp_avg_sq']),'Adam moments not finite FP32')
            raw_path=raw if step==0 else out/f'step{global_step}.pth'
            if step:atomic_torch(raw_path,lora,torch)
            state=dict(protocol_id=identity,lora=lora,optimizer=opt.state_dict(),local_step=step,global_step=global_step,optimizer_step=step,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0))
            atomic_torch(out/'resume_latest.pt',state,torch);os.link(out/'resume_latest.pt',milestone)
            changed=sum(bool((p-r).abs().max()>0) for p,r in zip(params,anchors))
            atomic_json(marker,dict(protocol_id=identity,local_step=step,global_step=global_step,optimizer_step=step,raw_checkpoint=str(raw_path),raw_checkpoint_sha256=sha(raw_path),resume_checkpoint=str(milestone),resume_checkpoint_sha256=sha(milestone),all514_FP32_finite=True,all4_RNG_saved=True,changed_FP32_LoRA_tensors=changed,initial_A_checkpoint_sha256=A_RAW_SHA,no_teacher_outputs=True,no_val_GT=True,fresh_Adam_initial_step=0,input_catalog=str(out/'input_catalog.json'),input_catalog_sha256=sha(out/'input_catalog.json'),source_sha256=sources))
        if not a.resume:save(0)
        begin=time.time();first=True
        with (out/'train.jsonl').open('a',buffering=1) as log:
            for step in range(done+1,a.stop_at+1):
                started=time.time();case=random.randint(1,100);item=inputs[case-1]
                need(sha(item['path'])==item['sha256'],'Input changed')
                with Image.open(item['path']) as im:
                    im=im.convert('RGB');w,h=im.size
                    need(w>=a.crop_size and h>=a.crop_size,'Native input smaller than768 crop; no resize fallback')
                    left=random.randint(0,(w-a.crop_size)//a.crop_origin_alignment)*a.crop_origin_alignment
                    top=random.randint(0,(h-a.crop_size)//a.crop_origin_alignment)*a.crop_origin_alignment
                    arr=np.asarray(im.crop((left,top,left+a.crop_size,top+a.crop_size)),dtype=np.float32).copy()/255.
                x=torch.from_numpy(arr).permute(2,0,1)[None].to(device);x=((x.half()*2-1).float()+1)*.5
                opt.zero_grad(set_to_none=True)
                # Call only the original undecorated body: decorators would disable gradients.
                # This preserves ALL three original Gaussian BF16 tile accumulations.
                with torch.enable_grad(),torch.autocast('cuda',enabled=False):
                    rgb=enhance_grad(m,lq=x,prompt='',scale_by='factor',upscale=1,patch_size=512,stride=256,return_type='pt')
                need(rgb.requires_grad and rgb.grad_fn is not None,'Full original enhance lost training graph')
                export_rgb=(rgb.detach()*2-1).clamp(-1,1)*.5+.5
                quantized=torch.round(export_rgb*255)/255;nr_rgb=quantized+(rgb-rgb.detach())
                with torch.autocast('cuda',enabled=False):
                    l1=(rgb-x).abs().mean();lpips=metrics['lpips'](rgb,x).mean();spatial=helpers.gradient_loss(rgb,x)
                    parameter=torch.stack([(p-r).square().sum()/n for p,r,n in zip(params,anchors,anchor_norms)]).mean()
                    content=a.input_l1*l1+a.input_lpips*lpips+a.input_gradient*spatial
                    scores={n:metrics[n](nr_rgb).mean()/(100 if n=='musiq' else 1) for n in ['musiq','maniqa','clipiqa']}
                    reward=a.musiq_weight*scores['musiq']+scores['maniqa']+a.clip_weight*scores['clipiqa']
                    loss=content+a.parameter_anchor*parameter-a.reward_weight*reward
                need(torch.isfinite(loss),'Nonfinite loss');component={}
                if first:
                    for n,v in dict(scores,content=content).items():
                        g=torch.autograd.grad(v,rgb,retain_graph=True)[0];gn=float(g.float().norm())
                        need(torch.isfinite(g).all() and gn>0,'Missing real input gradient: '+n);component[n]=dict(input_gradient_norm=gn,finite=True,nonzero=True)
                    for n,v in [('weighted_content',content),('weighted_NR_reward',-a.reward_weight*reward),('weighted_parameter_anchor',a.parameter_anchor*parameter)]:
                        grads=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)
                        need(all(g is None or torch.isfinite(g).all() for g in grads),'Nonfinite component LoRA gradient: '+n)
                        norm=math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))
                        component[n]=dict(LoRA_gradient_L2=norm,finite=True,nonzero_tensors=sum(g is not None and bool(g.abs().max()>0) for g in grads),zero_at_initial_anchor_expected=(n=='weighted_parameter_anchor' and done==0))
                        del grads
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.)
                finite=sum(p.grad is not None and torch.isfinite(p.grad).all().item() for p in params);nonzero=sum(p.grad is not None and p.grad.abs().max().item()>0 for p in params)
                need(torch.isfinite(gn) and finite==514 and nonzero==514,'All514 actual LoRA gradients must be finite/nonzero')
                need(not any(p.grad is not None for mm in list(metrics.values())+[clip,m.vae,m.text_encoder] for p in mm.parameters()),'Frozen module got gradient')
                torch.cuda.synchronize()
                row=dict(local_step=step,global_step=START+step,optimizer_step=step,case=case,crop_xy=[left,top],loss=float(loss.detach()),content=float(content.detach()),L1_input=float(l1.detach()),LPIPS_input=float(lpips.detach()),gradient_input=float(spatial.detach()),parameter_anchor=float(parameter.detach()),reward=float(reward.detach()),scores={k:float(v.detach()) for k,v in scores.items()},gradient_norm=float(gn),finite_lora_gradients=finite,nonzero_lora_gradients=nonzero,lr=a.lr,step_seconds=time.time()-started,elapsed_seconds=time.time()-begin,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                if first:
                    atomic_json(out/'gradient_preflight.json',dict(status='PASS',components=component,row=row,no_optimizer_update_yet=True,teacher_output_loss=False,PNG_STE_forward_exact=torch.equal(nr_rgb.detach(),quantized),frozen_metrics=True));first=False
                if a.preflight_only:
                    atomic_json(out/'status.json',dict(stage='preflight_complete',completed_local_steps=done,optimizer_step=done,row=row));print(json.dumps(row),flush=True);return
                opt.step()
                with torch.no_grad():
                    delta2=torch.stack([(p-r).square().sum() for p,r in zip(params,anchors)])
                    relative2=delta2/torch.stack(anchor_norms)
                    row.update(update_RMS_from_initial=math.sqrt(float(delta2.sum())/259474432),
                        update_relative_L2_from_initial=math.sqrt(float(delta2.sum()/torch.stack(raw_anchor_norms).sum())),
                        update_max_tensor_relative_L2=math.sqrt(float(relative2.max())),
                        update_mean_tensor_relative_squared=float(relative2.mean()))
                    need(torch.isfinite(delta2).all() and torch.isfinite(relative2).all(),'Nonfinite actual update')
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                atomic_json(out/'status.json',dict(stage='training',completed_local_steps=step,total=100,row=row))
                if step%50==0 or step==a.stop_at:save(step)
                del x,rgb,nr_rgb,quantized,export_rgb,loss,reward,scores,l1,lpips,spatial,content,parameter
        for p,h in sources.items():need(sha(p)==h,'Source changed during training')
        atomic_json(out/'status.json',dict(stage='complete' if a.stop_at==100 else 'stopped_at_requested_boundary',completed_local_steps=a.stop_at,global_step=START+a.stop_at,optimizer_step=a.stop_at,requires_actual_four_previews_and_original_pilot=True,teacher_output_loss=False,distillation=False))

if __name__=='__main__':main()
