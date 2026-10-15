"""Registered fine-tuned original HYPIR UNet -> fixed64-latent FP16 TensorRT.

Default CPU plan. Root explicitly schedules --execute. No training/student,
fallback, resize, scheduler, posterior sampler or DDPM inside this engine.
"""
from pathlib import Path
import argparse, datetime, gc, hashlib, json, math, os, random, statistics, sys, time
R=Path('/cache/independent_sr_cuda_20261013');BASE=Path('/temp/zihaohe_stage/baselines/sd21b')
INITIAL=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
INITIAL_A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
BINDING_SHA='db2052014f941f517dee730dcb44e4d8e3d9939e0de15991f663bb098e6c451d'
PARENT=INITIAL.parent/'config.json';PARENT_SHA='86015a361f5fa1429f1d797f175d27450cb6f798e1df4483badf01d876b2ab5b'
RUNTIME_SHA='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4'
DFD_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
MODULES='to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj'.split(',')

def need(v,msg):
    if not v:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def p10(v):
    v=sorted(v);q=(len(v)-1)*.1;i=int(q);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(q-i)

def write(p,data):
    with Path(p).open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--native-manifest',type=Path,required=True);ap.add_argument('--native-manifest-sha256',required=True)
    ap.add_argument('--warm',type=int,default=30);ap.add_argument('--n',type=int,default=50);a=ap.parse_args()
    binding_path=Path(__file__).with_name('trained_native_binding.py');need(sha(binding_path)==BINDING_SHA,'Sealed native binding source changed')
    from trained_native_binding import bind_native
    native=bind_native(a.native_manifest,a.native_manifest_sha256,hash_raw=False)
    RAW=Path(native['checkpoint']);RAW_SHA=native['checkpoint_sha256']
    d=a.out_dir.resolve();need(not d.exists(),'Use a new immutable output directory')
    need(d.is_relative_to(R/'backends/push7_trt') and d.name!='unet_fp16_A7750','New trained-only engine directory required')
    need(0<=a.warm<=40 and 1<=a.n<=200,'Bounded diagnostic warm/n')
    runtime=Path(__file__).with_name('trt_vae_runtime_v2.py');need(sha(runtime)==RUNTIME_SHA,'Sealed dedicated-stream runtime changed')
    need(sha(PARENT)==PARENT_SHA,'Pinned A7750 source/base provenance changed')
    reference=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir/runner.py';need(sha(reference)==DFD_SHA,'Actual A deployment source changed')
    parent=json.loads(PARENT.read_text())['protocol'];repo=R/'repos/HYPIR_cuda'
    paths=[repo/'HYPIR/enhancer/base.py',repo/'HYPIR/enhancer/sd2.py',repo/'HYPIR/utils/common.py']
    source_hashes={str(p):sha(p) for p in [Path(__file__).resolve(),binding_path,runtime,reference,*paths]}
    for p in paths:need(source_hashes[str(p)]==parent['source_sha256'][str(p)],'Actual original HYPIR source changed')
    base_records=parent['base_provenance']['initial_core_asset_hashes']
    plan=dict(schema='hypir_finetuned_unet_fp16_trt_component_v1',execute=a.execute,out_dir=str(d),raw=str(RAW),raw_sha256=RAW_SHA,
        initial_A_sha256=INITIAL_A_SHA,initial_A_base_provenance_config_sha256=PARENT_SHA,native_candidate_binding=native,source_sha256=source_hashes,base_path=str(BASE),
        expected_base_UNet=base_records['unet/diffusion_pytorch_model.safetensors'],input_shape=[1,4,64,64],output_shape=[1,4,64,64],
        precision='FP16, only AFTER original BF16 trained-checkpoint load and safe PEFT fusion/unload',model_t=200,caption='',
        shape_policy='Original native512 tiles for every image; no resize or case/benchmark recognition',
        fixed_buffers='Original BF16 empty CLIP embedding converted FP16 and original integer timestep200',
        engine_scope='G(z_in,t200,empty_text).sample only; original VAE/posterior, latent scaling, scheduler/DDPM/noise, Gaussian tiling and wavelet stay outside',
        fallback=False,require_full_compilation=True,checkpoint_finetuned=True,backend_build_has_no_training=True,distillation=False,quality_reuse_claim=False,component_timing_not_tool_prediction=True)
    print(json.dumps(plan),flush=True)
    if not a.execute:return
    d.mkdir(parents=True);phase='artifact_hash_guards'
    try:
        need(sha(RAW)==RAW_SHA,'Actual registered trained raw changed')
        for records in [base_records,parent['base_provenance']['all_config_tokenizer_scheduler_metadata_hashes']]:
            for rel,record in records.items():need(sha(BASE/rel)==record['sha256'],'Actual base changed: '+rel)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        site=R/'backends/push7_trt/site';sys.path.insert(0,str(site));sys.path.insert(0,str(repo))
        import torch
        import numpy as np
        import torch_tensorrt
        import tensorrt as trt
        from HYPIR.enhancer.sd2 import SD2Enhancer
        from trt_vae_runtime_v2 import TensorRTEngine
        need(torch.cuda.is_available() and torch.__version__.split('+')[0]=='2.8.0' and torch_tensorrt.__version__.split('+')[0]=='2.8.0' and trt.__version__.startswith('10.12.'),'Reviewed native CUDA Torch2.8/TorchTRT2.8/TRT10.12 required')
        torch.cuda.set_device(a.device);device=torch.device(f'cuda:{a.device}');torch.set_num_threads(4)
        # Match original DFD initialization/fusion before any FP16 conversion.
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
        random.seed(231);np.random.seed(231);torch.manual_seed(231);torch.cuda.manual_seed_all(231)
        phase='original_BF16_trained_initialization_and_safe_fusion'
        m=SD2Enhancer(base_model_path=str(BASE),weight_path=str(RAW),lora_modules=MODULES,lora_rank=256,model_t=200,coeff_t=175,device=str(device))
        m.init_models();m.prepare_inputs(batch_size=1,prompt='')
        need(all(p.dtype==torch.bfloat16 for n,p in m.G.named_parameters() if 'lora' in n),'Original deployment LoRA BF16 load cast differs')
        m.G.fuse_lora(lora_scale=1.,safe_fusing=True,adapter_names=['default']);m.G.unload_lora();G=m.G.eval().requires_grad_(False)
        need(not any('lora' in n for n,_ in G.named_parameters()),'Merged G still has LoRA modules')
        embed_BF16=m.inputs['c_txt']['text_embed'].detach().clone();t=m.inputs['timesteps'].detach().clone()
        need(embed_BF16.dtype==torch.bfloat16 and tuple(embed_BF16.shape)==(1,77,1024) and t.tolist()==[200],'Original fixed empty conditioning differs')
        before=G.state_dict();need(all(not v.is_floating_point() or v.dtype==torch.bfloat16 for v in before.values()),'Merged G must be BF16 before FP16 conversion')
        def digest(state):
            h=hashlib.sha256();count=0;elements=0
            for name,v in state.items():
                cpu=v.detach().cpu().contiguous();h.update(json.dumps([name,list(cpu.shape),str(cpu.dtype)],separators=(',',':')).encode());h.update(b'\0');h.update(cpu.view(torch.uint8).numpy().tobytes());count+=1;elements+=v.numel()
            return dict(sha256=h.hexdigest(),tensors=count,elements=elements,recipe='ordered name/shape/dtype compactJSON, NUL, contiguous CPU raw tensor bytes')
        BF16_digest=digest(before);BF16_reference={}
        with torch.no_grad():
            for offset in [0,17,103]:
                v=(((torch.arange(16384,device=device,dtype=torch.float32).reshape(1,4,64,64)+offset).remainder(257)/128)-1).to(torch.bfloat16)
                BF16_reference[offset]=G(v,t,encoder_hidden_states=embed_BF16).sample.detach().cpu()
        G.half();after=G.state_dict();cast=[]
        for name,v in after.items():
            if not v.is_floating_point():continue
            original=before[name].float();new=v.float();delta=(new-original).abs()
            need(torch.isfinite(v).all(),'FP16 conversion overflow/nonfinite: '+name)
            cast.append(dict(name=name,elements=v.numel(),changed_elements=int((delta>0).sum()),underflow_to_zero=int(((original!=0)&(new==0)).sum()),max_abs=float(delta.max())))
        FP16_digest=digest(after);embedding=embed_BF16.half()
        need(torch.isfinite(embedding).all(),'Fixed conditioning FP16 overflow')
        conditioning=dict(BF16_sha256=digest({'empty_embedding':embed_BF16,'timestep':t}),FP16_sha256=digest({'empty_embedding':embedding,'timestep':t}),
            cast_changed_elements=int((embedding.float()!=embed_BF16.float()).sum()),cast_max_abs=float((embedding.float()-embed_BF16.float()).abs().max()))
        cast_summary=dict(changed_elements=sum(v['changed_elements'] for v in cast),underflow_to_zero=sum(v['underflow_to_zero'] for v in cast),max_abs=max(v['max_abs'] for v in cast),all_FP16_finite=True,
            policy='BF16 merged values converted FP16; tiny/subnormal rounding explicitly allowed by root; original FP32 weights are never reloaded for this cast',per_tensor=cast)
        del before,after,m,original,new,delta,embed_BF16;gc.collect();torch.cuda.empty_cache()
        class Component(torch.nn.Module):
            def __init__(self):
                super().__init__();self.G=G;self.register_buffer('empty_embedding',embedding);self.register_buffer('timestep',t)
            def forward(self,x):return self.G(x,self.timestep,encoder_hidden_states=self.empty_embedding).sample
        model=Component().eval().requires_grad_(False)
        def value(offset):return (((torch.arange(16384,device=device,dtype=torch.float32).reshape(1,4,64,64)+offset).remainder(257)/128)-1).half().contiguous()
        x=value(0);phase='torch_export_strictFalse';started=time.monotonic()
        with torch.no_grad():exported=torch.export.export(model,(x,),strict=False)
        phase='TensorRT_full_compilation'
        with torch.no_grad():
            engine=torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(exported,arg_inputs=[x],enabled_precisions={torch.float16},require_full_compilation=True,min_block_size=1,
                pass_through_build_failures=True,workspace_size=8*1024**3,optimization_level=3,max_aux_streams=0,immutable_weights=True,disable_tf32=True,truncate_double=False,timing_cache_path=str(d/'timing.cache'))
        engine_path=d/'component.engine'
        with engine_path.open('xb') as f:f.write(bytes(engine))
        build_s=time.monotonic()-started;engine_sha=sha(engine_path);phase='runtime_IO_and_synthetic_probes'
        runtime=TensorRTEngine(engine_path,engine_sha,device)
        need(runtime.input_shape==runtime.output_shape==(1,4,64,64) and runtime.input_dtype==runtime.output_dtype==torch.float16,'Fixed single latent64 FP16 IO changed')
        numerical=[];BF16_numerical=[];held=None
        with torch.no_grad():
            for offset in [0,17,103]:
                v=value(offset);copy=v.clone();expected=model(v);actual=runtime(v);torch.cuda.synchronize(device)
                need(torch.isfinite(actual).all() and torch.isfinite(expected).all() and torch.equal(v,copy),'Nonfinite synthetic/input mutated')
                if held is not None:need(torch.equal(held[0],held[1]),'Fresh output ownership broken')
                held=(actual,actual.clone());err=(actual.float()-expected.float()).abs()
                numerical.append(dict(input_offset=offset,float_exact=torch.equal(actual,expected),max_abs=float(err.max()),mean_abs=float(err.mean()),rmse=float(err.square().mean().sqrt())))
                original_error=(actual.detach().float().cpu()-BF16_reference[offset].float()).abs()
                BF16_numerical.append(dict(input_offset=offset,max_abs=float(original_error.max()),mean_abs=float(original_error.mean()),rmse=float(original_error.square().mean().sqrt())))
        phase='component_diagnostic_timing';baseline=torch.compile(model,mode='reduce-overhead',dynamic=False);timings={}
        with torch.no_grad():
            for label,fn in [('Torch_FP16_RO_component',baseline),('TensorRT_FP16_component',runtime)]:
                for _ in range(a.warm):fn(x)
                torch.cuda.synchronize(device);events=[];walls=[]
                for _ in range(a.n):
                    b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);wall=time.perf_counter();b.record();fn(x);e.record();torch.cuda.synchronize(device);events.append(b.elapsed_time(e));walls.append((time.perf_counter()-wall)*1000)
                need(all(math.isfinite(v) and v>0 for v in events+walls),'Invalid diagnostic times')
                timings[label]=dict(n=a.n,warm=a.warm,event_p10_ms=p10(events),event_median_ms=statistics.median(events),wall_p10_ms=p10(walls),raw_event_ms=events,raw_wall_ms=walls)
        result=dict(**plan,status='BUILT_COMPONENT_PROBED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_tensorrt=torch_tensorrt.__version__,tensorrt=trt.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device),
            actual_trained_raw_SHA_verified_by_builder=True,actual_base_UNet_sha256=sha(BASE/'unet/diffusion_pytorch_model.safetensors'),BF16_original_fused_G_state=BF16_digest,FP16_converted_fused_G_state=FP16_digest,FP16_cast_difference=cast_summary,fixed_conditioning=conditioning,
            build_s=build_s,engine_path=str(engine_path),engine_sha256=engine_sha,engine_bytes=engine_path.stat().st_size,actual_engine_IO=runtime.io_records,
            numerical_vs_same_FP16_merged_model=numerical,numerical_vs_original_BF16_merged_G=BF16_numerical,
            synthetic_probe_scope='Three deterministic latent64 tensors in[-1,1]; not a proof for all real native latent distributions',
            output_ownership_preserved=True,input_not_mutated=True,component_timings_diagnostic_only=timings,
            full_native_pipeline_quality_not_measured=True,original_A_quality_reuse_allowed=False,compiled_fallback=False,precision_change_from_A=True)
        write(d/'component_manifest.json',result);print(json.dumps(dict(status=result['status'],manifest=str(d/'component_manifest.json'),manifest_sha256=sha(d/'component_manifest.json'),build_s=build_s,engine_sha256=engine_sha,component_p10_ms={k:v['event_p10_ms'] for k,v in timings.items()},cast_changed_elements=cast_summary['changed_elements'])),flush=True)
    except BaseException as exc:
        if not (d/'failure.json').exists():write(d/'failure.json',dict(**plan,status='FAILED_NO_FALLBACK',phase=phase,error_type=type(exc).__name__,error=str(exc)))
        raise

if __name__=='__main__':main()
