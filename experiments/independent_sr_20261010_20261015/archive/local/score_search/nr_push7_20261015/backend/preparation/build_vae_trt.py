"""Build/probe a SAME-WEIGHT fixed512 VAE component; default CPU plan only.

Root explicitly schedules --execute on one GPU. BF16 first, no automatic FP16
fallback. Component timing is diagnostic and cannot be used in original predict.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import statistics
import time

BASE=Path('/temp/zihaohe_stage/baselines/sd21b')
VAE_SHA='a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815'


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def p10(v):
    v=sorted(v);p=(len(v)-1)*.1;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--component',choices=['decoder','encoder'],default='decoder')
    ap.add_argument('--precision',choices=['bf16','fp16'],default='bf16');ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--device',type=int,default=0);ap.add_argument('--warm',type=int,default=30);ap.add_argument('--n',type=int,default=50)
    a=ap.parse_args();d=a.out_dir.resolve();need(not (d/'component_manifest.json').exists(),'Immutable build already complete')
    need(1<=a.n<=200 and 0<=a.warm<=40,'Invalid probe count')
    shape=(1,4,64,64) if a.component=='decoder' else (1,3,512,512)
    outshape=(1,3,512,512) if a.component=='decoder' else (1,8,64,64)
    plan=dict(component=a.component,precision=a.precision,input_shape=list(shape),output_shape=list(outshape),out_dir=str(d),execute=a.execute,shape_policy='Original512 tile for ALL native input images; no input recognition or resolution scaling',reference_VAE_SHA256=VAE_SHA,script_sha256=sha(__file__),runtime_sha256=sha(Path(__file__).with_name('trt_vae_runtime.py')))
    print(json.dumps(plan),flush=True)
    if not a.execute:return
    import torch
    import torch_tensorrt
    import tensorrt as trt
    from diffusers import AutoencoderKL
    from trt_vae_runtime import TensorRTEngine
    need(torch.__version__.split('+')[0]=='2.8.0' and torch_tensorrt.__version__.split('+')[0]=='2.8.0' and trt.__version__.startswith('10.12.'),'Use the reviewed Torch2.8/TorchTRT2.8/TRT10.12 isolated stack')
    torch.cuda.set_device(a.device);device=torch.device(f'cuda:{a.device}');td=torch.bfloat16 if a.precision=='bf16' else torch.float16
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    need(sha(BASE/'vae/diffusion_pytorch_model.safetensors')==VAE_SHA,'Original VAE weights changed')
    vae=AutoencoderKL.from_pretrained(str(BASE),subfolder='vae',torch_dtype=td).to(device).eval().requires_grad_(False)
    need(not vae.use_tiling and not vae.use_slicing,'Original VAE does not use internal tiling/slicing')
    class Component(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if a.component=='decoder':self.pre=vae.post_quant_conv;self.core=vae.decoder
            else:self.core=vae.encoder;self.pre=vae.quant_conv
        def forward(self,x):
            if a.component=='decoder':return self.core(self.pre(x)) if self.pre is not None else self.core(x)
            x=self.core(x);return self.pre(x) if self.pre is not None else x
    model=Component().eval().requires_grad_(False)
    count=1
    for n in shape:count*=n
    def input_value(offset):
        q=torch.arange(count,device=device,dtype=torch.float32).reshape(shape)
        return (((q+offset).remainder(257)/128)-1).to(td).contiguous()
    x=input_value(0);d.mkdir(parents=True,exist_ok=True);engine_path=d/'component.engine';need(not engine_path.exists(),'Do not overwrite a partial/previous engine')
    began=time.monotonic()
    with torch.no_grad():
        exported=torch.export.export(model,(x,),strict=False)
        engine=torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported,arg_inputs=[x],enabled_precisions={td},require_full_compilation=True,
            min_block_size=1,pass_through_build_failures=True,workspace_size=8*1024**3,
            optimization_level=3,max_aux_streams=0,immutable_weights=True,disable_tf32=True,
            truncate_double=False,timing_cache_path=str(d/'timing.cache'))
    with engine_path.open('xb') as f:f.write(bytes(engine))
    build_s=time.monotonic()-began;engine_sha=sha(engine_path);runtime=TensorRTEngine(engine_path,engine_sha,device)
    need(runtime.input_shape==shape and runtime.output_shape==outshape and runtime.input_dtype==td and runtime.output_dtype==td,'Engine changed fixed component IO contract')
    numerical=[];held=None
    with torch.no_grad():
        for offset in [0,17,103]:
            value=input_value(offset);ref=model(value);y=runtime(value);torch.cuda.synchronize(device)
            need(tuple(y.shape)==outshape and bool(torch.isfinite(y).all()) and bool(torch.isfinite(ref).all()),'Nonfinite/wrong probe output')
            if held is not None:need(torch.equal(held[0],held[1]),'Engine output ownership broken')
            held=(y,y.clone());delta=(y.float()-ref.float()).abs();numerical.append(dict(input_offset=offset,float_exact=torch.equal(y,ref),max_abs=float(delta.max()),mean_abs=float(delta.mean()),rmse=float(delta.square().mean().sqrt())))
    # Same deterministic tensor and same precision in this component diagnostic.
    # This is not the original full Runner benchmark or a new quality result.
    baseline=torch.compile(model,mode='reduce-overhead',dynamic=False)
    measurements={}
    with torch.no_grad():
        for label,fn in [('Torch_RO_component',baseline),('TensorRT_component',runtime)]:
            for _ in range(a.warm):y=fn(x)
            torch.cuda.synchronize(device);samples=[];walls=[]
            for _ in range(a.n):
                b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);t=time.perf_counter();b.record();y=fn(x);e.record();torch.cuda.synchronize(device);samples.append(b.elapsed_time(e));walls.append((time.perf_counter()-t)*1000)
            measurements[label]=dict(n=a.n,warm=a.warm,event_p10_ms=p10(samples),event_median_ms=statistics.median(samples),wall_p10_ms=p10(walls),raw_event_ms=samples,raw_wall_ms=walls)
    result=dict(**plan,status='BUILT_COMPONENT_PROBED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_tensorrt=torch_tensorrt.__version__,tensorrt=trt.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device),build_s=build_s,engine_path=str(engine_path),engine_sha256=engine_sha,engine_bytes=engine_path.stat().st_size,numerical_vs_same_precision_original=numerical,output_ownership_preserved=True,component_timings_diagnostic_only=measurements,full_native_pipeline_quality_not_measured=True,original_A_quality_reuse_allowed=False,normalization='TorchTRT2.8 native_group_norm uses TensorRT INormalizationLayer; TRT10.x defaults FP32 accumulation. No FP16/INT8 requested for BF16 build.',posterior_policy='Encoder returns moments only. Original DiagonalGaussianDistribution and sample stay outside engine; decoder contains post_quant_conv+decoder only.',precision_change_from_A=(a.precision!='bf16'),references=['https://github.com/pytorch/TensorRT/releases/tag/v2.8.0','https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html'])
    with (d/'component_manifest.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'manifest':str(d/'component_manifest.json'),'manifest_sha256':sha(d/'component_manifest.json'),'build_s':build_s,'diagnostic_event_p10_ms':{k:v['event_p10_ms'] for k,v in measurements.items()},'numerical':numerical}),flush=True)


if __name__=='__main__':main()
