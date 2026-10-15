"""Root-scheduled single GPU parity/backward probe, default CPU-only plan.

Not training, IQA, official timing or a new score. Same A7750 architecture/raw.
Independent real PEFT fusion provides the expected BF16 weights/UNet forward.
"""
from pathlib import Path
import argparse
import datetime
import gc
import hashlib
import json
import random
import time

R=Path('/cache/independent_sr_cuda_20261013')
BASE=Path('/temp/zihaohe_stage/baselines/sd21b')
RAW=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
RAW_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
MODULES='to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj'.split(',')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',type=int,default=0)
    a=ap.parse_args();need(not a.out.exists(),'Use a new probe output path')
    module=Path(__file__).with_name('deployment_fused_lora.py')
    plan=dict(schema='hypir_deployment_fused_lora_probe_v1',execute=a.execute,raw=str(RAW),raw_sha256=RAW_SHA,base=str(BASE),script_sha256=sha(__file__),module_sha256=sha(module),out=str(a.out),GPU_only_if_execute=True,teacher_output_loss=False,training_updates=0)
    print(json.dumps(plan),flush=True)
    if not a.execute:return
    import torch
    import numpy as np
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig
    from deployment_fused_lora import install_deployment_fused_forward,installed_layers,deployment_merged_weight
    need(torch.cuda.is_available() and torch.__version__.split('+')[0]=='2.8.0','Reviewed native Torch2.8 CUDA required')
    torch.cuda.set_device(a.device);device=torch.device(f'cuda:{a.device}');torch.set_num_threads(4)
    # Original A Runner actual flags; no TensorRT diagnostic flags inherited.
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    random.seed(231);np.random.seed(231);torch.manual_seed(231);torch.cuda.manual_seed(231)
    need(sha(RAW)==RAW_SHA,'Immutable A7750 raw changed')
    raw=torch.load(RAW,map_location='cpu',weights_only=True,mmap=True)
    need(len(raw)==514 and sum(v.numel() for v in raw.values())==259474432,'Not original514 FP32 LoRA')
    need(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in raw.values()),'Raw masters must be finite FP32')
    def make(master):
        g=UNet2DConditionModel.from_pretrained(str(BASE),subfolder='unet',torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
        g.add_adapter(LoraConfig(r=256,lora_alpha=256,init_lora_weights='gaussian',target_modules=MODULES))
        need({n for n in g.state_dict() if 'lora' in n}==set(raw),'LoRA names differ from A7750')
        g.load_state_dict(raw,strict=False);g.requires_grad_(False)
        for n,p in g.named_parameters():
            if n in raw:
                if master:p.data=raw[n].to(device=device,dtype=torch.float32).clone();p.requires_grad_(True)
                else:need(p.dtype==torch.bfloat16 and torch.equal(p,raw[n].to(device=device,dtype=torch.bfloat16)),'Deployment loader cast differs')
        return g
    began=time.monotonic();master=make(True);reference=make(False)
    initial_master_exact=all(torch.equal(p,raw[n].to(device)) for n,p in master.named_parameters() if n in raw)
    need(initial_master_exact,'FP32 master restore differs')
    report=install_deployment_fused_forward(master)
    ref_modules={n:m for n,m in reference.named_modules() if hasattr(m,'lora_A')}
    # Independent actual installed PEFT CUDA safe-fusion, after BF16 load casts.
    reference.fuse_lora(lora_scale=1.0,safe_fusing=True,adapter_names=['default'])
    weight_checks=[]
    with torch.no_grad():
        for name,layer in installed_layers(master):
            expected=ref_modules[name].get_base_layer().weight
            merged=deployment_merged_weight(layer)
            exact=torch.equal(merged,expected);need(exact,'Deployment BF16 merged-weight parity failed: '+name)
            weight_checks.append(dict(name=name,shape=list(merged.shape),dtype=str(merged.dtype),exact=True))
    reference.unload_lora();reference.eval().requires_grad_(False)
    def deterministic(shape,mod,den):
        count=1
        for n in shape:count*=n
        return ((torch.arange(count,device=device,dtype=torch.float32).reshape(shape).remainder(mod)/den)-1).to(torch.bfloat16)
    latent=deterministic((1,4,64,64),257,128);cond=deterministic((1,77,1024),251,125);timestep=torch.tensor([200],device=device,dtype=torch.long)
    def states():
        # Hash all four actual states, no seed reset inside any forward.
        return dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),torchCPU=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),CUDA=hashlib.sha256(torch.cuda.get_rng_state(device).numpy().tobytes()).hexdigest())
    before_rng=states()
    with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
        expected=reference(latent,timestep,encoder_hidden_states=cond).sample
        actual=master(latent,timestep,encoder_hidden_states=cond).sample
    torch.cuda.synchronize(device);diff=(actual.float()-expected.float()).abs()
    exact=torch.equal(actual,expected)
    forward=dict(exact=exact,max_abs=float(diff.max()),mean_abs=float(diff.mean()),shape=list(actual.shape),dtype=str(actual.dtype),all_finite=bool(torch.isfinite(actual).all()))
    need(exact,'Eager fused UNet parity failed; do not start NR training')
    del expected,actual,reference,ref_modules;gc.collect();torch.cuda.empty_cache()
    master.enable_gradient_checkpointing()
    for name,layer in installed_layers(master):layer._deployment_forward_calls=0
    # A real dense forward/backward through every installed LoRA layer. This
    # scalar is an explicitly labeled gradient probe, not an IQA reward/loss.
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        out=master(latent,timestep,encoder_hidden_states=cond).sample
        loss=out.float().square().mean()+.01*out.float().mean()
    loss.backward();torch.cuda.synchronize(device)
    gradients=[]
    for name,p in master.named_parameters():
        if not p.requires_grad:continue
        need(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) and bool(p.grad.count_nonzero()),'Missing/nonfinite/zero FP32 master gradient: '+name)
        gradients.append(dict(name=name,dtype=str(p.grad.dtype),finite=True,nonzero=True,max_abs=float(p.grad.abs().max())))
    need(len(gradients)==514,'Gradient count differs')
    calls={name:layer._deployment_forward_calls for name,layer in installed_layers(master)}
    need(all(n>=1 for n in calls.values()) and any(n>1 for n in calls.values()),'Checkpoint recomputation was not exercised')
    after_rng=states();need(before_rng==after_rng,'UNet parity/backward unexpectedly changed default RNG')
    need(all(torch.equal(p,raw[n].to(device)) for n,p in master.named_parameters() if n in raw),'Probe unexpectedly updated master weights')
    result=dict(**plan,status='PASS_DEPLOYMENT_FUSED_PARITY_AND_514_GRADIENTS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,gpu=torch.cuda.get_device_name(device),numeric_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark),installation=report,initial_FP32_masters_exact=True,merged_weight_exact_count=len(weight_checks),merged_weight_checks=weight_checks,eager_fused_UNet_forward=forward,gradient_probe_loss=float(loss.detach()),gradient_probe_scope='deterministic UNet scalar, not NR/IQA; no optimization update',gradient_FP32_finite_nonzero_count=len(gradients),gradients=gradients,checkpoint_recompute_forward_call_counts=calls,all_fourRNG_before=before_rng,all_fourRNG_after=after_rng,all_fourRNG_unchanged=True,master_weights_unchanged=True,compiled_forward_parity_not_measured=True,full_enhance_and_NR_metrics_not_measured=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),wall_s=time.monotonic()-began)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],out=str(a.out),sha256=sha(a.out),merged_weight_exact_count=len(weight_checks),forward=forward,real_FP32_gradient_tensors=len(gradients),checkpoint_recomputed_modules=sum(n>1 for n in calls.values()),peak_allocated_bytes=result['peak_allocated_bytes'],wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
