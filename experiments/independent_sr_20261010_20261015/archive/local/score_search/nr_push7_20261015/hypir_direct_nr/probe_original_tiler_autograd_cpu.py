"""Small CPU check of the unchanged HYPIR Gaussian tiler's autograd graph.

This does not test CUDA LoRA arithmetic, VAE/UNet output quality, or speed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import torch
    from torch.utils.checkpoint import checkpoint
    torch.set_num_threads(2)
    source=R/'repos/HYPIR_cuda/HYPIR/utils/common.py'
    spec=importlib.util.spec_from_file_location('_unchanged_original_tiler',source)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    # FP64 finite-difference check includes overlapping in-place CopySlices and
    # the original FP32 count accumulation. No rewrite of the tiler formula.
    x=(torch.arange(3*3*4,dtype=torch.float64).reshape(1,3,3,4)/37).requires_grad_()
    p=torch.tensor([.7,.9,1.1],dtype=torch.float64,requires_grad=True)
    def fun(x,p):return mod.make_tiled_fn(lambda tile:tile*p[None,:,None,None],size=2,stride=1,progress=False)(x)
    gradcheck=torch.autograd.gradcheck(fun,(x,p),eps=1e-6,atol=1e-5,rtol=1e-4)
    rows=[]
    for dtype in [torch.float64,torch.bfloat16]:
        a=(torch.arange(3*5*7,dtype=torch.float32).reshape(1,3,5,7)/101).to(dtype).requires_grad_()
        w=torch.nn.Parameter(torch.tensor([.7,.9,1.1],dtype=torch.float32));calls=[0]
        def f(tile):calls[0]+=1;return tile*w.to(dtype)[None,:,None,None]
        plain=mod.make_tiled_fn(f,size=4,stride=2,progress=False)(a)
        gplain=torch.autograd.grad(plain.float().square().mean(),(a,w),retain_graph=False)
        a2=a.detach().clone().requires_grad_();w2=torch.nn.Parameter(w.detach().clone());calls2=[0]
        def fc(tile):calls2[0]+=1;return tile*w2.to(dtype)[None,:,None,None]
        checked=mod.make_tiled_fn(lambda tile:checkpoint(fc,tile,use_reentrant=False),size=4,stride=2,progress=False)(a2)
        gchecked=torch.autograd.grad(checked.float().square().mean(),(a2,w2))
        assert torch.equal(plain,checked) and all(torch.equal(u,v) for u,v in zip(gplain,gchecked))
        assert all(torch.isfinite(g).all() and g.count_nonzero()>0 for g in gchecked)
        assert calls2[0]>calls[0]
        rows.append(dict(dtype=str(dtype),output_dtype=str(checked.dtype),plain_calls=calls[0],checkpoint_calls=calls2[0],checkpoint_outputs_exact=True,input_and_FP32_parameter_gradients_exact=True,finite_nonzero_gradients=True))
    assert gradcheck and not torch.cuda.is_initialized()
    out=R/'analysis/push7/hypir_direct_nr/original_tiler_autograd_cpu.json'
    assert not out.exists()
    result=dict(status='PASS',CPU_only=True,CUDA_context_initialized=False,script_sha256=sha(__file__),original_tiler_source=str(source),original_tiler_source_sha256=sha(source),original_tiler_used_without_mutation=True,FP64_overlapping_CopySlices_gradcheck=gradcheck,checkpoint_tests=rows,scope='Small original-tiler CPU autograd/control only; actual CUDA full-HYPIR NR preflight remains required')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
