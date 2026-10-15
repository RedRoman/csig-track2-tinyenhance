"""CPU original NIQE forward/input-autograd feasibility; no scorer changes."""
from pathlib import Path
import hashlib, json, os, sys, time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
R=Path('/cache/independent_sr_cuda_20261013');os.environ['TORCH_HOME']=str(R/'scoring_tool/weights/torch')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(R/'code'));import numpy_legacy_compat
    import pyiqa
    import pyiqa.archs.niqe_arch as NA
    import pyiqa.archs.func_util as FU
    import pyiqa.models.inference_model as IM
    import pyiqa.matlab_utils.functions as MF
    stats=R/'scoring_tool/weights/torch/hub/checkpoints/niqe_modelparameters.mat'
    assert stats.is_file() and not torch.cuda.is_initialized()
    normal=pyiqa.create_metric('niqe',device='cpu',as_loss=False,pretrained_model_path=str(stats)).eval().requires_grad_(False)
    loss=pyiqa.create_metric('niqe',device='cpu',as_loss=True,loss_reduction='none',pretrained_model_path=str(stats)).eval().requires_grad_(False)
    results=[]
    for kind,size in [('textured',384),('textured',512),('flat',384)]:
        started=time.perf_counter();i,j=torch.meshgrid(torch.arange(size,dtype=torch.float32),torch.arange(size,dtype=torch.float32),indexing='ij')
        if kind=='flat':rgb=torch.full((1,3,size,size),.5)
        else:
            fine=((i*17+j*29+(i*j).remainder(71)).remainder(101)/100-.5)
            base=.5+.13*torch.sin(i*.041)+.12*torch.cos(j*.063)+.08*torch.sin((i+j)*.173)+.13*fine
            rgb=torch.stack([base,.9*base+.04+.03*torch.sin(j*.11),.85*base+.07+.03*torch.cos(i*.09)])[None].clamp(0,1)
            rgb=torch.round(rgb*255)/255
        x=rgb.clone().requires_grad_(True)
        row=dict(kind=kind,size=size,input_rgb_sha256=hashlib.sha256(rgb.numpy().tobytes()).hexdigest())
        try:
            y0=normal(x);y=loss(x)
            row.update(forward=float(y.detach()),as_loss_false=float(y0),forward_exact_equal=bool(torch.equal(y0,y.detach())),as_loss_true_requires_grad=y.requires_grad,as_loss_false_requires_grad=y0.requires_grad,output_dtype=str(y.dtype))
            gradient=torch.autograd.grad(y.sum(),x)[0]
            finite=torch.isfinite(gradient);row.update(gradient_dtype=str(gradient.dtype),gradient_finite=bool(finite.all()),gradient_nonzero_elements=int((gradient!=0).sum()),gradient_L2=float(gradient.double().norm()),gradient_max_abs=float(gradient.abs().max()),
                NIQE_over10_gradient_L2=float(gradient.double().norm()/10),coefficient_in_loss_when_reward_point05=.005,coefficient_in_loss_when_reward_point1=.01)
            row['status']='PASS_FINITE_NONZERO_INPUT_GRADIENT' if bool(finite.all()) and bool(gradient.abs().max()>0) and row['forward_exact_equal'] else 'NONFINITE_OR_ZERO_OR_MISMATCH'
            if bool(finite.all()) and bool(gradient.abs().max()>0):
                # Real re-evaluation, not a guarantee: quantized forward may be
                # piecewise flat/nonmonotone despite the installed STE gradient.
                changes=[]
                for eps in [1/255,2/255]:
                    candidate=(rgb-eps*gradient.sign()).clamp(0,1)
                    actual=normal(candidate)
                    changes.append(dict(epsilon=eps,NIQE_after_negative_sign_step=float(actual),delta=float(actual)-float(y0)))
                row['actual_original_forward_downhill_sign_probes']=changes
        except BaseException as e:row.update(status='FORWARD_OR_BACKWARD_FAILED',error_type=type(e).__name__,error=str(e))
        row['CPU_seconds']=time.perf_counter()-started;results.append(row);print(json.dumps(row,allow_nan=True),flush=True)
    assert not torch.cuda.is_initialized()
    sources={str(Path(m.__file__).resolve()):sha(m.__file__) for m in [NA,FU,IM,MF]}
    result=dict(schema='push7_original_niqe_cpu_input_gradient_probe_v1',status='COMPLETED_CPU_ONLY',CPU_only=True,CUDA_initialized=False,Torch_threads=1,torch=torch.__version__,pyiqa=pyiqa.__version__,
        installed_source_sha256=sources,NIQE_parameters=dict(path=str(stats),sha256=sha(stats)),probe_source_sha256=sha(__file__),results=results,
        estimator='Installed original NIQE unchanged; argmin alpha treated as piecewise constant and diff_round is installed STE. Gradient is not exact finite-difference derivative of quantized score.',
        previous_training_omission='Frozen helper NAMES=[musiq,maniqa,clipiqa,lpips] and new directNR explicit three-score dict omitted NIQE; prior claim formal-only/no-gradient was not verified for this installed implementation.',
        proposed_only_if_stable='Future HYPIR direct NR may add original niqe(as_lossTrue)/10 to loss with reward_weight; no new training launched. Crop statistics differ from native full score; exact formal original networks/formula remain unchanged.')
    # JSON must remain standard even a failure exposed NaN gradient statistics.
    def clean(v):
        if isinstance(v,float) and not __import__('math').isfinite(v):return str(v)
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    out=R/'analysis/push7/niqe_cpu_gradient/niqe_input_gradient_probe.json';out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists();out.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],out=str(out),sha256=sha(out),CUDA_initialized=False)),flush=True)

if __name__=='__main__':main()
