"""Original NIQE CPU autograd on existing frozen A7750 panel crop."""
from pathlib import Path
import hashlib,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
R=Path('/cache/independent_sr_cuda_20261013');D=R/'analysis/push7/niqe_cpu_gradient'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import torch,numpy as np
    from PIL import Image
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(R/'code'));import numpy_legacy_compat
    import pyiqa
    path=D/'A7750_case16_roi512.png';proof=D/'A7750_case16_roi512_provenance.json';p=json.loads(proof.read_text());assert sha(path)==p['output_sha256']
    with Image.open(path) as im:rgb=torch.from_numpy(np.asarray(im.convert('RGB'),dtype=np.float32).copy()/255).permute(2,0,1)[None]
    stats=R/'scoring_tool/weights/torch/hub/checkpoints/niqe_modelparameters.mat'
    m0=pyiqa.create_metric('niqe',device='cpu',as_loss=False,pretrained_model_path=str(stats)).eval().requires_grad_(False)
    m=pyiqa.create_metric('niqe',device='cpu',as_loss=True,loss_reduction='none',pretrained_model_path=str(stats)).eval().requires_grad_(False)
    started=time.perf_counter();x=rgb.clone().requires_grad_(True);y0=m0(x);y=m(x);g=torch.autograd.grad(y.sum(),x)[0]
    assert torch.equal(y0,y.detach()) and torch.isfinite(g).all() and bool(g.abs().max()>0) and not torch.cuda.is_initialized()
    changes=[dict(epsilon=e,actual_original_NIQE=float(m0((rgb-e*g.sign()).clamp(0,1)))) for e in [1/255,2/255]]
    result=dict(status='PASS_REAL_A_ROI_FINITE_NONZERO',CPU_only=True,CUDA_initialized=False,source_provenance=p,source_proof_sha256=sha(proof),input_sha256=sha(path),
        NIQE=float(y.detach()),normal_forward=float(y0),as_loss_forward_exact=True,gradient_dtype=str(g.dtype),gradient_L2=float(g.double().norm()),gradient_max_abs=float(g.abs().max()),gradient_nonzero_elements=int((g!=0).sum()),
        rw01_weighted_NIQE_gradient_L2=float(.01*g.double().norm()),rw005_weighted_NIQE_gradient_L2=float(.005*g.double().norm()),
        original_score_clamp_active=bool(y0<10),actual_original_forward_negative_sign_probes=changes,CPU_seconds=time.perf_counter()-started,
        source_sha256=sha(__file__),NIQE_parameters_sha256=sha(stats),limitations='One fixed512 ROI, not full/native metrics or a LoRA-gradient/GPU proof; installed alpha argmin and diff_round STE stay unchanged.')
    out=D/'niqe_real_A_roi_probe.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));print('SHA256',sha(out))
if __name__=='__main__':main()
