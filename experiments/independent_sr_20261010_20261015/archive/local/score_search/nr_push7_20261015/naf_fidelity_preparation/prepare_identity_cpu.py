"""CPU-only fresh identity initialization and portable probe registration.

No training, GPU initialization, old weight reads, or evaluation claims.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil

R=Path('/cache/independent_sr_cuda_20261013')
TEACHER=R/'nr_push4_full_evaluations/push4_a7750_compiled_all'
MANIFEST_SHA='35355c94cace390b5f5918cfb607d85b0f55c9cd1e1ba0a4f6a4d034abb54e16'
TEACHER_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--model-dir',type=Path,default=R/'submissions/naf_push7_identity_probe/model_dir')
    p.add_argument('--run',type=Path,default=R/'adaptation/runs/naf_push7_A7750_fidelity')
    a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='CPU_PREPARATION_PLAN',model_dir=str(a.model_dir),run=str(a.run))));return
    if a.model_dir.exists() or a.run.exists():raise RuntimeError('New isolated destinations must not exist')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(4);torch.manual_seed(231)
    src=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('naf',src/'naf_model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ln_checks=[]
    for dtype,tol in [(torch.float64,1e-10),(torch.float32,3e-5)]:
        x=torch.randn(2,7,5,9,dtype=dtype,requires_grad=True);x2=x.detach().clone().requires_grad_(True)
        norm=mod.LayerNorm2d(7).to(dtype);w=norm.weight.detach().clone().requires_grad_(True);b=norm.bias.detach().clone().requires_grad_(True)
        y=norm(x);ref=F.layer_norm(x2.movedim(1,-1),(7,),w,b,1e-6).movedim(-1,1)
        v=torch.randn_like(y);g=torch.autograd.grad((y*v).sum(),(x,norm.weight,norm.bias));gr=torch.autograd.grad((ref*v).sum(),(x2,w,b))
        err=max((y-ref).abs().max().item(),*((u-v).abs().max().item() for u,v in zip(g,gr)))
        assert err<tol,(dtype,err)
        ln_checks.append(dict(dtype=str(dtype),forward_gradient_max_abs_error=err,tolerance=tol))
    model=mod.build_model().eval()
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    assert len(list(model.parameters()))==664 and sum(v.numel() for v in model.parameters())==17111907
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
    with torch.no_grad():
        x=torch.rand(1,3,33,47);y=model(x)
        assert torch.equal(x,y)
    class Identity(torch.nn.Module):
        def __init__(self,dtype):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros((),dtype=dtype))
        def forward(self,x):return x+self.dummy*0
    q=torch.arange(512,dtype=torch.float32)-255.5;g=torch.exp(-.5*(q/128).square());gaussian=(g[:,None]*g[None,:])[None,None]
    tiling=[]
    for shape in [(33,47),(512,512),(513,777)]:
        x=torch.rand(1,3,*shape)
        for dtype in [torch.float32,torch.float16]:
            y=mod.native_tiled(Identity(dtype),x,gaussian)
            assert torch.equal(x,y) and x.data_ptr()!=y.data_ptr()
            tiling.append(dict(shape=list(shape),NN_dtype=str(dtype),exact_native_identity=True,fresh_output=True))
    manifest=TEACHER/'render_manifest.json';assert sha(manifest)==MANIFEST_SHA
    catalog=json.loads(manifest.read_text());records=[r for r in catalog['records'] if r['split']=='test']
    assert [r['case'] for r in records]==list(range(1,101))
    assert catalog['protocol']['runner']['checkpoint_sha256']==TEACHER_SHA
    teacher=[]
    for rec in records:
        path=TEACHER/rec['output'];assert sha(path)==rec['sha256']
        with Image.open(path) as im:assert list(im.size)==rec['size'] and im.mode=='RGB'
        inp=R/'source_payload/assets/input'/f"case{rec['case']}.jpg"
        with Image.open(inp) as im:assert list(im.size)==rec['size']
        teacher.append(dict(case=rec['case'],input=str(inp),input_sha256=sha(inp),teacher=str(path),teacher_sha256=rec['sha256'],size=rec['size']))
    origin=dict(initialization='fresh random NAFNet32 parameters; zero ending identity; no pretrained model',G4_used=False,
                purpose='untrained speed gate only; no claimed teacher fit',teacher_checkpoint_sha256=TEACHER_SHA)
    a.run.mkdir(parents=True);a.model_dir.mkdir(parents=True)
    raw=a.run/'step0.pth'
    torch.save(dict(schema='nafnet32_native_fidelity_raw_v1',cfg=mod.ARCHITECTURE,state=state,local_step=0,optimizer_step=0,origin=origin),raw)
    os.link(raw,a.model_dir/'model.pth')
    for name in ['naf_model.py','runner.py']:shutil.copyfile(src/name,a.model_dir/name)
    config=dict(model_type='nafnet32_native_fidelity_v1',label='naf_push7_identity_probe',core=512,halo=64,stride=256,seed=231,
                local_step=0,optimizer_step=0,checkpoint_sha256=sha(raw),architecture_source_sha256=sha(src/'naf_model.py'),
                teacher_checkpoint_sha256=TEACHER_SHA,teacher_manifest=str(manifest),teacher_manifest_sha256=MANIFEST_SHA,
                training_provenance=None)
    (a.model_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (a.run/'teacher_catalog.json').write_text(json.dumps(teacher,indent=2)+'\n')
    proof=dict(schema='nafnet32_identity_cpu_proof_v1',status='PASS',CPU_only=True,CUDA_context_initialized=torch.cuda.is_initialized(),
               source_sha256={n:sha(src/n) for n in ['naf_model.py','runner.py','prepare_identity_cpu.py','ORIGINAL_NAFNet_arch.py']},
               LayerNorm_forward_gradient=ln_checks,parameter_tensors=664,parameter_elements=17111907,all_parameters_FP32_finite=True,
               actual_model_FP32_zero_ending_exact_identity=True,beta_gamma_initial_zero=True,first_step_all_parameters_nonzero_not_required=True,
               native_tiler=tiling,no_resize=True,core=512,halo=64,NN_context=640,stride=256,benchmark_same_640_context=True,
               teacher_actual_test_PNGs_checked=100,validation_data_used=False,teacher_catalog_sha256=sha(a.run/'teacher_catalog.json'),
               raw=str(raw),raw_sha256=sha(raw),model_dir=str(a.model_dir),config_sha256=sha(a.model_dir/'config.json'),origin=origin)
    assert not proof['CUDA_context_initialized']
    (a.run/'identity_cpu_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
