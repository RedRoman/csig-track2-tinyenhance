"""Fresh NAFNet32 native fidelity student, fixed640-context/native512 core tiling.

No G4 architecture/weights, no case recognition, no external image downscale.
Training provenance is metadata only; inference reads this model_dir and Torch.
"""
from pathlib import Path
import hashlib
import importlib.util
import json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


class Runner:
    def __init__(self,modeldir):
        import torch
        self.model_dir=Path(modeldir).resolve()
        c=json.loads((self.model_dir/'config.json').read_text());self.config=c
        if (c['model_type'],c['core'],c['halo'],c['stride'],c['seed']) != ('nafnet32_native_fidelity_v1',512,64,256,231):
            raise ValueError('Unsupported model/native tiling policy')
        if not torch.cuda.is_available():raise RuntimeError('Native CUDA required')
        self.device=torch.device('cuda:0');torch.cuda.set_device(self.device);torch.set_num_threads(4)
        torch.manual_seed(231);torch.cuda.manual_seed(231)
        path=self.model_dir/'naf_model.py'
        if sha(path)!=c['architecture_source_sha256']:raise RuntimeError('Architecture source changed')
        spec=importlib.util.spec_from_file_location('_push7_naf_model',path)
        source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
        self.native_tiled=source.native_tiled
        raw=self.model_dir/'model.pth'
        if sha(raw)!=c['checkpoint_sha256']:raise RuntimeError('Student checkpoint changed')
        ckpt=torch.load(raw,map_location='cpu',weights_only=False)
        if ckpt['cfg']!=source.ARCHITECTURE or ckpt['local_step']!=c['local_step']:raise RuntimeError('Checkpoint/config differs')
        self.model=source.build_model();self.model.load_state_dict(ckpt['state'],strict=True)
        self.model=self.model.eval().requires_grad_(False).to(device=self.device,dtype=torch.float16)
        self.model=torch.compile(self.model,mode='reduce-overhead',dynamic=False)
        a=torch.arange(512,device=self.device,dtype=torch.float32)-255.5
        g=torch.exp(-.5*(a/128).square());self.gaussian=(g[:,None]*g[None,:])[None,None]
        self._path='NAFNet32_native_core512_halo64_stride256_gaussian_compileRO_fp16'
        self.protocol=dict(schema='nafnet32_native_fidelity_runner_v1',model_type=c['model_type'],label=c['label'],runner_path=self._path,
            weight_path=str(raw),checkpoint_sha256=sha(raw),runner_sha256=sha(__file__),repo_path=str(self.model_dir),
            source_sha256={'naf_model.py':sha(path)},model_dir_config_sha256=sha(self.model_dir/'config.json'),
            origin=ckpt['origin'],local_step=ckpt['local_step'],optimizer_step=ckpt['optimizer_step'],
            input='CUDA FP16[-1,1] native -> FP32[0,1] -> fixed native640 context NN FP16; no resize',
            output='fresh CUDA FP32[-1,1] native; native FP32 identity plus Gaussian-averaged NN residual, final clamp',
            internal_precision='NN FP16; channel LayerNorm statistics FP32; accumulation FP32',
            seed='231 before initialization; deterministic student inference, no stochastic forward',
            core=512,halo=64,NN_context=640,stride=256,no_resize=True,Gaussian_sigma=128,
            pad='minimum512 replicate if needed, then64 reflect halo; NAF internal mod16 constant pad; crop original native',
            tiling='same positions/minimum-pad/640context/Gaussian formula for every input, including512 benchmark',
            no_warmup_during_initialization=True,output_ownership='new accumulation/output each call; no image/output cache',
            teacher_checkpoint_sha256=c['teacher_checkpoint_sha256'],teacher_manifest=c['teacher_manifest'],
            teacher_manifest_sha256=c['teacher_manifest_sha256'],training_provenance=c.get('training_provenance'),
            training_data_scope='test100 fixed native A7750 pseudo labels; not validation GT or a claim of new GT generalization')

    def infer(self,x):
        import torch
        if not isinstance(x,torch.Tensor) or x.dtype!=torch.float16 or x.device!=self.device or x.ndim!=4 or x.shape[:2]!=(1,3):
            raise TypeError('Expected CUDA0 FP16 batch1 RGB')
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            native=(x.float()+1)*.5
            y=self.native_tiled(self.model,native,self.gaussian)
            return y.clamp(0,1)*2-1
