"""HYPIR native512 hybrid: existing FP16 TensorRT VAE, original BF16 UNet.

No training/distillation, downscale, new tiling, sampler substitution or RNG reset
inside infer. The frozen original Runner remains a byte-identical separate file.
Changing VAE arithmetic requires new full103/418 and visual evaluation.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys

REFERENCE_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
RUNTIME_SHA='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4'
VAE_SHA='a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815'
ENGINES={
    'encoder':('14c206896a7caa6b6491dae33133d149283025cfb8e15b6cf129d8854176ba55','21d3781a5cc092062d4ef87844e74ce916c461ea5b9939658917333d74a2f0ae',(1,3,512,512),(1,8,64,64)),
    'decoder':('fe81087086d76aa8d0111cb80070b65b897e0438ebb2a8a020ef0ced8bc6bf30','1a3a349641e65ca5ad5d92301b25b3b1eae214e760645b5a7c294460f0ac08fd',(1,4,64,64),(1,3,512,512)),
}

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

class Runner:
    def __init__(self,modeldir):
        md=Path(modeldir).resolve();c=json.loads((md/'config.json').read_text())
        need(c.get('backend')=='HYPIR_BF16_UNet_FP16_TRT_VAE_native512','Explicit hybrid backend required')
        need(sha(md/'reference_runner.py')==REFERENCE_SHA and sha(md/'trt_vae_runtime_v2.py')==RUNTIME_SHA,'Pinned original runner/runtime source changed')
        need(sha(c['weight_path'])==c['checkpoint_sha256'],'Explicit HYPIR LoRA checkpoint SHA differs')
        need(sha(c['base_vae_path'])==VAE_SHA,'Original VAE base file changed')
        manifests={}
        for name,(eh,mh,inp,out) in ENGINES.items():
            spec=c['TRT_engines'][name];need(spec['sha256']==eh and spec['manifest_sha256']==mh,'Unreviewed engine declaration')
            need(sha(spec['path'])==eh and sha(spec['manifest'])==mh,'Actual engine/manifest SHA differs')
            m=json.loads(Path(spec['manifest']).read_text());manifests[name]=m
            need(m['status']=='BUILT_COMPONENT_PROBED' and m['component']==name and m['precision']=='fp16' and m['engine_sha256']==eh
                 and m['reference_VAE_SHA256']==VAE_SHA and tuple(m['input_shape'])==inp and tuple(m['output_shape'])==out,'Sealed component identity/shape changed')
        ref=load('_push7_hypir_trt_reference',md/'reference_runner.py')
        # Original construction seed231 and LoRA initialization/fusion/compilation.
        self.original=ref.Runner(md)
        import torch
        import numpy as np
        import random
        self.device=self.original.device;self.model=self.original.model;self.model_dir=md
        def capture():
            return dict(python=random.getstate(),numpy=np.random.get_state(),TorchCPU=torch.get_rng_state().clone(),CUDA=torch.cuda.get_rng_state(self.device).clone())
        def restore(v):
            random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['TorchCPU']);torch.cuda.set_rng_state(v['CUDA'],self.device)
        def same(a,b):
            return a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['TorchCPU'],b['TorchCPU']) and torch.equal(a['CUDA'],b['CUDA'])
        before=capture()
        flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)
        try:
            site=Path(c['TRT_site']).resolve();need(site.is_dir(),'Isolated TRT site absent');sys.path.insert(0,str(site))
            runtime=load('_push7_hypir_trt_runtime',md/'trt_vae_runtime_v2.py')
            self.engines={name:runtime.TensorRTEngine(c['TRT_engines'][name]['path'],spec[0],self.device) for name,spec in ENGINES.items()}
            import tensorrt as trt
            import torch_tensorrt
            need(torch.__version__.split('+')[0]=='2.8.0' and torch_tensorrt.__version__.split('+')[0]=='2.8.0' and trt.__version__.startswith('10.12.'),'Reviewed Torch/TRT versions required')
            backend_versions=dict(torch=torch.__version__,torch_tensorrt=torch_tensorrt.__version__,tensorrt=trt.__version__)
            for name,engine in self.engines.items():
                need(engine.input_shape==ENGINES[name][2] and engine.output_shape==ENGINES[name][3]
                     and engine.input_dtype==torch.float16 and engine.output_dtype==torch.float16,'Actual TensorRT fixed tile FP16 IO differs')
            from diffusers.models.modeling_outputs import AutoencoderKLOutput
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, DecoderOutput
            def encode(x,return_dict=True):
                need(x.device==self.device and x.dtype==torch.bfloat16 and tuple(x.shape)==(1,3,512,512),'Original native512 BF16 encoder tile required')
                moments=self.engines['encoder'](x.to(torch.float16)).to(torch.bfloat16)
                posterior=DiagonalGaussianDistribution(moments)
                return AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)
            def decode(z,return_dict=True,generator=None):
                need(generator is None and z.device==self.device and z.dtype==torch.bfloat16 and tuple(z.shape)==(1,4,64,64),'Original native64 BF16 decoder tile required')
                image=self.engines['decoder'](z.to(torch.float16)).to(torch.bfloat16)
                return DecoderOutput(sample=image) if return_dict else (image,)
            # Parent compiled VAE callables have never run. Replace their tile
            # implementations; original full enhance and all Gaussian merges stay.
            self.model.vae.encode=encode;self.model.vae.decode=decode
        finally:
            restore(before)
            torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32']
            torch.backends.cudnn.benchmark=flags['cudnn_benchmark'];torch.backends.cudnn.deterministic=flags['cudnn_deterministic']
        after=capture();need(same(before,after),'TRT initialization changed default RNG')
        self.protocol=dict(self.original.protocol)
        self.protocol.update(schema='hypir_trt_vae_hybrid_runner_v1',runner_sha256=sha(__file__),model_dir_config_sha256=sha(md/'config.json'),
            runner_path='HYPIR_native512_BF16_fused_UNet_RO_wavelet_FP16_TRT_VAE',model_type='HYPIR_SD2_rank256_TRT_VAE_native512',
            internal_precision='UNet original BF16; TensorRT VAE FP16 IO/weights, original posterior/sample and tile merges BF16; native wavelet/output FP32',
            compilation='Original UNet.forward+wavelet reduce-overhead; VAE.encode/decode replaced by fixed TensorRT runtime v2; original full tile loops/sampling/scheduler',
            speed_candidate='HYPIR same architecture/native512; VAE backend/precision changed, no student/distillation or scale change',
            quality_reuse_claim=False,requires_new_full103_418=True,no_training=True,no_distillation=True,
            implementation_sources={str(md/'runner.py'):sha(__file__),str(md/'reference_runner.py'):REFERENCE_SHA,str(md/'trt_vae_runtime_v2.py'):RUNTIME_SHA},
            TRT=dict(site=str(site),versions=backend_versions,runtime_sha256=RUNTIME_SHA,engines={name:dict(path=c['TRT_engines'][name]['path'],sha256=ENGINES[name][0],manifest=c['TRT_engines'][name]['manifest'],manifest_sha256=ENGINES[name][1],actual_IO=engine.io_records) for name,engine in self.engines.items()},
                original_VAE_sha256=VAE_SHA,initialization_all_four_RNG_restored=True,forward_RNG_reset=False,
                encoder='Original BF16 tile -> FP16 engine moments -> BF16 DiagonalGaussianDistribution; original sample at its original call site',
                decoder='Original BF16 latent -> FP16 engine including post_quant_conv -> BF16 DecoderOutput; original sample.float and Gaussian BF16 out/FP32 count merge',
                execution='Dedicated nondefault worker stream waits caller; caller waits worker; fresh Torch-owned outputs; no input/output cache',
                component_manifests_are_historical_v1_probes=True,runtime_v2_full_pipeline_not_previously_measured=True),
            numerical_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic))
        self.original.protocol=self.protocol

    def infer(self,x):
        return self.original.infer(x)
