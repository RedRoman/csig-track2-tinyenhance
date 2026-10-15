"""Unregistered HYPIR TRT UNet+VAE template; requires REAL sealed engine config.

No placeholder engine/hash is accepted. Preserve original scheduler/DDPM,
posterior sampling, native512/p256 Gaussian tiling and wavelet through parent.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys

A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
PARENT_SHA='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9'
RUNTIME_SHA='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4'
BUILDER_SHA='4281d9be015142919b51b84d946ad8d23b8bd49451ad54adb9ada8f1d42cdcca'

def need(v,msg):
    if not v:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

class Runner:
    def __init__(self,modeldir):
        md=Path(modeldir).resolve();c=json.loads((md/'config.json').read_text())
        need(c.get('UNet_TRT_backend')=='A7750_FP16_TRT_fixed64_empty_t200','Explicit UNet backend required')
        need(c['checkpoint_sha256']==A_SHA and sha(c['weight_path'])==A_SHA,'This engine may only be used with its real A7750 checkpoint')
        need(sha(md/'vae_hybrid_runner.py')==PARENT_SHA and sha(md/'trt_vae_runtime_v2.py')==RUNTIME_SHA,'Frozen hybrid/runtime source differs')
        spec=c['TRT_UNet_engine'];mp=Path(spec['manifest']);ep=Path(spec['path'])
        need(sha(mp)==spec['manifest_sha256'] and sha(ep)==spec['sha256'],'Actual UNet engine/manifest SHA differs')
        m=json.loads(mp.read_text())
        need(m['status']=='BUILT_COMPONENT_PROBED' and m['schema']=='hypir_A7750_unet_fp16_trt_component_v1' and m['raw_sha256']==A_SHA,'Wrong completed UNet engine/initial weights')
        need(m['engine_sha256']==spec['sha256'] and m['engine_bytes']==ep.stat().st_size and Path(m['engine_path']).resolve()==ep.resolve(),'Engine provenance/path/bytes differ')
        need(tuple(m['input_shape'])==tuple(m['output_shape'])==(1,4,64,64) and m['model_t']==200 and m['caption']=='' and m['compiled_fallback'] is False,'Fixed64/empty/t200 no-fallback contract required')
        producer=Path('/cache/independent_sr_cuda_20261013/code/push7/trt_unet/build_hypir_unet_trt.py')
        need(m['source_sha256'].get(str(producer))==BUILDER_SHA and sha(producer)==BUILDER_SHA,'Sealed actual UNet engine producer changed')
        need(sha(Path(c['base_model_path'])/'unet/diffusion_pytorch_model.safetensors')==m['actual_base_UNet_sha256'],'Actual original UNet base differs')
        parent=load('_push7_trt_unet_vae_parent',md/'vae_hybrid_runner.py');self.parent=parent.Runner(md)
        self.model=self.parent.model;self.device=self.parent.device;self.model_dir=md
        import torch
        import numpy as np
        import random
        text=self.model.inputs['c_txt']['text_embed'];timestep=self.model.inputs['timesteps']
        need(text.dtype==torch.bfloat16 and tuple(text.shape)==(1,77,1024) and timestep.tolist()==[200],'Original BF16 empty/t200 cached inputs required')
        def conditioning_digest():
            h=hashlib.sha256()
            for name,v in [('empty_embedding',text),('timestep',timestep)]:
                cpu=v.detach().cpu().contiguous();h.update(json.dumps([name,list(cpu.shape),str(cpu.dtype)],separators=(',',':')).encode());h.update(b'\0');h.update(cpu.view(torch.uint8).numpy().tobytes())
            return h.hexdigest()
        actual_conditioning=conditioning_digest()
        need(actual_conditioning==m['fixed_conditioning']['BF16_sha256']['sha256'],'Actual empty text/timestep buffers differ from fixed engine')
        def capture():return dict(python=random.getstate(),numpy=np.random.get_state(),TorchCPU=torch.get_rng_state().clone(),CUDA=torch.cuda.get_rng_state(self.device).clone())
        def restore(s):
            random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['TorchCPU']);torch.cuda.set_rng_state(s['CUDA'],self.device)
        def same(a,b):return a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['TorchCPU'],b['TorchCPU']) and torch.equal(a['CUDA'],b['CUDA'])
        flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)
        before=capture()
        try:
            runtime=load('_push7_trt_unet_runtime',md/'trt_vae_runtime_v2.py');self.engine=runtime.TensorRTEngine(ep,spec['sha256'],self.device)
            need(self.engine.input_shape==self.engine.output_shape==(1,4,64,64) and self.engine.input_dtype==self.engine.output_dtype==torch.float16,'Actual fixed64 FP16 UNet IO differs')
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
            def forward(sample,t,encoder_hidden_states,return_dict=True,**kwargs):
                need(not kwargs and sample.device==self.device and sample.dtype==torch.bfloat16 and tuple(sample.shape)==(1,4,64,64),'Original BF16 z_in tile only')
                need(t is timestep and encoder_hidden_states is text,'Only constructor-verified fixed original cached t200/empty text are accepted')
                eps=self.engine(sample.to(torch.float16)).to(torch.bfloat16)
                return UNet2DConditionOutput(sample=eps) if return_dict else (eps,)
            # No conversion of scheduler or posterior. The original SD2 forward
            # calls this with its already-scaled BF16 z_in and continues itself.
            self.model.G.forward=forward
        finally:
            restore(before);torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32'];torch.backends.cudnn.benchmark=flags['cudnn_benchmark'];torch.backends.cudnn.deterministic=flags['cudnn_deterministic']
        need(same(before,capture()),'UNet engine construction changed default RNG')
        self._path='HYPIR_A7750_native512_FP16_TRT_UNet_and_VAE_original_BF16_sampler'
        self.protocol=dict(self.parent.protocol)
        self.protocol.update(schema='hypir_trt_unet_vae_hybrid_runner_v1',runner_path=self._path,runner_sha256=sha(__file__),model_dir_config_sha256=sha(md/'config.json'),
            model_type='HYPIR_SD2_rank256_TRT_UNet_VAE_native512',internal_precision='FP16 TensorRT UNet+VAE, BF16 eps/latent/DiagonalGaussianDistribution/scheduler/tiling, FP32 wavelet/output',
            compilation='Original whole native enhance/Gaussian/sampling/scheduler/wavelet; only VAE and UNet tile tensor functions use fixed TensorRT engines',
            UNet_TRT=dict(engine_path=str(ep),engine_sha256=spec['sha256'],manifest=str(mp),manifest_sha256=spec['manifest_sha256'],actual_IO=self.engine.io_records,
                fixed_original_BF16_conditioning_sha256=actual_conditioning,initial_A_sha256=A_SHA,BF16_fused_G_digest_from_builder=m['BF16_original_fused_G_state'],
                construction_all_four_RNG_restored=True,forward_RNG_reset=False,scheduler_DDPM_original_BF16_outside_engine=True,producer_sha256=BUILDER_SHA),
            quality_reuse_claim=False,requires_new_full103_418=True,no_training=True,no_distillation=True)
        self.protocol['implementation_sources']=dict(self.parent.protocol['implementation_sources'],**{str(md/'runner.py'):sha(__file__),str(md/'vae_hybrid_runner.py'):PARENT_SHA})

    def infer(self,x):return self.parent.infer(x)
