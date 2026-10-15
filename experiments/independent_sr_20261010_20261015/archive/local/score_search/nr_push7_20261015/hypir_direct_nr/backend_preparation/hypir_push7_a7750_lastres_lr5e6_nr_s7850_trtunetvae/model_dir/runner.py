"""Registered fine-tuned HYPIR TRT UNet+VAE template; requires REAL sealed engine config.

No placeholder engine/hash is accepted. Preserve original scheduler/DDPM,
posterior sampling, native512/p256 Gaussian tiling and wavelet through parent.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys

A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
PARENT_SHA='4b225fbe82c556abf4b409927d16709e771a69fb811d7c6803f7281792875862'
BINDING_SHA='db2052014f941f517dee730dcb44e4d8e3d9939e0de15991f663bb098e6c451d'
RUNTIME_SHA='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4'
BUILDER_SHA='b7dcc1f4fa986506eda23651b94c3f65b463fb46a9fe45e464833e6017fda34d'

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
        need(c.get('UNet_TRT_backend')=='finetuned_FP16_TRT_fixed64_empty_t200','Explicit fine-tuned UNet backend required')
        need(sha(md/'trained_native_binding.py')==BINDING_SHA,'Native binding source changed')
        binding=load('_push7_finetuned_native_binding',md/'trained_native_binding.py')
        native=binding.bind_native(c['native_candidate_manifest']['path'],c['native_candidate_manifest']['sha256'],hash_raw=False)
        need(c['checkpoint_sha256']==native['checkpoint_sha256'] and c['weight_path']==native['checkpoint'] and
             sha(c['weight_path'])==native['checkpoint_sha256'],'Actual trained checkpoint identity differs')
        need(c['training_provenance']==native['training_provenance'],'Training provenance differs')
        need(sha(md/'vae_hybrid_runner.py')==PARENT_SHA and sha(md/'trt_vae_runtime_v2.py')==RUNTIME_SHA,'Frozen hybrid/runtime source differs')
        spec=c['TRT_UNet_engine'];mp=Path(spec['manifest']);ep=Path(spec['path'])
        need(sha(mp)==spec['manifest_sha256'] and sha(ep)==spec['sha256'],'Actual UNet engine/manifest SHA differs')
        m=json.loads(mp.read_text())
        need(m['status']=='BUILT_COMPONENT_PROBED' and m['schema']=='hypir_finetuned_unet_fp16_trt_component_v1' and m['raw_sha256']==native['checkpoint_sha256'],'Wrong completed UNet engine/initial weights')
        need(m['native_candidate_binding']==native and m['initial_A_sha256']==A_SHA and m['checkpoint_finetuned'] is True and
             m['backend_build_has_no_training'] is True and m['quality_reuse_claim'] is False and
             m['require_full_compilation'] is True and m['actual_trained_raw_SHA_verified_by_builder'] is True,
             'Engine was not built from this registered fine-tune')
        need(ep.resolve()!=Path('/cache/independent_sr_cuda_20261013/backends/push7_trt/unet_fp16_A7750/component.engine'),'A-only UNet engine is forbidden')
        need(m['BF16_original_fused_G_state']['tensors']==686 and m['BF16_original_fused_G_state']['elements']==865910724 and
             m['fixed_conditioning']['BF16_sha256']['sha256']=='f65bb78e0c2376a4df09f17d8671ba7dfe163375d3349a1db49f12c5bbd9d0a9' and
             m['fixed_conditioning']['FP16_sha256']['sha256']=='0008da3ec7f91844e17610a5929d68b151938ac5d895bacbb088f9c098d5003b','Architecture/fixed conditioning differs')
        need(m['engine_sha256']==spec['sha256'] and m['engine_bytes']==ep.stat().st_size and Path(m['engine_path']).resolve()==ep.resolve(),'Engine provenance/path/bytes differ')
        need(tuple(m['input_shape'])==tuple(m['output_shape'])==(1,4,64,64) and m['model_t']==200 and m['caption']=='' and m['compiled_fallback'] is False,'Fixed64/empty/t200 no-fallback contract required')
        producer=Path('/cache/independent_sr_cuda_20261013/code/push7/hypir_direct_nr/trained_backend_v1/build_finetuned_unet_trt.py')
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
        self._path='HYPIR_finetuned_native512_FP16_TRT_UNet_and_VAE_original_BF16_sampler'
        self.protocol=dict(self.parent.protocol)
        self.protocol.update(schema='hypir_finetuned_trt_unet_vae_hybrid_runner_v1',runner_path=self._path,runner_sha256=sha(__file__),model_dir_config_sha256=sha(md/'config.json'),
            model_type='HYPIR_SD2_rank256_finetuned_TRT_UNet_VAE_native512',internal_precision='FP16 TensorRT UNet+VAE, BF16 eps/latent/DiagonalGaussianDistribution/scheduler/tiling, FP32 wavelet/output',
            compilation='Original whole native enhance/Gaussian/sampling/scheduler/wavelet; only VAE and UNet tile tensor functions use fixed TensorRT engines',
            UNet_TRT=dict(engine_path=str(ep),engine_sha256=spec['sha256'],manifest=str(mp),manifest_sha256=spec['manifest_sha256'],actual_IO=self.engine.io_records,
                fixed_original_BF16_conditioning_sha256=actual_conditioning,initial_A_sha256=A_SHA,checkpoint_sha256=native['checkpoint_sha256'],native_candidate_manifest=native['native_candidate_manifest'],BF16_fused_G_digest_from_builder=m['BF16_original_fused_G_state'],
                construction_all_four_RNG_restored=True,forward_RNG_reset=False,scheduler_DDPM_original_BF16_outside_engine=True,producer_sha256=BUILDER_SHA),
            quality_reuse_claim=False,requires_new_full103_418=True,checkpoint_finetuned=True,backend_build_has_no_training=True,no_distillation=True,
            training_provenance=native['training_provenance'])
        self.protocol['implementation_sources']=dict(self.parent.protocol['implementation_sources'],**{str(md/'runner.py'):sha(__file__),str(md/'vae_hybrid_runner.py'):PARENT_SHA,str(md/'trained_native_binding.py'):BINDING_SHA})

    def infer(self,x):return self.parent.infer(x)
