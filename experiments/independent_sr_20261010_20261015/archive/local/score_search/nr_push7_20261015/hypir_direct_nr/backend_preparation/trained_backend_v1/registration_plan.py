"""CPU registration PLAN only: no candidate writes and no unbuilt-engine placeholders."""
from pathlib import Path
import argparse, json, os, re
os.environ['CUDA_VISIBLE_DEVICES']=''
from trained_native_binding import bind_native, need, sha, ref, R
BINDING_SHA='db2052014f941f517dee730dcb44e4d8e3d9939e0de15991f663bb098e6c451d'
BUILDER_SHA='b7dcc1f4fa986506eda23651b94c3f65b463fb46a9fe45e464833e6017fda34d'
WRAPPER_SHA='c98c406884cf07bc5541e315393b98e11351165b35c94d33f926d5443f9563c3'
VAE_PARENT_SHA='4b225fbe82c556abf4b409927d16709e771a69fb811d7c6803f7281792875862'
RUNTIME_SHA='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4'
DFD_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'

def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--native-manifest',type=Path,required=True);q.add_argument('--native-manifest-sha256',required=True)
    q.add_argument('--engine-manifest',type=Path);q.add_argument('--engine-manifest-sha256');q.add_argument('--label');a=q.parse_args()
    folder=Path(__file__).resolve().parent
    for name,h in [('trained_native_binding.py',BINDING_SHA),('build_finetuned_unet_trt.py',BUILDER_SHA),
                   ('finetuned_unet_vae_runner.py',WRAPPER_SHA),('finetuned_trtvae_runner.py',VAE_PARENT_SHA),('trt_vae_runtime_v2.py',RUNTIME_SHA)]:
        need(sha(folder/name)==h,'Prepared source changed: '+name)
    native=bind_native(a.native_manifest,a.native_manifest_sha256,hash_raw=False)
    label=a.label or native['label']+'_trt_unetvae';need(re.fullmatch(r'push7_[A-Za-z0-9_]+',label),'New explicit label required')
    md=R/'submissions'/('hypir_'+label)/'model_dir';need(not md.parent.exists(),'Planned immutable destination already exists')
    files={'runner.py':ref(folder/'finetuned_unet_vae_runner.py'), 'vae_hybrid_runner.py':ref(folder/'finetuned_trtvae_runner.py'),
        'reference_runner.py':ref(R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir/runner.py'),
        'trt_vae_runtime_v2.py':ref(folder/'trt_vae_runtime_v2.py'),'trained_native_binding.py':ref(folder/'trained_native_binding.py')}
    need(files['reference_runner.py']['sha256']==DFD_SHA,'Frozen original reference changed')
    base=dict(status='CPU_PLAN_WAITING_REAL_NEW_UNET_ENGINE',model_dir=str(md),label=label,files_to_copy=files,
        native_candidate_binding=native,planned_new_engine_dir=str(R/'backends/push7_trt'/(native['label']+'_unet_fp16')),
        new_builder=ref(folder/'build_finetuned_unet_trt.py'),CPU_only=True,GPU_used=False,engine_built_here=False,
        candidate_registered_here=False,quality_reuse_claim=False,requires_actual_fresh_full103_418=True,
        gate_owned_by_root='Root must review complete matched pilot and actual native ROI before scheduling a new build',
        runtime_metadata_dependencies='Keep native manifest, config, marker, input catalog, parity proof and their sealed source files at the referenced restored paths')
    if not a.engine_manifest:
        need(a.engine_manifest_sha256 is None,'Engine manifest path missing')
        base['runtime_config_not_emitted_because_engine_missing']=True;print(json.dumps(base,indent=2));return
    need(a.engine_manifest_sha256 and sha(a.engine_manifest)==a.engine_manifest_sha256,'Actual new engine manifest SHA differs')
    m=json.loads(a.engine_manifest.read_text());ep=Path(m['engine_path']).resolve()
    need(m['status']=='BUILT_COMPONENT_PROBED' and m['schema']=='hypir_finetuned_unet_fp16_trt_component_v1' and
         m['native_candidate_binding']==native and m['raw_sha256']==native['checkpoint_sha256'] and
         m['checkpoint_finetuned'] is True and m['backend_build_has_no_training'] is True and
         m['require_full_compilation'] is True and m['compiled_fallback'] is False and m['quality_reuse_claim'] is False and
         m['actual_trained_raw_SHA_verified_by_builder'] is True,'Not a real full-compiled engine for this native fine-tune')
    need(ep.is_relative_to(R/'backends/push7_trt') and ep!=R/'backends/push7_trt/unet_fp16_A7750/component.engine' and
         ep.stat().st_size==m['engine_bytes'] and sha(ep)==m['engine_sha256'],'Actual new engine bytes/path/hash differs; A-only engine forbidden')
    producer=R/'code/push7/hypir_direct_nr/trained_backend_v1/build_finetuned_unet_trt.py'
    need(m['source_sha256'].get(str(producer))==BUILDER_SHA and sha(producer)==BUILDER_SHA and
         m['model_t']==200 and m['caption']=='' and m['input_shape']==m['output_shape']==[1,4,64,64], 'Producer/fixed64/t200/empty differs')
    need(len(m['actual_engine_IO'])==2 and all(x['shape']==[1,4,64,64] and x['dtype']=='DataType.HALF' for x in m['actual_engine_IO']), 'Actual engine IO differs')
    c=json.loads((Path(native['model_dir'])/'config.json').read_text())
    vae=json.loads((R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16/model_dir/config.json').read_text())
    c.update(label=label,weight_path=native['checkpoint'],checkpoint_sha256=native['checkpoint_sha256'],
        training_provenance=native['training_provenance'],native_candidate_manifest=native['native_candidate_manifest'],
        backend='HYPIR_BF16_UNet_FP16_TRT_VAE_native512',UNet_TRT_backend='finetuned_FP16_TRT_fixed64_empty_t200',
        TRT_site=vae['TRT_site'],base_vae_path=vae['base_vae_path'],TRT_engines=vae['TRT_engines'],
        TRT_UNet_engine=dict(path=str(ep),sha256=m['engine_sha256'],manifest=str(a.engine_manifest.resolve()),manifest_sha256=a.engine_manifest_sha256))
    base.update(status='CPU_READY_REGISTRATION_PLAN_REAL_NEW_ENGINE_BOUND',runtime_config=c,new_engine_manifest=ref(a.engine_manifest))
    print(json.dumps(base,indent=2))

if __name__=='__main__':main()
