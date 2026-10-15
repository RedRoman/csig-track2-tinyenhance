"""Metadata/source/engine-byte audit only. Does not import Torch/TRT or run GPU."""
from pathlib import Path
import ast, hashlib, importlib.util, json
R=Path('/cache/independent_sr_cuda_20261013')
MD=R/'submissions/hypir_push7_a7750_trtvae_fp16/model_dir'

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    path=MD/'runner.py';ast.parse(path.read_text());s=importlib.util.spec_from_file_location('hybrid_cpu_only',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    c=json.loads((MD/'config.json').read_text());records={}
    for p in [path,MD/'config.json',MD/'reference_runner.py',MD/'trt_vae_runtime_v2.py']:
        if p.suffix=='.py':ast.parse(p.read_text())
        records[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    need(records[str(MD/'reference_runner.py')]['sha256']==m.REFERENCE_SHA,'Original DFD runner')
    need(records[str(MD/'trt_vae_runtime_v2.py')]['sha256']==m.RUNTIME_SHA,'Dedicated-stream v2 runtime')
    need(sha(c['weight_path'])==c['checkpoint_sha256'],'Actual explicit HYPIR checkpoint')
    need(sha(c['base_vae_path'])==m.VAE_SHA,'Actual original VAE')
    for name,(eh,mh,inp,out) in m.ENGINES.items():
        spec=c['TRT_engines'][name];ep=Path(spec['path']);mp=Path(spec['manifest']);j=json.loads(mp.read_text())
        need(sha(ep)==spec['sha256']==eh and sha(mp)==spec['manifest_sha256']==mh,'Actual engine/manifest')
        need(j['component']==name and j['precision']=='fp16' and j['status']=='BUILT_COMPONENT_PROBED' and j['reference_VAE_SHA256']==m.VAE_SHA and tuple(j['input_shape'])==inp and tuple(j['output_shape'])==out,'Declared shape/provenance')
        records[str(ep)]=dict(sha256=eh,bytes=ep.stat().st_size);records[str(mp)]=dict(sha256=mh,bytes=mp.stat().st_size)
    candidate=MD.parent/'candidate_manifest.json';need(not candidate.exists(),'Do not overwrite candidate manifest')
    manifest=dict(schema='hypir_trt_vae_hybrid_candidate_v1',status='CPU_PREPARED_ACTUAL_ARTIFACT_HASHES',label=c['label'],model_type='HYPIR_SD2_rank256_TRT_VAE_native512',model_dir=str(MD),
        checkpoint=c['weight_path'],checkpoint_sha256=c['checkpoint_sha256'],runner_sha256=sha(path),config_sha256=sha(MD/'config.json'),
        reference_runner_sha256=m.REFERENCE_SHA,TRT_runtime_sha256=m.RUNTIME_SHA,original_VAE_sha256=m.VAE_SHA,
        actual_engines={name:dict(path=c['TRT_engines'][name]['path'],sha256=sha(c['TRT_engines'][name]['path']),bytes=Path(c['TRT_engines'][name]['path']).stat().st_size,
            manifest=c['TRT_engines'][name]['manifest'],manifest_sha256=sha(c['TRT_engines'][name]['manifest'])) for name in m.ENGINES},
        no_training=True,no_distillation=True,scale=1,patch=512,stride=256,quality_reuse_claim=False,requires_new_actual_full103_418=True,
        CPU_preparer_sha256=sha(__file__),formal_latency_required_extra_hash_files=[c['weight_path']]+[v['path'] for v in c['TRT_engines'].values()])
    candidate.write_text(json.dumps(manifest,indent=2)+'\n');records[str(candidate)]=dict(sha256=sha(candidate),bytes=candidate.stat().st_size)
    out=R/'analysis/push7/trt_vae_hybrid/preparation_cpu_audit.json';need(not out.exists(),'Do not overwrite CPU audit')
    result=dict(status='PASS_CPU_PREPARATION_ONLY',schema='hypir_trt_hybrid_cpu_preparation_v1',model_dir=str(MD),CPU_only=True,Torch_imported=False,TRT_imported=False,GPU_executed=False,
        no_training=True,no_distillation=True,original_DFD_source_byte_identical=True,checkpoint_sha256=c['checkpoint_sha256'],
        encoder_manifest_IO=[list(m.ENGINES['encoder'][2]),list(m.ENGINES['encoder'][3])],decoder_manifest_IO=[list(m.ENGINES['decoder'][2]),list(m.ENGINES['decoder'][3])],
        actual_runtime_IO_RNG_restore_and_output_ownership='Strictly guarded in Runner but not executed by this CPU audit; root must GPU smoke',
        quality_reuse_claim=False,requires_new_actual_quality_evaluation=True,file_records=records,auditor_source_sha256=sha(__file__))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],out=str(out),sha256=sha(out),model_dir=str(MD))))

if __name__=='__main__':main()
