"""CPU-only, exclusive registration from the frozen real-trained-engine plan.

No GPU/import of Torch, engine construction, image rendering, quality reuse or
current-pointer update. Default prints the complete actual-engine-bound plan.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013')
PREP=R/'code/push7/hypir_direct_nr/trained_backend_v1'
PLAN_SHA='97a08c5e3291d8ae9182f0bf31f23a9ec976fa4a79c77df7bfc36afb34325335'
BUILDER_SHA='b7dcc1f4fa986506eda23651b94c3f65b463fb46a9fe45e464833e6017fda34d'
RUNNER_SHA='c98c406884cf07bc5541e315393b98e11351165b35c94d33f926d5443f9563c3'
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'

def need(v,m):
    if not v:raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)

def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-manifest',type=Path,required=True);p.add_argument('--native-manifest-sha256',required=True)
    p.add_argument('--engine-manifest',type=Path,required=True);p.add_argument('--engine-manifest-sha256',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    need(sha(PREP/'registration_plan.py')==PLAN_SHA,'Frozen registration plan changed')
    need(sha(a.native_manifest)==a.native_manifest_sha256,'Actual native manifest changed')
    native_cm=json.loads(a.native_manifest.read_text());label=native_cm['label']+'_trtunetvae'
    cmd=[sys.executable,str(PREP/'registration_plan.py'),'--native-manifest',str(a.native_manifest.resolve()),
         '--native-manifest-sha256',a.native_manifest_sha256,'--engine-manifest',str(a.engine_manifest.resolve()),
         '--engine-manifest-sha256',a.engine_manifest_sha256,'--label',label]
    result=subprocess.run(cmd,capture_output=True,text=True,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    plan=json.loads(result.stdout)
    need(plan['status']=='CPU_READY_REGISTRATION_PLAN_REAL_NEW_ENGINE_BOUND' and plan['label']==label and
         plan['candidate_registered_here'] is False and plan['GPU_used'] is False,'No completed source-bound registration plan')
    md=Path(plan['model_dir']);cfg=plan['runtime_config'];native=plan['native_candidate_binding']
    need(md==R/'submissions'/('hypir_'+label)/'model_dir' and not md.parent.exists(),'Use a new immutable candidate directory')
    need(native['checkpoint_sha256']==native_cm['checkpoint_sha256'] and native['initial_A_sha256']==A_SHA and
         cfg['weight_path']==native['checkpoint'] and cfg['checkpoint_sha256']==native['checkpoint_sha256'] and
         cfg['label']==label,'Actual native/current checkpoint or label differs')
    need(sha(cfg['weight_path'])==native['checkpoint_sha256'],'Actual current raw bytes changed')
    engine=json.loads(a.engine_manifest.read_text())
    need(sha(a.engine_manifest)==a.engine_manifest_sha256 and engine['native_candidate_binding']==native and
         engine['raw_sha256']==native['checkpoint_sha256'] and engine['actual_trained_raw_SHA_verified_by_builder'] is True,
         'Actual trained engine manifest changed')
    producer=PREP/'build_finetuned_unet_trt.py'
    need(sha(producer)==engine['source_sha256'][str(producer)]==BUILDER_SHA,'Actual new engine producer differs')
    actual_engines={name:dict(x,bytes=Path(x['path']).stat().st_size) for name,x in cfg['TRT_engines'].items()}
    actual_engines['UNet']=dict(cfg['TRT_UNet_engine'],bytes=Path(cfg['TRT_UNet_engine']['path']).stat().st_size)
    need(set(actual_engines)=={'encoder','decoder','UNet'},'Exactly three original-architecture component engines required')
    for name,x in actual_engines.items():
        need(sha(x['path'])==x['sha256'] and sha(x['manifest'])==x['manifest_sha256'],'Actual engine bytes/manifest changed: '+name)
    for name,source in plan['files_to_copy'].items():
        need(sha(source['path'])==source['sha256'],'Actual prepared source changed: '+name)
    if not a.execute:
        print(json.dumps(dict(status='CPU_ACTUAL_PLAN_ONLY',plan=plan,actual_three_engine_SHA_verified=True,
                              actual_trained_raw_SHA_verified=True,GPU_used=False,candidate_registered=False),ensure_ascii=False));return
    md.parent.mkdir(parents=False,exist_ok=False);md.mkdir(exist_ok=False)
    for name,source in plan['files_to_copy'].items():
        with Path(source['path']).open('rb') as src,(md/name).open('xb') as dst:shutil.copyfileobj(src,dst)
        need(sha(md/name)==source['sha256'],'Copied source SHA differs: '+name)
    write(md/'config.json',cfg)
    need(sha(md/'runner.py')==RUNNER_SHA,'Registered runner differs')
    manifest=dict(schema='hypir_finetuned_trt_unet_vae_candidate_v1',
        model_type='HYPIR_SD2_rank256_finetuned_TRT_UNet_VAE_native512',label=label,model_dir=str(md),
        checkpoint=native['checkpoint'],checkpoint_sha256=native['checkpoint_sha256'],initial_A_checkpoint_sha256=A_SHA,
        native_candidate_binding=native,training_provenance=native['training_provenance'],training_run=native['training_run'],
        local_step=native['local_step'],global_step=native['global_step'],optimizer_step=native['optimizer_step'],
        optimizer_tensor_count=native['optimizer_tensor_count'],optimizer_parameter_keys=native['optimizer_parameter_keys'],
        optimizer_parameter_elements=native['optimizer_parameter_elements'],frozen_unselected_exact_A_count=native['unselected_exact_A_count'],
        runner_sha256=sha(md/'runner.py'),config_sha256=sha(md/'config.json'),files={x:ref(md/x) for x in plan['files_to_copy']},
        actual_engines=actual_engines,UNet_component_manifest=ref(a.engine_manifest),UNet_engine_producer=ref(producer),
        BF16_original_fused_G_state=engine['BF16_original_fused_G_state'],fixed_conditioning=engine['fixed_conditioning'],
        actual_base_UNet_sha256=engine['actual_base_UNet_sha256'],actual_trained_raw_SHA_verified_here=True,
        raw_resume_tensors_not_reopened_here=True,actual_three_engine_SHA_verified_here=True,
        checkpoint_finetuned=True,backend_build_has_no_training=True,no_teacher_outputs=True,no_val_GT_training=True,
        no_distillation=True,quality_reuse_claim=False,requires_actual_fresh_full103_418=True,
        CPU_only=True,GPU_used=False,CUDA_initialized=False,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        registrar_source=ref(__file__),frozen_registration_plan=ref(PREP/'registration_plan.py'),
        runtime_metadata_dependencies=plan['runtime_metadata_dependencies'])
    path=md.parent/'candidate_manifest.json';write(path,manifest)
    print(json.dumps(dict(status='REGISTERED_CPU_ONLY_NO_QUALITY_REUSE',label=label,model_dir=str(md),
        candidate_manifest=ref(path),runner_sha256=manifest['runner_sha256'],config_sha256=manifest['config_sha256'],
        checkpoint_sha256=native['checkpoint_sha256'],UNet_engine_sha256=actual_engines['UNet']['sha256'],GPU_used=False)))

if __name__=='__main__':main()
