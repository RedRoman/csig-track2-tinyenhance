"""CPU-only registration of real NIQE768 LoRA plus unchanged TRT VAE engines.
Default plan; root alone executes after positive native pilot. No A UNet engine.
"""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
RUN=R/'adaptation/runs/hypir_push7_A7750_niqe768_100'
BASE=R/'submissions/hypir_push7_a7750_trtvae_fp16/model_dir'
TEMPLATE=R/'code/push7/hypir_direct_nr/finetuned_trtvae_runner.py'
TEMPLATE_SHA='4b225fbe82c556abf4b409927d16709e771a69fb811d7c6803f7281792875862'
A='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
SEEN={}
def need(v,m):
    if not v:raise RuntimeError(m)
def sha(p):
    p=Path(p).resolve();s=p.stat();st=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
    if p in SEEN:
        need(SEEN[p][0]==st,'Source changed: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    s=p.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed during SHA')
    SEEN[p]=(st,h.hexdigest());return h.hexdigest()
def read(p):sha(p);return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--step',type=int,choices=[7800,7850],required=True);q.add_argument('--execute',action='store_true');a=q.parse_args()
    label=f'push7_a7750_niqe768_s{a.step}_trtvae';dest=R/'submissions'/f'hypir_{label}'
    native=R/'submissions'/f'hypir_push7_a7750_niqe768_s{a.step}'
    need(not dest.exists(),'Immutable backend candidate already exists')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',native_candidate=str(native),model_dir=str(dest/'model_dir'),label=label,
          requires_actual_native_registration=True,requires_positive_complete_native_pilot_before_evaluation=True,
          new_training_checkpoint=True,backend_has_no_training=True,reuses_only_two_VAE_engines=True,no_A_UNet_engine=True,GPU=False)));return
    os.environ['CUDA_VISIBLE_DEVICES']=''
    m=read(native/'candidate_manifest.json');t=m['training_provenance'];p=read(t['config']['path'])['protocol'];marker=read(t['marker']['path'])
    need(m['schema']=='hypir_direct_nr_candidate_v1' and m['training_run']==str(RUN) and
      m['registrar_source_sha256']=='2b8874ed3a1575af1a40f7567fccaf3abac35bdc075ae06e857b9ac39eff75fb','Only actual approved NIQE768 native registration allowed')
    need(p['schema']=='hypir_push7_direct_nr_niqe768_full_gaussian_fused_bf16_v4' and p['teacher_output_loss'] is False and
      p['distillation'] is False and p['validation_GT_training'] is False and p['new_Adam_initial_step']==0,'Not direct input-only HYPIR NIQE768')
    source=str(R/'code/push7/hypir_direct_nr/train_hypir_direct_nr_niqe768_v4.py')
    need(p['source_sha256'][source]=='3dd7388abd78659de9a22831ead98a75e88a9b8558473eb5cd7178a89e51b109','Wrong v4 training producer')
    need(marker['global_step']==m['global_step']==a.step and marker['local_step']==marker['optimizer_step']==m['local_step']==a.step-7750 and
      marker['initial_A_checkpoint_sha256']==m['initial_checkpoint_sha256']==A and marker['no_teacher_outputs'] is True and marker['no_val_GT'] is True,'Actual marker lineage/counters differ')
    need(marker['raw_checkpoint_sha256']==m['checkpoint_sha256']!=A and Path(marker['raw_checkpoint']).resolve()==Path(m['checkpoint']).resolve(),'Raw checkpoint identity differs')
    need(m['actual_raw_and_resume_lora_equal'] is True and m['raw_FP32_tensor_count']==514 and marker['changed_FP32_LoRA_tensors']>0,'Actual native514 registration proof absent')
    for key in ['config','marker','input_catalog','resume']:need(sha(t[key]['path'])==t[key]['sha256'],'Training immutable reference changed')
    for path,digest in p['source_sha256'].items():need(sha(path)==digest,'Actual training source changed')
    raw=Path(m['checkpoint']).resolve();need(sha(raw)==m['checkpoint_sha256'],'Actual new LoRA SHA changed')
    need(sha(native/'model_dir/runner.py')=='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b','Native HYPIR source changed')
    need(sha(BASE/'runner.py')=='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9' and
      sha(BASE/'config.json')=='661bf9bd6a7ecb2ae5b475dda9dcd855700bb52be87ab0f2b6d97740f32308f4' and sha(TEMPLATE)==TEMPLATE_SHA,'Frozen VAE source/template changed')
    base=read(BASE.parent/'candidate_manifest.json');engines=base['actual_engines'];need(set(engines)=={'encoder','decoder'},'Only VAE engines may be reused')
    for name,e in engines.items():
        need(sha(e['path'])==e['sha256'] and sha(e['manifest'])==e['manifest_sha256'],'Actual VAE engine identity changed')
    cfg=read(BASE/'config.json')
    provenance=dict(t,training_run=str(RUN),global_step=a.step,local_step=a.step-7750,optimizer_step=a.step-7750,initial_A_checkpoint_sha256=A,no_teacher_outputs=True,no_val_GT=True)
    cfg.update(label=label,weight_path=str(raw),checkpoint_sha256=m['checkpoint_sha256'],training_provenance=provenance)
    b=(json.dumps(cfg,indent=2)+'\n').encode()
    manifest=dict(schema='hypir_finetuned_trt_vae_candidate_v1',model_type='HYPIR_SD2_rank256_finetuned_TRT_VAE_native512',
      label=label,model_dir=str(dest/'model_dir'),checkpoint=str(raw),checkpoint_sha256=m['checkpoint_sha256'],initial_A_checkpoint_sha256=A,
      global_step=a.step,local_step=a.step-7750,optimizer_step=a.step-7750,runner_sha256=TEMPLATE_SHA,config_sha256=hashlib.sha256(b).hexdigest(),
      fine_tuned_checkpoint=True,backend_has_no_training=True,no_distillation=True,no_teacher_outputs=True,no_val_GT_training=True,
      training_provenance=provenance,native_candidate_manifest=ref(native/'candidate_manifest.json'),actual_engines=engines,
      original_VAE_sha256=base['original_VAE_sha256'],TRT_runtime_sha256=base['TRT_runtime_sha256'],
      quality_reuse_claim=False,requires_new_actual_full103_418=True,no_A_UNet_engine=True,
      native_raw_resume_tensor_recheck_in_this_backend_registration=False,registration_source_sha256=sha(__file__))
    for path,(st,digest) in SEEN.items():
        s=path.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed before commit')
    tmp=Path(tempfile.mkdtemp(prefix='.finetuned_trtvae_',dir=dest.parent))
    try:
        md=tmp/'model_dir';md.mkdir()
        shutil.copyfile(TEMPLATE,md/'runner.py')
        for name in ['reference_runner.py','trt_vae_runtime_v2.py']:shutil.copyfile(BASE/name,md/name)
        (md/'config.json').write_bytes(b);(tmp/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        need(sha(md/'runner.py')==TEMPLATE_SHA,'Source copy failed');os.rename(tmp,dest)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print(json.dumps(dict(status='REGISTERED_ACTUAL_FINETUNED_HYPIR_TRT_VAE',model_dir=str(dest/'model_dir'),checkpoint_sha256=m['checkpoint_sha256'],
      candidate_manifest_sha256=sha(dest/'candidate_manifest.json'),runner_sha256=TEMPLATE_SHA,GPU=False)))
if __name__=='__main__':main()

