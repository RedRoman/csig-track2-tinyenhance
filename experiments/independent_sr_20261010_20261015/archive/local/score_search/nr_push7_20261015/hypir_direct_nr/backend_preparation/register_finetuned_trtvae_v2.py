"""Explicit CPU registry for direct NIQE milestones or audited HYPIR parameter blends.
Root owns execute and positive pilot/visual gate. No copied raw/engine or A UNet.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
BASE=R/'submissions/hypir_push7_a7750_trtvae_fp16/model_dir'
TEMPLATE=R/'code/push7/hypir_direct_nr/finetuned_trtvae_runner.py'
TS='4b225fbe82c556abf4b409927d16709e771a69fb811d7c6803f7281792875862'
DIRECT=R/'code/push7/hypir_direct_nr/register_finetuned_trtvae.py'
DS='0854d50fac7bcac2e05bf75fca2e634d7ae17c385f45e3bdd531ebb7cdd5ac7c'
PROOF=R/'audits/push7/parameter_blends/independent514_cpu.json'
PS='1286bac0cf9e70a90a63c10b5a727a24cacec71976c416f6148c1fa3c683e20b'
A='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
N='763950727fe29a009245cd9cd855743a99803fa9fbf6824a62d38994fa8e0650'
EXPECTED={'alpha025':(.25,'daa6e782eb219222f7becc6c5d2a26038fa2a7897c90db7736c8ba38c885d993'),
          'alpha050':(.5,'3438c5913333729ade0560dde38fa25813932754ddabefffde7529c511cd2f58')}
SEEN={}
def need(x,m):
    if not x:raise RuntimeError(m)
def sha(p):
    p=Path(p).resolve();s=p.stat();st=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
    if p in SEEN:
        need(SEEN[p][0]==st,'Source changed');return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    s=p.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed during SHA')
    SEEN[p]=(st,h.hexdigest());return h.hexdigest()
def read(p):sha(p);return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def main():
    q=argparse.ArgumentParser(description=__doc__);g=q.add_mutually_exclusive_group(required=True)
    g.add_argument('--step',type=int,choices=[7800,7850]);g.add_argument('--blend',choices=list(EXPECTED))
    q.add_argument('--execute',action='store_true');a=q.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.step:
        need(sha(DIRECT)==DS,'Frozen direct registry changed')
        cmd=[sys.executable,str(DIRECT),'--step',str(a.step)]+(['--execute'] if a.execute else [])
        subprocess.run(cmd,check=True);return
    alpha,raw_sha=EXPECTED[a.blend];native=R/'submissions'/('hypir_push7_a7750_niqe100_'+a.blend)
    label='push7_a7750_niqe100_'+a.blend+'_trtvae';dest=R/'submissions'/('hypir_'+label)
    need(not dest.exists(),'Immutable candidate exists')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',native_candidate=str(native),model_dir=str(dest/'model_dir'),label=label,
          alpha=alpha,requires_actual_514_formula_proof=True,requires_root_positive_pilot_visual_gate=True,
          no_derived_optimizer_or_resume=True,reuses_only_two_VAE_engines=True,no_A_UNet_engine=True,GPU=False)));return
    m=read(native/'candidate_manifest.json');proof=read(PROOF)
    need(proof['status']=='PASS_EXACT_FP32_LERP514_BOTH' and sha(proof['auditor']['path'])==proof['auditor']['sha256']==PS,'Actual independent interpolation audit absent')
    need(m['schema']=='HYPIR_FP32_parameter_interpolation_v1' and m['alpha']==alpha and m['checkpoint_sha256']==raw_sha and
      m['initial_checkpoint_sha256']==A and m['finetuned_checkpoint_sha256']==N,'Explicit A toNIQE100 blend identity differs')
    need(m['recipe']=='torch.lerp(initial_A_FP32,NIQE100_FP32,alpha) on CPU per514 ordered tensors' and m['new_training'] is False and
      m['distillation'] is False and m['lightweight_model'] is False and m['no_teacher_outputs'] is True and m['no_val_GT'] is True and
      m['optimizer_state']=='not applicable: derived parameters, no new Adam/step/resume','Derived lineage/optimizer mislabeled')
    records=[x for x in proof['candidates'] if x['checkpoint']['sha256']==raw_sha]
    need(len(records)==1 and records[0]['alpha']==alpha and records[0]['exact_FP32_lerp_tensors']==514 and
      records[0]['native_candidate_manifest']['sha256']==sha(native/'candidate_manifest.json'),'Formula proof is for another native candidate')
    need(sha(m['checkpoint'])==raw_sha and sha(m['training_marker']['path'])==m['training_marker']['sha256'],'Actual derived raw/upstream marker changed')
    need(sha(proof['producer']['path'])==proof['producer']['sha256']==m['derivation_source_sha256']=='2bb83af6f7cd988b7cbf210aa241f7be48f93337c126ca0813ec4a02e802a2d3','Actual torch.lerp producer changed')
    need(sha(native/'model_dir/runner.py')==m['runner_sha256']=='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b' and
      sha(native/'model_dir/config.json')==m['config_sha256'],'Native original HYPIR registration changed')
    need(sha(BASE/'runner.py')=='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9' and
      sha(BASE/'config.json')=='661bf9bd6a7ecb2ae5b475dda9dcd855700bb52be87ab0f2b6d97740f32308f4' and sha(TEMPLATE)==TS,'Frozen backend/template changed')
    base=read(BASE.parent/'candidate_manifest.json');engines=base['actual_engines'];need(set(engines)=={'encoder','decoder'},'Only VAE engines may be reused')
    for e in engines.values():need(sha(e['path'])==e['sha256'] and sha(e['manifest'])==e['manifest_sha256'],'Actual VAE engine changed')
    cfg=read(BASE/'config.json');need(set(cfg['TRT_engines'])=={'encoder','decoder'} and 'TRT_UNet_engine' not in cfg,'A UNet engine forbidden')
    provenance=dict(checkpoint_kind='HYPIR_FP32_parameter_interpolation',checkpoint_derivation=dict(m),
      independent514_formula_audit=ref(PROOF),native_candidate_manifest=ref(native/'candidate_manifest.json'),
      upstream_finetuning_marker=dict(m['training_marker']),derived_optimizer_or_resume='not applicable; no new training step/RNG')
    cfg.update(label=label,weight_path=m['checkpoint'],checkpoint_sha256=raw_sha,training_provenance=provenance)
    b=(json.dumps(cfg,indent=2)+'\n').encode()
    manifest=dict(schema='hypir_parameter_blend_trt_vae_candidate_v1',model_type='HYPIR_SD2_rank256_finetuned_TRT_VAE_native512',
      label=label,model_dir=str(dest/'model_dir'),checkpoint=m['checkpoint'],checkpoint_sha256=raw_sha,initial_A_checkpoint_sha256=A,
      alpha=alpha,fine_tuned_checkpoint=True,checkpoint_parameter_interpolation=True,directly_trained_checkpoint=False,
      derived_optimizer_or_resume=False,backend_has_no_training=True,no_distillation=True,no_teacher_outputs=True,no_val_GT_training=True,
      runner_sha256=TS,config_sha256=hashlib.sha256(b).hexdigest(),training_provenance=provenance,
      actual_engines=engines,original_VAE_sha256=base['original_VAE_sha256'],TRT_runtime_sha256=base['TRT_runtime_sha256'],
      quality_reuse_claim=False,requires_new_actual_full103_418=True,no_A_UNet_engine=True,registration_source_sha256=sha(__file__))
    for path,(st,d) in SEEN.items():
        s=path.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed before commit')
    tmp=Path(tempfile.mkdtemp(prefix='.blend_trtvae_',dir=dest.parent))
    try:
        md=tmp/'model_dir';md.mkdir();shutil.copyfile(TEMPLATE,md/'runner.py')
        for name in ['reference_runner.py','trt_vae_runtime_v2.py']:shutil.copyfile(BASE/name,md/name)
        (md/'config.json').write_bytes(b);(tmp/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        need(sha(md/'runner.py')==TS,'Runner byte copy differs');os.rename(tmp,dest)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print(json.dumps(dict(status='REGISTERED_ACTUAL_HYPIR_PARAMETER_BLEND_TRT_VAE',model_dir=str(dest/'model_dir'),
      candidate_manifest_sha256=sha(dest/'candidate_manifest.json'),checkpoint_sha256=raw_sha,alpha=alpha,GPU=False)))
if __name__=='__main__':main()

