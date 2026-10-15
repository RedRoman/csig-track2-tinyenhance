"""Prepare immutable native HYPIR coefficient controls; no CUDA or training."""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
BASE=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
RAW=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
RS='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
CS='8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8'
WS='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
SEEN={}
def need(v,m):
    if not v: raise RuntimeError(m)
def sha(p):
    p=Path(p).resolve();s=p.stat();st=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
    if p in SEEN:
        need(SEEN[p][0]==st,'Mutated source: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    t=p.stat();need(st==(t.st_size,t.st_mtime_ns,t.st_ino,t.st_dev),'Source changed during read')
    SEEN[p]=(st,h.hexdigest());return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--coeff-t',type=int,choices=[165,185],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    label=f'push7_a7750_ct{a.coeff_t}';dest=R/'submissions'/f'hypir_{label}'
    need(not dest.exists(),'Immutable candidate already exists')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',model_dir=str(dest/'model_dir'),label=label,coeff_t=a.coeff_t,runner_sha256=RS,checkpoint_sha256=WS,GPU=False)));return
    need(sha(BASE/'runner.py')==RS and sha(BASE/'config.json')==CS and sha(RAW)==WS,'Actual frozen base/source/raw mismatch')
    base=json.loads((BASE/'config.json').read_text());cfg=dict(base,label=label,coeff_t=a.coeff_t)
    need({k for k in base if base[k]!=cfg[k]}=={'label','coeff_t'},'Only label and coeff_t may change')
    rb=(json.dumps(cfg,indent=2)+'\n').encode()
    manifest=dict(schema='push7_hypir_native_coefficient_control_v1',model_type='HYPIR_SD2_rank256_native512',
        label=label,model_dir=str(dest/'model_dir'),checkpoint=str(RAW),checkpoint_sha256=WS,
        runner_sha256=RS,config_sha256=hashlib.sha256(rb).hexdigest(),
        base_config=dict(path=str(BASE/'config.json'),sha256=CS),config_changed_fields=['label','coeff_t'],
        source_sha256={str(BASE/'runner.py'):RS},model_t=200,coeff_t=a.coeff_t,patch_size=512,stride=256,
        internal_precision='bfloat16',native_resolution=True,upscale=1,external_resize=False,
        no_new_training=True,no_distillation=True,architecture_unchanged=True,
        quality_reuse_claim=False,requires_fresh_quality_evaluation=True,
        render_seed=231,scorer_seed=1234,CPU_only_preparation=True,GPU_started_by_preparation_agent=False,
        registrar_source_sha256=sha(__file__))
    tmp=Path(tempfile.mkdtemp(prefix='.hypir_coeff_',dir=dest.parent))
    try:
        md=tmp/'model_dir';md.mkdir();shutil.copyfile(BASE/'runner.py',md/'runner.py');(md/'config.json').write_bytes(rb)
        (tmp/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        need(sha(md/'runner.py')==RS and sha(md/'config.json')==manifest['config_sha256'],'Copy verification failed')
        for path,(st,d) in SEEN.items():
            s=path.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed before commit')
        os.rename(tmp,dest)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print(json.dumps(dict(status='REGISTERED_ACTUAL_SAME_WEIGHT_COEFF_CONTROL',model_dir=str(dest/'model_dir'),candidate_manifest_sha256=sha(dest/'candidate_manifest.json'),runner_sha256=RS,config_sha256=manifest['config_sha256'],checkpoint_sha256=WS,coeff_t=a.coeff_t,GPU=False)))
if __name__=='__main__': main()

