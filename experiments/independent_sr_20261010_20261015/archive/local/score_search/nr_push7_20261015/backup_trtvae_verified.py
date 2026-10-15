"""Small selected TRT-VAE backup; stdlib CPU only, default cache plan.

New SFS prefix only. No old archive mutation, raw/resume/site/fullPNG copying.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil
R=Path('/cache/independent_sr_cuda_20261013');P=Path('/temp/independent_sr_cuda_20261013_artifacts')
B=R/'backups/nr_push7_trtvae_verified_20261015_bundle';D=P/'nr_push7_trtvae_verified_20261015'
SEL=R/'results/nr_push7_selection.json';SEL_SHA='6ecc19473fbc5b457da8b420c272e121cc32d932e5c12c072926f5ff54b89db5'
OLD=P/'nr_push4_20261014';OLD_M='3e1c41dc447fcb242aac85d7e6834b7bcb5f6d3f5a23ff63f255e682506cd35a';OLD_D='ac09208eef5b26f75f5bdf55ccfc5a02b694673ed3540df67607a5a6488e722f'
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'

def need(v,msg):
    if not v:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,j):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        f.write(json.dumps(j,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def heads():
    result={}
    for d in sorted(P.iterdir()):
        if not d.is_dir() or d==D:continue
        mf=d/'manifest.json';done=d/'DONE' if (d/'DONE').exists() else d/'DONE.json'
        if mf.is_file() and done.is_file():result[d.name]=dict(manifest_sha256=sha(mf),DONE_filename=done.name,DONE_sha256=sha(done))
        elif mf.is_file() and (d/'final_commit/DONE').is_file():
            result[d.name]=dict(manifest_sha256=sha(mf),final_manifest_sha256=sha(d/'final_commit/manifest.json'),DONE_filename='final_commit/DONE',DONE_sha256=sha(d/'final_commit/DONE'),stable_payload_receipt_sha256=sha(d/'STABLE_PAYLOAD_RECEIPT.json'))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    need(not D.exists(),'New immutable SFS destination already exists; never overwrite')
    need(sha(SEL)==SEL_SHA,'Pinned verified selection changed');s=json.loads(SEL.read_text())
    need(s['selected_label']=='push7_a7750_trtvae_fp16_all' and s['checkpoint_sha256']==A_SHA and s['distillation'] is False and s['quality_reused'] is False,'Selected HYPIR backend identity')
    q=Path(s['quality_dir']);md=Path(s['model_dir']);cal=Path(s['formal_speed_dir']);qa=Path(s['quality_audit']['remote']);sa=Path(s['speed_audit']['remote'])
    need(sha(qa)==s['quality_audit']['sha256']=='1e54ac9a0681b67144c2f173e458d9c23c8f1d9d41fec17e7c278cc80470c63b' and sha(sa)==s['speed_audit']['sha256'],'Actual quality/speed audits')
    j=json.loads(qa.read_text());need(j['status']=='PASS_FULL103_418' and j['candidates'][0]['image_rows']==103 and j['candidates'][0]['scalar_metrics']==418,'Fresh full103/418 required')
    need(sha(md/'runner.py')=='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9' and sha(md/'config.json')=='661bf9bd6a7ecb2ae5b475dda9dcd855700bb52be87ab0f2b6d97740f32308f4','Selected runtime pins')
    before_heads=heads();need(sha(OLD/'manifest.json')==OLD_M and sha(OLD/'DONE')==OLD_D,'Old Push4 commit heads changed')
    old=json.loads((OLD/'manifest.json').read_text());old_receipt=R/'results/nr_push4_backup_receipt_20261014.json';receipt=json.loads(old_receipt.read_text())
    need(receipt['status']=='verified_complete' and receipt['manifest_sha256']==OLD_M and receipt['DONE_sha256']==OLD_D and receipt['all_file_actual_SFS_bytes_full_SHA_and_size_reread'] is True and receipt['selected_raw_sha256']==A_SHA,'Prior full-SFS verification proof missing')
    old_records={v['path']:v for v in old['files']};resume=receipt['selected_immutable_resume_path'];need((OLD/resume).is_file() and (OLD/resume).stat().st_size==old_records[resume]['bytes'] and old_records[resume]['sha256']==receipt['selected_immutable_resume_sha256'],'Retained exact immutable resume absent')
    files=B/'files';files.mkdir(parents=True,exist_ok=True);sources={}
    def add(src,rel=None):
        src=Path(src);need(src.is_file() and not src.is_symlink(),'Regular source only: '+str(src));rel=rel or str(src.relative_to(R))
        need('..' not in Path(rel).parts and not Path(rel).is_absolute(),'Unsafe relative key')
        if rel in sources:need(sources[rel]['source']==str(src),'Duplicate payload key');return
        target=files/rel;target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(src);shutil.copyfile(src,target);need(sha(target)==digest,'Cache bundle copy mismatch')
        sources[rel]=dict(source=str(src),path=rel,bytes=src.stat().st_size,sha256=digest)
    add(SEL);add(qa);add(sa);add(old_receipt,'prior_A7750/nr_push4_independent_receipt.json')
    for n in ['manifest.json','DONE']:add(OLD/n,'prior_A7750/push4_'+n)
    for rel in ['audits/push4/raw_recovery/push4_a7750_compiled_all/recipe.json','audits/push4/raw_recovery/push4_a7750_compiled_all/raw_zip_gaps.bin','audits/push4/raw_recovery/push4_a7750_compiled_all/probe_result.json','code/push4/checkpoint_archive_recipe.py']:
        need(sha(OLD/rel)==old_records[rel]['sha256'],'Actual retained exact-raw recipe changed');add(OLD/rel,'prior_A7750/'+rel)
    reference=dict(scope='Reference to old immutable Push4 archive; new backup contains NO LoRA raw/resume',archive=str(OLD),manifest_sha256=OLD_M,DONE_sha256=OLD_D,
        resume_path=resume,resume_sha256=old_records[resume]['sha256'],resume_bytes=old_records[resume]['bytes'],resume_exists_size_checked_now=True,resume_full_SHA_reread_this_run=False,
        resume_full_SHA_previously_independently_verified=True,previous_receipt_sha256=sha(old_receipt),raw_recovery=receipt['selected_raw_exact_recovery_from_retained_resume_and_recipe'],raw_sha256=A_SHA)
    write(B/'A7750_WEIGHT_REFERENCE.json',reference);add(B/'A7750_WEIGHT_REFERENCE.json','A7750_WEIGHT_REFERENCE.json')
    c=json.loads((md/'config.json').read_text());engine_paths=[]
    for name,v in c['TRT_engines'].items():
        need(sha(v['path'])==v['sha256'] and sha(v['manifest'])==v['manifest_sha256'],'Actual VAE engine changed')
        engine_paths.append(str(Path(v['path'])));add(v['path']);add(v['manifest'])
    need(sum(Path(p).stat().st_size for p in engine_paths)==173701256,'Exactly two selected VAE engines expected')
    for p in md.iterdir():
        if p.is_file() and p.suffix in ['.py','.json']:add(p)
    add(md.parent/'candidate_manifest.json')
    for p in q.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl']:add(p)
    for p in cal.iterdir():
        if p.is_file() and p.suffix in ['.json','.log']:add(p)
    for p in (R/'audits/push7/speed').glob('formal4*.json'):add(p)
    for rel in ['code/render_cuda_submission.py','code/score_cuda_stream.py','code/benchmark_cuda.py','code/numpy_legacy_compat.py','code/maniqa_offline_init.py','code/push7/launch_formal_push7.py','code/push7/audit_formal4_cpu.py','code/push7/audit_full_trt.py','code/push7/trt_vae/build_vae_trt.py','code/push7/trt_vae/trt_vae_runtime_v2.py','code/push7/trt_vae/reprobe_vae_trt_stream_v2.py']:
        add(R/rel)
    for folder in [R/'scoring_tool',R/'repos/HYPIR_cuda/HYPIR']:
        for p in folder.rglob('*.py'):add(p)
    site=R/'backends/push7_trt/site'
    for dist in site.glob('*.dist-info'):
        for name in ['METADATA','WHEEL','direct_url.json','INSTALLER','RECORD']:
            if (dist/name).is_file():add(dist/name,'environment/wheel_metadata/'+dist.name+'/'+name)
    visual=B/'incoming_visuals';need(sha(visual/'preview_provenance.json')==s['visual']['sha256'],'Selected reviewed visual proof changed')
    for p in visual.iterdir():add(p,'visuals/'+p.name)
    add(Path(__file__).resolve(),'code/backup_trtvae_verified.py')
    note='This archive preserves ONLY the verified HYPIR A7750 TensorRT VAE backend: two FP16 engines, runtime/source/config, fresh103/418 metadata, formal4 timings and four fixed 1:1 ROI panels. Tool prediction62.32913337707007; target65 NOT achieved. No distillation/new teacher training. No raw/resume/base weights/TRT installation/full103 native PNGs are copied. Recover exact A7750 raw with prior_A7750 recipe+gaps+helper and the referenced immutable Push4 resume; verify raw SHA589324...3977d. Restore the archived relative /cache tree to the original root or update explicit paths while retaining artifact SHA guards. Install Linux x86_64 Python3.13 Torch2.8.0+cu128, Torch-TensorRT2.8.0 and TensorRT10.12.0.36; pinned actual wheel METADATA/WHEEL/direct_url/RECORD are included, not binaries. Engines were built on A100-SXM4-80GB; new hardware/runtime needs its own build/verification. Original SD2 base and metric weights remain shared/prior-backup references. Full native output PNGs remain in /cache; only four comparison ROI panels are persisted here.'
    (B/'SCOPE_AND_RESTORE.txt').write_text(note+'\n');add(B/'SCOPE_AND_RESTORE.txt','SCOPE_AND_RESTORE.txt')
    data=sorted(sources.values(),key=lambda r:r['path']);total=sum(r['bytes'] for r in data)
    need(total<=200000000,'Payload exceeds200MB budget')
    need([r['source'] for r in data if r['path'].endswith('.engine')]==sorted(engine_paths),'Engine scope')
    need(not any(Path(r['path']).suffix in ['.pt','.pth','.safetensors','.so','.whl'] for r in data),'No tensor archives/site binaries')
    need(sum(r['path'].endswith('.png') for r in data)==4,'Exactly four ROI panels, no native full PNGs')
    reserve=32*1024**2;free=shutil.disk_usage(P).free;plan=dict(schema='push7_trtvae_small_backup_v1',status='CACHE_BUNDLE_READY',selection_sha256=SEL_SHA,destination=str(D),scope='two_FP16_VAE_engines_small_proofs_and_ROIs_only_A_raw_resume_referenced',files=data,count=len(data),bytes=total,free_before=free,reserve_bytes=reserve,metadata_allowance_bytes=1024**2,space_sufficient=free>=total+reserve+1024**2,prior_backup_heads=before_heads,weight_reference=reference,raw_persisted=False,resume_persisted=False,full103_PNGs_persisted=False,ROI_PNGs=4,engine_count=2)
    write(B/'backup_plan.json',plan);print(json.dumps({k:plan[k] for k in ['status','count','bytes','free_before','space_sufficient']}),flush=True)
    if not a.execute or not plan['space_sufficient']:return
    D.mkdir();copied=0
    for row in data:
        src=files/row['path'];dst=D/row['path'];need(sha(src)==row['sha256'],'Cache source guard')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as fi,dst.open('xb') as fo:
            for chunk in iter(lambda:fi.read(8*1024**2),b''):fo.write(chunk)
            fo.flush();os.fsync(fo.fileno())
        need(dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256'],'Actual SFS SHA/size readback failed');copied+=1
    for row in data:need(sha(row['source'])==row['sha256'],'Original source changed during backup')
    need(heads()==before_heads and sha(SEL)==SEL_SHA,'Old heads or selection changed')
    actual={str(p.relative_to(D)) for p in D.rglob('*') if p.is_file()}
    need(not any(p.is_symlink() for p in D.rglob('*')) and actual=={r['path'] for r in data},'Actual new-prefix inventory differs from payload')
    need(shutil.disk_usage(P).free>=reserve+1024**2,'Insufficient remaining free before commit')
    plan.update(status='complete',actual_all_file_SFS_SHA_size_readback_verified=True,actual_verified_files=copied,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(D/'manifest.json',plan);mh=sha(D/'manifest.json');done=dict(status='complete',manifest_sha256=mh,selection_sha256=SEL_SHA,files=len(data),bytes=total,scope=plan['scope'],raw_persisted=False,resume_persisted=False,full103_PNGs_persisted=False,ROI_PNGs=4,engine_count=2)
    write(D/'DONE',done);need(json.loads((D/'DONE').read_text())==done and sha(D/'manifest.json')==mh,'Commit readback')
    actual={str(p.relative_to(D)) for p in D.rglob('*') if p.is_file()}
    need(actual=={r['path'] for r in data}|{'manifest.json','DONE'},'Postcommit new-prefix inventory mismatch')
    final=dict(status='verified_complete',destination=str(D),manifest_sha256=mh,DONE_sha256=sha(D/'DONE'),files=len(data),bytes=total,all_file_actual_SFS_SHA_readback=True,old_heads_unchanged=True,raw_persisted=False,resume_persisted=False,full103_PNGs_persisted=False,engine_count=2,ROI_PNGs=4,free_after=shutil.disk_usage(P).free,selection_sha256=SEL_SHA)
    need(final['free_after']>=reserve,'Actual remaining free below32MiB')
    write(R/'results/nr_push7_trtvae_backup_receipt_20261015.json',final);write(B/'backup_receipt.json',final);print(json.dumps(final),flush=True)

if __name__=='__main__':main()
