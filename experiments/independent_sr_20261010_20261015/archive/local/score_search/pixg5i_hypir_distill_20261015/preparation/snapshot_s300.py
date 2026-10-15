"""New CACHE-only exact step300 snapshot; logs are frozen prefixes, not live files."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile
R=Path('/cache/independent_sr_cuda_20261013');E=R/'experiments/pixg5i_hypir_distill_20261015';C=Path('/cache/pixg5i_s300_snapshot_20261015')
import argparse
_ap=argparse.ArgumentParser(description=__doc__);_ap.add_argument('--audit-sha256',required=True)
AUDIT_SHA=_ap.parse_args().audit_sha256
def need(x,m):
    if not x:raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
need(not C.exists(),'New immutable CACHE snapshot only');C.mkdir();records={};source_guards={}
def add(source,expected=None,key=None):
    source=Path(source);need(source.is_file() and source.stat().st_size<60*1024**2,'Bounded actual files only')
    rel=key or str(source.relative_to(R));need(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Safe snapshot relative path')
    digest=sha(source);need(expected is None or digest==expected,'Actual source SHA differs')
    if rel in records:need(records[rel]['sha256']==digest,'Snapshot collision');return
    s=source.stat();token=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev);out=C/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,out)
    t=source.stat();need(token==(t.st_size,t.st_mtime_ns,t.st_ino,t.st_dev) and sha(out)==digest,'Copy/source changed')
    records[rel]=dict(path=rel,bytes=out.stat().st_size,sha256=digest,source=str(source));source_guards[str(source)]=token
audit_path=E/'audits/training_steps300_cpu.json';need(sha(audit_path)==AUDIT_SHA,'Actual independent training audit')
audit=read(audit_path);need(audit['status']=='PASS_PIXG5I_FP32_ADAM_SCALER_RNG','Training audit not PASS');add(audit_path,AUDIT_SHA)
add('/cache/audit_pixg5i_training_cpu.py',audit['auditor_source_sha256'],'backup_proofs/audit_training_cpu.py')
for model in audit['models']:
    d=Path(model['run']);m=read(d/'checkpoint_step300.json');proof=model['milestones']['300']
    need(m['raw_sha256']==proof['raw_sha256'] and m['resume_sha256']==proof['resume_sha256'] and m['local_step']==m['optimizer_step']==300,'Actual audited checkpoint100')
    for key,hkey in [('raw_checkpoint','raw_sha256'),('resume_checkpoint','resume_sha256')]:add(m[key],m[hkey])
    for n in ['checkpoint_step300.json','config.json','input_catalog.json','teacher_catalog.json','initial_deployment_relation.json','gradient_preflight_step1.json']:
        add(d/n)
    for path,h in read(d/'config.json')['protocol']['source_sha256'].items():add(path,h)
    lines=(d/'train.jsonl').read_text().splitlines();need(len(lines)>=300,'Actual300 log rows missing');prefix=('\n'.join(lines[:300])+'\n').encode()
    need(hashlib.sha256(prefix).hexdigest()==model['log_prefix_sha256'],'Actual300 log prefix differs')
    rel=str(d.relative_to(R)/'train_step300_snapshot.jsonl');out=C/rel;out.write_bytes(prefix)
    records[rel]=dict(path=rel,bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),source=str(d/'train.jsonl'),scope='first300 actual lines; future appended lines deliberately excluded')
    cm=R/'submissions'/('pixg5i_hypir_'+model['variant']+'_s300')/'candidate_manifest.json'
    if cm.is_file():
        add(cm);registered=read(cm)
        for name,row in registered['files'].items():
            if name!='model.pth':add(row['path'],row['sha256'])
write(C/'RECOVERY_SCOPE.json',dict(scope='Two actual step300 raw weights+matching immutable FP32 Adam/scaler/fourRNG resumes, small sources/proofs and frozen100 logs',
    initial_package_checkpoint_sha256=audit['initial_package_sha256'],initial_package_weights_copied=False,
    initial_package_location='Local original_package/model_dir/model.pth or remote source_payload/pixg5i_original/model_dir/model.pth',
    teacher_raw_sha256=audit['teacher_raw_sha256'],teacher_PNGs_copied=False,
    teacher_training_directory='/cache/independent_sr_cuda_20261013/nr_push7_recovered_full_evaluations/push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all/render/test',
    external_for_further_training='Fixed100 teacher PNGs and original100 JPEGs, two frozen LPIPS weights and native CUDA runtime remain external; no teacher needed to deploy student.',
    deployment_weight_reconstruction='Copy each backed-up runs/pixg5i_hypir_VARIANT/step300.pth to corresponding submissions/pixg5i_hypir_VARIANT_s300/model_dir/model.pth; identical actual SHA.',
    future_log_append_allowed=True,SFS_written=False,GPU_started=False,quality_conclusion=False))
add(Path(__file__),key='backup_code/snapshot_s300.py')
for path,token in source_guards.items():
    s=Path(path).stat();need((s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==token,'Immutable source changed before commit')
rows=[]
for p in sorted(C.rglob('*')):
    need(not p.is_symlink(),'No snapshot symlinks')
    if p.is_file():rows.append(dict(path=str(p.relative_to(C)),bytes=p.stat().st_size,sha256=sha(p)))
manifest=dict(status='complete',schema='pixg5i_actual_s300_CACHE_MAC_snapshot_v1',files=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows),
    training_audit_sha256=AUDIT_SHA,all_actual_snapshot_file_SHA_verified=True,source_records=list(records.values()),
    mutable_log_scope='first300 lines frozen independently; later append does not invalidate snapshot',SFS_written=False,GPU_started=False)
write(C/'manifest.json',manifest);write(C/'DONE',dict(status='complete',manifest_sha256=sha(C/'manifest.json'),files=len(rows),bytes=manifest['bytes']))
archive=Path('/cache/pixg5i_s300_verified.tar');need(not archive.exists(),'New cache tar only')
with tarfile.open(archive,'x') as tar:tar.add(C,arcname='.')
receipt=dict(status='CACHE_SNAPSHOT_READY_LOCAL_TRANSFER_PENDING',directory=str(C),files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(C/'manifest.json'),DONE_sha256=sha(C/'DONE'),tar=str(archive),tar_sha256=sha(archive),SFS_written=False,GPU_started=False)
write(Path('/cache/pixg5i_s300_snapshot_receipt.json'),receipt);print(json.dumps(receipt),flush=True)
