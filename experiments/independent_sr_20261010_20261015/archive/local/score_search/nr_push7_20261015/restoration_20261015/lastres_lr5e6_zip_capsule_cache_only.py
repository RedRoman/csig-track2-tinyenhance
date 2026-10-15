"""Six-parameter LR5e-6 exact ZIP capsule; CACHE/MAC ONLY, SFS readonly, stdlib/no GPU."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,zipfile,zlib
R=Path('/cache/independent_sr_cuda_20261013');P=Path('/temp/independent_sr_cuda_20261013_artifacts')
OLD=P/'nr_push4_20261014';REL='adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/resume_step7750.pt';BASE=OLD/REL
BASE_SHA='e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b';BASE_BYTES=3114332685
C=Path('/cache/push7_lastres_lr5e6_nr_capsule_20261015');V=Path('/cache/push7_lastres_lr5e6_nr_capsule_verification_20261015');DEST=None # This lastres package is CACHE/MAC ONLY; no SFS destination exists.
ROOT_RAW_SHA={7800:'dc184b14f549fc01aeafbdc3bdcf6a7112e6bc3944d6c047c557ed850074d7b7',7850:'23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'}
RUN=R/'adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100';LIMIT=160*1024**2;RESERVE=32*1024**2
def need(v,m):
 if not v:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,j):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:f.write(json.dumps(j,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def read(p):return json.loads(p.read_text())
def stamp(p):
 s=p.stat();return [s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev]
def data_start(stream,info):
 stream.seek(info.header_offset);h=stream.read(30);need(h[:4]==b'PK\x03\x04','ZIP local header');return info.header_offset+30+int.from_bytes(h[26:28],'little')+int.from_bytes(h[28:30],'little')
def fixed_base():
 need(sha(P/'nr_push7_lastconv_nr_verified_20261015/manifest.json')=='5d748f2fe9061d21ebb9e84ccf41044bf8578178eae44b78955377b7ba9e23c9' and sha(P/'nr_push7_lastconv_nr_verified_20261015/DONE')=='889eec0bb66d7f140cb835d7219d5b0b6b434be6908001cc8e5c5a3167aa7e47','Frozen v1 commit heads changed')
 need(BASE.stat().st_size==BASE_BYTES and sha(BASE)==BASE_SHA,'Actual persistent A resume SHA/size differs')
 need(sha(OLD/'manifest.json')=='3e1c41dc447fcb242aac85d7e6834b7bcb5f6d3f5a23ff63f255e682506cd35a' and sha(OLD/'DONE')=='ac09208eef5b26f75f5bdf55ccfc5a02b694673ed3540df67607a5a6488e722f','Old A commit heads differ')
def build():
 need(not C.exists(),'New cache capsule only');fixed_base();C.mkdir();(C/'blobs').mkdir();blobs={};source_records=[]
 recipe_dir=OLD/'audits/push4/raw_recovery/push4_a7750_compiled_all';need(sha(recipe_dir/'recipe.json')=='6ae28dcb95be58d91a518f3706944ba39210ec1c0af17fb7dbf8474023547b67','Prior A raw recipe')
 oldrecipe=read(recipe_dir/'recipe.json');need(oldrecipe['resume_sha256']==BASE_SHA,'Recipe base binding');reference={}
 with zipfile.ZipFile(BASE) as z,BASE.open('rb') as f:
  for s in oldrecipe['segments']:
   if s['kind']!='resume_entry':continue
   info=z.getinfo(s['name']);need(info.compress_type==zipfile.ZIP_STORED and info.file_size==info.compress_size==s['bytes'],'Stored base payload only')
   reference[(s['bytes'],s['sha256'])]=dict(offset=data_start(f,info),bytes=s['bytes'],sha256=s['sha256'],entry_name=info.filename)
 def store(data):
  digest=hashlib.sha256(data).hexdigest()
  if digest not in blobs:
   packed=zlib.compress(data,6);path=C/'blobs'/(digest+'.z');path.write_bytes(packed)
   blobs[digest]=dict(path=str(path.relative_to(C)),bytes=len(data),compressed_bytes=len(packed),compressed_sha256=sha(path))
  return dict(kind='blob',sha256=digest,bytes=len(data))
 def metadata(src,key=None):
  src=Path(src);need(src.is_file() and src.stat().st_size<2*1024**2,'Small metadata source only')
  rel=key or str(src.relative_to(R));need(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Metadata key')
  out=C/'metadata'/rel
  if out.exists():need(sha(out)==sha(src),'Metadata collision');return
  out.parent.mkdir(parents=True,exist_ok=True);digest=sha(src);shutil.copyfile(src,out);need(sha(out)==digest,'Metadata copy')
  source_records.append(dict(source=str(src),capsule_path=str(out.relative_to(C)),bytes=src.stat().st_size,sha256=digest,source_stamp=stamp(src)))
 files=[]
 for step in [7800,7850]:
  marker=read(RUN/('checkpoint_step%d.json'%step));need(marker['raw_checkpoint_sha256']==ROOT_RAW_SHA[step],'Root actual registered raw SHA differs');need(marker['global_step']==step and marker['local_step']==marker['optimizer_step']==step-7750 and marker['optimizer_tensor_count']==6 and marker['unselected508_exact_A_count']==508 and marker['changed_FP32_LoRA_tensors']==6,'Actual six-parameter marker scope');metadata(RUN/('checkpoint_step%d.json'%step))
  for pathkey,hashkey in [('raw_checkpoint','raw_checkpoint_sha256'),('resume_checkpoint','resume_checkpoint_sha256')]:
   path=Path(marker[pathkey]);token=stamp(path);segments=[];whole=hashlib.sha256();position=0;base_count=0
   with zipfile.ZipFile(path) as z,path.open('rb') as f:
    ranges=[]
    for info in z.infolist():
     need(not info.is_dir() and info.compress_type==zipfile.ZIP_STORED and info.file_size==info.compress_size,'All target ZIP entries must be uncompressed')
     ranges.append((data_start(f,info),info.file_size))
    for offset,length in sorted(ranges):
     need(offset>=position and offset+length<=token[0],'Disjoint bounded ZIP payloads')
     f.seek(position);gap=f.read(offset-position);need(len(gap)==offset-position,'Complete ZIP gap');whole.update(gap)
     if gap:segments.append(store(gap))
     f.seek(offset);h=hashlib.sha256();remaining=length
     while remaining:
      b=f.read(min(8*1024**2,remaining));need(b,'Complete ZIP payload');h.update(b);whole.update(b);remaining-=len(b)
     key=(length,h.hexdigest())
     if key in reference:segments.append(dict(kind='base',**reference[key]));base_count+=1
     else:
      need(length<=16*1024**2,'Unexpected large unique payload; refuse capsule');f.seek(offset);segments.append(store(f.read(length)))
     position=offset+length
    f.seek(position);tail=f.read();whole.update(tail)
    if tail:segments.append(store(tail))
   need(stamp(path)==token and whole.hexdigest()==marker[hashkey] and base_count>=508,'Actual source file SHA or shared508 proof differs')
   files.append(dict(name=path.name,original_path=str(path),bytes=token[0],sha256=whole.hexdigest(),source_stamp=token,base_payload_entries=base_count,segments=segments))
  metadata(RUN/('config.json'));metadata(RUN/'input_catalog.json')
 for name in ['train.jsonl','initialization_proof.json','gradient_preflight.json','status.json','checkpoint_step7750.json','checkpoint_step7751.json']:
  if (RUN/name).is_file():metadata(RUN/name)
 protocol=read(RUN/'config.json')['protocol']
 for path,digest in protocol['source_sha256'].items():need(sha(Path(path))==digest,'Actual frozen training source');metadata(path)
 metadata(R/'code/push7/hypir_direct_nr/config_hypir_lastres_lr5e6_nr100_v4.json')
 metadata('/cache/audit_lastres_lr5e6_log_rng_cpu.py','proofs/audit_lastres_lr5e6_log_rng_cpu.py');metadata('/cache/lastres_lr5e6_log_rng_cpu_audit.json','proofs/lastres_lr5e6_log_rng_cpu_audit.json')
 logproof=read(Path('/cache/lastres_lr5e6_log_rng_cpu_audit.json'));need(logproof['status']=='PASS_LOG100_RNG_AND_FROZEN_SOURCE' and logproof['source_sha256']=='5f99fcebc6cb50ed5c3ae18ae8df0cdf10cb2ab3b62e124c5097b0c81cf5cd9b' and logproof['updates']==100 and sha(Path('/cache/audit_lastres_lr5e6_log_rng_cpu.py'))==logproof['auditor_sha256'],'Actual minimal source/log/RNG audit')
 for name in ['recipe.json','raw_zip_gaps.bin','probe_result.json']:metadata(recipe_dir/name,'prior_A7750/'+name)
 metadata(OLD/'code/push4/checkpoint_archive_recipe.py','prior_A7750/checkpoint_archive_recipe.py')
 metadata(R/'code/push7/hypir_direct_nr/train_hypir_lastres_nr_v3.py','source_reference/train_hypir_lastres_nr_v3.py')
 metadata('/cache/lastres_lr5e6_source.diff','proofs/source_diff_lastresv3_to_lr5e6.diff')
 for step in [7800,7850]:metadata(R/('submissions/hypir_push7_a7750_lastres_lr5e6_nr_s%d/candidate_manifest.json'%step))
 metadata(Path(__file__),'code/lastres_lr5e6_zip_capsule_cache_only.py')
 metadata(R/'code/push7/audit_lastres_lr5e6_recovered_matched.py')
 metadata(R/'audits/push7/hypir_pilot/lastres_lr5e6_50_100_recovered_matched.json')
 quality=read(R/'audits/push7/hypir_pilot/lastres_lr5e6_50_100_recovered_matched.json');need(quality['status']=='PASS_LASTRES_LR5E6_FRESH_RECOVERED_MATCHED23_98' and quality['auditor_source_sha256']==sha(R/'code/push7/audit_lastres_lr5e6_recovered_matched.py'),'Actual matched pilot audit identity')
 capsule=dict(schema='lastres_lr5e6_exact_zip_capsule_v1',base=dict(path=str(BASE),bytes=BASE_BYTES,sha256=BASE_SHA,reference_scope='Original immutable A7750 resume stored in prior Push4; NOT copied'),files=files,blobs=blobs,metadata_source_records=source_records,no_GPU=True,no_raw_or_resume_archives_stored=True)
 write(C/'capsule.json',capsule);total=sum(p.stat().st_size for p in C.rglob('*') if p.is_file());need(total<LIMIT,'Actual capsule exceeds160MiB')
 print(json.dumps(dict(status='BUILT_NOT_YET_RESTORATION_VERIFIED',files=len(files),blobs=len(blobs),payload_bytes=total,capsule_sha256=sha(C/'capsule.json'))),flush=True)
def verify():
 need(C.is_dir() and not V.exists(),'New cache verification directory required');fixed_base();c=read(C/'capsule.json');V.mkdir();records=[]
 with BASE.open('rb') as base:
  for row in c['files']:
   need(Path(row['name']).name==row['name'],'Safe restore leaf');out=V/row['name'];whole=hashlib.sha256();written=0
   with out.open('xb') as f:
    for s in row['segments']:
     if s['kind']=='blob':
      b=c['blobs'][s['sha256']];p=C/b['path'];need(sha(p)==b['compressed_sha256'] and p.stat().st_size==b['compressed_bytes'],'Actual compressed blob')
      raw=zlib.decompress(p.read_bytes());need(len(raw)==s['bytes'] and hashlib.sha256(raw).hexdigest()==s['sha256'],'Actual blob decoded SHA');f.write(raw);whole.update(raw);written+=len(raw)
     else:
      need(s['kind']=='base' and s['offset']>=0 and s['offset']+s['bytes']<=BASE_BYTES,'Bounded base reference');base.seek(s['offset']);left=s['bytes'];h=hashlib.sha256()
      while left:
       raw=base.read(min(left,8*1024**2));need(raw,'Complete base segment');h.update(raw);whole.update(raw);f.write(raw);written+=len(raw);left-=len(raw)
      need(h.hexdigest()==s['sha256'],'Actual retained base segment SHA')
    f.flush();os.fsync(f.fileno())
   need(written==out.stat().st_size==row['bytes'] and whole.hexdigest()==row['sha256'] and sha(out)==row['sha256'],'Actual restored complete file full reread SHA')
   records.append(dict(name=row['name'],verification_path=str(out),bytes=written,sha256=row['sha256'],actual_disk_full_SHA_reread=True,verification_stamp=stamp(out)))
 proof=dict(status='PASS_RESTORED_ALL4_EXACT_BYTES',capsule_sha256=sha(C/'capsule.json'),base_sha256=BASE_SHA,restored=records,CPU_only=True,GPU_started=False,original_files_modified=False)
 write(C/'verification_proof.json',proof);print(json.dumps(dict(status=proof['status'],files=4,restored_bytes=sum(r['bytes'] for r in records),proof_sha256=sha(C/'verification_proof.json'))),flush=True)
def old_heads():
 result={}
 for d in P.iterdir():
  if not d.is_dir() or d==DEST:continue
  for rel in ['manifest.json','DONE','DONE.json','STABLE_PAYLOAD_RECEIPT.json','final_commit/manifest.json','final_commit/DONE']:
   p=d/rel
   if p.is_file():result[str(p)]=sha(p)
 if (P/'DONE').is_file():result[str(P/'DONE')]=sha(P/'DONE')
 return result
def package():
 need(not (C/'manifest.json').exists() and not (C/'DONE').exists(),'New immutable cache package only')
 c=read(C/'capsule.json');proof=read(C/'verification_proof.json')
 need(proof['status']=='PASS_RESTORED_ALL4_EXACT_BYTES' and proof['capsule_sha256']==sha(C/'capsule.json') and len(proof['restored'])==4,'Actual four-file restore proof')
 for r in proof['restored']:need(stamp(Path(r['verification_path']))==r['verification_stamp'],'Verification file changed')
 for r in c['metadata_source_records']:need(stamp(Path(r['source']))==r['source_stamp'] and sha(Path(r['source']))==r['sha256'],'Original metadata changed')
 for r in c['files']:need(stamp(Path(r['original_path']))==r['source_stamp'],'Original checkpoint changed')
 for digest,b in c['blobs'].items():
  path=C/b['path'];need(path.stat().st_size==b['compressed_bytes'] and sha(path)==b['compressed_sha256'],'Declared compressed CAS changed')
  raw=zlib.decompress(path.read_bytes());need(len(raw)==b['bytes'] and hashlib.sha256(raw).hexdigest()==digest,'Declared decoded CAS changed')
 for r in c['metadata_source_records']:need((C/r['capsule_path']).stat().st_size==r['bytes'] and sha(C/r['capsule_path'])==r['sha256'],'Declared packaged metadata changed')
 need(not any(p.is_symlink() for p in C.rglob('*')),'No capsule symlinks')
 rows=[dict(path=str(p.relative_to(C)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(C.rglob('*')) if p.is_file()];total=sum(r['bytes'] for r in rows)
 need(total<LIMIT,'Cache payload must remain below160MiB');heads=old_heads()
 manifest=dict(status='complete',schema='lastres_lr5e6_exact_zip_CACHE_ONLY_capsule_v1',files=rows,count=len(rows),bytes=total,base=c['base'],restored_files=[{k:v for k,v in r.items() if k in ['name','bytes','sha256','actual_disk_full_SHA_reread']} for r in proof['restored']],verification_proof_sha256=sha(C/'verification_proof.json'),CAS_and_metadata_declared_SHA_checked=True,old_SFS_heads=heads,SFS_written=False,no_raw_or_resume_archive_copied=True,GPU_started=False)
 write(C/'manifest.json',manifest);done=dict(status='complete',manifest_sha256=sha(C/'manifest.json'),files=len(rows),bytes=total,capsule_sha256=sha(C/'capsule.json'),exact_restored_files=4,SFS_written=False);write(C/'DONE',done)
 need(read(C/'DONE')==done and old_heads()==heads,'Local commit and old SFS heads')
 receipt=dict(status='verified_complete_CACHE_ONLY',directory=str(C),manifest_sha256=sha(C/'manifest.json'),DONE_sha256=sha(C/'DONE'),files=len(rows),bytes=total,all4_restored_original_full_SHA_PASS=True,CAS_metadata_declared_SHA_PASS=True,base_A_resume_sha256=BASE_SHA,base_A_resume_not_copied=True,SFS_written=False,old_SFS_heads_unchanged=True,GPU_started=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(Path('/cache/lastres_lr5e6_capsule_receipt.json'),receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=['build','verify','package']);a=q.parse_args();{'build':build,'verify':verify,'package':package}[a.command]()
