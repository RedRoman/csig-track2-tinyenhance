"""Authorized CPU restoration of parent A7750 and original6000 only; SFS readonly."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,datetime
P=Path('/temp/independent_sr_cuda_20261013_artifacts');R=Path('/cache/independent_sr_cuda_20261013')
P4=P/'nr_push4_20261014';REL=Path('adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500');OUT=R/REL
def need(v,m):
 if not v:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def stamp(p):
 s=p.stat();return s.st_size,s.st_mtime_ns,s.st_ino
records=[]
def verified_copy(src,dst,expected):
 before=stamp(src);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():
  need(sha(dst)==expected,'Existing target differs; never overwrite '+str(dst))
 else:
  tmp=dst.with_name(dst.name+'.restoring');need(not tmp.exists(),'Unexpected partial target')
  h=hashlib.sha256()
  with src.open('rb') as fi,tmp.open('xb') as fo:
   for b in iter(lambda:fi.read(8*1024**2),b''):h.update(b);fo.write(b)
   fo.flush();os.fsync(fo.fileno())
  need(stamp(src)==before and h.hexdigest()==expected and tmp.stat().st_size==before[0] and sha(tmp)==expected,'Copy source/target SHA mismatch')
  os.rename(tmp,dst)
 records.append(dict(source=str(src),target=str(dst),bytes=dst.stat().st_size,sha256=expected,actual_full_target_SHA_verified=True))
 print(json.dumps(records[-1]),flush=True)
need(sha(P4/'manifest.json')=='3e1c41dc447fcb242aac85d7e6834b7bcb5f6d3f5a23ff63f255e682506cd35a','Push4 manifest')
need(sha(P4/'DONE')=='ac09208eef5b26f75f5bdf55ccfc5a02b694673ed3540df67607a5a6488e722f','Push4 DONE')
helper=P4/'code/push4/checkpoint_archive_recipe.py';recipe=P4/'audits/push4/raw_recovery/push4_a7750_compiled_all'
for p,d in [(helper,'43b135a68047a9319e355d9967b4c1bcddcab16ff58d775a8c8f7ef261d932ce'),(recipe/'recipe.json','6ae28dcb95be58d91a518f3706944ba39210ec1c0af17fb7dbf8474023547b67'),(recipe/'raw_zip_gaps.bin','f3a4be7d6b5ca3a232829b2d3423186862d0e071aedb82688b79bd8e800f1c0d')]:need(sha(p)==d,'Actual restore helper/recipe identity')
verified_copy(P4/REL/'resume_step7750.pt',OUT/'resume_step7750.pt','e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b')
verified_copy(P4/REL/'config.json',OUT/'config.json','86015a361f5fa1429f1d797f175d27450cb6f798e1df4483badf01d876b2ab5b')
raw=OUT/'step7750.pth';rawsha='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
if not raw.exists():subprocess.run([sys.executable,str(helper),'restore','--directory',str(recipe),'--resume',str(OUT/'resume_step7750.pt'),'--output',str(raw)],check=True)
need(raw.stat().st_size==1038079419 and sha(raw)==rawsha,'Actual restored A raw SHA')
records.append(dict(target=str(raw),bytes=raw.stat().st_size,sha256=rawsha,actual_full_target_SHA_verified=True,exact_raw_byte_recovery=True))
rel6000=Path('source_payload/adaptation/runs/ascend_balanced/step6000.pth')
verified_copy(P/'nr_microtune_20261014'/rel6000,R/rel6000,'2c82a71a94aa754bd4a26a2a474c2bbe4989bde1d38d2e15d9e58eb0c3e408bc')
result=dict(status='PASS_RESTORED_EXACT_A7750_PARENT_AND_ORIGINAL6000',records=records,SFS_written=False,GPU_started=False,scope='Only A7750 parent resume/config/raw and original6000 raw targets',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=Path('/cache/push7_A7750_restore_receipt.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
