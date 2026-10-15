"""Restore selected existing HYPIR runtime only; cache writes, SFS readonly, no GPU."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,os,tarfile
R=Path('/cache/independent_sr_cuda_20261013');P=Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push7_trtvae_verified_20261015')
STAGE=Path('/cache/push7_trtunetvae_restore_metadata');TAR=Path('/cache/push7_trtunetvae_small_metadata.tar')
TAR_SHA='be2a6684a1a951ea59b6a64790b71e2e1bd4b5540f5182ee99fbb07e0dda70f8'
U_SHA='08fca16f286c604614e9322ded93f848b0c1d3dd40de7ca793f10b98f286aec8';U_BYTES=1656387116
def need(v,m):
 if not v:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,j):
 with p.open('w') as f:f.write(json.dumps(j,indent=2)+'\n');f.flush();os.fsync(f.fileno())
rows=[]
def exact_copy(src,dst,digest,size):
 dst.parent.mkdir(parents=True,exist_ok=True)
 need(str(dst.parent.resolve()).startswith(str(R)+'/'),'Only original cache root target allowed')
 if dst.exists():need(dst.is_file() and sha(dst)==digest and dst.stat().st_size==size,'Existing target differs; never overwrite '+str(dst))
 else:
  need(not dst.is_symlink(),'Dangling target symlink');tmp=dst.with_name(dst.name+'.restoring');need(not tmp.exists(),'Unexpected partial metadata/VAE target')
  h=hashlib.sha256()
  with src.open('rb') as fi,tmp.open('xb') as fo:
   for b in iter(lambda:fi.read(8*1024**2),b''):h.update(b);fo.write(b)
   fo.flush();os.fsync(fo.fileno())
  need(h.hexdigest()==digest and tmp.stat().st_size==size and sha(tmp)==digest,'Actual copied target SHA/size differs')
  os.rename(tmp,dst)
 rows.append(dict(target=str(dst),bytes=size,sha256=digest,actual_target_full_SHA_verified=True))
q=argparse.ArgumentParser();q.add_argument('--finalize-unet',action='store_true');a=q.parse_args()
need(sha(TAR)==TAR_SHA,'Actual local metadata upload changed');STAGE.mkdir(exist_ok=True)
with tarfile.open(TAR) as tf:
 members=tf.getmembers();names=set()
 for m in members:
  rel=PurePosixPath(m.name);need(m.isfile() and not rel.is_absolute() and '..' not in rel.parts and m.name not in names,'Unsafe metadata tar member');names.add(m.name)
 for m in members:
  dst=STAGE/m.name;dst.parent.mkdir(parents=True,exist_ok=True)
  need(str(dst.resolve()).startswith(str(STAGE)+'/') and not dst.is_symlink(),'Unsafe stage destination')
  if not dst.exists():
   with tf.extractfile(m) as fi,dst.open('xb') as fo:
    for b in iter(lambda:fi.read(1024**2),b''):fo.write(b)
inv=json.loads((STAGE/'SOURCE_INVENTORY.json').read_text());need(inv['count']==37,'Metadata inventory differs')
for r in inv['files']:
 src=STAGE/r['path'];need(src.stat().st_size==r['bytes'] and sha(src)==r['sha256'],'Actual source metadata changed')
 if r['path'].startswith(('submissions/','backends/','code/')):exact_copy(src,R/r['path'],r['sha256'],r['bytes'])
need(sha(P/'manifest.json')=='4736be77e7e0d957455c445f31f223ce07e3454f1118f985d6de8e47fec15a29' and sha(P/'DONE')=='ee6a0fb6460890a9734f6cd2f402fc75deeffc36be4423af855a009a595b6b5a','Frozen VAE archive heads')
for name,size,digest in [('encoder_fp16',71304884,'14c206896a7caa6b6491dae33133d149283025cfb8e15b6cf129d8854176ba55'),('decoder_fp16',102396372,'fe81087086d76aa8d0111cb80070b65b897e0438ebb2a8a020ef0ced8bc6bf30')]:
 rel=Path('backends/push7_trt')/name/'component.engine';exact_copy(P/rel,R/rel,digest,size)
if a.finalize_unet:
 dst=R/'backends/push7_trt/unet_fp16_A7750/component.engine';partial=dst.with_name(dst.name+'.restoring')
 if not dst.exists():
  need(partial.is_file() and partial.stat().st_size==U_BYTES and sha(partial)==U_SHA,'UNet upload incomplete or actual SHA differs');os.rename(partial,dst)
 need(dst.stat().st_size==U_BYTES and sha(dst)==U_SHA,'Actual complete UNet target SHA')
 rows.append(dict(target=str(dst),bytes=U_BYTES,sha256=U_SHA,actual_target_full_SHA_verified=True))
result=dict(status='PASS_BEST63_RUNTIME_FILES_AND_3ENGINES_RESTORED' if a.finalize_unet else 'PASS_SMALL_RUNTIME_AND_2VAE_ENGINES_UNET_UPLOAD_PENDING',targets=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows),engine_count=3 if a.finalize_unet else 2,original_absolute_config_and_checkpoint_path_preserved=True,selection_or_historical_scores_changed=False,SFS_written=False,GPU_started=False,TRT_site_not_installed_by_this_helper=True,original_quality_and_timing_metadata_restored=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=Path('/cache/push7_best63_runtime_restore_receipt.json' if a.finalize_unet else '/cache/push7_best63_runtime_restore_stage_receipt.json');write(out,result)
print(json.dumps(dict(status=result['status'],count=result['count'],bytes=result['bytes'],receipt=str(out),receipt_sha256=sha(out))),flush=True)
