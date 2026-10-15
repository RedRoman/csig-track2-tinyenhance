"""Own CPU-only incremental artifact synchronization, with no GPU calls."""
from pathlib import Path
import os,json,time,hashlib,datetime,fcntl,subprocess,traceback
R=Path("/cache/independent_sr_cuda_20261013")
A=Path("/temp/independent_sr_cuda_20261013_artifacts")
A.mkdir(parents=True,exist_ok=True);os.nice(10)
os.environ["OMP_NUM_THREADS"]="1"
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8388608),b""):h.update(b)
 return h.hexdigest()
def free():
 v=os.statvfs(A);return v.f_bavail*v.f_frsize
def guard(png_bytes=0,extra=0):
 assert free()>=40*1024**3,"SFS free below 40 GiB; synchronization paused"
 assert png_bytes+extra<10_000_000_000,"Artifact PNG total would reach 10 GB; synchronization paused"
def direct_json(path,value):
 assert not path.is_symlink()
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def copy(source,relative=None):
 destination=A/(relative or source.relative_to(R));destination.parent.mkdir(parents=True,exist_ok=True)
 assert not destination.is_symlink()
 with source.open("rb") as stream:body=stream.read()
 digest=hashlib.sha256(body).hexdigest()
 if destination.exists() and sha(destination)==digest:return False
 with destination.open("wb") as stream:stream.write(body)
 assert sha(destination)==digest,destination
 return True
with (R/"preflight/cuda_artifact_sync.lock").open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 while True:
  copied=0;counts={};complete=[];references={}
  try:
   guard()
   png_bytes=sum(p.stat().st_size for p in (A/"evaluations").glob("*/render/*/*.png")) if (A/"evaluations").exists() else 0
   for E in sorted((R/"evaluations").iterdir()):
    if not E.is_dir():continue
    target=A/"evaluations"/E.name;target.mkdir(parents=True,exist_ok=True)
    protocol_path=E/"render_protocol.json"
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    if protocol:
     runner=protocol["protocol"]["runner"]
     references[E.name]={"checkpoint_path":runner.get("weight_path"),"checkpoint_sha256":runner.get("checkpoint_sha256"),"base_model_path":runner.get("base_model_path"),"runner_sha256":runner.get("runner_sha256"),"source_sha256":runner.get("source_sha256"),"protocol_id":protocol["protocol_id"],"external_base_hash_evidence":"independent preflight manifests copied below; base assets are already persistent"}
    progress=E/"render_progress.pt";proof_path=target/"render_commit_sync.json"
    previous=json.loads(proof_path.read_text()) if proof_path.exists() else {"records":[]}
    if progress.exists():
     state=torch.load(progress,map_location="cpu",weights_only=False)
     assert protocol and state["protocol_id"]==protocol["protocol_id"],E.name
     assert state["checkpoint_sha256"]==protocol["protocol"]["runner"]["checkpoint_sha256"],E.name
     assert state["input_sha256"]==[x["input_sha256"] for x in protocol["protocol"]["tasks"]],E.name
     records=state["records"]
     assert records[:len(previous["records"])]==previous["records"],"Commit prefix changed: "+E.name
     for row in records[len(previous["records"]):]:
      relative=Path(row["output"]);assert relative.parts[0]=="render" and ".." not in relative.parts
      source=E/relative;destination=target/relative
      guard(png_bytes,source.stat().st_size)
      assert sha(source)==row["sha256"],source
      already=destination.exists();copied+=copy(source)
      assert sha(destination)==row["sha256"],destination
      if not already:png_bytes+=destination.stat().st_size
     direct_json(proof_path,{"time_bjt":now(),"protocol_id":state["protocol_id"],"checkpoint_sha256":state["checkpoint_sha256"],"input_sha256":state["input_sha256"],"verified_png_count":len(records),"records":records})
     counts[E.name]=len(records)
    for source in E.rglob("*"):
     if not source.is_file() or source.is_symlink() or "render" in source.relative_to(E).parts:continue
     if source.suffix in (".json",".jsonl",".csv",".log",".txt") or source.name in ("render_progress.pt","resume_state.pt","scoring_progress.pt","render_complete"):
      copied+=copy(source)
    result=E/"scores/result.json";manifest=E/"render_manifest.json"
    if result.exists() and manifest.exists():
     score=json.loads(result.read_text());m=json.loads(manifest.read_text())
     if score.get("status")=="complete":
      rows=[json.loads(l) for l in (E/"scores/measurements.jsonl").read_text().splitlines() if l.strip()]
      assert len(rows)==103 and len(m["records"])==103 and counts[E.name]==103
      assert len({(x["split"],x["case"]) for x in rows})==103
      for row in m["records"]:assert sha(target/row["output"])==row["sha256"],row["output"]
      for name in ("scores/result.json","scores/measurements.jsonl","scores/protocol.json","render_manifest.json","render_protocol.json"):
       assert sha(E/name)==sha(target/name),name
      direct_json(target/"artifact_complete.json",{"completed_bjt":now(),"label":E.name,"verified_png_count":103,"test_n":score["test"]["n"],"val_n":score.get("val",score.get("validation"))["n"],"S_proxy":score.get("S_proxy"),"pred":score.get("pred"),"not_official_score":score.get("not_official_score"),"result_sha256":sha(result),"render_manifest_sha256":sha(manifest),"render_protocol_sha256":sha(protocol_path)})
      complete.append(E.name)
   for root in (R/"code",R/"preflight",R/"calibration",R/"logs",R/"results",R/"submissions",R/"scoring_tool",R/"repos"):
    if not root.exists():continue
    for source in root.rglob("*"):
     if source.is_file() and not source.is_symlink() and "__pycache__" not in source.parts and source.suffix in (".py",".sh",".json",".jsonl",".csv",".md",".txt",".log"):
      copied+=copy(source)
   for source in R.iterdir():
    if source.is_file() and source.suffix in (".json",".md",".txt"):copied+=copy(source)
   direct_json(A/"asset_references.json",{"time_bjt":now(),"references":references,"no_environment_or_checkpoint_payload_copied":True,"npu_persistent_clone":"/temp/independent_sr_score_search_20261013_transfer/payload"})
   state={"time_bjt":now(),"source_root":str(R),"persistent_root":str(A),"copied_files_this_cycle":copied,"committed_png_counts":counts,"png_bytes":png_bytes,"sfs_free_bytes":free(),"complete_candidates":complete,"errors":[]}
   direct_json(A/"sync_status.json",state);direct_json(R/"preflight/cuda_artifact_sync_status.json",state)
   if (A/"sync_error.json").exists():(A/"sync_error.json").unlink()
   print(json.dumps(state),flush=True)
   if (R/"preflight/cuda_artifact_sync.finish").exists():
    direct_json(A/"DONE",dict(state,stage="done",sync_pid=os.getpid()))
    direct_json(R/"preflight/cuda_artifact_sync_done.json",dict(state,stage="done",sync_pid=os.getpid()))
    break
  except Exception as error:
   state={"time_bjt":now(),"error":repr(error),"sfs_free_bytes":free(),"context_file":str(locals().get("source","unknown"))}
   direct_json(A/"sync_error.json",state);print(json.dumps(state),traceback.format_exc(),flush=True)
  time.sleep(50)
