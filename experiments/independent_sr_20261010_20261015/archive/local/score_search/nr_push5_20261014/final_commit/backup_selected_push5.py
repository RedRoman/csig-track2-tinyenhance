"""Small CPU-only selected-Pix archive; dry-run unless --execute. Log outside R."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import shutil

R=Path('/cache/independent_sr_cuda_20261013')
D=Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push5_20261014')
G4_SHA='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
PRIOR=[('nr_microtune_20261014','DONE.json','99f461e6e95045cf0af053dc3b24bde74cd3531715225d0ccf95eef2d2dbd0d4','92cadf2e81aba1864e027e1fded512c008b15585018e1c1eda7d9658b45b8d12'),
 ('nr_push2_20261014','DONE','ef284432a338b0a4574f52be645f4a40571ef94e775e1e39a2c0d1cfc1d13e8d','0a9cb512123ca402baaf43e21db2204853b2033fbcb61a1739f802306b36703f'),
 ('nr_push3_20261014','DONE','6c4cc6db3d963762cb1868d83d89969083326cc0e3932d64743a23b598c48eaa','15846dd6f53e3327bf7331d00a27d4928b2078edca4244ee29186623cf85655d'),
 ('nr_push4_20261014','DONE','3e1c41dc447fcb242aac85d7e6834b7bcb5f6d3f5a23ff63f255e682506cd35a','ac09208eef5b26f75f5bdf55ccfc5a02b694673ed3540df67607a5a6488e722f')]
SMALL={'.py','.json','.jsonl','.md','.txt','.yaml','.yml','.csv','.sh','.diff'}
CHUNK=8*1024**2


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def token(p):
    s=Path(p).stat();return s.st_size,s.st_mtime_ns,s.st_ino


def sha(p):
    before=token(p);h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    need(token(p)==before,'Source changed while hashing: '+str(p));return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def heads():
    out=[]
    for name,marker,mh,dh in PRIOR:
        q=D.parent/name;m=q/'manifest.json';d=q/marker
        need(sha(m)==mh and sha(d)==dh and read(d)['status']=='complete' and read(d)['manifest_sha256']==mh,'Old backup head changed: '+name)
        out.append(dict(prefix=name,manifest_sha256=mh,DONE_sha256=dh))
    return out


def tree(base,files):
    if base.exists():
        for p in base.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in SMALL and p.stat().st_size<=8*1024**2 and not any(x in {'__pycache__','cache','_work'} for x in p.relative_to(base).parts):files.add(p)


def direct(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:f.write(data);f.flush()
    need(path.read_bytes()==data,'Metadata write/readback differs')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--selection-json',type=Path,default=R/'results/nr_push5_selection.json')
    ap.add_argument('--reserve-mib',type=int,default=128,help='Self-chosen reserve, not a user requirement')
    ap.add_argument('--plan-out',type=Path,default=Path('/cache/push5_selected_backup_plan.json'));a=ap.parse_args()
    import fcntl
    lock=Path('/cache/push5_backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    need(0<=a.reserve_mib<=1024 and a.plan_out.resolve().is_relative_to(Path('/cache')) and not a.plan_out.resolve().is_relative_to(R),'Keep plan/log in /cache outside experiment R')
    need(not D.is_symlink(),'New prefix must not be a symlink')
    prior=heads();s=read(a.selection_json);selection_sha=sha(a.selection_json)
    need(s['status']=='selected_new_student_full103_paired_prediction_ge70' and s['reaches_target'] is True and s['predicted_tool_score']>=70,'Actual selected new student must reach70')
    need(s['local_step']==s['optimizer_step']==250,'Only actual student250 selected')
    model=Path(s['model_dir']);raw=Path(s['checkpoint']);resume=Path(s['resume']);run=raw.parent
    need(run==R/'adaptation/runs/pix_push5_g4_nr250' and raw==run/'step250.pth' and resume==run/'resume_step250.pt','Wrong actual immutable student milestone')
    need(sha(raw)==s['checkpoint_sha256'] and sha(resume)==s['resume_sha256'] and sha(model/'model.pth')==s['checkpoint_sha256'],'Raw/model/resume actual SHA differs')
    need(sha(s['config'])==s['config_sha256'],'Actual training config differs')
    marker=read(run/'checkpoint_step250.json');binding=read(model/'student_binding.json')
    need(marker['raw_sha256']==binding['raw_sha256']==s['checkpoint_sha256'] and marker['resume_sha256']==binding['resume_sha256']==s['resume_sha256'],'Marker/binding mismatch')
    need(binding['local_step']==binding['optimizer_step']==250 and binding['all100_actual_FP32_parameters_changed_and_raw_equal_resume'] is True,'New real training proof absent')
    for key in ['training_audit','full_audit','paired_prediction']:
        p=Path(s[key]);need(p.exists() and sha(p)==s[key+'_sha256'] and read(p)['status']=='PASS','Independent actual audit/join missing: '+key)
    pair=read(s['paired_prediction']);pred=pair['newly_trained_student']
    need(all(abs(pred[k]-s[k])<=1e-10 for k in ['NR','FR','S_proxy']) and pair['qualities']['student']['checkpoint_sha256']==s['checkpoint_sha256'] and pair['qualities']['student']['model_dir']==s['model_dir'] and pair['qualities']['student']['evaluation_dir']==s['quality_dir'] and pair['training_proof']['resume_sha256']==s['resume_sha256'],'Paired model/quality/resume identity differs')
    need(pair['newly_trained_student_reaches70'] is True and abs(pred['original_tool_prediction']-s['predicted_tool_score'])<=1e-10 and pred['original_tool_prediction']>=70,'Paired actual prediction is below70/different')
    q=Path(s['quality_dir']);result=read(q/'scores/result.json');manifest=read(q/'render_manifest.json')
    rows=[json.loads(x) for x in (q/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
    order=[('validation',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
    need(result['status']=='complete' and result['completed']==result['expected']==len(rows)==103 and result['official_full_count'] is True and [(x['split'],x['case']) for x in rows]==order,'Native full103 required')
    need(sum(len(x['metrics']) for x in rows)==418 and len(manifest['records'])==103,'Full418 metrics/manifest missing')
    means={split:{k:sum(x['metrics'][k] for x in rows if x['split']==split)/n for k in keys} for split,n,keys in [('test',100,['clipiqa','maniqa','musiq','niqe']),('validation',3,['lpips','dists'])]}
    t,v=means['test'],means['validation'];nr=t['clipiqa']+t['maniqa']+t['musiq']/100+max(0,(10-t['niqe'])/10);fr=2-v['lpips']-v['dists']
    need(all(math.isfinite(x) and abs(x-s[k])<=1e-10 for k,x in [('NR',nr),('FR',fr),('S_proxy',nr+fr)]),'Full quality differs from selection')
    files={a.selection_json,R/'results/status_cuda_nr_push5_20261014.json',Path(s['config']),resume,R/'calibration/g4/model_dir/model.pth',Path(__file__).resolve(),*(Path(s[k]) for k in ['training_audit','full_audit','paired_prediction'])};png_guards={}
    for rec,row in zip(manifest['records'],rows):
        need(('validation' if rec['split']=='val' else rec['split'],rec['case'])==(row['split'],row['case']) and rec['sha256']==row['sr_sha256'],'Render/scored PNG identity differs')
        p=q/rec['output'];need(sha(p)==rec['sha256'],'Actual full PNG SHA differs');png_guards[str(p)]=token(p)
        if rec['split']=='test' and rec['case'] in [7,16,41,83]:files.add(p)
    for base in [model,run,R/'nr_push5_20261014/preparation',R/'code/push5',R/'analysis/push5',R/'audits/push5',q,R/'nr_push5_full_evaluations/push5_g4_reference_full',R/'nr_push4_full_evaluations/push4_a7750_compiled_all']:
        tree(base,files)
    files.add(model/'model.pth');calibration=[R/f'calibration/push5_final_{name}.json' for name in ['g4_latency','hypir_latency','student_latency','latency_launch','paired_plan','paired_prediction']];need(all(p.is_file() for p in calibration),'All six final paired JSONs required');files.update(calibration)
    reports=list((R/'reports').glob('*PUSH5*'));need(reports,'Final Chinese Push5 report required');files.update(p for p in reports if p.is_file())
    need(sha(R/'calibration/g4/model_dir/model.pth')==G4_SHA,'Supplied reference init changed')
    records=[];guards={str(raw):token(raw),**png_guards}
    for p in sorted(files):
        p=p.resolve();need(p.is_relative_to(R) and p.is_file(),'Scoped payload must stay under R')
        guards[str(p)]=token(p);records.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
    expected={r['path'] for r in records};need(len(expected)==len(records) and str(raw.relative_to(R)) not in expected,'Duplicate records/raw copy forbidden')
    tensor={r['path'] for r in records if Path(r['path']).suffix in {'.pt','.pth','.ckpt','.safetensors'}}
    need(tensor=={str(resume.relative_to(R)),str((model/'model.pth').relative_to(R)),'calibration/g4/model_dir/model.pth'} and sum(r['path'].endswith('.png') for r in records)==4,'Only selected model/resume/reference+4PNG allowed')
    if D.exists():need(all(not p.is_symlink() and (not p.is_file() or str(p.relative_to(D)) in expected|{'manifest.json','DONE'}) for p in D.rglob('*')),'Unexpected partial/symlink in new prefix')
    actions=[r for r in records if not (D/r['path']).is_file() or (D/r['path']).stat().st_size!=r['bytes'] or sha(D/r['path'])!=r['sha256']]
    reserve=a.reserve_mib*1024**2;growth=sum(max(0,r['bytes']-((D/r['path']).stat().st_size if (D/r['path']).exists() else 0)) for r in actions);free=shutil.disk_usage(D if D.exists() else D.parent).free
    plan=dict(schema='push5_selected_portable_backup_v1',destination=str(D),selection_sha256=selection_sha,files=records,count=len(records),bytes=sum(r['bytes'] for r in records),prior_backup_heads=prior,reserve_bytes=reserve,reserve_is_user_requirement=False,free_bytes_before=free,net_growth_bytes=growth,space_sufficient=free>=growth+reserve+1024**2,full103_PNGs_persisted=False,selected_preview_cases=[7,16,41,83],raw_recoverable_exactly_from_deployment_modelpth=True,duplicate_raw_persisted=False,backup_script_sha256=sha(__file__))
    a.plan_out.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'dry_run':not a.execute,'plan':plan},ensure_ascii=False),flush=True)
    if not a.execute:return
    need(plan['space_sufficient'],'Actual space below payload+self-chosen reserve')
    if (D/'DONE').exists():need(not actions and read(D/'DONE')['manifest_sha256']==sha(D/'manifest.json') and read(D/'manifest.json')['files']==records,'Committed prefix immutable');return
    remaining=growth
    for r in actions:
        need(shutil.disk_usage(D if D.exists() else D.parent).free>=remaining+reserve+1024**2,'Free space changed before write')
        target=D/r['path'];prior_size=target.stat().st_size if target.exists() else 0;target.parent.mkdir(parents=True,exist_ok=True)
        with (R/r['path']).open('rb') as src,target.open('wb') as dst:
            for b in iter(lambda:src.read(CHUNK),b''):dst.write(b)
            dst.flush()
        need(target.stat().st_size==r['bytes'] and sha(target)==r['sha256'],'SFS full-byte readback differs');remaining-=max(0,r['bytes']-prior_size)
        print(json.dumps({'copied_full_SHA_verified':r['path'],'bytes':r['bytes']}),flush=True)
    need(all(token(p)==sig for p,sig in guards.items()) and heads()==prior,'Source/old heads changed before commit')
    manifest={**plan,'status':'complete','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_destination_bytes_full_SHA_readback':True}
    direct(D/'manifest.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
    done=dict(status='complete',destination=str(D),manifest_sha256=sha(D/'manifest.json'),selection_sha256=selection_sha,files=len(records),bytes=plan['bytes'],full103_PNGs_persisted=False,reserve_bytes=reserve)
    direct(D/'DONE',(json.dumps(done,indent=2)+'\n').encode());print(json.dumps({'commit_complete':done,'free_bytes_after':shutil.disk_usage(D).free}),flush=True)


if __name__=='__main__':main()
