"""Independent post-commit read-only SFS verification; writes one cache receipt."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import shutil
import struct
import subprocess

R=Path('/cache/independent_sr_cuda_20261013')
D=Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push5_20261014')
MANIFEST='dd1c4201736386223cc162a0d57fc8d9c87cb6b07c1d1146ac11165ba8c0eb87'
SELECTION='aedfa48c0bc557efc97545a0d3259372598e6c3ec1e39a5c4db5371994e9c99a'
REPORT='e6b73dd38d2f2090b41e1e86de0bffab96c16165817f44eb308f6df7eca9f426'
RAW='8c95e115e12219bca8bea2b2d04e0a01552f072de56619fab878ae61a9643237'
RESUME='38e2d27d200fb85a654da137935b1943f223c484570b84643f03b237f098d9fb'
G4='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
PRIOR=[('nr_microtune_20261014','DONE.json','99f461e6e95045cf0af053dc3b24bde74cd3531715225d0ccf95eef2d2dbd0d4','92cadf2e81aba1864e027e1fded512c008b15585018e1c1eda7d9658b45b8d12'),
 ('nr_push2_20261014','DONE','ef284432a338b0a4574f52be645f4a40571ef94e775e1e39a2c0d1cfc1d13e8d','0a9cb512123ca402baaf43e21db2204853b2033fbcb61a1739f802306b36703f'),
 ('nr_push3_20261014','DONE','6c4cc6db3d963762cb1868d83d89969083326cc0e3932d64743a23b598c48eaa','15846dd6f53e3327bf7331d00a27d4928b2078edca4244ee29186623cf85655d'),
 ('nr_push4_20261014','DONE','3e1c41dc447fcb242aac85d7e6834b7bcb5f6d3f5a23ff63f255e682506cd35a','ac09208eef5b26f75f5bdf55ccfc5a02b694673ed3540df67607a5a6488e722f')]


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    p=Path(p);s=p.stat();before=(s.st_size,s.st_mtime_ns,s.st_ino);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    s=p.stat();need((s.st_size,s.st_mtime_ns,s.st_ino)==before,'Changed during hash: '+str(p));return h.hexdigest()


def heads():
    out=[]
    for name,marker,mh,dh in PRIOR:
        p=D.parent/name
        need(sha(p/'manifest.json')==mh and sha(p/marker)==dh and read(p/marker)['status']=='complete' and read(p/marker)['manifest_sha256']==mh,'Old backup head changed')
        out.append(dict(prefix=name,manifest_sha256=mh,DONE_sha256=dh))
    return out


def gpu_empty():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    rows=[x for x in p.stdout.splitlines() if x.strip()];need(not rows,'Visible GPU compute process remains');return {'NVML_visible_compute_processes':rows,'all_GPUs_have_no_visible_compute_processes':True}


def main():
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt=R/'results/nr_push5_backup_receipt_20261014.json'
    need(not receipt.exists(),'Receipt already exists; do not overwrite');g0=gpu_empty();old=heads()
    need(sha(D/'manifest.json')==MANIFEST,'Unexpected committed manifest');m=read(D/'manifest.json');done=read(D/'DONE');done_sha=sha(D/'DONE')
    need(m['status']==done['status']=='complete' and done['manifest_sha256']==MANIFEST and done['selection_sha256']==m['selection_sha256']==SELECTION,'DONE binding differs')
    need(m['count']==done['files']==83 and m['bytes']==done['bytes']==141526567,'Payload count/bytes differ')
    records=m['files'];index={x['path']:x for x in records};need(len(index)==len(records)==83,'Duplicate manifest paths')
    def inventory():
        actual=set()
        need(not D.is_symlink(),'Prefix is symlink')
        for p in D.rglob('*'):
            need(not p.is_symlink(),'SFS symlink forbidden')
            if p.is_file():actual.add(str(p.relative_to(D)))
        need(actual==set(index)|{'manifest.json','DONE'},'Physical file inventory differs')
    inventory();total=0
    for name,rec in index.items():
        p=D/name;need(not Path(name).is_absolute() and '..' not in Path(name).parts and p.resolve().is_relative_to(D),'Payload escaped prefix')
        need(p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],'Actual SFS SHA/size mismatch: '+name);total+=rec['bytes']
    need(total==141526567,'Actual bytes differ');print(json.dumps({'all_actual_SFS_files_SHA_size_PASS':83,'bytes':total}),flush=True)
    s=read(D/'results/nr_push5_selection.json');need(sha(D/'results/nr_push5_selection.json')==sha(R/'results/nr_push5_selection.json')==SELECTION,'Selection source/archive differs')
    rp=Path('reports/NR_PUSH5_REPORT_20261014.md');need(sha(D/rp)==sha(R/rp)==REPORT,'Report source/archive differs')
    def archived(p):
        rel=Path(p).relative_to(R);need(str(rel) in index,'Required evidence absent: '+str(rel));return D/rel
    model=Path(s['model_dir']);resume=Path(s['resume']);q=Path(s['quality_dir']);raw=Path(s['checkpoint'])
    tensor={n for n in index if Path(n).suffix.lower() in {'.pt','.pth','.ckpt','.safetensors'}}
    need(tensor=={str((model/'model.pth').relative_to(R)),str(resume.relative_to(R)),'calibration/g4/model_dir/model.pth'},'Three tensor archives required')
    need(str(raw.relative_to(R)) not in index and sha(archived(model/'model.pth'))==RAW==s['checkpoint_sha256'] and sha(archived(resume))==RESUME==s['resume_sha256'] and sha(D/'calibration/g4/model_dir/model.pth')==G4,'Selected archive identity differs')
    need(s['local_step']==s['optimizer_step']==250 and s['reaches_target'] is True and s['predicted_tool_score']>=70,'Wrong actual selection')
    audits={}
    for k in ['training_audit','full_audit','paired_prediction']:
        p=archived(s[k]);need(sha(p)==s[k+'_sha256'] and read(p)['status']=='PASS','Archived audit differs');audits[k]=read(p)
    tr,full,pair=[audits[k] for k in ['training_audit','full_audit','paired_prediction']]
    need(tr['selected_student_candidate_step250_raw_sha256']==RAW and tr['selected_student_candidate_step250_resume_sha256']==RESUME and [x['local_step'] for x in tr['milestones']]==[0,125,250] and [x['changed_state_tensors_vs_G4'] for x in tr['milestones']]==[0,100,100],'Actual new training audit differs')
    need(full['image_rows']==103 and full['scalar_metrics']==418 and full['actual_checkpoint_sha256']==RAW and full['actual103_PNG_SHA_size_scored_manifest_PASS'] is True,'Full audit binding differs')
    rows=[json.loads(x) for x in archived(q/'scores/measurements.jsonl').read_text().splitlines() if x.strip()];manifest=read(archived(q/'render_manifest.json'))
    order=[('validation',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
    need([(x['split'],x['case']) for x in rows]==order and sum(len(x['metrics']) for x in rows)==418 and len(manifest['records'])==103,'Archived full103/418 order differs')
    need(all(math.isfinite(v) for x in rows for v in x['metrics'].values()),'Nonfinite archived metric')
    t={k:sum(x['metrics'][k] for x in rows if x['split']=='test')/100 for k in ['clipiqa','maniqa','musiq','niqe']};v={k:sum(x['metrics'][k] for x in rows if x['split']=='validation')/3 for k in ['lpips','dists']}
    nr=t['clipiqa']+t['maniqa']+t['musiq']/100+max(0,(10-t['niqe'])/10);fr=2-v['lpips']-v['dists']
    need(all(abs(value-s[k])<=1e-12 and abs(value-full[k])<=1e-12 for k,value in [('NR',nr),('FR',fr),('S_proxy',nr+fr)]),'Actual full quality differs')
    png={n for n in index if n.endswith('.png')};expected={str((q/f'render/test/case{i}.png').relative_to(R)) for i in [7,16,41,83]};need(png==expected,'Only four native previews allowed')
    for rec,row in zip(manifest['records'],rows):
        need(('validation' if rec['split']=='val' else rec['split'],rec['case'])==(row['split'],row['case']) and rec['sha256']==row['sr_sha256'],'Archived manifest/score identity differs')
        rel=str((q/rec['output']).relative_to(R))
        if rel in expected:
            need(index[rel]['sha256']==rec['sha256'],'Native preview SHA differs')
            with (D/rel).open('rb') as f:h=f.read(24)
            need(h[:8]==b'\x89PNG\r\n\x1a\n' and list(struct.unpack('>II',h[16:24]))==rec['size'],'Native preview dimensions differ')
    pred=pair['newly_trained_student'];need(pair['physical_card_proof']['same_physical_UUID_PCI_all_three'] is True and pair['training_proof']['resume_sha256']==RESUME and pair['qualities']['student']['checkpoint_sha256']==RAW and pair['qualities']['student']['evaluation_dir']==str(q) and pair['qualities']['student']['model_dir']==str(model) and pair['newly_trained_student_reaches70'] is True and abs(pred['original_tool_prediction']-s['predicted_tool_score'])<=1e-12,'Actual paired prediction differs')
    rootexit=read('/cache/push5_backup_root_exit.json');log=Path('/cache/push5_backup_execute.log');log_sha=sha(log)
    need(rootexit['exit_code']==0 and rootexit['session_id']==7198 and rootexit['manifest_sha256']==MANIFEST and rootexit['log_sha256']==log_sha and Path(rootexit['log'])==log,'Root observed exit/log differs')
    entries=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')];commit=[x['commit_complete'] for x in entries if 'commit_complete' in x]
    need(len(commit)==1 and commit[0]==done,'Actual execution log commit differs')
    plan=read('/cache/push5_selected_backup_plan.json');need(plan['files']==records and plan['selection_sha256']==SELECTION and plan['bytes']==total,'Reviewed plan differs')
    need(heads()==old and m['prior_backup_heads']==old and sha(D/'manifest.json')==MANIFEST and sha(D/'DONE')==done_sha,'Old/new heads changed during verify');inventory();g1=gpu_empty()
    summary=dict(status='verified_complete',scope='selected_portable_student_resume_reference_and_four_previews',destination=str(D),manifest_sha256=MANIFEST,DONE_sha256=done_sha,files=83,bytes=total,selected_local_step=250,selected_optimizer_step=250,selected_raw_sha256=RAW,selected_resume_sha256=RESUME,predicted_tool_score=s['predicted_tool_score'],full103_PNGs_persisted=False)
    result=dict(schema='nr_push5_independent_final_backup_receipt_v1',**summary,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_started_utc=started,CPU_only=True,CUDA_context_initialized=False,all_file_actual_SFS_bytes_full_SHA_and_size_reread=True,actual_physical_inventory_equal_manifest_plus_heads=True,tensor_archive_count=3,persisted_preview_PNG_count=4,duplicate_raw_persisted=False,raw_exact_bytes_available_as_deployment_modelpth=True,full_metric_rows=103,full_metric_scalars=418,NR=nr,FR=fr,S_proxy=nr+fr,full103_original_PNGs_rehashed_in_this_postcommit_verifier=False,full103_actual_PNG_audit_bound_to_archived_manifest=True,selected_training_audit_sha256=s['training_audit_sha256'],selected_full_audit_sha256=s['full_audit_sha256'],paired_prediction_sha256=s['paired_prediction_sha256'],selection_sha256=SELECTION,report_sha256=REPORT,selection_and_report_source_bytes_match_archive=True,previous_backup_heads_reverified=old,SFS_free_bytes_at_independent_verify=shutil.disk_usage(D).free,self_chosen_reserve_bytes=m['reserve_bytes'],reserve_is_user_requirement=False,root_backup_exit=rootexit,root_backup_exit_marker_sha256=sha('/cache/push5_backup_root_exit.json'),root_backup_log_sha256=log_sha,root_backup_log_commit_matches_DONE=True,selected_backup_plan_sha256=sha('/cache/push5_selected_backup_plan.json'),GPU_process_evidence={'start':g0,'end':g1},verification_script=str(Path(__file__)),verification_script_sha256=sha(__file__),verification_scope='Postcommit independent complete SFS byte reread; original training and native103 audit are bound by their retained proof hashes. No CUDA imports or SFS writes. Receipt is new cache metadata only.',backup=summary)
    with receipt.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'verified_complete','receipt':str(receipt),'receipt_sha256':sha(receipt),'files':83,'bytes':total}),flush=True)


if __name__=='__main__':main()
