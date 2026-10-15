"""Independent post-commit full-byte verification of the Push4 recipe backup.

All SFS paths are read-only. Only the new post-commit receipt is created under
/cache. This script never imports tensor libraries, restores a duplicate raw
file, starts a GPU job, waits for a backup, or modifies archived metadata.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import zipfile

R=Path('/cache/independent_sr_cuda_20261013')
D=Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014')
ROOT_EXIT=Path('/cache/push4_backup_root_exit.json')
ROOT_LOG=Path('/cache/nr_push4_backup_execute.log')
OUTPUT=R/'results/nr_push4_backup_receipt_20261014.json'
MANIFEST_SCOPE='selected_exact_resume_plus_exact_raw_recipe_and_optional_previews'
RECEIPT_SCOPE='resume_plus_exact_raw_recipe'
HELPER_SHA='afc2f54bffddd1e9d03cf263881f1ba4beb6022c745fc6639b7f902e0dee6cf4'
RECIPE_HELPER_SHA='43b135a68047a9319e355d9967b4c1bcddcab16ff58d775a8c8f7ef261d932ce'
PREVIEWS=[7,16,41,83]
CHUNK=8*1024**2
PRIOR=[
 {'directory':'/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014','marker':'DONE',
  'manifest_sha256':'ef284432a338b0a4574f52be645f4a40571ef94e775e1e39a2c0d1cfc1d13e8d',
  'DONE_sha256':'0a9cb512123ca402baaf43e21db2204853b2033fbcb61a1739f802306b36703f'},
 {'directory':'/temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014','marker':'DONE.json',
  'manifest_sha256':'99f461e6e95045cf0af053dc3b24bde74cd3531715225d0ccf95eef2d2dbd0d4',
  'DONE_sha256':'92cadf2e81aba1864e027e1fded512c008b15585018e1c1eda7d9658b45b8d12'},
 {'directory':'/temp/independent_sr_cuda_20261013_artifacts/nr_push3_20261014','marker':'DONE',
  'manifest_sha256':'6c4cc6db3d963762cb1868d83d89969083326cc0e3932d64743a23b598c48eaa',
  'DONE_sha256':'15846dd6f53e3327bf7331d00a27d4928b2078edca4244ee29186623cf85655d'},
]


def need(ok,message):
    if not ok:raise RuntimeError(message)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def token(p):
    s=Path(p).stat();return s.st_size,s.st_mtime_ns,s.st_ino


def sha(p):
    p=Path(p);before=token(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    need(token(p)==before,'File changed while hashing: '+str(p))
    return h.hexdigest()


def relative(source):
    p=Path(source)
    need(p.is_absolute() and p.is_relative_to(R),'Original source outside experiment root')
    need('..' not in p.parts,'Escaped original source')
    return str(p.relative_to(R))


def safe(rel):
    p=Path(rel);need(not p.is_absolute() and '..' not in p.parts,'Unsafe manifest path')
    q=D/p;need(q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(D.resolve()),'Missing/escaped archive path: '+str(p))
    for a in [D,*q.parents]:
        if a==D.parent:break
        need(not a.is_symlink(),'Symlinked archive parent')
    return q


def archived(source):return safe(relative(source))


def actual_prefix_inventory(expected):
    actual=set()
    for p in D.rglob('*'):
        need(not p.is_symlink(),'Unexpected symlink in committed backup prefix: '+str(p))
        if p.is_file():actual.add(str(p.relative_to(D)))
    need(actual==set(expected)|{'manifest.json','DONE'},
         'Actual prefix files differ from manifest plus commit heads; extra/missing payload exists')


def previous_heads():
    out=[]
    for row in PRIOR:
        p=Path(row['directory']);m=p/'manifest.json';d=p/row['marker']
        need(sha(m)==row['manifest_sha256'] and sha(d)==row['DONE_sha256'],'Prior immutable backup head changed')
        mm,dd=read(m),read(d)
        need(dd['status']=='complete' and dd['manifest_sha256']==row['manifest_sha256'] and
             dd['destination']==mm['destination']==str(p),'Prior manifest/DONE binding differs')
        out.append(dict(row,reverified_manifest_and_DONE_byte_SHA=True))
    return out


def gpu_evidence():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader,nounits'],text=True)
    apps=[{'uuid':r[0].strip(),'pid':r[1].strip(),'process':','.join(r[2:]).strip()}
          for r in csv.reader(io.StringIO(output)) if len(r)>=3]
    ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    names=['train_cuda_nr_stage_v5.py','render_cuda_submission.py','score_cuda_stream.py',
           'benchmark_cuda.py','paired_latency_push4.py']
    own=[line.strip() for line in ps.splitlines() if str(R) in line and
         any(n in line for n in names) and 'ps -eo' not in line]
    need(not own,'This round still has a training/render/scoring/benchmark command')
    return {'NVML_visible_compute_processes':apps,'all_GPUs_have_no_visible_compute_processes':not apps,
            'this_round_GPU_command_lines':own,'this_round_GPU_processes_absent':True,
            'method':'read-only nvidia-smi NVML process query plus ps; no CUDA/model imported'}


def reconstruct_raw(selection,index,resume):
    directory=Path(selection['raw_recipe_dir'])
    need(directory.is_relative_to(R/'audits/push4'),'Recipe is not in new Push4 audits')
    recipe_path=archived(directory/'recipe.json');recipe=read(recipe_path)
    gap=archived(directory/recipe['gaps_file'])
    need(Path(recipe['gaps_file']).name==recipe['gaps_file'] and gap.parent==recipe_path.parent,'Escaped literal gaps')
    need(recipe['schema']=='raw_torch_zip_exact_byte_recipe_v1' and recipe['tensor_entries']==514,'Recipe schema/count differs')
    need(recipe['raw_name']==Path(selection['checkpoint']).name and recipe['raw_sha256']==selection['checkpoint_sha256']
         and recipe['resume_name']==resume.name and recipe['resume_sha256']==selection['resume_sha256']
         and recipe['resume_bytes']==resume.stat().st_size,'Recipe raw/resume identity differs')
    need(sha(gap)==recipe['gaps_sha256'] and gap.stat().st_size<=16*1024**2,'Literal gaps identity/size differs')
    proof_path=archived(selection['raw_recipe_restore_proof']);proof=read(proof_path)
    need(sha(proof_path)==selection['raw_recipe_restore_proof_sha256'] and
         sha(proof_path)==sha(Path(selection['raw_recipe_restore_proof'])),'Archived/source actual restore proof differs')
    need(proof['status']=='PASS' and proof['CPU_only'] is True and
         proof['raw_restore_complete_byte_SHA256']==selection['checkpoint_sha256'] and
         proof['resume_sha256']==selection['resume_sha256'] and proof['raw_bytes']==recipe['raw_bytes'] and
         proof['helper_sha256']==RECIPE_HELPER_SHA,'Not the selected checkpoint own actual full-byte restore proof')
    need(sha(safe('code/push4/checkpoint_archive_recipe.py'))==RECIPE_HELPER_SHA,'Archived recipe helper source differs')
    h=hashlib.sha256();total=0;entries=0;gap_total=0
    with zipfile.ZipFile(resume) as z,gap.open('rb') as gaps:
        need(len(z.namelist())==len(set(z.namelist())),'Duplicate resume ZIP members')
        for segment in recipe['segments']:
            count=segment['bytes'];need(isinstance(count,int) and count>=0,'Invalid segment byte count')
            if segment['kind']=='gap':
                need(isinstance(segment['offset'],int) and segment['offset']==gap_total and count<=16*1024**2,
                     'Noncontiguous or oversized literal gap')
                gaps.seek(segment['offset']);data=gaps.read(count);need(len(data)==count,'Truncated gap')
                h.update(data);total+=count;gap_total+=count
            else:
                need(segment['kind']=='resume_entry','Unknown recipe segment kind')
                info=z.getinfo(segment['name']);need(info.file_size==count,'Resume tensor entry size differs')
                eh=hashlib.sha256();n=0
                with z.open(info) as source:
                    for b in iter(lambda:source.read(CHUNK),b''):eh.update(b);h.update(b);n+=len(b)
                need(n==count and eh.hexdigest()==segment['sha256'],'Actual retained resume tensor bytes differ')
                entries+=1;total+=n
    need(entries==514 and gap_total==gap.stat().st_size and total==recipe['raw_bytes'] and
         h.hexdigest()==recipe['raw_sha256']==selection['checkpoint_sha256'],
         'Archived resume plus gaps cannot reconstruct exact selected raw bytes')
    return {'status':'PASS','tensor_entries':entries,'raw_bytes':total,'raw_sha256':h.hexdigest(),
            'literal_gap_bytes':gap_total,'recipe_sha256':sha(recipe_path),'gaps_sha256':sha(gap),
            'recipe_helper_sha256':RECIPE_HELPER_SHA,'selected_actual_restore_proof_sha256':sha(proof_path),
            'method':'Independent stdlib ZIP streaming into SHA256 from retained SFS resume+gaps; no output raw file',
            'duplicate_raw_file_written':False}


def full_quality(selection,index):
    q=relative(selection['quality_dir']);render=read(safe(q+'/render_manifest.json'))
    rp=read(safe(q+'/render_protocol.json'));protocol=rp['protocol']
    pid=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    need(render['protocol']==protocol and render['protocol_id']==rp['protocol_id']==pid,
         'Archived full render protocol identity differs')
    runner=protocol['runner']
    need(runner['checkpoint_sha256']==selection['checkpoint_sha256'] and
         Path(runner['weight_path'])==Path(selection['checkpoint']) and
         Path(protocol['model_dir'])==Path(selection['model_dir']),
         'Full render runner/model/checkpoint differs from selection')
    model=Path(selection['model_dir'])
    need(sha(archived(model/'runner.py'))==runner['runner_sha256'] and
         Path(read(archived(model/'config.json'))['weight_path'])==Path(selection['checkpoint']),
         'Retained inference runner/config checkpoint binding differs')
    records=render['records'];expected=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
    need([(x['split'],x['case']) for x in records]==expected and len({r['output'] for r in records})==103,
         'Full render103 split/order/uniqueness differs')
    pngs=[r for r in index.values() if Path(r['path']).suffix.lower()=='.png']
    previews=[r for r in records if r['split']=='test' and r['case'] in PREVIEWS]
    need(len(pngs)==4 and {r['path'] for r in pngs}=={str(Path(q)/r['output']) for r in previews},
         'Must persist exactly selected native test7/16/41/83 previews, not full103 PNGs')
    for r in previews:
        rel=str(Path(q)/r['output']);need(index[rel]['sha256']==r['sha256'],'Preview/manifest SHA differs')
        with safe(rel).open('rb') as f:header=f.read(24)
        need(header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR' and
             list(struct.unpack('>II',header[16:24]))==r['size'],'Native PNG dimensions differ from manifest')
    result=read(safe(q+'/scores/result.json'))
    rows=[json.loads(x) for x in safe(q+'/scores/measurements.jsonl').read_text().splitlines() if x.strip()]
    need(result['status']=='complete' and result['completed']==result['expected']==103 and
         result['official_full_count'] is True and len(rows)==103,'Full scoring incomplete')
    need([(x['split'],x['case']) for x in rows]==[('validation' if s=='val' else s,c) for s,c in expected],
         'Metric ordering differs')
    means={};scalars=0
    for split,n,keys in [('test',100,['clipiqa','maniqa','musiq','niqe']),
                         ('validation',3,['lpips','dists','clipiqa','maniqa','musiq','niqe'])]:
        subset=[r for r in rows if r['split']==split];need(len(subset)==n,'Split metric count differs')
        need(all(set(x['metrics'])==set(keys) for x in subset),'Metric key set differs')
        means[split]={}
        for k in keys:
            values=[r['metrics'][k] for r in subset];need(all(math.isfinite(v) for v in values),'Nonfinite metric')
            means[split][k]=sum(values)/n;need(abs(means[split][k]-result[split][k])<=1e-12,'Mean differs')
        scalars+=len(keys)*n
    need(scalars==418,'Expected418 scalar metrics')
    for row,rec in zip(rows,records):
        need(row['label']==selection['label'] and row['sr_sha256']==rec['sha256'] and
             Path(row['sr_path'])==Path(selection['quality_dir'])/rec['output'],'Scored full PNG/manifest identity differs')
    t,v=means['test'],means['validation']
    nr=t['clipiqa']+t['maniqa']+t['musiq']/100+max(0,(10-t['niqe'])/10);fr=2-v['lpips']-v['dists']
    for k,value in [('NR',nr),('FR',fr),('S_proxy',nr+fr)]:
        need(abs(selection[k]-value)<=1e-12,'Full component differs from selection: '+k)
    need(abs(result['test']['nr']-nr)<=1e-12 and abs(result['validation']['fr']-fr)<=1e-12 and
         abs(result['S_proxy']-nr-fr)<=1e-12,'Published full NR/FR/S formula differs')
    audit=read(archived(selection['full_audit']))
    need(sha(archived(selection['full_audit']))==selection['full_audit_sha256'] and audit['status']=='PASS' and
         audit['image_rows']==103 and audit['scalar_metrics']==418 and audit['label']==selection['label'] and
         audit['actual_checkpoint_sha256']==selection['checkpoint_sha256'] and
         audit['actual_png_manifest_scored_sha']=='PASS103' and audit['actual_source_code_sha']=='PASS',
         'Archived independent full audit differs')
    actual={(r['split'],r['case']):r for r in audit['records']}
    need(len(actual)==103,'Independent actual full PNG audit records incomplete')
    for rec in records:
        key=('validation' if rec['split']=='val' else rec['split'],rec['case'])
        need(actual[key]['actual_png_sha256']==rec['sha256'],'Full manifest differs from independent actual PNG audit')
    return {'rows':103,'scalars':418,'NR':nr,'FR':fr,'S_proxy':nr+fr,
            'preview_cases':PREVIEWS,'preview_native_PNG_count':4,'preview_native_PNG_bytes':sum(p['bytes'] for p in pngs),
            'full103_native_PNGs_persisted':False,'full103_original_PNGs_rehashed_in_this_postcommit_verifier':False,
            'full103_metric_rows_bound_to_archived_manifest_and_independent_actual_PNG_audit':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expected-manifest-sha',required=True)
    ap.add_argument('--expected-selection-sha',required=True)
    ap.add_argument('--expected-report-sha',required=True)
    ap.add_argument('--expected-root-session',type=int,required=True)
    ap.add_argument('--expected-helper-sha',default=HELPER_SHA)
    ap.add_argument('--plan-path',type=Path,default=R/'nr_push4_20261014/preparation/selected_backup_plan.json')
    args=ap.parse_args();started=now()
    # No wait loop: root must explicitly dispatch after the final commit/exit0.
    need((D/'DONE').is_file() and ROOT_EXIT.is_file(),'Dispatch only after root DONE plus exit0 marker; no waiting')
    need(not OUTPUT.exists(),'Post-commit receipt already exists; do not overwrite')
    root_exit=read(ROOT_EXIT);need(root_exit['exit_code']==0 and root_exit['session_id']==args.expected_root_session,
                                 'Root backup session did not confirm expected exit0')
    need(ROOT_LOG.is_file(),'Root execution log missing')
    need(sha(D/'manifest.json')==args.expected_manifest_sha,'Manifest differs from root-provided commit SHA')
    done=read(D/'DONE');done_sha=sha(D/'DONE');manifest=read(D/'manifest.json')
    need(done['status']==manifest['status']=='complete' and done['manifest_sha256']==args.expected_manifest_sha and
         done['selection_sha256']==manifest['selection_sha256']==args.expected_selection_sha,'Commit binding differs')
    need(done['scope']==manifest['scope']==MANIFEST_SCOPE and done['destination']==manifest['destination']==str(D),
         'Scope/destination differs')
    need(manifest['backup_script_sha256']==args.expected_helper_sha,'Committed backup helper differs')
    need(manifest['full103_PNGs_persisted'] is False and done['full103_PNGs_persisted'] is False and
         manifest['raw_checkpoint_persisted'] is False and done['raw_checkpoint_persisted'] is False and
         manifest['raw_exact_byte_recovery_available'] is True and done['raw_exact_byte_recovery_available'] is True,
         'Recipe-only scope flags differ')
    need(manifest['persisted_preview_cases']==PREVIEWS and manifest['persisted_preview_PNG_count']==done['persisted_preview_PNG_count']==4,
         'Expected four native previews')
    prior=previous_heads();gpu_start=gpu_evidence()
    rows=manifest['files'];index={x['path']:x for x in rows}
    need(len(rows)==len(index)==done['files']==manifest['count'] and sum(x['bytes'] for x in rows)==done['bytes']==manifest['bytes'],
         'Manifest count/total/uniqueness differs')
    actual_prefix_inventory(index)
    total=0;tokens={}
    for n,record in enumerate(rows,1):
        path=safe(record['path']);tokens[str(path)]=token(path)
        need(path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],'Actual SFS size/SHA differs: '+record['path'])
        total+=record['bytes']
        if n%50==0 or n==len(rows):print(json.dumps({'verified_SFS_full_byte_files':n,'expected_files':len(rows),'verified_bytes':total}),flush=True)
    sp=safe('results/nr_push4_selection.json');selection=read(sp)
    need(sha(sp)==sha(R/'results/nr_push4_selection.json')==args.expected_selection_sha,'Archived/source selection differs')
    need(selection['status']=='selected_after_independent_full_audit' and selection['local_step'] in (250,500) and
         selection['global_step']==7250+selection['local_step'] and selection['optimizer_step']==1250+selection['local_step'],
         'Selected new stage5 milestone counters differ')
    run=Path(selection['config']).parent;need(run.parent==R/'adaptation/runs' and run.name.startswith('cuda_nr_stage5_'),'Selected run scope differs')
    resume_rel=relative(selection['resume']);raw_rel=relative(selection['checkpoint']);resume=safe(resume_rel)
    need(Path(selection['resume'])==run/f"resume_step{selection['global_step']}.pt" and
         Path(selection['checkpoint'])==run/f"step{selection['global_step']}.pth",'Wrong mutable/nonmatching checkpoint path')
    tensor_archives=[p for p in index if Path(p).suffix.lower() in ('.pt','.pth','.ckpt','.safetensors')]
    need(tensor_archives==[resume_rel] and raw_rel not in index,'Must persist only one selected immutable resume archive')
    need(index[resume_rel]['sha256']==selection['resume_sha256'],'Retained resume SHA differs')
    need(sha(safe('code/push4/backup_selected_push4.py'))==sha(R/'code/push4/backup_selected_push4.py')==args.expected_helper_sha,
         'Actual committed/source backup helper bytes differ')
    recipe=reconstruct_raw(selection,index,resume)
    need(manifest['raw_recipe_proof']['raw_sha256']==recipe['raw_sha256'] and
         manifest['raw_recipe_proof']['recipe_sha256']==recipe['recipe_sha256'] and
         manifest['raw_recipe_proof']['gaps_sha256']==recipe['gaps_sha256'],'Manifest recipe proof differs')
    quality=full_quality(selection,index)
    ta=archived(selection['training_audit']);audit=read(ta)
    need(sha(ta)==selection['training_audit_sha256'] and audit['status']=='PASS' and audit['CPU_only'] is True and
         audit['CUDA_context_initialized'] is False,'Archived independent CPU training audit differs')
    bb=[b for b in audit['branches'] if b['run']==run.name and b['audited_global_step']==selection['global_step']]
    need(len(bb)==1,'Selected training audit branch missing/ambiguous');b=bb[0]
    need(b['status']=='PASS' and b['audited_stage_local_step']==selection['local_step'] and
         b['audited_Adam_step']==selection['optimizer_step'] and b['final_weight_exact_equal_resume_FP32_LoRA'] is True and
         b['LoRA']['keys']==514 and b['LoRA']['dtype']=='torch.float32' and b['LoRA']['all_finite'] is True and
         b['Adam']['state_entries']==514 and b['Adam']['all_steps']==[selection['optimizer_step']] and
         b['Adam']['all_moments_FP32_CPU_finite'] is True and
         set(b['final_rng']['state_sha256'])=={'python','numpy','cpu','cuda'},'Archived all514/Adam/fourRNG proof incomplete')
    need(b['file_records'][Path(selection['resume']).name]['sha256']==selection['resume_sha256'] and
         b['file_records'][Path(selection['checkpoint']).name]['sha256']==selection['checkpoint_sha256'],
         'Training audit raw/resume SHA differs')
    config_path=archived(selection['config']);config=read(config_path)
    need(sha(config_path)==b['file_records']['config.json']['sha256'] and
         hashlib.sha256(json.dumps(config['protocol'],sort_keys=True).encode()).hexdigest()==config['protocol_id']==b['protocol_id'],
         'Archived training config/protocol differs from independent audit')
    marker=read(archived(run/f"checkpoint_step{selection['global_step']}.json"))
    need((marker['local_step'],marker['global_step'],marker['optimizer_step'])==
         (selection['local_step'],selection['global_step'],selection['optimizer_step']) and
         marker['raw_checkpoint_sha256']==selection['checkpoint_sha256'] and
         marker['resume_checkpoint_sha256']==selection['resume_sha256'],
         'Archived immutable checkpoint marker differs')
    report_rel='reports/NR_PUSH4_REPORT_20261014.md';report_sha=sha(safe(report_rel))
    need(report_sha==sha(R/report_rel)==args.expected_report_sha,'Final archived/source report bytes differ')
    plan=read(args.plan_path)
    need(plan['selection_sha256']==args.expected_selection_sha and plan['files']==manifest['files'] and
         plan['count']==manifest['count'] and plan['bytes']==manifest['bytes'],'Reviewed backup plan differs from commit')
    commits=[]
    for line in ROOT_LOG.read_text().splitlines():
        try:v=json.loads(line)
        except json.JSONDecodeError:continue
        if 'commit_complete' in v:commits.append(v['commit_complete'])
    need(commits and commits[-1]==done,'Root successful execution log not bound to final DONE')
    need(sha(D/'manifest.json')==args.expected_manifest_sha and sha(D/'DONE')==done_sha and
         sha(sp)==args.expected_selection_sha and sha(safe(report_rel))==args.expected_report_sha,
         'Committed heads/selection/report changed during verification')
    need(all(token(p)==t for p,t in tokens.items()),'Archived payload changed during verification')
    actual_prefix_inventory(index)
    need(previous_heads()==prior,'Prior heads changed during verification')
    gpu_end=gpu_evidence();free=shutil.disk_usage(D).free
    need(free>=done['self_chosen_reserve_bytes'],'SFS free below recorded self-chosen reserve')
    receipt={'schema':'nr_push4_independent_final_backup_receipt_v1','status':'verified_complete','verified_utc':now(),
             'verification_started_utc':started,'CPU_only':True,'CUDA_context_initialized':False,'scope':RECEIPT_SCOPE,
             'archived_manifest_scope':MANIFEST_SCOPE,'destination':str(D),'manifest_sha256':args.expected_manifest_sha,
             'DONE_sha256':done_sha,'files':len(rows),'bytes':total,'all_file_actual_SFS_bytes_full_SHA_and_size_reread':True,
             'all_file_reread_files':len(rows),'all_file_reread_bytes':total,'tensor_archive_count':1,
             'selected_global_step':selection['global_step'],'selected_local_step':selection['local_step'],
             'selected_optimizer_step':selection['optimizer_step'],'selected_raw_path':raw_rel,
             'selected_raw_sha256':selection['checkpoint_sha256'],'selected_raw_persisted':False,
             'selected_raw_exact_recovery_from_retained_resume_and_recipe':recipe,
             'selected_immutable_resume_path':resume_rel,'selected_immutable_resume_sha256':selection['resume_sha256'],
             'persisted_preview_PNG_count':4,'native_full_PNG_count':0,'full103_PNGs_persisted':False,
             'full_quality_metadata_verification':quality,'full_metric_rows':103,'full_metric_scalars':418,
             'archived_exact514_FP32_Adam_and_four_RNG_audit_PASS':True,'selected_training_audit_sha256':sha(ta),
             'previous_backup_heads_reverified':prior,'SFS_free_bytes_at_independent_verify':free,
             'self_chosen_reserve_bytes':done['self_chosen_reserve_bytes'],'reserve_is_user_requirement':False,
             'report_sha256':report_sha,'report_source_bytes_match_archive':True,
             'selection_sha256':args.expected_selection_sha,'selection_source_bytes_match_archive':True,
             'selected_backup_plan_sha256':sha(args.plan_path),'reviewed_plan_files_equal_committed_manifest':True,
             'root_backup_exit':root_exit,'root_backup_exit_marker_sha256':sha(ROOT_EXIT),
             'root_backup_log_sha256':sha(ROOT_LOG),'root_backup_log_commit_matches_DONE':True,
             'GPU_process_evidence':{'start':gpu_start,'end':gpu_end},'verification_script':str(Path(__file__).resolve()),
             'verification_script_sha256':sha(__file__),
             'verification_scope':'Independent complete SFS payload reread after root exit0; exact raw ZIP reconstructed into SHA256 only. Receipt is new post-commit metadata; archived manifest/DONE/report/status and all prior prefixes remain read-only.'}
    receipt['backup']={k:receipt[k] for k in ['status','scope','destination','manifest_sha256','DONE_sha256','files','bytes',
        'selected_global_step','selected_local_step','selected_optimizer_step','selected_raw_sha256',
        'selected_immutable_resume_sha256','tensor_archive_count','persisted_preview_PNG_count','full103_PNGs_persisted',
        'all_file_actual_SFS_bytes_full_SHA_and_size_reread','SFS_free_bytes_at_independent_verify']}
    receipt['backup'].update(verified_complete=True,raw_checkpoint_persisted=False,raw_exact_byte_recovery_available=True,
                            native_full_PNG_count=0,full_metric_rows=103,full_metric_scalars=418)
    with OUTPUT.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    need(read(OUTPUT)==receipt,'Postcommit receipt readback differs')
    print(json.dumps({'status':'verified_complete','receipt':str(OUTPUT),'receipt_sha256':sha(OUTPUT),
                      'manifest_sha256':args.expected_manifest_sha,'DONE_sha256':done_sha,'files':len(rows),'bytes':total,
                      'selected_raw_reconstructed_sha256':recipe['raw_sha256'],'full_quality':quality,
                      'GPU_process_evidence':gpu_end},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
