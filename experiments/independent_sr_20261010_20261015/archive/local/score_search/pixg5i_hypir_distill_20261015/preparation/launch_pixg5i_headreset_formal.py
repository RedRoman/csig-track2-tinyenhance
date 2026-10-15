"""Five fresh single-GPU0 processes: G4pre, original, previous, new, G4post.

Default CPU plan requires REAL registered candidates/full quality, never guessed
hashes. Root alone schedules --execute after training/render/score have exited.
The common min(two G4 eventp10) normalizer is fixed before measuring anything.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time
R=Path('/cache/independent_sr_cuda_20261013')
BENCH=R/'code/benchmark_cuda.py'
BENCH_SHA='3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
G4='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
PREVIOUS='45c7f6bb414567f31b906fc996c037f31df7ce6aebb615c19183e02886950fab'
PREVIOUS_CM='1c5a7813c254ace9415486dd95fec8bf4412860f8ed7286e6989f1ab83b11b35'
PHASE_PARENT='97e6736da384a1320398995443f441ae92cdb9bf59a50a725e894e1ad9b5cd90'
PHASE_PARENT_CM='812c2b19021a24cbcd4de993fb8da328c85c3f4dfe9502a239a71674a55fb7c0'
AUDITOR='702dda1d6385973f27dc3802a7c3c0907333bc8822b8cff21b84defb17d37f09'
REGISTRAR='3a2a15092356d653f6ce173b709f2b681b03207a7ecf8a877cb279f670ce440b'
TRAIN='3e64743ff117de1934ba7da04f3c24e822ba8573d91e842230eeb86a1c6c05cb'
CLI='10e6a2ffeb7293f1e8b989d1cff582a332dbcf13bbbfc5ff1e2f8968bec458f0'
REFERENCE_RULE='min(G4_pre.event_p10,G4_post.event_p10); fixed before measurements'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,m):
    if not x:raise RuntimeError(m)
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True).strip()
def snapshot():
    return dict(utc=now(),gpus=query(['--query-gpu=index,uuid,pci.bus_id,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader']),
        processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))
def atomic(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-model-dir',type=Path,required=True)
    p.add_argument('--previous-model-dir',type=Path,default=R/'submissions/pixg5i_hypir_medium_s600/model_dir')
    p.add_argument('--quality-audit',type=Path,required=True);p.add_argument('--quality-audit-sha256',required=True)
    p.add_argument('--quality-auditor-sha256',required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args();out=a.out_dir.resolve()
    need(out.is_relative_to(R/'calibration') and not out.exists(),'Use a new immutable calibration directory')
    need(sha(BENCH)==BENCH_SHA and sha(a.quality_audit)==a.quality_audit_sha256,'Actual original benchmark/full quality audit changed')
    quality=json.loads(a.quality_audit.read_text());producer=quality['auditor_source']
    need(quality['status']=='PASS_PIXG5I_ORIGINAL_PREVIOUS_HEADRESET_FULL103_418' and len(quality['models'])==3 and
         [x['role'] for x in quality['models']]==['original','previous','new'] and
         producer['sha256']==a.quality_auditor_sha256==AUDITOR==sha(producer['path']),'Three actual full-quality results from reviewed source required')
    models=[R/'submissions/pixg5i_original_20261015/model_dir',a.previous_model_dir.resolve(),a.new_model_dir.resolve()]
    need(models[1]==R/'submissions/pixg5i_hypir_medium_s600/model_dir','Fixed actual medium600 previous model required')
    need(len(set(models))==3,'Three independent actual model directories required')
    jobs=[('g4_pre','g4_pre',R/'calibration/g4/model_dir',[])];bindings={}
    for i,(role,md) in enumerate(zip(['original','previous','new'],models)):
        cp=md.parent/'candidate_manifest.json';cm=json.loads(cp.read_text());q=quality['models'][i];label=cm['label']
        need(Path(q['model_dir']).resolve()==md and q['registration_lineage']['candidate_manifest']['sha256']==sha(cp) and
             len(q['records'])==103 and sum(len(r['metrics']) for r in q['records'])==418 and
             q['role']==role and q['label']==label+'_all','Actual full quality/model manifest differs')
        need(cm['schema']=='pixg5i_registered_candidate_v1' and cm['used_as_G4_reference'] is False and
             cm['initial_package_checkpoint_sha256']==INITIAL and cm['quality_reuse_claim'] is False and
             cm['stage']==('original_baseline' if i==0 else 'distilled_checkpoint'),'Actual original/distilled model-family binding differs')
        need(sha(md/'runner.py')==cm['runner_sha256'] and sha(md/'model.pth')==cm['checkpoint_sha256'],'Actual deployment source/weights changed')
        if i==0:need(cm['checkpoint_sha256']==INITIAL and sha(cp)=='f47a1b5318be533c221d612a2baf99d90ee49267d4728e50bfb8e42b395bfbff','Original package baseline changed')
        elif i==1:need(cm['checkpoint_sha256']==PREVIOUS and sha(cp)==PREVIOUS_CM and cm['local_step']==cm['optimizer_step']==600,'Actual fixed previous medium600 differs')
        else:
            need(cm['checkpoint_sha256'] not in [INITIAL,PREVIOUS,PHASE_PARENT] and cm['training_phase']==cm['variant']=='headreset_v3' and
                 cm['phase_initial_checkpoint_sha256']==PHASE_PARENT and cm['parent_candidate_manifest']['sha256']==PHASE_PARENT_CM and
                 cm['local_step']==cm['optimizer_step'] and cm['local_step'] in [100,300,600] and
                 cm['parent_distillation_steps']==1200 and cm['global_step']==cm['total_distillation_steps']==1200+cm['local_step'],
                 'Real head-reset checkpoint/actual structure600 phase parent required')
            need(cm['registration_source']['sha256']==REGISTRAR==sha(cm['registration_source']['path']) and
                 cm['training_provenance']['training_source']['sha256']==TRAIN==sha(cm['training_provenance']['training_source']['path']),
                 'Reviewed head-reset registration/training source required')
            prov=cm['training_provenance']['config'];need(sha(prov['path'])==prov['sha256'],'Actual head-reset config changed')
            runtime=json.loads(Path(prov['path']).read_text());tp=runtime['protocol']
            reset=dict(names=['outc.weight','outc.bias'],elements=3468,initial_outc_zero=True,body110_exact_parent=True,first_step_body_gradient_zero_expected=True)
            need(runtime['protocol_id']==hashlib.sha256(json.dumps(tp,sort_keys=True).encode()).hexdigest() and
                 tp['schema']=='pixg5i_HYPIR_headreset_fidelity_v3' and tp['reset_head']==cm['reset_head']==reset and
                 tp['phase_initial_checkpoint_sha256']==PHASE_PARENT and tp['parent_distillation_steps']==1200 and
                 tp['lr_schedule']==dict(kind='two_group_cosine',body_initial=1e-5,body_final=2e-6,head_initial=5e-5,head_final=1e-5,total_steps=600) and
                 TRAIN in tp['source_sha256'].values() and CLI in tp['source_sha256'].values() and
                 cm['registration_CPU_proof']['actual112_raw_resume_equal_FP32_finite'] is True and
                 cm['registration_CPU_proof']['actual112_Adam_moments_finite_and_counters'] is True,
                 'Actual reset-head/two-group fresh Adam phase differs')
            for path,h in tp['source_sha256'].items():need(sha(path)==h,'Frozen head-reset source/config changed')
        extras=[Path(cm['checkpoint']),cp];jobs.append((role,label,md,extras))
        bindings[role]=dict(model_label=label,candidate_manifest=dict(path=str(cp),sha256=sha(cp)),checkpoint_sha256=cm['checkpoint_sha256'])
    jobs.append(('g4_post','g4_post',R/'calibration/g4/model_dir',[]))
    need(sha(R/'calibration/g4/model_dir/model.pth')==G4,'Actual G4 calibration reference changed; pixG5i is not the reference')
    plan=dict(status='CPU_PLAN_ONLY',out_dir=str(out),order=[x[0] for x in jobs],n=200,warm=30,seed=231,
        reference_rule=REFERENCE_RULE,actual_model_bindings=bindings,actual_full_quality=dict(path=str(a.quality_audit.resolve()),sha256=a.quality_audit_sha256))
    if not a.execute:print(json.dumps(plan));return
    initial=snapshot();need(not initial['processes'],'All GPUs must be idle before formal timing')
    gpu0=initial['gpus'].splitlines()[0].split(',');need(gpu0[0].strip()=='0' and gpu0[1].strip().startswith('GPU-'),'Physical GPU0 absent')
    uuid=gpu0[1].strip();out.mkdir(parents=True)
    evidence=dict(plan,schema='pixg5i_headreset_five_fresh_serial_cuda_v1',status='RUNNING',start_utc=now(),physical_GPU0_UUID=uuid,
        benchmark_sha256=BENCH_SHA,launcher_sha256=sha(__file__),initial_snapshot=initial,jobs=[])
    receipt=out/'root_launch_evidence.json';atomic(receipt,evidence)
    for role,label,md,extras in jobs:
        before=snapshot();need(not before['processes'],'GPU work active before '+label)
        need(before['gpus'].splitlines()[0].split(',')[1].strip()==uuid,'Physical GPU0 changed')
        cmd=[sys.executable,str(BENCH),'--model-dir',str(md),'--out',str(out/(role+'_latency.json')),
             '--n','200','--warm','30','--seed','231','--device','0']
        for x in extras:cmd+=['--hash-file',str(x)]
        row=dict(role=role,label=label,command=cmd,before=before,start_utc=now(),CUDA_VISIBLE_DEVICES='0',process_samples=[],interference=[])
        began=time.monotonic()
        with (out/(role+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
            row['container_pid']=child.pid
            while child.poll() is None:
                sample=dict(utc=now(),processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))
                row['process_samples'].append(sample);lines=sample['processes'].splitlines() if sample['processes'] else []
                if len(lines)>1 or any(x.split(',')[0].strip()!=uuid for x in lines):row['interference'].append(sample)
                time.sleep(2)
        after=snapshot()
        for _ in range(10):
            if not after['processes']:break
            time.sleep(1);after=snapshot()
        row.update(exit_code=child.returncode,end_utc=now(),elapsed_seconds=time.monotonic()-began,after=after)
        if child.returncode==0:row['latency_512']=json.loads((out/(role+'_latency.json')).read_text())['latency_512']
        evidence['jobs'].append(row);atomic(receipt,evidence)
        print(json.dumps(dict(role=role,label=label,exit_code=child.returncode,latency=row.get('latency_512'),interference_count=len(row['interference']))),flush=True)
        need(child.returncode==0 and not row['interference'] and not after['processes'],'Fresh timing failed or interfered: '+label)
    evidence.update(status='COMPLETE_FIVE_FRESH_PROCESSES',end_utc=now(),final_snapshot=snapshot())
    atomic(receipt,evidence)
if __name__=='__main__':main()
