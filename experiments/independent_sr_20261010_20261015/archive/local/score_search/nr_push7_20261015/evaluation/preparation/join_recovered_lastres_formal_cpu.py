"""CPU-only fresh A/B full-quality plus four-fresh-process timing audit/join.

G4pre -> A -> B -> G4post; fixed min(two reference eventp10) normalizer.
No benchmark, score, old-quality reuse, large engine rehash or pointer write.
"""
import argparse,csv,datetime,hashlib,importlib.util,json,math,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/cache/independent_sr_cuda_20261013')
D=R/'calibration/push7_lastres_recovered_formal_20261015T0840'
LAUNCH=R/'code/push7/launch_recovered_lastres_formal.py'
LS='98835860247a867aca08d603c9c8ff1a0366b602e035382c34e58d80f6a16738'
BENCH=R/'code/benchmark_cuda.py';BS='3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
PRED=R/'scoring_tool/predict.py';PS='0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
QUALITY=R/'audits/push7/hypir_full/lastres_lr5e6_s7850_trtunetvae_recovered_matched.json'
QS='985bc0ff192244d01e531daea871195c9626c1647122cf66184796c0bb5ed9b4'
GPU0='GPU-823adbe3-cd4b-354f-3797-a941e3cabc9f'
A='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
B='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'
G4='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
MODELS=dict(g4_pre=R/'calibration/g4/model_dir',reference=R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16/model_dir',
            candidate=R/'submissions/hypir_push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae/model_dir',g4_post=R/'calibration/g4/model_dir')
CMS=dict(reference='3e422d0e8474532a7fc20838504c743c32e19b03770e440eaada6eee003d38f3',
         candidate='d7e03918355aa9cb459c1d37f8ffafcee89856bb8b23cdf0346b5e440b3046cb')
SEEN={}
def need(v,m):
    if not v:raise RuntimeError(m)
def stamp(p):
    s=Path(p).stat();return s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev
def sha(p):
    p=Path(p).resolve();st=stamp(p)
    if p in SEEN:
        need(SEEN[p][0]==st,'File changed: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    need(stamp(p)==st,'File changed during read');SEEN[p]=(st,h.hexdigest());return h.hexdigest()
def read(p):sha(p);return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def pct(a,q):
    a=sorted(a);f=(len(a)-1)*q;i=math.floor(f);j=math.ceil(f);return a[i]+(a[j]-a[i])*(f-i)
def utc(s):return datetime.datetime.fromisoformat(s)
def snapshot(x):
    need(not x['processes'].strip(),'Nonempty boundary GPU process snapshot')
    rows=list(csv.reader(x['gpus'].strip().splitlines()));ids=[tuple(v.strip() for v in r[:4]) for r in rows]
    need(len(ids)==4 and [x[0] for x in ids]==['0','1','2','3'] and ids[0][1]==GPU0 and
         all(x[1].startswith('GPU-') and 'A100' in x[3] for x in ids),'Physical GPU inventory differs')
    return ids
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--out',type=Path,default=R/'audits/push7/speed/recovered_lastres_formal_cpu.json');a=p.parse_args()
    out=a.out.resolve();need(out.is_relative_to(R/'audits/push7') and not out.exists(),'Use a new immutable CPU receipt')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',quality=str(QUALITY),timing_dir=str(D),out=str(out),
            order=list(MODELS),n=200,warm=30,seed=231,reference_rule='min(G4_pre.event_p10,G4_post.event_p10)',GPU=False)));return
    need(sha(LAUNCH)==LS and sha(BENCH)==BS and sha(PRED)==PS,'Frozen original timing/predictor producer changed')
    evidence=read(D/'root_launch_evidence.json')
    if evidence['status']!='COMPLETE_FOUR_FRESH_PROCESSES':
        result=dict(status='INCOMPLETE_FORMAL_BATCH_NO_PREDICTION',evidence=ref(D/'root_launch_evidence.json'),
            actual_exits=[dict(label=x['label'],exit_code=x.get('exit_code')) for x in evidence['jobs']],predictions_available=False)
    else:
        need(evidence['schema']=='recovered_hypir_lastres_serial_cuda_v1' and evidence['launcher_sha256']==LS and
             evidence['benchmark_sha256']==BS and evidence['physical_GPU0_UUID']==GPU0 and
             evidence['n']==200 and evidence['warm']==30 and evidence['seed']==231 and
             evidence['reference_rule']=='min(G4_pre.event_p10,G4_post.event_p10); fixed before measurements','Formal protocol changed')
        quality=read(QUALITY)
        need(quality['status']=='PASS_LASTRES_LR5E6_TRTUNETVAE_FRESH_RECOVERED_FULL103_418' and
             quality['auditor_source_sha256']==sha(R/'code/push7/audit_lastres_lr5e6_trtunetvae_recovered_full.py')==QS and
             quality['full_quality_measured_here'] is True and quality['old_pre_reset_parent_score_not_used'] is True and
             quality['terminal_score_fourRNG_same'] is True and len(quality['models'])==2,'Actual fresh A/B full audit missing')
        qualities=dict(zip(['reference','candidate'],quality['models']));binding={}
        for key,z in qualities.items():
            need(len(z['records'])==103 and sum(len(x['metrics']) for x in z['records'])==418 and
                 z['registration_lineage']['actual_three_engine_SHA_rehashed_here'] is True and
                 z['registration_lineage']['actual_raw_SHA_rehashed_here'] is True,'Wrong actual full quality scope')
            cm=MODELS[key].parent/'candidate_manifest.json';need(sha(cm)==CMS[key]==z['registration_lineage']['candidate_manifest']['sha256'],'Quality/current CM changed')
            c=read(cm);binding[key]=c
            need(c['checkpoint_sha256']==z['registration_lineage']['checkpoint_sha256']==(A if key=='reference' else B) and
                 read(MODELS[key]/'config.json')==z['config'] and sha(MODELS[key]/'config.json')==c['config_sha256'] and
                 sha(MODELS[key]/'runner.py')==c['runner_sha256']==z['pipeline']['runner_sha256'],'Actual full and timed runtime binding differs')
            for field in ['render_manifest','score_result','measurements']:
                need(sha(z[field]['path'])==z[field]['sha256'],'Frozen quality metadata changed: '+field)
        jobs=evidence['jobs'];need([x['label'] for x in jobs]==list(MODELS),'Fresh G4/A/B/G4 order differs')
        identities=snapshot(evidence['initial_snapshot']);need(snapshot(evidence['final_snapshot'])==identities,'Physical UUID inventory changed')
        previous=utc(evidence['initial_snapshot']['utc']);timings={};pids=[]
        for job in jobs:
            key=job['label'];md=MODELS[key];extras=[]
            if key in binding:
                c=binding[key];extras=[Path(c['checkpoint']),md.parent/'candidate_manifest.json']
                for x in c['actual_engines'].values():extras.extend([Path(x['path']),Path(x['manifest'])])
            cmd=[str(R/'env/bin/python'),str(BENCH),'--model-dir',str(md),'--out',str(D/(key+'_latency.json')),
                 '--n','200','--warm','30','--seed','231','--device','0']
            for x in extras:cmd.extend(['--hash-file',str(x)])
            need(Path(job['command'][0]).resolve()==(R/'env/bin/python').resolve(),'Fresh Python interpreter identity differs')
            cmd[0]=job['command'][0] # Equivalent argv spelling of the same restored interpreter.
            need(job['command']==cmd and job['exit_code']==0 and job['CUDA_VISIBLE_DEVICES']=='0' and
                 job['elapsed_seconds']>0 and not job['interference'],'Wrong/failed/interfered fresh command')
            need(snapshot(job['before'])==snapshot(job['after'])==identities and
                 previous<=utc(job['before']['utc'])<=utc(job['start_utc'])<=utc(job['after']['utc'])<=utc(job['end_utc']),
                 'Fresh-process timestamps or UUID continuity differ')
            previous=utc(job['end_utc']);pids.append(job['container_pid'])
            for sample in job['process_samples']:
                lines=sample['processes'].splitlines() if sample['processes'] else []
                need(len(lines)<=1 and all(x.split(',')[0].strip()==GPU0 for x in lines),'Recorded concurrent/other-GPU process')
                need(utc(job['start_utc'])<=utc(sample['utc'])<=utc(job['after']['utc']),'Process sample outside fresh job')
            q=read(D/(key+'_latency.json'))
            need(q['status']=='complete' and Path(q['model_dir']).resolve()==md and q['seed']==231 and
                 q['benchmark_script']['sha256']==BS and Path(q['benchmark_script']['resolved_path']).resolve()==BENCH,'Wrong original benchmark result')
            need(q['cuda']['logical_device']==0 and q['cuda']['name']==identities[0][3] and
                 q['cuda']['torch_version']=='2.8.0+cu128' and
                 q['input']==dict(device='cuda:0',shape=[1,3,512,512],dtype='torch.float16',distribution='uniform [-1,1)',allocation_excluded_from_timing=True),'512 CUDA input/runtime changed')
            o=q['output'];need(o['device']=='cuda:0' and o['shape']==[1,3,512,512] and
                 o['dtype'] in (['torch.float32'] if key in binding else ['torch.float16','torch.float32']) and
                 o['all_finite'] is True and o['within_minus1_plus1'] is True and o['clamping_applied_by_benchmark'] is False,'Output contract failed')
            stats=q['latency_512'];need(stats==job['latency_512'] and stats['n']==200 and stats['warm']==30,'Wrong n/warm/stat snapshot')
            recomputed={}
            for kind in ['event','wall']:
                xs=q['raw_samples'][kind+'_ms']
                need(len(xs)==200 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in xs),'Invalid actual timing sample')
                recomputed.update({kind+'_p10':pct(xs,.1),kind+'_med':pct(xs,.5),kind+'_min':min(xs)})
            need(all(abs(stats[k]-v)<1e-10 for k,v in recomputed.items()),'Original actual200 statistics mismatch')
            artifact=q['artifacts'];records={Path(x['resolved_path']).resolve():x for x in artifact['files']+artifact['extra_files']}
            need([Path(x['resolved_path']).resolve() for x in artifact['extra_files']]==[x.resolve() for x in extras],'Timed artifact coverage differs')
            for x in artifact['files']:
                path=Path(x['resolved_path']);need(path.stat().st_size==x['bytes'],'Timed model-dir artifact bytes changed')
                need(sha(path)==x['sha256'],'Actual timed model-dir source/config/G4 weight changed')
            if key in binding:
                c=binding[key];z=qualities[key]
                for filename,h in [('runner.py',c['runner_sha256']),('config.json',c['config_sha256'])]:
                    need(records[md/filename]['sha256']==h,'Quality and timed runtime source/config differ')
                need(records[Path(c['checkpoint']).resolve()]['sha256']==c['checkpoint_sha256'] and
                     records[Path(c['checkpoint']).resolve()]['bytes']==Path(c['checkpoint']).stat().st_size,'Quality and timing raw identity/bytes differ')
                need(records[(md.parent/'candidate_manifest.json').resolve()]['sha256']==CMS[key],'Timed CM differs from audited CM')
                for x in c['actual_engines'].values():
                    e=records[Path(x['path']).resolve()];em=records[Path(x['manifest']).resolve()]
                    need(e['sha256']==x['sha256'] and e['bytes']==Path(x['path']).stat().st_size and
                         em['sha256']==sha(x['manifest'])==x['manifest_sha256'],'Full quality and timed actual engine identity differ')
            else:
                need(any(x['sha256']==G4 for x in artifact['files']),'Wrong actual G4 reference weight')
            timings[key]=dict(result=ref(D/(key+'_latency.json')),recomputed=recomputed,output=o,cuda=q['cuda'],artifacts=artifact,
                actual200_event_and_wall_recomputed=True)
        need(len(set(pids))==4 and all(isinstance(x,int) and x>0 for x in pids) and previous<=utc(evidence['final_snapshot']['utc']),'Not four distinct sequential fresh processes')
        # Both G4 passes must have the identical actual model-dir artifact bytes.
        need(timings['g4_pre']['artifacts']==timings['g4_post']['artifacts'],'G4 reference artifact changed between endpoints')
        spec=importlib.util.spec_from_file_location('_original_recovered_predict',PRED);predict=importlib.util.module_from_spec(spec);spec.loader.exec_module(predict)
        need(predict.A==.49 and predict.K==2.17 and predict.ANCHOR==dict(score=152.95,S=4.0841,g=.431),'Original prediction constants changed')
        refs={k:timings[k]['recomputed']['event_p10'] for k in ['g4_pre','g4_post']};r=min(refs.values());predictions={}
        def estimate(S,t,ref):
            g=t*.431/ref;a=predict.predict(S,g);b=152.95*2**(S-4.0841)*((.49+2.17*.431)/(.49+2.17*g))**.2
            need(abs(a-b)<1e-10,'Original predictor versus explicit formula differs');return a,g
        for key,z in qualities.items():
            S=z['NR']+z['FR'];need(abs(S-z['S_full'])<1e-12,'Fresh audited quality sum differs')
            t=timings[key]['recomputed']['event_p10'];value,g=estimate(S,t,r)
            predictions[key]=dict(NR=z['NR'],FR=z['FR'],S_full=S,event_p10_ms=t,normalized_g_ms=g,
                original_tool_prediction=value,reaches65=value>=65,
                G4_endpoint_predictions={k:estimate(S,t,x)[0] for k,x in refs.items()})
        quality_only=estimate(predictions['candidate']['S_full'],predictions['reference']['event_p10_ms'],r)[0]
        result=dict(status='PASS_FRESH_RECOVERED_A_B_FULL_AND_4_SERIAL_TIMINGS',auditor=ref(__file__),CPU_only=True,GPU_started=False,
            launcher=ref(LAUNCH),evidence=ref(D/'root_launch_evidence.json'),benchmark=ref(BENCH),predictor=ref(PRED),quality=ref(QUALITY),
            predictions=predictions,timings=timings,normalization=dict(rule=evidence['reference_rule'],G4_event_p10_ms=refs,common_conservative_G4_event_p10_ms=r,
                formula='event_p10_ms*0.431/min(G4pre_eventp10,G4post_eventp10)',preselected_before_measurements=True),
            quality_only_candidate_at_reference_latency=quality_only,
            total_prediction_gain=predictions['candidate']['original_tool_prediction']-predictions['reference']['original_tool_prediction'],
            quality_only_gain=quality_only-predictions['reference']['original_tool_prediction'],
            latency_change_percent=100*(predictions['candidate']['event_p10_ms']/predictions['reference']['event_p10_ms']-1),
            physical_GPU0_UUID=GPU0,actual_event_samples=800,actual_wall_samples=800,
            physical_continuity_scope='Boundary inventory/empty snapshots plus every launcher process sample; fixed CVD0 and logical0; no in-process UUID telemetry claimed',
            HYPIR_raw_and_engine_bytes_rehashed_here=False,G4_model_dir_files_rehashed_here=True,engine_identity_scope='Actual SHA/bytes independently produced by original benchmark and this round actual full audit; small source/CM/manifests rehashed in join',
            old_quality_or_speed_reused=False,no_official_score_claim=True,no_current_pointer_changed=True)
    for path,(st,h) in SEEN.items():need(stamp(path)==st,'Source mutated before publish')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],out=str(out),sha256=sha(out),predictions=result.get('predictions'))))
if __name__=='__main__':main()
