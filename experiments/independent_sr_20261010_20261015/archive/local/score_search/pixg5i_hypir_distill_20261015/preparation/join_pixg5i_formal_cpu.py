"""CPU-only join of three actual PixG5i full qualities and five fresh timings.

No GPU, scoring, training, teacher inference, preference write or official score.
Full audit owns PNG/input/GT/RNG verification; this join rechecks small metadata
and current model files, all 1000 event/wall samples, and original predict.py.
"""
import argparse,csv,datetime,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013')
LAUNCH=R/'code/pixg5i_distill/launch_pixg5i_formal.py'
LS='422f7919521b85f71642ccca700fd579233a116a3556dd0d07bdbe77b1edc058'
AUDIT='66768368c0d83d48e84b3b7f9f72682b18c3a612c6bad10ce720b5b1a78ebb86'
BENCH=R/'code/benchmark_cuda.py';BS='3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
PRED=R/'scoring_tool/predict.py';PS='0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
G4='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
ADAPTER='70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
RULE='min(G4_pre.event_p10,G4_post.event_p10); fixed before measurements'
ORDER=['g4_pre','original','mild','medium','g4_post']
SEEN={}
def need(x,m):
    if not x:raise RuntimeError(m)
def stamp(p):
    s=Path(p).stat();return s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev
def sha(p):
    p=Path(p).resolve();st=stamp(p)
    if p in SEEN:
        need(SEEN[p][0]==st,'File changed: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    need(stamp(p)==st,'File changed while hashing')
    SEEN[p]=(st,h.hexdigest());return h.hexdigest()
def read(p):sha(p);return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def close(a,b):return abs(a-b)<1e-10
def pct(xs,q):
    a=sorted(xs);f=(len(a)-1)*q;i=math.floor(f);j=math.ceil(f);return a[i]+(a[j]-a[i])*(f-i)
def utc(s):return datetime.datetime.fromisoformat(s)
def snapshot(x,uuid):
    need(not x['processes'].strip(),'GPU processes present at formal boundary')
    ids=[tuple(v.strip() for v in row[:4]) for row in csv.reader(x['gpus'].strip().splitlines())]
    need(len(ids)==4 and [x[0] for x in ids]==['0','1','2','3'] and ids[0][1]==uuid and
         all(x[1].startswith('GPU-') and x[3]=='NVIDIA A100-SXM4-80GB' for x in ids),'Physical A100 inventory differs')
    return ids
def recompute_quality(q):
    rows=q['records'];need(len(rows)==103 and sum(len(x['metrics']) for x in rows)==418,'Full103/418 missing')
    need([(x['split'],x['case']) for x in rows]==[('validation',i) for i in range(1,4)]+[('test',i) for i in range(1,101)],'Quality row order differs')
    for row in rows:
        need(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row['metrics'].values()),'Nonfinite quality scalar')
    test={k:sum(x['metrics'][k] for x in rows[3:])/100 for k in ['niqe','musiq','maniqa','clipiqa']}
    val={k:sum(x['metrics'][k] for x in rows[:3])/3 for k in ['lpips','dists']}
    nr=test['clipiqa']+test['maniqa']+test['musiq']/100+max(0.,(10-test['niqe'])/10)
    fr=(1-val['lpips'])+(1-val['dists'])
    need(close(nr,q['NR']) and close(fr,q['FR']) and close(nr+fr,q['S_full']),'Audited quality formula differs')
    return dict(NR=nr,FR=fr,S_full=nr+fr)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timing-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    d=a.timing_dir.resolve();out=a.out.resolve()
    need(d.is_relative_to(R/'calibration') and out.is_relative_to(R/'audits/pixg5i_distill') and not out.exists(),'Use new experiment calibration/proof paths')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',timing_dir=str(d),out=str(out),order=ORDER,n=200,warm=30,seed=231,
            rule=RULE,actual_full103_418_models=3,actual_event_samples=1000,actual_wall_samples=1000,GPU=False)));return
    need(sha(LAUNCH)==LS and sha(BENCH)==BS and sha(PRED)==PS,'Original launcher/benchmark/predict changed')
    evidence=read(d/'root_launch_evidence.json')
    need(evidence['status']=='COMPLETE_FIVE_FRESH_PROCESSES' and evidence['schema']=='pixg5i_five_fresh_serial_cuda_v1' and
         evidence['launcher_sha256']==LS and evidence['benchmark_sha256']==BS and evidence['reference_rule']==RULE and
         evidence['n']==200 and evidence['warm']==30 and evidence['seed']==231 and evidence['order']==ORDER,'Five complete fresh processes required')
    qp=evidence['actual_full_quality'];quality=read(qp['path'])
    need(sha(qp['path'])==qp['sha256'] and quality['status']=='PASS_FRESH_PIXG5I_FULL103_418' and
         quality['auditor_source']['sha256']==sha(quality['auditor_source']['path'])==AUDIT and
         quality['fresh_quality_measured'] is True and quality['quality_reuse_claim'] is False and
         quality['score_fourRNG_same'] is True and len(quality['models'])==3,'Actual fresh PixG5i quality proof differs')
    models={'g4_pre':R/'calibration/g4/model_dir','g4_post':R/'calibration/g4/model_dir'}
    qualities={};cms={}
    for i,(key,z) in enumerate(zip(['original','mild','medium'],quality['models'])):
        md=Path(z['model_dir']).resolve();cm_path=md.parent/'candidate_manifest.json';cm=read(cm_path)
        need(sha(cm_path)==z['registration_lineage']['candidate_manifest']['sha256']==evidence['actual_model_bindings'][key]['candidate_manifest']['sha256'] and
             Path(evidence['actual_model_bindings'][key]['candidate_manifest']['path']).resolve()==cm_path,'Quality/current/timed CM differs')
        need(cm['schema']=='pixg5i_registered_candidate_v1' and cm['model_type']=='supplied_pixG5i_pixel_UNet' and
             cm['used_as_G4_reference'] is False and cm['quality_reuse_claim'] is False and
             cm['initial_package_checkpoint_sha256']==INITIAL and cm['stage']==('original_baseline' if i==0 else 'distilled_checkpoint') and
             cm['label']+'_all'==z['label'] and cm['checkpoint_sha256']==z['registration_lineage']['checkpoint_sha256']==
             evidence['actual_model_bindings'][key]['checkpoint_sha256'],'PixG5i stage/current checkpoint binding differs')
        need(cm['checkpoint_sha256']==INITIAL if i==0 else cm['checkpoint_sha256']!=INITIAL and cm['variant']==key,'Original and distilled identities confused')
        need(sha(md/'runner.py')==cm['runner_sha256']==z['pipeline']['runner_sha256']==ADAPTER and
             sha(md/'model.pth')==sha(cm['checkpoint'])==cm['checkpoint_sha256'],'Current actual deployment changed')
        for name,x in cm['files'].items():
            need(Path(x['path']).resolve()==md/name and sha(md/name)==x['sha256'],'Registered model file changed')
        need({str(x.relative_to(md)) for x in md.rglob('*') if x.is_file()}==set(cm['files']),'Extra/missing runtime model file')
        for key2 in ['render_manifest','score_result','measurements']:
            need(sha(z[key2]['path'])==z[key2]['sha256'],'Frozen full-quality metadata changed')
        models[key]=md;cms[key]=cm;qualities[key]=recompute_quality(z)
    need(cms['mild']['local_step']==cms['medium']['local_step'] and
         cms['mild']['checkpoint_sha256']!=cms['medium']['checkpoint_sha256'],'Unequal stages or duplicate candidate')
    jobs=evidence['jobs'];need([x['label'] for x in jobs]==ORDER,'Fresh process order changed')
    uuid=evidence['physical_GPU0_UUID'];identities=snapshot(evidence['initial_snapshot'],uuid)
    need(snapshot(evidence['final_snapshot'],uuid)==identities,'Physical inventory changed')
    previous=utc(evidence['initial_snapshot']['utc']);timings={};pids=[]
    for job in jobs:
        key=job['label'];md=models[key];extras=[]
        if key in cms:extras=[Path(cms[key]['checkpoint']),md.parent/'candidate_manifest.json']
        cmd=[str(R/'env/bin/python'),str(BENCH),'--model-dir',str(md),'--out',str(d/(key+'_latency.json')),
             '--n','200','--warm','30','--seed','231','--device','0']
        for x in extras:cmd+=['--hash-file',str(x)]
        need(Path(job['command'][0]).resolve()==(R/'env/bin/python').resolve(),'Fresh interpreter differs')
        cmd[0]=job['command'][0]
        need(job['command']==cmd and job['exit_code']==0 and job['CUDA_VISIBLE_DEVICES']=='0' and
             job['elapsed_seconds']>0 and not job['interference'],'Failed/interfered/changed fresh command')
        need(snapshot(job['before'],uuid)==snapshot(job['after'],uuid)==identities and
             previous<=utc(job['before']['utc'])<=utc(job['start_utc'])<=utc(job['after']['utc'])<=utc(job['end_utc']),'Fresh boundary/time continuity differs')
        previous=utc(job['end_utc']);pids.append(job['container_pid'])
        for sample in job['process_samples']:
            lines=sample['processes'].splitlines() if sample['processes'] else []
            need(len(lines)<=1 and all(x.split(',')[0].strip()==uuid for x in lines) and
                 utc(job['start_utc'])<=utc(sample['utc'])<=utc(job['after']['utc']),'Recorded GPU interference/sample time differs')
        q=read(d/(key+'_latency.json'));s=q['latency_512']
        need(q['status']=='complete' and Path(q['model_dir']).resolve()==md and q['seed']==231 and
             q['benchmark_script']['sha256']==BS and Path(q['benchmark_script']['resolved_path']).resolve()==BENCH and
             s==job['latency_512'] and s['n']==200 and s['warm']==30,'Original actual benchmark metadata differs')
        need(q['cuda']['logical_device']==0 and q['cuda']['name']==identities[0][3] and q['cuda']['torch_version']=='2.8.0+cu128' and
             q['input']==dict(device='cuda:0',shape=[1,3,512,512],dtype='torch.float16',distribution='uniform [-1,1)',allocation_excluded_from_timing=True),'Actual CUDA input/runtime differs')
        o=q['output'];need(o['device']=='cuda:0' and o['shape']==[1,3,512,512] and o['dtype'] in ['torch.float16','torch.float32'] and
             o['all_finite'] is True and o['within_minus1_plus1'] is True and o['clamping_applied_by_benchmark'] is False,'Output contract failed')
        stats={}
        for kind in ['event','wall']:
            xs=q['raw_samples'][kind+'_ms']
            need(len(xs)==200 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in xs),'Invalid actual 200 timing samples')
            stats.update({kind+'_p10':pct(xs,.1),kind+'_med':pct(xs,.5),kind+'_min':min(xs)})
        need(all(close(v,s[k]) for k,v in stats.items()),'Actual 200 timing statistics differ')
        artifacts=q['artifacts'];files=artifacts['files'];extra=artifacts['extra_files']
        need({Path(x['resolved_path']).resolve() for x in files}=={x.resolve() for x in md.rglob('*') if x.is_file()} and
             [Path(x['resolved_path']).resolve() for x in extra]==[x.resolve() for x in extras],'Timed artifact coverage differs')
        for x in files+extra:
            path=Path(x['resolved_path']);need(path.stat().st_size==x['bytes'] and sha(path)==x['sha256'],'Timed artifact bytes differ now')
        if key in cms:
            records={Path(x['resolved_path']).resolve():x for x in files+extra};cm=cms[key]
            need(records[(md.parent/'candidate_manifest.json').resolve()]['sha256']==sha(md.parent/'candidate_manifest.json') and
                 records[md/'model.pth']['sha256']==cm['checkpoint_sha256'] and records[md/'runner.py']['sha256']==ADAPTER,'Timed PixG5i identity differs')
        else:need(any(x['sha256']==G4 and Path(x['resolved_path']).name=='model.pth' for x in files),'PixG5i cannot serve as G4 reference')
        timings[key]=dict(result=ref(d/(key+'_latency.json')),recomputed=stats,output=o,cuda=q['cuda'],artifacts=artifacts)
    need(len(set(pids))==5 and all(isinstance(x,int) and x>0 for x in pids) and
         previous<=utc(evidence['final_snapshot']['utc']),'Not five distinct sequential fresh processes')
    need(timings['g4_pre']['artifacts']==timings['g4_post']['artifacts'],'G4 changed between endpoints')
    spec=importlib.util.spec_from_file_location('_original_pixg5i_predict',PRED);predict=importlib.util.module_from_spec(spec);spec.loader.exec_module(predict)
    need(predict.A==.49 and predict.K==2.17 and predict.ANCHOR==dict(score=152.95,S=4.0841,g=.431),'Original predictor constants changed')
    refs={k:timings[k]['recomputed']['event_p10'] for k in ['g4_pre','g4_post']};normalizer=min(refs.values())
    def estimate(S,t,r):
        g=t*.431/r;v=predict.predict(S,g);explicit=152.95*2**(S-4.0841)*((.49+2.17*.431)/(.49+2.17*g))**.2
        need(close(v,explicit),'Original prediction formula differs');return v,g
    predictions={}
    for key,z in qualities.items():
        t=timings[key]['recomputed']['event_p10'];v,g=estimate(z['S_full'],t,normalizer)
        predictions[key]=dict(z,event_p10_ms=t,normalized_g_ms=g,original_tool_prediction=v,
            G4_endpoint_predictions={k:estimate(z['S_full'],t,r)[0] for k,r in refs.items()})
    changes={}
    for key in ['mild','medium']:
        qonly=estimate(qualities[key]['S_full'],predictions['original']['event_p10_ms'],normalizer)[0]
        changes[key]=dict(NR=qualities[key]['NR']-qualities['original']['NR'],FR=qualities[key]['FR']-qualities['original']['FR'],
            S_full=qualities[key]['S_full']-qualities['original']['S_full'],
            total_prediction_gain=predictions[key]['original_tool_prediction']-predictions['original']['original_tool_prediction'],
            quality_only_prediction_at_original_latency=qonly,quality_only_gain=qonly-predictions['original']['original_tool_prediction'],
            latency_change_percent=100*(predictions[key]['event_p10_ms']/predictions['original']['event_p10_ms']-1))
    result=dict(status='PASS_PIXG5I_3_FULL_5_FRESH_TIMINGS',auditor=ref(__file__),launcher=ref(LAUNCH),evidence=ref(d/'root_launch_evidence.json'),
        benchmark=ref(BENCH),predictor=ref(PRED),quality=ref(qp['path']),predictions=predictions,deltas_vs_original=changes,timings=timings,
        normalization=dict(rule=RULE,G4_event_p10_ms=refs,common_conservative_G4_event_p10_ms=normalizer,preselected_before_measurements=True),
        physical_GPU0_UUID=uuid,actual_event_samples=1000,actual_wall_samples=1000,CPU_only=True,GPU_started=False,
        current_model_files_rehashed_here=True,PNG_input_GT_and_metric_weights_rehashed_here=False,
        quality_scope='Fresh103/418 owned by referenced full audit; this join rechecks source/model files and small quality metadata, not another PNG audit',
        physical_continuity_scope='All boundary inventory/empty snapshots and every recorded process sample; CVD0/logical0; no in-process UUID telemetry claim',
        used_pixG5i_as_G4_reference=False,old_quality_or_speed_reused=False,no_official_score_claim=True,no_current_pointer_changed=True)
    for path,(st,h) in SEEN.items():need(stamp(path)==st,'Audited source mutated before commit')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],out=str(out),sha256=sha(out),predictions=predictions)))
if __name__=='__main__':main()
