"""Independent CPU audit of four completed fresh same-GPU0 benchmark processes.
Default plan does no artifact IO. Only Push6 and TRT-VAE have full quality here.
"""
import argparse,csv,datetime,hashlib,importlib.util,json,math,os
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
D=R/'calibration/push7_teacher_formal_20261015T0604'
LAUNCH=R/'code/push7/launch_formal_push7.py'
LAUNCH_SHA='3661c6cf04b53b47fc4d4403e91458a98b5b9223651dcf19c8d6d793baee7d80'
BENCH=R/'code/benchmark_cuda.py'
BENCH_SHA='3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
PRED=R/'scoring_tool/predict.py'
PRED_SHA='0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
QUALITY=R/'audits/push7/hypir_full/trtvae_fp16_all.json'
QUALITY_SHA='1e54ac9a0681b67144c2f173e458d9c23c8f1d9d41fec17e7c278cc80470c63b'
RAW=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
A='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
ENG=R/'backends/push7_trt'
MODELS=dict(g4=R/'calibration/g4/model_dir',push6=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir',
 trtvae=R/'submissions/hypir_push7_a7750_trtvae_fp16/model_dir',
 trtunetvae=R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16/model_dir')
EXTRAS=dict(g4=[],push6=[RAW],trtvae=[RAW,ENG/'encoder_fp16/component.engine',ENG/'decoder_fp16/component.engine'],
 trtunetvae=[RAW,ENG/'encoder_fp16/component.engine',ENG/'decoder_fp16/component.engine',ENG/'unet_fp16_A7750/component.engine'])
SEEN={}
def need(x,m):
    if not x:raise RuntimeError(m)
def stamp(p):
    s=Path(p).stat();return s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev
def sha(p):
    p=Path(p).resolve();st=stamp(p)
    if p in SEEN:
        need(SEEN[p][0]==st,'Source changed: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    need(stamp(p)==st,'Source changed during read');SEEN[p]=(st,h.hexdigest());return h.hexdigest()
def read(p):
    sha(p);return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def pct(xs,q):
    a=sorted(xs);f=(len(a)-1)*q;i=math.floor(f);j=math.ceil(f)
    return a[i]+(a[j]-a[i])*(f-i)
def utc(s):return datetime.datetime.fromisoformat(s)
def snapshot(v):
    need(not v['processes'].strip(),'Nonempty boundary GPU process snapshot')
    rows=list(csv.reader(v['gpus'].strip().splitlines()))
    ids=[tuple(x.strip() for x in row[:4]) for row in rows]
    need(len(ids)==4 and [x[0] for x in ids]==['0','1','2','3'],'Unexpected physical GPU inventory')
    need(all('A100' in x[3] and x[1].startswith('GPU-') for x in ids),'Expected four A100 physical UUIDs')
    return ids
def load_predict():
    need(sha(PRED)==PRED_SHA,'Original predictor changed')
    s=importlib.util.spec_from_file_location('original_push7_predict',PRED);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    need(m.A==.49 and m.K==2.17 and m.ANCHOR==dict(score=152.95,S=4.0841,g=.431),'Original predictor constants changed')
    return m
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--out',type=Path,default=R/'audits/push7/speed/formal4_independent_cpu.json');a=p.parse_args()
    out=a.out.resolve();need(out.is_relative_to(R/'audits/push7') and not out.exists(),'Use a new immutable Push7 audit path')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',timing_dir=str(D),out=str(out),requires_actual_completed_four_processes=True,
          samples_per_model=200,full_quality_models=['push6','trtvae'],unetvae_prediction_available=False,GPU=False)));return
    os.environ['CUDA_VISIBLE_DEVICES']=''
    evidence=read(D/'root_launch_evidence.json')
    if evidence.get('status')!='COMPLETE_FOUR_FRESH_PROCESSES':
        result=dict(status='INCOMPLETE_FORMAL_RUN_NO_PREDICTION',evidence=ref(D/'root_launch_evidence.json'),
          actual_jobs=[dict(label=x['label'],exit_code=x.get('exit_code')) for x in evidence.get('jobs',[])],
          predictions_available=False,no_current_model_pointer_changed=True)
    else:
        need(sha(LAUNCH)==evidence['launcher_sha256']==LAUNCH_SHA and sha(BENCH)==evidence['benchmark_sha256']==BENCH_SHA,'Formal producer source mismatch')
        need(evidence['schema']=='push7_root_serial_idle_gpu0_v1' and evidence['CUDA_VISIBLE_DEVICES']=='0' and
             evidence['n']==200 and evidence['warm']==30 and evidence['seed']==231 and evidence['device']==0,'Formal launch protocol changed')
        jobs=evidence['jobs'];need([x['label'] for x in jobs]==list(MODELS),'Wrong four-model order')
        identities=snapshot(evidence['initial_snapshot']);need(snapshot(evidence['final_snapshot'])==identities,'Physical UUID inventory changed')
        timings={};previous=utc(evidence['initial_snapshot']['utc'])
        for job in jobs:
            label=job['label'];md=MODELS[label]
            need(job['exit_code']==0 and job['CUDA_VISIBLE_DEVICES']=='0' and job['elapsed_seconds']>0,'Fresh process failed')
            need(snapshot(job['before'])==snapshot(job['after'])==identities,'Physical UUID continuity mismatch')
            need(previous<=utc(job['before']['utc'])<=utc(job['start_utc'])<=utc(job['end_utc'])<=utc(job['after']['utc']),'Fresh process order/timestamps overlap')
            previous=utc(job['after']['utc'])
            cmd=job['command'];expected=[str(R/'env/bin/python'),str(BENCH),'--model-dir',str(md),'--out',str(D/(label+'_latency.json')),'--n','200','--warm','30','--seed','231','--device','0']
            for path in EXTRAS[label]:expected+=['--hash-file',str(path)]
            need(cmd==expected,'Actual fresh command changed: '+label)
            f=D/(label+'_latency.json');q=read(f)
            need(q['status']=='complete' and Path(q['model_dir']).resolve()==md and q['seed']==231,'Wrong benchmark result: '+label)
            need(q['benchmark_script']['sha256']==BENCH_SHA and Path(q['benchmark_script']['resolved_path']).resolve()==BENCH,'Wrong original benchmark artifact')
            need(q['cuda']['logical_device']==0 and q['cuda']['name']==identities[0][3] and q['cuda']['torch_version']=='2.8.0+cu128','Wrong benchmark CUDA device/runtime')
            need(q['input']==dict(device='cuda:0',shape=[1,3,512,512],dtype='torch.float16',distribution='uniform [-1,1)',allocation_excluded_from_timing=True),'Original512 input changed')
            o=q['output'];need(o['device']=='cuda:0' and o['shape']==[1,3,512,512] and o['dtype'] in ['torch.float16','torch.float32'] and
              o['all_finite'] is True and o['within_minus1_plus1'] is True and o['clamping_applied_by_benchmark'] is False,'Output contract failed')
            raw=q['raw_samples'];stats=q['latency_512'];need(stats['n']==200 and stats['warm']==30 and job['latency_512']==stats,'Wrong original n/warm/stat snapshot')
            recomputed={}
            for kind in ['event','wall']:
                xs=raw[kind+'_ms'];need(len(xs)==200 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in xs),'Raw timing samples invalid')
                recomputed.update({kind+'_med':pct(xs,.5),kind+'_p10':pct(xs,.1),kind+'_min':min(xs)})
            need(all(abs(stats[k]-v)<1e-10 for k,v in recomputed.items()),'200-sample original statistics mismatch')
            manifest=q['artifacts'];need([Path(x['resolved_path']).resolve() for x in manifest['extra_files']]==[x.resolve() for x in EXTRAS[label]],'Hash-file coverage changed')
            for record in manifest['files']+manifest['extra_files']:
                actual=Path(record['resolved_path']);need(sha(actual)==record['sha256'] and actual.stat().st_size==record['bytes'],'Actual timed artifact changed: '+str(actual))
            if label=='g4':need(any(x['sha256']=='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83' for x in manifest['files']),'Wrong G4 calibration checkpoint')
            else:need(any(x['sha256']==A and Path(x['resolved_path']).resolve()==RAW for x in manifest['extra_files']),'Wrong HYPIR A checkpoint')
            if label.startswith('trt'):
                binding=read(md.parent/'candidate_manifest.json');need(binding['checkpoint_sha256']==A and binding['quality_reuse_claim'] is False,'Wrong HYPIR backend manifest')
                for key in ['runner','config']:
                    actual=md/(key+'.py' if key=='runner' else key+'.json');need(sha(actual)==binding[key+'_sha256'],'Backend source/config mismatch')
                for name,engine in binding['actual_engines'].items():
                    need(sha(engine['path'])==engine['sha256'] and sha(engine['manifest'])==engine['manifest_sha256'],'Timed engine provenance changed')
                need({str(Path(x['path']).resolve()) for x in binding['actual_engines'].values()}=={str(x.resolve()) for x in EXTRAS[label] if x.suffix=='.engine'},'Missing extra engine hash')
            timings[label]=dict(result=ref(f),actual_200_event_and_wall_recomputed=True,recomputed=recomputed,output=o,cuda=q['cuda'],artifacts=manifest)
        need(previous<=utc(evidence['final_snapshot']['utc']),'Final snapshot precedes jobs')
        need(sha(QUALITY)==QUALITY_SHA,'TRT full103/418 quality audit changed');quality=read(QUALITY)
        need(quality['status']=='PASS_FULL103_418' and len(quality['candidates'])==1,'TRT fresh full quality absent')
        parent=quality['parent'];candidate=quality['candidates'][0]
        need(parent['image_rows']==candidate['image_rows']==103 and parent['scalar_metrics']==candidate['scalar_metrics']==418 and
             Path(candidate['model_dir']).resolve()==MODELS['trtvae'],'Full quality scope/model mismatch')
        need(candidate['actual_checkpoint_sha256']==parent['actual_checkpoint_sha256']==A,'Full quality checkpoint mismatch')
        need(sha(MODELS['trtvae'].parent/'candidate_manifest.json')==candidate['candidate_manifest']['sha256'],'Quality/timing backend manifest differ')
        predictor=load_predict();r=timings['g4']['recomputed']['event_p10'];predictions={}
        for label,q in [('push6',parent),('trtvae',candidate)]:
            S=q['NR']+q['FR'];t=timings[label]['recomputed']['event_p10'];g=t*.431/r
            estimate=predictor.predict(S,g);direct=152.95*2**(S-4.0841)*((.49+2.17*.431)/(.49+2.17*g))**.2
            need(abs(estimate-direct)<1e-10,'Original predictor formula mismatch')
            predictions[label]=dict(NR=q['NR'],FR=q['FR'],S_proxy=S,event_p10_ms=t,normalized_g_ms=g,original_tool_prediction=estimate,reaches65=estimate>=65)
        result=dict(status='PASS_FORMAL4_TIMING_AND_2_QUALITY_PREDICTIONS',cpu_only=True,GPU_started=False,launcher=ref(LAUNCH),
          evidence=ref(D/'root_launch_evidence.json'),benchmark=ref(BENCH),predictor=ref(PRED),quality=ref(QUALITY),
          timings=timings,predictions=predictions,unetvae_prediction_available=False,unetvae_reason='No fresh full103/418 quality audit yet; timing only',
          normalization=dict(formula='event_p10_ms*0.431/new_same_batch_G4_event_p10_ms',G4_event_p10_ms=r),
          actual_event_samples=800,actual_wall_samples=800,physical_GPU0_UUID=identities[0][1],
          physical_continuity_scope='All actual before/after boundary snapshots plus fixed launcher CUDA_VISIBLE_DEVICES0/logical0 commands; no in-process UUID telemetry claimed',
          no_official_score_claim=True,no_current_model_pointer_changed=True)
    for path,(st,d) in SEEN.items():need(stamp(path)==st,'Source mutated before audit publish')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],out=str(out),sha256=sha(out),predictions=result.get('predictions'))))
if __name__=='__main__':main()

