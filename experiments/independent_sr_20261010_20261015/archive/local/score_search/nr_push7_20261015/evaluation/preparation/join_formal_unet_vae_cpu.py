"""CPU-only join of fresh UNet-VAE full quality with the already fixed formal batch.
No new benchmark, no old output overwrite, no model preference writes.
"""
import argparse,hashlib,importlib.util,json,math,os
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
T=R/'audits/push7/speed/formal4_independent_cpu.json'
TS='e3db1f53c9b316b982a9b8e7fb5e740ef948e4174a2ca7718b659a6696020275'
Q=R/'audits/push7/hypir_full/trt_unet_vae_fp16_all.json'
QSOURCE='cfdaf826ea9fe36662141bf248af2df6e000d8b1eca61fcb3521cb8a04715b0d'
MD=R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16/model_dir'
P=R/'scoring_tool/predict.py'
PS='0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
SEEN={}
def need(x,m):
    if not x:raise RuntimeError(m)
def sha(p):
    p=Path(p).resolve();s=p.stat();st=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    t=p.stat();need(st==(t.st_size,t.st_mtime_ns,t.st_ino,t.st_dev),'Source changed during read')
    SEEN[p]=(st,h);return h
def read(p):
    sha(p);return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def pct(v,q):
    a=sorted(v);f=(len(a)-1)*q;i=math.floor(f);j=math.ceil(f)
    return a[i]+(a[j]-a[i])*(f-i)
def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--execute',action='store_true')
    q.add_argument('--out',type=Path,default=R/'audits/push7/speed/formal4_unet_vae_quality_join.json');a=q.parse_args()
    out=a.out.resolve();need(out.is_relative_to(R/'audits/push7') and not out.exists(),'Use new immutable join output')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',quality=str(Q),fixed_timings=str(T),out=str(out),GPU=False,new_benchmark=False)));return
    os.environ['CUDA_VISIBLE_DEVICES']=''
    need(sha(T)==TS,'Fixed formal timing audit changed');timing=read(T)
    need(timing['status']=='PASS_FORMAL4_TIMING_AND_2_QUALITY_PREDICTIONS' and timing['actual_event_samples']==timing['actual_wall_samples']==800,'Formal batch not complete')
    quality=read(Q);need(quality['schema']=='push7_actual_native_hypir_trt_unet_vae_full_v1' and quality['status']=='PASS_FULL103_418' and
      quality['auditor']['sha256']==sha(quality['auditor']['path'])==QSOURCE and len(quality['candidates'])==1,'Fresh full103/418 quality audit missing')
    c=quality['candidates'][0];parent=quality['parent'];need(Path(c['model_dir']).resolve()==MD and
      c['image_rows']==parent['image_rows']==103 and c['scalar_metrics']==parent['scalar_metrics']==418,'Wrong full quality cohort/model')
    need(c['official_full_count'] is True and c['actual_checkpoint_sha256']==parent['actual_checkpoint_sha256']=='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d','Wrong A/HYPIR identity')
    need(sha(c['candidate_manifest']['path'])==c['candidate_manifest']['sha256'] and
      sha(c['config']['path'])==c['config']['sha256']=='63e227319f2c2fb327257a810f6709b4f1d738732571f6c100e22b75174694e5','Quality model metadata changed')
    samples={}
    for label,record in timing['timings'].items():
        need(sha(record['result']['path'])==record['result']['sha256'],'Fixed actual timing JSON changed')
        data=read(record['result']['path']);stats=data['latency_512'];need(stats['n']==200 and stats['warm']==30,'Wrong original n/warm')
        for kind in ['event','wall']:
            v=data['raw_samples'][kind+'_ms'];need(len(v)==200 and all(math.isfinite(x) and x>0 for x in v),'Actual timing samples invalid')
            for suffix,value in [('p10',pct(v,.1)),('med',pct(v,.5)),('min',min(v))]:
                need(abs(stats[kind+'_'+suffix]-value)<1e-10,'Original raw statistics drifted')
        samples[label]=data
    unet=samples['trtunetvae'];need(Path(unet['model_dir']).resolve()==MD,'Timed UNet model differs')
    artifacts={Path(x['resolved_path']).resolve():x for x in unet['artifacts']['files']+unet['artifacts']['extra_files']}
    for name,digest in [('runner.py','270d2434179d17658e9f1df228c8ef99124a2695141c92f0b5fecfa73d8d3080'),
                        ('config.json','63e227319f2c2fb327257a810f6709b4f1d738732571f6c100e22b75174694e5')]:
        need(artifacts[MD/name]['sha256']==sha(MD/name)==digest,'Fresh quality and timed source/config differ')
    for engine in c['registered_backend_provenance']['actual_TRT_engines'].values():
        r=artifacts[Path(engine['path']).resolve()]
        need(r['sha256']==engine['sha256'] and r['bytes']==engine['bytes'],'Fresh quality and timed engine bytes differ')
    for key in ['NR','FR','S_proxy']:need(abs(parent[key]-timing['predictions']['push6'][key])<1e-12,'Parent full quality differs from same batch')
    need(sha(P)==PS,'Original predictor changed')
    spec=importlib.util.spec_from_file_location('original_predict_unet_full',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    need(m.A==.49 and m.K==2.17 and m.ANCHOR==dict(score=152.95,S=4.0841,g=.431),'Original predictor constants differ')
    reference=samples['g4']['latency_512']['event_p10'];t=unet['latency_512']['event_p10'];g=t*.431/reference;S=c['NR']+c['FR'];estimate=m.predict(S,g)
    explicit=152.95*2**(S-4.0841)*((.49+2.17*.431)/(.49+2.17*g))**.2
    need(abs(estimate-explicit)<1e-10,'Original prediction formula mismatch')
    predictions=dict(timing['predictions']);predictions['trtunetvae']=dict(NR=c['NR'],FR=c['FR'],S_proxy=S,event_p10_ms=t,
      normalized_g_ms=g,original_tool_prediction=estimate,reaches65=estimate>=65)
    result=dict(status='PASS_FIXED_FORMAL_BATCH_WITH_FRESH_UNET_VAE_FULL',cpu_only=True,GPU_started=False,new_benchmark=False,
      formal_timing_audit=ref(T),fresh_full_quality_audit=ref(Q),auditor=ref(__file__),predictor=ref(P),
      predictions=predictions,G4_event_p10_ms=reference,normalization='t*0.431/fixed_same_batch_G4_event_p10',
      actual_raw_event_and_wall_samples_recomputed=800,quality_reused_for_UNetVAE=False,
      big_weight_or_engine_bytes_rehashed_in_this_join=False,engine_identity_scope='Equal actual SHA/bytes in fixed timing audit and fresh full quality audit',
      no_current_preference_changed=True,no_official_score_claim=True)
    for p,(st,h) in SEEN.items():
        s=p.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed before publish')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],out=str(out),sha256=sha(out),predictions=predictions)))
if __name__=='__main__':main()

