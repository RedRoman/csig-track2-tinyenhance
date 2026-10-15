"""Existing-engine stream/format diagnostic. CPU plan default; root schedules GPU.

Reuses sealed component.engine read-only. No rebuild, no quality/official speed
claim. Both v1/v2 timings include all work through stream dependencies/events.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import statistics
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def p10(v):
    v=sorted(v);p=(len(v)-1)*.1;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--component-manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',type=int,default=0);ap.add_argument('--warm',type=int,default=30);ap.add_argument('--n',type=int,default=50)
    a=ap.parse_args();need(not a.out.exists(),'Do not overwrite a prior diagnostic');need(1<=a.n<=200 and 0<=a.warm<=40,'Probe counts')
    m=json.loads(a.component_manifest.read_text());need(m['status']=='BUILT_COMPONENT_PROBED','Unsealed component')
    ep=Path(m['engine_path']);need(sha(ep)==m['engine_sha256'],'Engine changed')
    plan=dict(component=m['component'],precision=m['precision'],component_manifest=str(a.component_manifest),component_manifest_sha256=sha(a.component_manifest),engine_path=str(ep),engine_sha256=sha(ep),execute=a.execute,script_sha256=sha(__file__),runtime_v1_sha256=sha(Path(__file__).with_name('trt_vae_runtime.py')),runtime_v2_sha256=sha(Path(__file__).with_name('trt_vae_runtime_v2.py')),out=str(a.out))
    print(json.dumps(plan),flush=True)
    if not a.execute:return
    import torch
    import tensorrt as trt
    from trt_vae_runtime import TensorRTEngine as V1
    from trt_vae_runtime_v2 import TensorRTEngine as V2
    torch.cuda.set_device(a.device);device=torch.device(f'cuda:{a.device}')
    r1=V1(ep,m['engine_sha256'],device);r2=V2(ep,m['engine_sha256'],device)
    need(r1.input_shape==r2.input_shape and r1.output_shape==r2.output_shape,'IO shapes differ')
    count=1
    for d in r1.input_shape:count*=d
    def inp(offset):return ((torch.arange(count,device=device,dtype=torch.float32).reshape(r1.input_shape).add(offset).remainder(257)/128)-1).to(r1.input_dtype)
    numerical=[];held=[]
    with torch.no_grad():
        for offset in [0,17,103]:
            x=inp(offset);y1=r1(x);y2=r2(x);torch.cuda.synchronize(device)
            need(bool(torch.isfinite(y1).all()) and bool(torch.isfinite(y2).all()),'Nonfinite output')
            for y,s in held:need(torch.equal(y,s),'Returned output overwritten')
            delta=(y1.float()-y2.float()).abs();numerical.append(dict(offset=offset,exact=torch.equal(y1,y2),max_abs=float(delta.max()),mean_abs=float(delta.mean())))
            held.extend([(y1,y1.clone()),(y2,y2.clone())])
    # Round-trip stream events include async work, not just enqueue overhead.
    timings={};x=inp(0)
    with torch.no_grad():
        for label,fn in [('v1_caller_stream',r1),('v2_dedicated_nondefault_stream',r2)]:
            for _ in range(a.warm):y=fn(x)
            torch.cuda.synchronize(device);events=[];walls=[]
            for _ in range(a.n):
                b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);t=time.perf_counter();b.record();y=fn(x);e.record();torch.cuda.synchronize(device);events.append(b.elapsed_time(e));walls.append((time.perf_counter()-t)*1000)
            timings[label]=dict(warm=a.warm,n=a.n,event_p10_ms=p10(events),event_median_ms=statistics.median(events),wall_p10_ms=p10(walls),raw_event_ms=events,raw_wall_ms=walls)
    # Inspector may lack tactic/precision details: original build did not set
    # detailed profiling verbosity. Record availability rather than infer them.
    inspector=r2.engine.create_engine_inspector();inspector.execution_context=r2.context
    try:layers=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    except Exception as ex:layers=dict(unavailable=str(ex))
    class Profiler(trt.IProfiler):
        def __init__(self):trt.IProfiler.__init__(self);self.rows=[]
        def report_layer_time(self,layer_name,ms):self.rows.append(dict(name=layer_name,ms=float(ms)))
    prof=Profiler();r2.context.profiler=prof;r2.context.enqueue_emits_profile=True
    with torch.no_grad():y=r2(x);torch.cuda.synchronize(device)
    # Profiling is a separate untimed pass and can add synchronization.
    result=dict(**plan,status='STREAM_FORMAT_DIAGNOSTIC_COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,tensorrt=trt.__version__,gpu=torch.cuda.get_device_name(device),io_records=r2.io_records,caller_stream_handle=int(torch.cuda.current_stream(device).cuda_stream),worker_stream_handle=int(r2.stream.cuda_stream),same_engine_stream_numeric_comparisons=numerical,output_ownership_preserved=True,timings_diagnostic_only=timings,engine_num_layers=r2.engine.num_layers,engine_inspector=layers,untimed_layer_profile=prof.rows,untimed_profile_sum_ms=sum(r['ms'] for r in prof.rows),precision_or_tactic_inference='Only inspector-reported facts; original non-detailed engine may omit details',official_runner_benchmark=False,new_quality_not_measured=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],out=str(a.out),sha256=sha(a.out),timing_p10={k:v['event_p10_ms'] for k,v in timings.items()},stream_numeric=numerical,io_records=r2.io_records,layer_profile_sum_ms=result['untimed_profile_sum_ms'])),flush=True)


if __name__=='__main__':main()
