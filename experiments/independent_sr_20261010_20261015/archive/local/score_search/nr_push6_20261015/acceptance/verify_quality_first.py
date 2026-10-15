"""Independent CPU read-back of Push6 timing/prediction; never imports its join.

Default is a plan. --execute writes only a new Push6 audit JSON. No Torch/GPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys

sys.dont_write_bytecode = True
ROOT=Path('/cache/independent_sr_cuda_20261013')
CAL=ROOT/'calibration/push6_quality_first'
PARENT=ROOT/'nr_push4_full_evaluations/push4_a7750_compiled_all'
CAND=ROOT/'nr_push6_full_evaluations/push6_a7750_ro_wavelet_all'
PRED_SHA='0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
BENCH_SHA='3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
RAW_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
AUDIT_SHA='ae5b2234f5e110240507fbec515139d1b2a23bf8f664e7477e34c377e09c507b'
LABELS=['g4','parent','candidate']
ORDER=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
OBSERVED={}


def must(ok,message):
    if not ok: raise RuntimeError(message)


def stamp(path):
    s=Path(path).stat(); return s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev


def digest(path):
    p=Path(path).resolve(); before=stamp(p)
    if p in OBSERVED:
        must(OBSERVED[p][0]==before,'Changed source: '+str(p)); return OBSERVED[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            block=f.read(4*1024*1024)
            if not block: break
            h.update(block)
    must(stamp(p)==before,'Changed while reading: '+str(p)); OBSERVED[p]=(before,h.hexdigest()); return h.hexdigest()


def load(path):
    digest(path); return json.loads(Path(path).read_text())


def artifact(record):
    p=Path(record['path']); must(digest(p)==record['sha256'],'Reference SHA mismatch: '+str(p))
    if 'bytes' in record: must(p.stat().st_size==record['bytes'],'Reference size mismatch')
    return p


def means(directory,manifest,frozen=None):
    result=load(directory/'scores/result.json'); rows_path=directory/'scores/measurements.jsonl'; digest(rows_path)
    rows=[json.loads(s) for s in rows_path.read_text().splitlines() if s.strip()]
    must(len(rows)==103 and result['status']=='complete' and result['completed']==result['expected']==103
         and result['official_full_count'] is True,'Not full quality')
    if frozen: must(result==frozen['published_result'],'Frozen parent result changed')
    total=0
    for i,((split,case),row,image) in enumerate(zip(ORDER,rows,manifest['records'])):
        must((row['split'],row['case'])==('validation' if split=='val' else split,case)
             and row['sr_sha256']==image['sha256'],'Measurement/image differs')
        keys={'niqe','musiq','maniqa','clipiqa'}|({'lpips','dists'} if split=='val' else set())
        must(set(row['metrics'])==keys and all(math.isfinite(v) for v in row['metrics'].values()),'Wrong scalar metrics'); total+=len(keys)
        if split=='val':
            gt=ROOT/'source_payload/scoring_assets/data/验证集'/(f'case{case}_gt.jpg' if case==3 else f'case{case}_gt.png')
            must(digest(gt)==row['gt_sha256'],'GT differs')
            if frozen: must(row['gt_sha256']==frozen['records'][i]['actual_gt_sha256'],'Frozen GT differs')
    must(total==418,'Wrong scalar count')
    nr=statistics.fmean(r['metrics']['clipiqa'] for r in rows[3:])+statistics.fmean(r['metrics']['maniqa'] for r in rows[3:])+statistics.fmean(r['metrics']['musiq'] for r in rows[3:])/100+max(0,(10-statistics.fmean(r['metrics']['niqe'] for r in rows[3:]))/10)
    fr=2-statistics.fmean(r['metrics']['lpips'] for r in rows[:3])-statistics.fmean(r['metrics']['dists'] for r in rows[:3])
    must(abs(nr-result['test']['nr'])<1e-12 and abs(fr-result['validation']['fr'])<1e-12 and abs(nr+fr-result['S_proxy'])<1e-12,'Mean-first formula differs')
    return {'NR':nr,'FR':fr,'S_proxy':nr+fr}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--prediction',type=Path,default=CAL/'paired_prediction.json')
    p.add_argument('--out',type=Path,default=ROOT/'audits/push6/speed/paired_prediction_independent_cpu.json')
    p.add_argument('--execute',action='store_true'); a=p.parse_args()
    if not a.execute:
        print(json.dumps({'status':'CPU_PLAN_ONLY','prediction':str(a.prediction),'out':str(a.out),'GPU':False})); return
    must(a.prediction.resolve().is_relative_to(ROOT/'calibration') and a.out.resolve().is_relative_to(ROOT/'audits/push6'),'New Push6 paths required')
    x=load(a.prediction); must(x['schema']=='push6_quality_first_fresh_triplet_v1' and x['CPU_only'] is True,'Wrong join schema')
    artifact(x['source']); event={}; verified={}
    for label in LABELS:
        bound=x['timings'][label]; q=load(artifact(bound['artifact']))
        must(q['status']=='complete' and q['seed']==231 and q['latency_512']['n']==200 and q['latency_512']['warm']==30,'Wrong timing protocol')
        must(q['input']['shape']==[1,3,512,512] and q['input']['device']=='cuda:0' and q['input']['dtype']=='torch.float16','Wrong input')
        y=q['output']; must(y['device']=='cuda:0' and y['shape']==[1,3,512,512] and y['all_finite'] is True
             and y['within_minus1_plus1'] is True and -1<=y['min']<=y['max']<=1,'Invalid timing output')
        must(q['benchmark_script']['sha256']==BENCH_SHA and digest(ROOT/'code/benchmark_cuda.py')==BENCH_SHA,'Benchmark source mismatch')
        for rec in q['artifacts']['files']+q['artifacts']['extra_files']:
            path=Path(rec['resolved_path']); must(digest(path)==rec['sha256'] and path.stat().st_size==rec['bytes'],'Actual timing artifact differs')
        must(Path(q['model_dir']).resolve()==Path(bound['model_dir']).resolve()
             and digest(Path(q['model_dir'])/'runner.py')==bound['runner_sha256'],'Runner binding differs')
        for kind in ['event','wall']:
            values=q['raw_samples'][kind+'_ms']; must(len(values)==200 and all(math.isfinite(t) and t>0 for t in values),'Raw200 invalid')
            nums={'p10':statistics.quantiles(values,n=10,method='inclusive')[0],'med':statistics.median(values),'min':min(values)}
            for suffix,v in nums.items():
                k=kind+'_'+suffix; must(abs(v-q['latency_512'][k])<1e-9 and abs(v-bound['recomputed'][k])<1e-9,'Independent raw200 recompute differs')
        event[label]=q['latency_512']['event_p10']; verified[label]=dict(path=str(bound['artifact']['path']),sha256=digest(bound['artifact']['path']),p10=event[label],raw200_recomputed=True)
    physical=x['launch']; launch=load(artifact(physical['root_evidence']))
    must(launch['serial'] is True and launch['order']==LABELS and launch['other_root_gpu_jobs_active_during_pair'] is False,'Launch isolation differs')
    for index,command in enumerate(launch['commands']):
        must(command['label']==LABELS[index] and command['order']==index+1 and command['exit_code']==0 and command['separate_fresh_process'] is True,'Launch sequence/exit differs')
        argv=shlex.split(command['command']); must(argv[0]=='CUDA_VISIBLE_DEVICES=0' and argv[argv.index('--device')+1]=='0','Physical/logical device selection differs')
        must(Path(argv[argv.index('--out')+1]).resolve()==Path(verified[LABELS[index]]['path']).resolve(),'Launch JSON differs')
    snapshots=[artifact(r) for r in physical['snapshots']]
    must(snapshots[0].read_text()==snapshots[1].read_text() and all(not p.read_text().strip() for p in snapshots[2:]),'Candidate physical/process snapshots differ')
    reduced=1-event['candidate']/event['parent']; must(abs(reduced-x['speed_only']['latency_reduction_fraction'])<1e-12,'Speed reduction differs')
    result=dict(schema='push6_independent_paired_CPU_verification_v1',status='PASS_TIMING_ONLY_PENDING_QUALITY',CPU_only=True,
                verified_utc=datetime.now(timezone.utc).isoformat(),prediction_path=str(a.prediction),prediction_sha256=digest(a.prediction),
                verifier_source_sha256=digest(__file__),timings=verified,launch_evidence_sha256=digest(physical['root_evidence']['path']),
                latency_reduction_fraction=reduced,candidate_prediction_available=False)
    if x['status']=='PENDING_QUALITY':
        must(x['candidate_prediction_available'] is False and x['candidate_quality_confirmed'] is False
             and 'predictions' not in x and 'quality' not in x,'Pending result prematurely claims quality/prediction')
    else:
        must(x['status']=='PASS' and x['candidate_quality_confirmed'] is True,'Unknown final status')
        quality=x['quality']; proof=load(artifact(quality['independent_proof']))
        frozen_path=ROOT/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json'
        must(digest(frozen_path)==AUDIT_SHA,'Frozen parent audit SHA differs'); frozen=load(frozen_path)
        manifests={}
        for label, directory in [('parent',PARENT),('candidate',CAND)]:
            m=load(directory/'render_manifest.json'); artifact(quality[label+'_manifest'])
            must((directory/'render_complete').is_file() and len(m['records'])==len(m['protocol']['tasks'])==103,'Formal render incomplete')
            must(m['protocol_id']==hashlib.sha256(json.dumps(m['protocol'],sort_keys=True).encode()).hexdigest(),'Formal manifest identity differs')
            must(m['protocol']['runner']['runner_sha256']==x['timings'][label]['runner_sha256'] and m['protocol']['runner']['checkpoint_sha256']==RAW_SHA,'Formal/timing model differs')
            for i,((split,case),row,task) in enumerate(zip(ORDER,m['records'],m['protocol']['tasks'])):
                must((row['split'],row['case'])==(split,case) and row['output']==f'render/{split}/case{case}.png','Formal order differs')
                must(digest(directory/row['output'])==row['sha256'] and digest(task['input'])==task['input_sha256'],'Actual formal PNG/input differs')
                if label=='parent': must(row['sha256']==frozen['records'][i]['actual_png_sha256'] and task['input_sha256']==frozen['records'][i]['actual_input_sha256'],'Frozen parent records differ')
            manifests[label]=m
        must(manifests['parent']['protocol']['tasks']==manifests['candidate']['protocol']['tasks'],'Formal inputs differ')
        if quality['mode']=='exact103':
            must(proof['schema']=='push6_formal_png103_comparison_v2' and proof['status']=='PASS_EXACT103'
                 and proof['PNG_exact_count']==103 and proof['final_fourRNG_equal'] is True and proof['metric_input_reuse_eligible'] is True,'Exact103 independent proof incomplete')
            must(proof['parent_final_fourRNG']==proof['candidate_final_fourRNG'],'Exact final RNG differs')
            must(all(a['sha256']==b['sha256'] for a,b in zip(manifests['parent']['records'],manifests['candidate']['records'])),'Candidate PNG103 differs')
            scores={'parent':means(PARENT,manifests['parent'],frozen)}; scores['candidate']=dict(scores['parent'])
        else:
            must(proof['status']=='PASS' and proof['image_rows']==103 and proof['scalar_metrics']==418,'Rescored independent proof incomplete')
            scores={'parent':means(PARENT,manifests['parent'],frozen),'candidate':means(CAND,manifests['candidate'])}
        predpath=artifact(x['predict_source']); must(digest(predpath)==PRED_SHA,'Original predictor changed')
        spec=importlib.util.spec_from_file_location('push6_independent_original_predict',predpath); original=importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
        numbers={}
        for label in ['parent','candidate']:
            s=scores[label]; g=event[label]*.431/event['g4']; prediction=original.predict(s['S_proxy'],g)
            for key in ['NR','FR','S_proxy']: must(abs(s[key]-x['predictions'][label][key])<1e-12,'Independent quality recompute differs')
            must(abs(prediction-x['predictions'][label]['original_tool_prediction'])<1e-12 and abs(g-x['predictions'][label]['normalized_g_ms'])<1e-12,'Original prediction/normalization differs')
            numbers[label]=prediction
        factor=2**(scores['candidate']['S_proxy']-scores['parent']['S_proxy']); gA=event['parent']*.431/event['g4']; gN=event['candidate']*.431/event['g4']; speed=((.49+2.17*gA)/(.49+2.17*gN))**.2
        must(abs(factor-x['quality_factor'])<1e-12 and abs(speed-x['speed_factor'])<1e-12
             and abs(factor*speed-numbers['candidate']/numbers['parent'])<1e-12,'Quality/speed factor decomposition differs')
        must(x['reaches_target']==(numbers['candidate']>=70),'Target flag differs')
        result.update(status='PASS',candidate_prediction_available=True,predictions=numbers,quality_mode=quality['mode'],
                      quality_proof_sha256=quality['independent_proof']['sha256'],actual_PNG103_each_cohort_rehashed=True,
                      target70=70,reaches_target=numbers['candidate']>=70,quality_factor=factor,speed_factor=speed)
    for path,(s,h) in OBSERVED.items(): must(stamp(path)==s,'Source changed before audit publish')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(a.out)+'.tmp'); temp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); os.replace(temp,a.out)
    print(json.dumps({'status':result['status'],'out':str(a.out),'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}))


if __name__=='__main__': main()
