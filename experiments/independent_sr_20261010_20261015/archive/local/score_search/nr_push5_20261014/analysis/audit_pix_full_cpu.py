"""Small independent native103/418 audit; no HYPIR-specific model schema."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import struct


def need(ok,message):
    if not ok:raise RuntimeError(message)


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256();s=Path(p).stat();before=(s.st_size,s.st_mtime_ns,s.st_ino)
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    s=Path(p).stat();need((s.st_size,s.st_mtime_ns,s.st_ino)==before,'File changed during SHA')
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--expected-checkpoint-sha',required=True);ap.add_argument('--reference',action='store_true')
    args=ap.parse_args();R=args.root;D=args.directory
    need(D.resolve().is_relative_to(R.resolve()),'Full dir outside experiment')
    need((D/'render_complete').is_file(),'Renderer not complete')
    paths=[D/'render_manifest.json',D/'render_protocol.json',D/'scores/result.json',D/'scores/protocol.json',D/'scores/measurements.jsonl']
    heads={str(p):sha(p) for p in paths};m,p,result,sp=map(read,paths[:4])
    protocol=p['protocol'];pid=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    need(m['protocol']==protocol and m['protocol_id']==p['protocol_id']==pid,'Render protocol digest differs')
    runner=protocol['runner'];need(runner['checkpoint_sha256']==args.expected_checkpoint_sha and
        sha(runner['weight_path'])==args.expected_checkpoint_sha,'Actual checkpoint differs')
    md=Path(protocol['model_dir']);need(sha(md/'runner.py')==runner['runner_sha256'],'Actual adapter source differs')
    for name,digest in runner['source_sha256'].items():need(sha(md/name)==digest,'Actual model source differs')
    need(sha(R/'code/render_cuda_submission.py')==protocol['source_sha256'],'Renderer source changed')
    need(sha(R/'code/score_cuda_stream.py')==sp['script_sha256'],'Scorer source changed')
    for row in sp['original_sources']:need(sha(row['resolved_path'])==row['sha256'],'Original metric source changed')
    need(result['label']==args.label and result['status']=='complete' and
        result['completed']==result['expected']==103 and result['official_full_count'] is True,'Full score incomplete')
    order=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
    need([(r['split'],r['case']) for r in m['records']]==order and
         [(r['split'],r['case']) for r in protocol['tasks']]==order,'Native103 task order differs')
    rows=[json.loads(x) for x in paths[4].read_text().splitlines() if x.strip()]
    score_order=[('validation' if s=='val' else s,i) for s,i in order]
    need([(r['split'],r['case']) for r in rows]==score_order and sp['sequence']==[list(x) for x in score_order],
         'Full score ordering differs')
    need(len({r['output'] for r in m['records']})==103,'Duplicate rendered output paths')
    gt_names=['case1_gt.png','case2_gt.png','case3_gt.jpg'];checked=[]
    for idx,(rec,task,row) in enumerate(zip(m['records'],protocol['tasks'],rows),1):
        out=(D/rec['output']).resolve();need(out.is_relative_to(D.resolve()),'Output escaped full dir')
        png_sha=sha(out);input_sha=sha(task['input'])
        need(png_sha==rec['sha256']==row['sr_sha256'] and input_sha==task['input_sha256'],'Actual PNG/input SHA differs')
        need(Path(row['sr_path']).resolve()==out and row['label']==args.label,'Scored source path/label differs')
        with out.open('rb') as f:header=f.read(24)
        need(header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR','Non-PNG native output')
        size=list(struct.unpack('>II',header[16:24]));need(size==rec['size']==task['size'],'Native dimensions differ')
        gt=None
        if rec['split']=='val':
            gt=sha(R/'source_payload/scoring_assets/data/验证集'/gt_names[rec['case']-1]);need(gt==row['gt_sha256'],'Actual GT SHA differs')
        checked.append([rec['split'],rec['case'],png_sha,input_sha,gt])
        if idx%25==0:print(json.dumps({'actual_native_PNG_input_SHA_verified':idx}),flush=True)
    means={};scalars=0;maxdiff=0
    for split,n,keys in [('validation',3,['lpips','dists','niqe','musiq','maniqa','clipiqa']),
                         ('test',100,['niqe','musiq','maniqa','clipiqa'])]:
        subset=[r for r in rows if r['split']==split]
        need(len(subset)==n and all(set(r['metrics'])==set(keys) for r in subset),'Metric count/key set differs')
        means[split]={}
        for key in keys:
            values=[r['metrics'][key] for r in subset];need(all(math.isfinite(v) for v in values),'Nonfinite metric')
            value=sum(values)/n;means[split][key]=value;diff=abs(value-result[split][key]);maxdiff=max(maxdiff,diff)
            need(diff<=1e-12,'Published component differs from actual row mean')
        scalars+=n*len(keys)
    need(scalars==418,'Expected418 scalars')
    t,v=means['test'],means['validation'];nr=t['clipiqa']+t['maniqa']+t['musiq']/100+max(0,(10-t['niqe'])/10)
    fr=2-v['lpips']-v['dists'];need(abs(nr-result['test']['nr'])<=1e-12 and abs(fr-result['validation']['fr'])<=1e-12 and
        abs(nr+fr-result['S_proxy'])<=1e-12,'Published aggregate differs')
    # Same original metric assets as the already frozen previous full audit.
    prior=read(R/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json');weights={}
    for name,record in prior['actual_metric_weights_sha256'].items():
        digest=sha(R/'scoring_tool/weights'/name);need(digest==record['sha256'],'Actual metric weight changed')
        weights[name]=digest
    need(all(sha(path)==digest for path,digest in heads.items()),'Committed full metadata changed during audit')
    report={'schema':'push5_independent_pixel_full_native103_418_CPU_audit_v1','status':'PASS',
        'CPU_only':True,'GPU_used':False,'audited_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'label':args.label,'evaluation_dir':str(D),'provenance':'supplied reference replication, no new training' if args.reference else 'new trained student',
        'image_rows':103,'scalar_metrics':418,'actual103_PNG_SHA_size_scored_manifest_PASS':True,
        'actual103_input_SHA_and_order_PASS':True,'actual3_GT_SHA_PASS':True,
        'all_metric_values_finite':True,'component_means_recomputed':means,'max_published_component_delta':maxdiff,
        'NR':nr,'FR':fr,'S_proxy':nr+fr,'actual_checkpoint_sha256':args.expected_checkpoint_sha,
        'actual_runner_sha256':runner['runner_sha256'],'original_renderer_scorer_metric_sources_PASS':True,
        'actual_metric_weights_sha256':weights,'render_protocol_id':pid,'metadata_file_SHA256':heads,
        'all_actual_PNG_input_GT_records_digest_SHA256':hashlib.sha256(json.dumps(checked,sort_keys=True).encode()).hexdigest(),
        'score_seed':sp['seed'],'score_seed_scope':sp['seed_scope'],'metric_order':sp['metric_order'],
        'image_policy':sp['image_policy'],'not_official_leaderboard_score':True,
        'prediction_not_computed_without_fresh_same_card_latency':True}
    print('AUDIT_JSON_BEGIN',flush=True);print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
