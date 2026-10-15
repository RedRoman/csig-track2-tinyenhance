"""CPU-only formula inversion plus supplied G4 asset/active-protocol review.

No model inference, GPU initialization, score editing, old writes, or big copies.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import math
import os

os.environ['CUDA_VISIBLE_DEVICES']=''


def need(ok,message):
    if not ok:raise RuntimeError(message)


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root
    predictor=r/'scoring_tool/predict.py';need(sha(predictor)=='0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1','Original predictor differs')
    p=load_module('unmodified_predictor_for_cpu_math',predictor)
    pair=read(r/'calibration/push4_paired_prediction.json')
    S=pair['selected_quality']['S'];t=pair['selected_event_p10_ms'];ref=pair['G4_event_p10_ms'];target=70
    need(abs(S-3.7614565663241937)<1e-12,'Frozen best S differs')
    a,k,anc=p.A,p.K,p.ANCHOR
    prediction=lambda ss,ms,rf:p.predict(ss,ms*anc['g']/rf)
    needed_S=lambda ms,rf:anc['S']+math.log2(target/anc['score'])+.2*math.log2((a+k*ms*anc['g']/rf)/(a+k*anc['g']))
    max_ms=lambda ss,rf:(((a+k*anc['g'])*(anc['score']*2**(ss-anc['S'])/target)**5-a)/k)*rf/anc['g']
    hist=[]
    for name in ['g4_latency.json','push2_g4_latency.json','push3_g4_latency.json','push4_g4_latency.json']:
        path=r/'calibration'/name;b=read(path);rf=b['latency_512']['event_p10']
        hist.append({'source':str(path),'source_sha256':sha(path),'reference_p10_ms':rf,
                     'reference_min_ms':b['latency_512']['event_min'],'reference_median_ms':b['latency_512']['event_med'],
                     'hypothetical_same_frozen_S_and_t_prediction':prediction(S,t,rf),
                     'target70_max_ms_at_frozen_S':max_ms(S,rf),
                     'scope':'Calibration sensitivity holding candidate S/time fixed; not a new paired score.'})
    matrix=[]
    for ms in [1,5,10,12,15,16,18,20,22,25,30,40,t]:
        ss=needed_S(ms,ref);need(abs(prediction(ss,ms,ref)-target)<1e-10,'Inversion check')
        matrix.append({'single_GPU512_event_p10_ms':ms,'required_full_S_for70':ss,'delta_S_vs_best':ss-S,
                       'prediction_at_unchanged_best_S':prediction(S,ms,ref)})
    need(abs(prediction(S,max_ms(S,ref),ref)-target)<1e-10,'Latency inversion check')
    md=r/'submissions/push5_g4_reference/model_dir';cal=r/'calibration/g4/model_dir'
    pins={'model.pth':'283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83',
          'reference_runner.py':'6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e',
          'pix_model.py':'5b1ba75929d25c2663ec5aba5f7cd4d4a56b79f214dd8e7a49e48c34eca50f5f',
          'config.yaml':'e13bab32185ac95f07ac36133eff89133a33ed90cc5ca534b7d8496bc457250d',
          'runner.py':'276da6fe32a8e67f595f6ecf8a95b4b3ad2d87ec2beb0ca9b8fb1822777ff972'}
    for name,digest in pins.items():need(sha(md/name)==digest,'G4 reference artifact differs: '+name)
    for name in ['model.pth','pix_model.py','config.yaml']:need(sha(md/name)==sha(cal/name),'Reference/calibration asset differs')
    need(sha(md/'reference_runner.py')==sha(cal/'runner.py'),'Supplied original runner changed')
    tree=ast.parse((md/'runner.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runner')
    infer=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='infer')
    need(len(infer.body)==1 and isinstance(infer.body[0],ast.Return) and
         ast.unparse(infer.body[0].value)=='self.impl.infer(x)','Adapter changes inference arithmetic')
    import torch
    torch.set_num_threads(2);need(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    cp=torch.load(md/'model.pth',map_location='cpu',weights_only=False);state=cp['state']
    pm=load_module('g4_asset_strict_CPU_check',md/'pix_model.py')
    net=pm.build_pix(md/'model.pth',device='cpu',dtype=torch.float32);fold=pm.fold_inference(net)
    need(len(state)==100 and all(v.dtype==torch.float32 and torch.isfinite(v).all().item() for v in state.values()),'G4 state invalid')
    asset={'status':'PASS','files_sha256':pins,'cfg':cp['cfg'],'historical_training_step':cp['step'],
           'state_tensors':len(state),'FP32_parameters':sum(v.numel() for v in state.values()),
           'folded_parameters':sum(v.numel() for v in fold.parameters()),'all_finite':True,
           'outc_weight_norm':state['outc.weight'].norm().item(),'strict_CPU_load_and_fold_PASS':True,
           'reference_replication_not_new_training':True,'adapter_infer_delegates_original_unchanged':True}
    new=r/'nr_push5_full_evaluations/push5_g4_reference_full';old=r/'nr_push4_full_evaluations/push4_a7750_compiled_all'
    rp,op=read(new/'render_protocol.json'),read(old/'render_protocol.json')
    need(hashlib.sha256(json.dumps(rp['protocol'],sort_keys=True).encode()).hexdigest()==rp['protocol_id'],'New render protocol digest')
    need(rp['protocol']['tasks']==op['protocol']['tasks'],'Inputs or input order differ from frozen fullA')
    for row in rp['protocol']['tasks']:need(sha(row['input'])==row['input_sha256'],'Actual input differs')
    sp,osp=read(new/'scores/protocol.json'),read(old/'scores/protocol.json')
    changed=sorted(k for k in set(sp)|set(osp) if sp.get(k)!=osp.get(k))
    need(set(changed)=={'label','render_dir','cuda'},'Metric protocol changed beyond label/path/device')
    need(sp['sequence']==[['validation',i] for i in range(1,4)]+[['test',i] for i in range(1,101)],'Full103 score order differs')
    for row in sp['original_sources']:need(sha(row['resolved_path'])==row['sha256'],'Original metric source differs')
    need(sha(r/'code/score_cuda_stream.py')==sp['script_sha256'],'Stream scorer source changed')
    need(rp['protocol']['source_sha256']==op['protocol']['source_sha256'] and
         sha(r/'code/render_cuda_submission.py')==rp['protocol']['source_sha256'],'Original renderer source differs')
    oldaudit=read(r/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json')
    gt_names=['case1_gt.png','case2_gt.png','case3_gt.jpg'];gt=[]
    for i,name in enumerate(gt_names,1):
        path=r/'source_payload/scoring_assets/data/验证集'/name;digest=sha(path)
        oldrec=next(x for x in oldaudit['records'] if x['split']=='validation' and x['case']==i)
        need(digest==oldrec['actual_gt_sha256'],'Original GT changed');gt.append({'case':i,'sha256':digest})
    protocol={'status':'PASS_static_and_actual_inputs','full_output_completion_not_yet_claimed':True,
              'render_protocol_id':rp['protocol_id'],'render_protocol_sha256':sha(new/'render_protocol.json'),
              'score_protocol_sha256':sha(new/'scores/protocol.json'),'actual103_input_SHA_verified_and_same_order':True,
              'actual3_GT_SHA_verified_unchanged':gt,'score_protocol_only_changes_vs_frozen_A':changed,
              'original_renderer_and_scorer_and_metric_sources_SHA_verified':True,
              'image_policy':sp['image_policy'],'seed_scope':sp['seed_scope'],'metric_order':sp['metric_order']}
    need(not torch.cuda.is_initialized(),'CUDA initialized at audit end')
    result={'schema':'push5_target70_formula_and_G4_CPU_review_v1','status':'PASS','CPU_only':True,
            'CUDA_context_initialized':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'original_predictor_SHA256':sha(predictor),'empirical_prediction_not_official_score':True,
            'current_best':{'S':S,'single_GPU512_p10_ms':t,'G4_ref_ms':ref,'prediction':prediction(S,t,ref)},
            'target':target,'fixed_S_max_p10_ms_for70':max_ms(S,ref),'required_speedup_at_fixed_S':t/max_ms(S,ref),
            'fixed_p10_required_S_for70':needed_S(t,ref),'fixed_p10_required_delta_S':needed_S(t,ref)-S,
            'tradeoff_matrix_current_reference':matrix,'historical_G4_calibration_sensitivity':hist,
            'G4_same_speed_reference_required_S_for70':needed_S(ref,ref),
            'G4_README_declared_quality_only_not_remeasured_here':{'NR':2.7620,'FR':1.3221,'S':4.0841,
                'conditional_prediction_at_reference_speed':prediction(4.0841,ref,ref)},
            'G4_asset_and_adapter_CPU_review':asset,'active_reference_full_protocol_review':protocol}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
