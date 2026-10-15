"""CPU-only read audit of completed CUDA100 pilots; no models/metrics loaded."""
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import statistics
import datetime

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import numpy as np
torch.set_num_threads(4)
assert not torch.cuda.is_initialized()

R=Path('/cache/independent_sr_cuda_20261013')
RUNS=['cuda_nr_s6000_ct175_conservative_pilot100','cuda_nr_s6000_ct175_aligned_pilot100']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def delta(a,b):
    maximum=0.;ss=0.;abs_sum=0.;changed=0;count=0;tensors_changed=0
    for k,x in a.items():
        y=b[k];d=(x.float()-y.float());n=d.numel();count+=n
        c=int((d!=0).sum().item());changed+=c;tensors_changed+=int(c>0)
        maximum=max(maximum,float(d.abs().max().item()));ss+=float(d.double().square().sum().item())
        abs_sum+=float(d.double().abs().sum().item())
    return dict(elements=count,changed_elements=changed,changed_fraction=changed/count,
                tensors_changed=tensors_changed,max_abs=maximum,l2=math.sqrt(ss),mean_abs=abs_sum/count)


def state_digest(s):
    n=s['numpy_rng']
    return dict(python=hashlib.sha256(json.dumps(s['python_rng']).encode()).hexdigest(),
                numpy=hashlib.sha256(n[1].tobytes()+repr((n[0],n[2],n[3],n[4])).encode()).hexdigest(),
                cpu=hashlib.sha256(s['torch_rng'].numpy().tobytes()).hexdigest(),
                cuda=hashlib.sha256(s['cuda_rng'].numpy().tobytes()).hexdigest())


initial=R/'source_payload/adaptation/runs/ascend_balanced/step6000.pth'
raw=torch.load(initial,map_location='cpu',weights_only=True)
assert len(raw)==514 and all(x.dtype==torch.float32 and x.device.type=='cpu' for x in raw.values())
quant={k:x.to(torch.bfloat16).float() for k,x in raw.items()}
initial_quantization=delta(quant,raw)
results=[];logs=[];protocols=[];final_rng=[]
for name in RUNS:
    d=R/'adaptation/runs'/name
    cfg=json.loads((d/'config.json').read_text());p=cfg['protocol'];protocols.append(p)
    status=json.loads((d/'status.json').read_text());audit=json.loads((d/'gradient_preflight.json').read_text())
    rows=[json.loads(x) for x in (d/'train.jsonl').read_text().splitlines() if x.strip()];logs.append(rows)
    assert status['stage']=='complete' and len(rows)==100
    assert [x['local_step'] for x in rows]==list(range(1,101))
    assert all(x['nonzero_lora_gradients']==514 and math.isfinite(x['gradient_norm']) for x in rows)
    assert audit['status']=='passed' and audit['no_optimizer_update_yet'] and audit['frozen_metrics'] and audit['frozen_same_latent_anchor']
    assert all(x['input_gradient_norm']>0 and x['input_gradient_finite'] for x in audit['components'].values())
    path=d/'step6100.pth';final=torch.load(path,map_location='cpu',weights_only=True)
    assert set(final)==set(raw) and all(final[k].shape==raw[k].shape and final[k].dtype==torch.float32 for k in raw)
    deployed={k:x.to(torch.bfloat16).float() for k,x in final.items()}
    item=dict(run=name,status='passed',checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size,
              keys=514,relative_to_raw_FP32_total_delta=delta(final,raw),
              relative_to_actual_BF16_then_FP32_start_optimizer_delta=delta(final,quant),
              deployment_BF16_values_delta=delta(deployed,quant),
              first_gradient_audit=audit,completed_steps=100,
              all100_steps_514_nonzero_finite_lora_gradients=True,
              teacher_and_metrics_no_parameter_grad='All100 iterations passed explicit source checks before opt.step; completion proves no detected teacher/metric parameter gradient.',
              crop_sequence_sha256=hashlib.sha256(json.dumps([(x['case'],x['crop_xy']) for x in rows]).encode()).hexdigest(),
              steady_median_step_seconds=statistics.median(x['step_seconds'] for x in rows[1:]),
              elapsed100_seconds=rows[-1]['elapsed_seconds'],peak_allocated_bytes=max(x['gpu_peak_allocated_bytes'] for x in rows),
              training_config=p)
    assert item['relative_to_actual_BF16_then_FP32_start_optimizer_delta']['changed_elements']>0
    saved=torch.load(d/'resume_latest.pt',map_location='cpu',weights_only=False)
    assert saved['protocol_id']==cfg['protocol_id'] and saved['local_step']==100 and saved['global_step']==6100
    assert set(saved['lora'])==set(final) and all(torch.equal(saved['lora'][k],final[k]) for k in final)
    steps={int(x['step'].item()) for x in saved['optimizer']['state'].values()}
    assert steps=={100}
    item['resume_lora_equals_raw_FP32_saved_checkpoint']=True;item['adamw_state_all_step100']=True
    digest=state_digest(saved);final_rng.append(digest);item['final_rng_state_sha256']=digest
    results.append(item);del saved,final,deployed;gc.collect()

expected={'lr','reward_weight','clip_weight'}
different={k:[protocols[0].get(k),protocols[1].get(k)] for k in protocols[0].keys()|protocols[1].keys() if protocols[0].get(k)!=protocols[1].get(k)}
assert set(different)==expected
assert [(r['case'],r['crop_xy']) for r in logs[0]]==[(r['case'],r['crop_xy']) for r in logs[1]]
assert final_rng[0]==final_rng[1]
assert not torch.cuda.is_initialized()
report=dict(audited_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            cpu_only=True,cuda_context_initialized=False,initial_sha256=sha(initial),
            initial_load_precision='Raw FP32 checkpoint -> BF16 plain HYPIR load -> FP32 trainable parameters; same rounding for frozen current6000 anchor.',
            initial_FP32_to_BF16_rounding_delta=initial_quantization,
            branch_differences=different,cases_and_crop100_sequence_equal=True,final_all4_rng_states_equal=True,
            formal_eval='not yet established by this audit; train crop20 scores are not formal100-crop full-resolution NR',
            resume_review='New resume copies saved FP32 lora into already-FP32 current state after initialization, then loads new CUDA AdamW and restores all four RNG states; preserves FP32 optimization continuity. No NPU optimizer/RNG restore.',
            results=results)
print(json.dumps(report,ensure_ascii=False,indent=2))
