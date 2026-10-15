"""Read-only stdlib/Pillow/NumPy audit; does not import Torch or use GPU."""
from pathlib import Path
import json,hashlib,math,datetime
from PIL import Image
import numpy as np
R=Path('/cache/independent_sr_cuda_20261013')
LABELS=['s6000_ct175_compiled_control_pilot','nr_conservative_s6100_compiled_pilot','nr_aligned_s6100_compiled_pilot']
CASES=[1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100]
NR=['niqe','musiq','maniqa','clipiqa'];VK=['lpips','dists']+NR
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
results=[];render=[];score=[]
for name in LABELS:
    d=R/'nr_pilot_evaluations'/name;res=json.loads((d/'scores/result.json').read_text())
    rp=json.loads((d/'render_protocol.json').read_text());p=rp['protocol'];render.append(p)
    sp=json.loads((d/'scores/protocol.json').read_text());score.append(sp)
    assert hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()==rp['protocol_id']
    rows=[json.loads(x) for x in (d/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
    assert res['status']=='complete' and res['completed']==res['expected']==23 and not res['official_full_count']
    assert len(rows)==23 and sum(len(x['metrics']) for x in rows)==98
    assert [(x['split'],x['case']) for x in rows]==[('validation',i) for i in range(1,4)]+[('test',i) for i in CASES]
    m=json.loads((d/'render_manifest.json').read_text());assert len(m['records'])==23
    for row,rec,inp in zip(rows,m['records'],p['tasks']):
        assert list(row['metrics'])==(VK if row['split']=='validation' else NR)
        assert all(math.isfinite(x) for x in row['metrics'].values())
        assert (row['split'].replace('validation','val'),row['case'])==(rec['split'],rec['case'])==(inp['split'],inp['case'])
        assert row['sr_sha256']==rec['sha256']==sha(d/rec['output'])
        assert row['size_hw']==rec['size'][::-1]==inp['size'][::-1]
        assert sha(Path(inp['input']))==inp['input_sha256']
        if row['split']=='validation':
            gt=R/'scoring_tool/data/验证集'/('case3_gt.jpg' if row['case']==3 else f'case{row["case"]}_gt.png')
            assert row['gt_sha256']==sha(gt)
    means={}
    for split,keys in [('test',NR),('validation',VK)]:
        subset=[x for x in rows if x['split']==split]
        means[split]={k:sum(x['metrics'][k] for x in subset)/len(subset) for k in keys}
    nr=lambda z:z['clipiqa']+z['maniqa']+z['musiq']/100+max(0,(10-z['niqe'])/10)
    a,b=means['test'],means['validation'];N=nr(a);F=2-b['lpips']-b['dists'];S=N+F
    ds=[abs(res['test'][k]-v) for k,v in a.items()]+[abs(res['validation'][k]-v) for k,v in b.items()]
    ds+=[abs(res['test']['nr']-N),abs(res['validation']['fr']-F),abs(res['S_proxy']-S)]
    assert max(ds)<1e-12
    results.append(dict(label=name,image_rows=23,scalar_metrics=98,NR=N,FR=F,S_subset=S,max_formula_delta=max(ds),
                        actual_png_hash='PASS23',actual_input_hash='PASS23',actual_validation_GT_hash='PASS3',
                        component_means=means,checkpoint_sha256=p['runner']['checkpoint_sha256']))
assert all(p['tasks']==render[0]['tasks'] for p in render)
RK=['model_t','coeff_t','lora_rank','patch','stride','scale','caption','latent','fusion','runner_sha256','source_sha256','numerical_flags','versions','seed','compilation']
for p in render[1:]:assert all(p['runner'][k]==render[0]['runner'][k] for k in RK)
SK=['seed','seed_scope','metric_order','environment_versions','environment_protocol_sha256','original_tool_deployment_sha256','original_sources','script_sha256','image_policy','numpy_legacy_compat','maniqa_offline_init_proof']
for p in score[1:]:assert all(p[k]==score[0][k] for k in SK)
old=R/'evaluations/s6000_mt200_ct175_compiled_cuda_all';oldp=json.loads((old/'render_protocol.json').read_text())['protocol']
oldsp=json.loads((old/'scores/protocol.json').read_text())
assert oldp['runner']==render[0]['runner']
assert all(oldp[k]==render[0][k] for k in ['source_sha256','model_dir','model_dir_sha256','input_encoding','output_encoding','warmup','inference_rng','gpu','torch'])
assert oldp['tasks'][:3]==render[0]['tasks'][:3]
assert all(oldsp[k]==score[0][k] for k in SK)
oldrows=[json.loads(x) for x in (old/'scores/measurements.jsonl').read_text().splitlines()[:3]]
newrows=[json.loads(x) for x in (R/'nr_pilot_evaluations'/LABELS[0]/'scores/measurements.jsonl').read_text().splitlines()[:3]]
prefix=[]
for i,(a,b) in enumerate(zip(oldrows,newrows),1):
    pa=old/f'render/val/case{i}.png';pb=R/'nr_pilot_evaluations'/LABELS[0]/f'render/val/case{i}.png'
    xa=np.asarray(Image.open(pa).convert('RGB')).astype(np.int16);xb=np.asarray(Image.open(pb).convert('RGB')).astype(np.int16);diff=xb-xa
    prefix.append(dict(case=i,old_png_sha256=sha(pa),control_png_sha256=sha(pb),byteequal=sha(pa)==sha(pb),channel_elements=diff.size,
                        changed_channel_elements=int(np.count_nonzero(diff)),changed_pixel_fraction=float(np.mean(np.any(diff!=0,axis=2))),
                        uint8_max_abs=int(np.max(np.abs(diff))),uint8_mean_abs=float(np.mean(np.abs(diff))),
                        metric_delta={k:b['metrics'][k]-a['metrics'][k] for k in VK}))
report=dict(audited_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    scope='matched fixed20test+3validation pilot; S_subset only, not full100 proxy or official score',cpu_only=True,no_GPU_or_rerender=True,
    results=results,matched23_input_GT_seed_metric_and_renderer_protocol='PASS',
    scorefile_schema_note='Raw S_proxy field is explicitly partial_or_preflight_only/official_full_countFalse; canonical comparison labels it S_subset.',
    control_vs_old_full_validation_prefix=dict(records=prefix,recorded_runner_renderer_config_flags_versions_equal=True,recorded_metric_protocol_equal=True,
       source_and_input_path_recipe_changes_found=False,
       verdict='Val1/2 PNG pixels differ while val3 is byteexact. FR delta entirely DISTS; LPIPS unchanged. No recorded input/algorithm/weight/source/flag change. Small compiled-CUDA floating/kernel/cache drift is consistent with evidence, exact cause unproven; no GPU rerun.'),
    independent_mean_formula='means first; NR=C+M+U/100+max(0,(10-N)/10); FR=2-L-D; S_subset=NR_test20+FR_val3')
print(json.dumps(report,ensure_ascii=False,indent=2))
