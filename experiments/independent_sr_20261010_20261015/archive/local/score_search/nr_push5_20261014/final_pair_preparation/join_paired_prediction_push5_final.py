"""Join actual full103 G4 reference, HYPIR A7750 and newly trained Pix with one fresh timing triplet.

CPU-only: standard library/Pillow verify artifacts and image headers; torch loads
training checkpoints on CPU only. Original predict.py is SHA-pinned. No GPU inference.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os

ROOT = Path('/cache/independent_sr_cuda_20261013')
PREDICT_SHA = '0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
BENCHMARK_SHA = '3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
METRIC_WEIGHTS = {'clip/RN50.pt': 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762', 'torch/hub/checkpoints/DISTS_weights-f5e65c96.pth': 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218', 'torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth': 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0', 'torch/hub/checkpoints/alexnet-owt-7be5be79.pth': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02', 'torch/hub/checkpoints/ckpt_koniq10k.pt': 'a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6', 'torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth': 'e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa', 'torch/hub/checkpoints/niqe_modelparameters.mat': 'fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d', 'torch/hub/checkpoints/vgg16-397923af.pth': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'}
CACHE = {}; TOKENS = {}


def need(ok, message):
    if not ok:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text())
def token(path):
    s = Path(path).stat();return s.st_size, s.st_mtime_ns, s.st_ino


def sha(path):
    p = Path(path).resolve(); before = token(p)
    if p not in CACHE or CACHE[p][0] != before:
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):h.update(block)
        need(token(p) == before, f'Source changed while hashing: {p}')
        CACHE[p] = before, h.hexdigest()
    TOKENS[str(p)] = before
    return CACHE[p][1]


def check_file(path, digest):
    need(sha(path) == digest, f'Actual source/artifact SHA differs: {path}')


def identity(value):return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def full_quality(directory):
    from PIL import Image
    d=Path(directory).resolve(strict=True); need(d.is_relative_to(ROOT),'Quality path outside experiment')
    rp=read(d/'render_protocol.json');p=rp['protocol'];sp=read(d/'scores/protocol.json')
    result=read(d/'scores/result.json');manifest=read(d/'render_manifest.json')
    need(identity(p)==rp['protocol_id']==manifest['protocol_id'] and manifest['protocol']==p,'Render protocol identity differs')
    for status in [read(d/'render_status.json'),read(d/'scores/status.json'),result]:
        need(status['status']=='complete' and status['completed']==status.get('expected',103)==103,'Only completed full103 accepted')
    need((d/'render_complete').is_file() and result['official_full_count'] is True
         and result['S_proxy_scope']=='full100_test_NR_plus3val_FR','Full quality scope differs')
    need(sp['label']==result['label']==d.name and Path(sp['render_dir']).resolve()==(d/'render').resolve(),'Full label/render path differs')
    records=manifest['records'];tasks=p['tasks']
    rows=[json.loads(t) for t in (d/'scores/measurements.jsonl').read_text().splitlines() if t.strip()]
    order=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
    need(len(records)==len(tasks)==len(rows)==103 and len({x['output'] for x in records})==103,'Unique full103 set absent')
    need([(x['split'],x['case']) for x in records]==order==[(x['split'],x['case']) for x in tasks],'Full native input order differs')
    need([(x['split'],x['case']) for x in rows]==[('validation' if s=='val' else s,i) for s,i in order]
         ==[tuple(x) for x in sp['sequence']],'Formal metric order differs')
    ledger=[]
    for row,record,task in zip(rows,records,tasks):
        png=d/record['output'];check_file(png,record['sha256'])
        need(row['sr_sha256']==record['sha256'] and Path(row['sr_path']).resolve()==png.resolve(),'Scored PNG differs')
        check_file(task['input'],task['input_sha256'])
        with Image.open(png) as im:need(im.mode=='RGB' and im.format=='PNG' and list(im.size)==record['size']==task['size'],'Native PNG mode/size differs')
        need(row['size_hw']==record['size'][::-1],'Metric native dimensions differ')
        gt=None
        if row['split']=='validation':
            gt=ROOT/'scoring_tool/data/验证集'/('case3_gt.jpg' if row['case']==3 else f'case{row["case"]}_gt.png')
            check_file(gt,row['gt_sha256'])
        ledger.append([row['split'],row['case'],record['sha256'],task['input_sha256'],row.get('gt_sha256')])
    means={};scalars=0
    for split,n,keys in [('validation',3,['lpips','dists','niqe','musiq','maniqa','clipiqa']),('test',100,['niqe','musiq','maniqa','clipiqa'])]:
        group=[r for r in rows if r['split']==split];need(len(group)==n and result[split]['n']==n,'Split count differs')
        need(all(list(r['metrics'])==keys and all(math.isfinite(x) for x in r['metrics'].values()) for r in group),'Wrong/nonfinite metric scalars')
        means[split]={k:sum(r['metrics'][k] for r in group)/n for k in keys};scalars+=n*len(keys)
        for k,v in means[split].items():need(abs(v-result[split][k])<1e-12,'Metric mean differs')
    t=means['test'];v=means['validation'];nr=t['clipiqa']+t['maniqa']+t['musiq']/100+max(0,(10-t['niqe'])/10);fr=2-v['lpips']-v['dists']
    need(scalars==418 and abs(nr-result['test']['nr'])<1e-12 and abs(fr-result['validation']['fr'])<1e-12 and abs(nr+fr-result['S_proxy'])<1e-12,'Original full418 NR/FR/S formula differs')
    model=Path(p['model_dir']).resolve();runner=p['runner'];check_file(model/'runner.py',runner['runner_sha256']);check_file(runner['weight_path'],runner['checkpoint_sha256'])
    for name,digest in runner['source_sha256'].items():check_file(Path(runner.get('repo_path',model))/name,digest)
    for name,digest in p['model_dir_sha256'].items():
        if '__pycache__' not in Path(name).parts and not name.endswith('.pyc'):check_file(model/name,digest)
    check_file(ROOT/'code/render_cuda_submission.py',p['source_sha256']);check_file(ROOT/'code/score_cuda_stream.py',sp['script_sha256'])
    need(p['source_sha256']=='1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
         and sp['script_sha256']=='20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074','Frozen render/scorer changed')
    for rec in sp['original_sources']:check_file(rec['resolved_path'],rec['sha256'])
    need(next(r['sha256'] for r in sp['original_sources'] if r['path']=='csig_metrics.py')==CM_SHA,'Original formal metric source changed')
    check_file(ROOT/'scoring_tool/deployment_manifest.json',sp['original_tool_deployment_sha256'])
    check_file(ROOT/'calibration/environment_protocol.json',sp['environment_protocol_sha256'])
    for key,pathkey,hashkey in [('numpy_legacy_compat','path','sha256'),('maniqa_offline_init_proof','adapter_path','adapter_sha256')]:check_file(sp[key][pathkey],sp[key][hashkey])
    proof=sp['maniqa_offline_init_proof'];need(sp['seed']==1234 and proof['test_sample']==100 and proof['all_final_parameters_and_buffers_equal_checkpoint'] and not proof['net_training'],'Formal MANIQA100 checkpoint/eval policy differs')
    for name,digest in METRIC_WEIGHTS.items():check_file(ROOT/'scoring_tool/weights'/name,digest)
    check_file(Path.home()/'.cache/clip/RN50.pt',METRIC_WEIGHTS['clip/RN50.pt'])
    for name in ['render_manifest.json','render_protocol.json','scores/protocol.json','scores/result.json','scores/measurements.jsonl']:sha(d/name)
    formal={k:sp[k] for k in ['metric_source','original_tool_deployment_sha256','original_sources','script_sha256','environment_protocol_sha256','environment_versions','numpy_legacy_compat','seed','seed_scope','metric_order','image_policy','resume_policy','interleaving_policy','sequence']}
    clean=json.loads(json.dumps(proof))
    for row in clean['before_init_records']:row.pop('before_original_checkpoint_load',None)
    formal['maniqa_loaded_checkpoint_proof']=clean
    return {'label':d.name,'NR':nr,'FR':fr,'S':nr+fr,'component_means':means,'checkpoint':str(Path(runner['weight_path']).resolve()),
            'checkpoint_sha256':runner['checkpoint_sha256'],'model_dir':str(model),'runner_sha256':runner['runner_sha256'],
            'evaluation_dir':str(d),'result_sha256':sha(d/'scores/result.json'),'render_protocol_sha256':sha(d/'render_protocol.json'),
            'actual103_PNG_input_GT_SHA_verified':True,'image_rows':103,'scalar_metrics':418,'ledger_sha256':identity(ledger),
            '_render_protocol':p,'_formal_policy':formal,'_input_tasks':tasks}

def training_proof(record,student,g4):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    need(not torch.cuda.is_initialized(),'CPU join must not initialize CUDA')
    check_file(record['marker'],record['marker_sha256']);marker=read(record['marker'])
    for key,hashkey in [('raw_checkpoint','raw_sha256'),('resume_checkpoint','resume_sha256')]:check_file(record[key],record[hashkey])
    need(record['raw_sha256']==student['checkpoint_sha256'] and record['local_step']>0,'Student is not the trained completed checkpoint')
    raw=torch.load(record['raw_checkpoint'],map_location='cpu',weights_only=False)
    resume=torch.load(record['resume_checkpoint'],map_location='cpu',weights_only=False)
    reference=torch.load(g4['checkpoint'],map_location='cpu',weights_only=False)
    step=record['local_step']
    check_file(record['config'],record['config_sha256']);cfg=read(record['config']);protocol=cfg['protocol']
    need(identity(protocol)==cfg['protocol_id']==record['protocol_id']==raw['protocol_id']==resume['protocol_id'],'Training protocol identity differs')
    need(cfg['initial_optimizer_step']==0 and cfg['original_reference_training_step']==30000,'G4 initialization/new Adam disclosure differs')
    check_file(protocol['initial_reference_checkpoint'],protocol['initial_reference_sha256'])
    need(protocol['initial_reference_sha256']==g4['checkpoint_sha256'] and protocol['reference_training_step']==30000,'Supplied G4 initialization differs')
    for path,digest in protocol['source_sha256'].items():check_file(path,digest)
    need(protocol['source_sha256'][str(ROOT/'code/push5/train_pix_g4_nr_stage1.py')]=='3a389e55e898be8a64cdda7637339126b1caf94d0b9e7fb19c9f66abe61e8a16','Frozen offline-teacher trainer differs')
    check_file(protocol['teacher_manifest'],protocol['teacher_manifest_sha256'])
    check_file(Path(protocol['teacher_manifest']).parent/'render_protocol.json',protocol['teacher_protocol_sha256'])
    need(protocol['teacher_raw_sha256']=='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d','Offline teacher is not sealed A7750')
    need([r['case'] for r in protocol['teacher_records']]==list(range(1,101))
         and [r['case'] for r in protocol['inputs']]==list(range(1,101)),'Training100 native input/teacher set differs')
    for row in protocol['teacher_records']+protocol['inputs']:check_file(row['path'],row['sha256'])
    check_file(record['initial_FP32_proof'],record['initial_FP32_proof_sha256']);initial=read(record['initial_FP32_proof'])
    need(initial['status']=='PASS' and initial['initial_student_exact_original_FP32'] is True
         and initial['new_Adam_initial_step']==0 and initial['no_validation_GT_training'] is True
         and initial['offline_A7750_teacher_100_actual_PNG_SHA_verified'] is True,'Initial reference/offline-teacher CPU proof absent')
    need(raw['step']==step and raw['reference_step']==30000 and bool(raw['origin']),'New training step/reference disclosure absent')
    need(raw['cfg']==reference['cfg'] and set(raw['state'])==set(reference['state'])==set(resume['state']),'Student cfg/model differs from supplied G4 initialization')
    need(resume['local_step']==resume['optimizer_step']==step,'Exact resume training counters differ')
    changed=0
    for key,x in raw['state'].items():
        need(x.dtype==torch.float32 and x.device.type=='cpu' and bool(torch.isfinite(x).all()),'Nonfinite/non-FP32 trained weights')
        need(torch.equal(x,resume['state'][key]),'Raw and immutable resume weights differ')
        changed+=not torch.equal(x,reference['state'][key].float())
    need(len(raw['state'])==100 and changed>0,'New training did not actually change supplied G4 weights')
    states=resume['optimizer']['state'];need(len(states)==100,'New Adam state count differs')
    for state in states.values():
        count=state['step'];count=int(count.item()) if torch.is_tensor(count) else int(count)
        need(count==step,'New Adam step differs')
        for name in ['exp_avg','exp_avg_sq']:
            x=state[name];need(x.dtype==torch.float32 and x.device.type=='cpu' and bool(torch.isfinite(x).all()),'Adam moments are not finite FP32 CPU')
    need(all(k in resume for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']),'Exact resume four RNG absent')
    need(not torch.cuda.is_initialized(),'CPU join initialized CUDA')
    return {'training_config':record['config'],'training_config_sha256':record['config_sha256'],
            'protocol_id':record['protocol_id'],'initial_FP32_proof_sha256':record['initial_FP32_proof_sha256'],
            'offline_teacher':protocol['teacher'],'teacher_raw_sha256':protocol['teacher_raw_sha256'],
            'teacher_manifest_sha256':protocol['teacher_manifest_sha256'],'new_Adam_initial_step':0,
            'all100_new_Adam_steps_match':True,'resume_four_RNG_present':True,'status':'PASS','CPU_only':True,'local_training_step':step,'optimizer_step':step,'reference_step':30000,
            'reference_checkpoint_sha256':g4['checkpoint_sha256'],'origin':raw['origin'],'FP32_parameter_tensors':len(raw['state']),
            'changed_parameter_tensors_vs_supplied_G4':changed,'raw_checkpoint_sha256':record['raw_sha256'],
            'resume_sha256':record['resume_sha256'],'marker_sha256':record['marker_sha256'],'exact_raw_resume_parameters_equal':True}


def percentile(values, percent):
    ordered = sorted(values); place = (len(ordered) - 1) * percent / 100
    lo = int(math.floor(place)); hi = int(math.ceil(place));f = place - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * f


def timing(path, quality=None, reference=None):
    sha(path)
    value = read(path)
    need(value['status'] == 'complete' and value['cuda']['logical_device'] == 0, 'Fresh benchmark incomplete/wrong logical GPU')
    need(value['input']['shape'] == [1, 3, 512, 512] and value['input']['dtype'] == 'torch.float16'
         and value['input']['device'] == 'cuda:0' and value['seed'] == 231, 'Benchmark input/seed differs')
    need(value['output']['all_finite'] is True and value['output']['within_minus1_plus1'] is True
         and value['output']['device'] == 'cuda:0' and value['output']['shape'] == [1, 3, 512, 512]
         and value['output']['clamping_applied_by_benchmark'] is False, 'Invalid benchmark output/range/device')
    stats = value['latency_512'];need(stats['n'] == 200 and stats['warm'] == 30, 'Benchmark must be fresh warm30/n200')
    for kind in ['event', 'wall']:
        samples = value['raw_samples'][kind + '_ms']
        need(len(samples) == 200 and all(math.isfinite(x) and x > 0 for x in samples), 'Invalid original timing sample vectors')
        for key, calculated in [('p10', percentile(samples, 10)), ('med', percentile(samples, 50)), ('min', min(samples))]:
            need(abs(stats[kind + '_' + key] - calculated) < 1e-12, 'Raw-sample timing statistic differs')
    need(value['benchmark_script']['sha256'] == BENCHMARK_SHA, 'Original benchmark SHA differs')
    files = value['artifacts']['files'] + value['artifacts']['extra_files'] + [value['benchmark_script']]
    for record in files:
        p = Path(record['resolved_path']);need(p.stat().st_size == record['bytes'], 'Actual timing artifact size differs')
        check_file(p, record['sha256'])
    if quality is not None:
        p = quality['_render_protocol'];runner = p['runner']
        need(value['model_dir'] == p['model_dir'] and Path(value['model_dir']).resolve() == Path(quality['model_dir']),
             'Latency measures a different model directory')
        need(any(r['sha256'] == runner['checkpoint_sha256'] and Path(r['resolved_path']).resolve() == Path(runner['weight_path']).resolve()
                 for r in value['artifacts']['extra_files']), 'Latency must explicitly hash exact full-quality checkpoint as extra-file')
        need(any(r['path'] == 'runner.py' and r['sha256'] == runner['runner_sha256'] for r in files), 'Latency runner differs from full-quality runner')
        for relative,digest in p['model_dir_sha256'].items():
            if '__pycache__' not in Path(relative).parts and not relative.endswith('.pyc'):
                need(any(r['path']==relative and r['sha256']==digest for r in files),'Latency model artifacts differ from full-quality source: '+relative)
    else:
        need(Path(value['model_dir']).resolve() == ROOT / 'calibration/g4/model_dir', 'Fresh reference must use original G4 model')
    if reference is not None:
        ref = reference['latency_512']['event_p10']
        need(value['cuda'] == reference['cuda'] and value['ref_ms'] == ref and value['anchor_g4_ms'] == .431,
             'G4 reference/hardware/runtime differs')
        need(abs(value['g_used_ms'] - stats['event_p10'] * .431 / ref) < 1e-12, 'Same-card normalization differs')
    return value


def launch_proof(path,qualities):
    sha(path);launch=read(path)
    need(launch['status']=='complete' and launch['schema']=='push5_final_fresh_paired_latency_launch_v1'
         and launch['same_UUID_PCI_all_three'] is True and launch['cuda_visible_devices']=='0','Final fresh paired launch incomplete')
    need([r['key'] for r in launch['runs']]==['g4','hypir','student'],'Final paired sequence differs')
    need(launch['G4_rule']=='Exactly one fixed fresh G4 run, warm30/n200/seed231/original p10; no target-driven sample/run selection','G4 rule changed')
    check_file(ROOT/'code/push5/final_pair/paired_latency_push5_final.py',launch['launcher_sha256'])
    check_file(ROOT/'results/nr_push4_selection.json',launch['hypir_selection_sha256'])
    physical=launch['initial_physical_gpu'];need(physical['index']==0 and not physical['active_compute_processes'],'Initial GPU0 not idle')
    previous_end=None
    for row,key in zip(launch['runs'],['g4','hypir','student']):
        out=ROOT/f'calibration/push5_final_{key}_latency.json'
        need(Path(row['out'])==out and row['returncode']==0 and row['cuda_visible_devices']=='0','Final subprocess failed or wrong output')
        check_file(out,row['out_sha256'])
        for moment in ['physical_before','physical_after']:
            p=row[moment];need(all(p[k]==physical[k] for k in ['index','uuid','pci_bus_id','name']) and not p['active_compute_processes'],'GPU identity/idle proof differs')
        for flag,value in [('--device','0'),('--warm','30'),('--n','200'),('--seed','231')]:need(row['argv'][row['argv'].index(flag)+1]==value,'Fixed benchmark sampling changed')
        need(row['argv'][1]==str(ROOT/'code/benchmark_cuda.py'),'Original benchmark path differs')
        start=datetime.datetime.fromisoformat(row['started_utc']);end=datetime.datetime.fromisoformat(row['ended_utc'])
        need(start<end and (previous_end is None or previous_end<start),'Separate benchmark processes overlap');previous_end=end
        need(launch[key if key!='g4' else 'g4_reference']['checkpoint_sha256']==qualities[key]['checkpoint_sha256'],'Launch checkpoint differs from full quality')
    return launch,{'uuid':physical['uuid'],'pci_bus_id':physical['pci_bus_id'],'name':physical['name'],
                   'same_physical_UUID_PCI_all_three':True,'fresh_sequential_processes':True,'launch_sha256':sha(path)}

def join():
    launch_path=ROOT/'calibration/push5_final_latency_launch.json';la=read(launch_path)
    need(la['status']=='complete','Final paired timings not complete')
    qualities={key:full_quality(la[field]['evaluation_dir']) for key,field in [('g4','g4_reference'),('hypir','hypir'),('student','student')]}
    need(qualities['g4']['checkpoint_sha256']=='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
         and qualities['g4']['runner_sha256']=='276da6fe32a8e67f595f6ecf8a95b4b3ad2d87ec2beb0ca9b8fb1822777ff972','Supplied G4 reference identity differs')
    for key in ['g4','student']:
        need(qualities[key]['_input_tasks']==qualities['hypir']['_input_tasks'],'Full native inputs/size/order differ')
        need(qualities[key]['_formal_policy']==qualities['hypir']['_formal_policy'],'Formal metrics/sources/RNG/crops differ')
    sealed=read(ROOT/'results/nr_push4_selection.json');need(sha(ROOT/'results/nr_push4_selection.json')=='b3148b21f466854f9fe8f81c7eca1348da5bf793141799573ff3be8b20e25c3f','Frozen HYPIR selection changed')
    for key,field in [('NR','NR'),('FR','FR'),('S_proxy','S')]:need(abs(sealed[key]-qualities['hypir'][field])<1e-12,'Frozen HYPIR quality differs')
    trained=training_proof(la['student_training'],qualities['student'],qualities['g4'])
    source=ROOT/'scoring_tool/predict.py';check_file(source,PREDICT_SHA);namespace={'__name__':'_unchanged_predict'}
    exec(compile(source.read_text(),str(source),'exec'),namespace)
    timings={};timings['g4']=timing(ROOT/'calibration/push5_final_g4_latency.json',qualities['g4'])
    for key in ['hypir','student']:timings[key]=timing(ROOT/f'calibration/push5_final_{key}_latency.json',qualities[key],timings['g4'])
    _,physical=launch_proof(launch_path,qualities);predictions={}
    for key in ['g4','hypir','student']:
        g=.431 if key=='g4' else timings[key]['g_used_ms']
        predictions[key]={'NR':qualities[key]['NR'],'FR':qualities[key]['FR'],'S_proxy':qualities[key]['S'],
                          'event_p10_ms':timings[key]['latency_512']['event_p10'],'normalized_g_ms':g,
                          'original_tool_prediction':namespace['predict'](qualities[key]['S'],g)}
    for path,before in TOKENS.items():need(token(path)==before,f'Artifact changed before join ended: {path}')
    return {'schema':'push5_final_fresh_paired_full103_prediction_v1','status':'PASS','CPU_only':True,
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_card_proof':physical,
            'benchmark_script_sha256':BENCHMARK_SHA,'original_predict_sha256':PREDICT_SHA,'join_source_sha256':sha(__file__),
            'g4_reference_reproduction':predictions['g4'],'hypir_A7750':predictions['hypir'],'newly_trained_student':predictions['student'],
            'newly_trained_student_reaches70':predictions['student']['original_tool_prediction']>=70,
            'training_proof':trained,'qualities':{k:{n:v for n,v in q.items() if not n.startswith('_')} for k,q in qualities.items()},
            'calibration_sha256':{k:sha(ROOT/f'calibration/push5_final_{k}_latency.json') for k in timings},
            'interpretation':'Original five-point empirical tool prediction, not official leaderboard score. Supplied G4 full reproduction is separate from new Pix training initialized from G4. Fixed one fresh G4 calibration, no target-driven reference/sample selection; same physical GPU0, fresh sequential warm30/n200/seed231.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'calibration/push5_final_paired_prediction.json');args=ap.parse_args()
    report=join();need(not args.out.exists(),'Immutable final join output exists');args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='qualities'},ensure_ascii=False,indent=2,allow_nan=False))
