"""Fresh original score_dir.py run; CPU sidecar logging never changes metric calls."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import weakref

EXPECTED = {
    'score_dir.py': 'e0b700b34c68d1bcdea9f6fe1d703eaa68e708cc29632f45082a0b50ce22c9f7',
    'csig_metrics.py': '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2',
    'numpy_legacy_compat.py': '87cdcdd5fe12b7824e219ac6a25c8358815d870d2c1f333c161c0cb41afd010f',
    'maniqa_offline_init.py': 'dbc61289d34eac63408badff9cb9adb401ccc32427a1578a74c23aedc4a10bbf',
}
WEIGHTS = {
    'clip/RN50.pt': 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762',
    'torch/hub/checkpoints/DISTS_weights-f5e65c96.pth': 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218',
    'torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth': 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0',
    'torch/hub/checkpoints/alexnet-owt-7be5be79.pth': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
    'torch/hub/checkpoints/ckpt_koniq10k.pt': 'a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6',
    'torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth': 'e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa',
    'torch/hub/checkpoints/niqe_modelparameters.mat': 'fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d',
    'torch/hub/checkpoints/vgg16-397923af.pth': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['tool', 'render-dir', 'label', 'out-dir']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    tool, render, out = [Path(x).resolve() for x in (args.tool,args.render_dir,args.out_dir)]
    if not args.label or '/' in args.label or args.label in {'.','..'}:
        parser.error('label must be a single directory name')
    if out == render or out.is_relative_to(render) or render.is_relative_to(out):
        parser.error('output and rendered PNG directories must be separate')
    out.mkdir(parents=True, exist_ok=False)  # Never resume/overwrite previous scores.
    rows = []
    status = {'status':'CPU_verifying', 'label':args.label, 'completed':0,
              'expected':103, 'expected_scalars':418, 'resume_supported':False}
    atomic(out/'status.json',status)
    try:
        import numpy as np
        from PIL import Image
        source = {}
        here = Path(__file__).resolve().parent
        for name, expected in EXPECTED.items():
            p = (tool if name in {'score_dir.py','csig_metrics.py'} else here)/name
            actual = sha(p)
            if actual != expected:
                raise RuntimeError(f'Frozen source changed: {p}')
            source[str(p)] = actual
        source[str(Path(__file__).resolve())] = sha(__file__)
        weight_proof = {}
        for name, expected in WEIGHTS.items():
            p = tool/'weights'/name
            actual = sha(p)
            if actual != expected:
                raise RuntimeError(f'Metric weight changed: {p}')
            weight_proof[name] = {'sha256':actual,'bytes':p.stat().st_size}
        evaluation = render.parent
        rp = json.loads((evaluation/'render_protocol.json').read_text())
        protocol = rp['protocol']
        if hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()!=rp['protocol_id']:
            raise RuntimeError('Render protocol ID mismatch')
        manifest = json.loads((evaluation/'render_manifest.json').read_text())
        rs = json.loads((evaluation/'render_status.json').read_text())
        if rs['status']!='complete' or rs['completed']!=103:
            raise RuntimeError('Require already completed native103 PNG render')
        sequence = [('validation',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
        if len(manifest['records'])!=103 or len(protocol['tasks'])!=103:
            raise RuntimeError('Require complete103 render records and inputs')
        expected_rows = []
        loads = []
        for (split,case), rec, inp in zip(sequence,manifest['records'],protocol['tasks']):
            local_split = split.replace('validation','val')
            if (rec['split'],rec['case'])!=(local_split,case) or (inp['split'],inp['case'])!=(local_split,case):
                raise RuntimeError('Render case ordering mismatch')
            png = render/local_split/f'case{case}.png'
            actual = sha(png)
            if actual!=rec['sha256'] or png!=(evaluation/rec['output']).resolve():
                raise RuntimeError(f'PNG not equal to committed manifest: {png}')
            size = list(Image.open(png).size)
            if size!=rec['size'] or size!=inp['size'] or sha(inp['input'])!=inp['input_sha256']:
                raise RuntimeError(f'Input SHA/native dimensions changed: {png}')
            gt = (tool/'data/验证集'/('case3_gt.jpg' if case==3 else f'case{case}_gt.png')).resolve() if split=='validation' else None
            item = {'label':args.label,'split':split,'case':case,'sr_path':str(png),
                    'sr_sha256':actual,'size_hw':size[::-1],'input_path':inp['input'],
                    'input_sha256':inp['input_sha256'],'gt_path':str(gt) if gt else None,
                    'gt_sha256':sha(gt) if gt else None}
            expected_rows.append(item)
            loads.append((str(png),actual))
            if gt:
                loads.append((str(gt),item['gt_sha256']))
        os.environ['TORCH_HOME']=str(tool/'weights/torch')
        os.environ['CSIG_MANIQA_SAMPLES']='100'
        os.environ['CSIG_DATA_ROOT']=str(tool/'data')
        sys.path.insert(0,str(here));sys.path.insert(0,str(tool))
        import numpy_legacy_compat
        import maniqa_offline_init
        import torch
        import csig_metrics as CM
        score_dir = importlib.import_module('score_dir')
        if Path(CM.__file__).resolve()!=tool/'csig_metrics.py' or Path(score_dir.__file__).resolve()!=tool/'score_dir.py':
            raise RuntimeError('Wrong original scoring module imported')
        CM.ensure_weights()
        if sha(Path.home()/'.cache/clip/RN50.pt')!=WEIGHTS['clip/RN50.pt']:
            raise RuntimeError('CLIP cache differs from original RN50')
        if len(list((tool/'data/测试集').glob('*.jpg')))!=100:
            raise RuntimeError('Original score_dir test enumeration must contain100 JPG inputs')
        maniqa_offline_init.install(tool/'weights/torch/hub/checkpoints/ckpt_koniq10k.pt')
        sidecar = {'schema':'original_score_dir_repeat_v1','label':args.label,
                   'started_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   'frozen_source_sha256':source,'metric_weight_sha256':weight_proof,
                   'render_protocol_id':rp['protocol_id'],'render_dir':str(render),
                   'expected_rows':expected_rows,'expected_image_rows':103,'expected_scalar_metrics':418,
                   'seed':1234,'seed_scope':'unchanged original score_dir: once at start of validation and test',
                   'measure_policy':'unchanged original CM.measure: val LPIPS,DISTS FP16 autocast; NIQE,MUSIQ,MANIQA100,CLIPIQA FP32',
                   'logging_policy':'CPU path/shape/hash metadata and original returned floats only; no added Tensor operations, forwards, seeding or random draws',
                   'fresh_output_required':True,'resume_supported':False,
                   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
        atomic(out/'protocol.json',sidecar)
        original_create, original_load, original_measure = CM.create_metrics, CM.load_img, CM.measure
        loaded = {}; load_index = 0
        def audited_create(*a,**kw):
            metrics = original_create(*a,**kw)
            proof = maniqa_offline_init.verify_final_state(metrics['maniqa'].net)
            if proof['test_sample']!=100 or proof['net_training'] or not proof['all_final_parameters_and_buffers_equal_checkpoint']:
                raise RuntimeError('MANIQA initialization proof failed')
            sidecar['maniqa_offline_init_proof']=proof
            sidecar['torch_version']=torch.__version__
            sidecar['metric_device']=str(next(metrics['maniqa'].net.parameters()).device)
            atomic(out/'protocol.json',sidecar)
            return metrics
        def audited_load(path,*a,**kw):
            nonlocal load_index
            path = Path(path).resolve()
            if load_index>=len(loads) or str(path)!=loads[load_index][0] or sha(path)!=loads[load_index][1]:
                raise RuntimeError(f'Unexpected loaded image/order/SHA: {path}')
            before = path.stat()
            value = original_load(str(path),*a,**kw)
            after = path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError(f'Image changed during load: {path}')
            loaded[id(value)] = (weakref.ref(value),str(path))
            load_index += 1
            return value
        def logged_measure(metrics,sr,gt=None):
            i = len(rows)
            if i>=103:
                raise RuntimeError('Unexpected extra metric call')
            expected = expected_rows[i]
            sr_meta = loaded.get(id(sr))
            gt_meta = loaded.get(id(gt)) if gt is not None else None
            if sr_meta is None or sr_meta[0]() is not sr or sr_meta[1]!=expected['sr_path']:
                raise RuntimeError('Metric SR does not match actual expected loaded PNG')
            if (gt_meta[1] if gt_meta else None)!=expected['gt_path'] or (gt_meta and gt_meta[0]() is not gt):
                raise RuntimeError('Metric GT does not match expected loaded GT')
            if list(sr.shape[-2:])!=expected['size_hw'] or str(sr.dtype)!='torch.float32':
                raise RuntimeError('Original FP32/native metric input changed')
            started = time.perf_counter()
            measured = original_measure(metrics,sr,gt)  # Single original call; untouched return.
            keys = list(CM.SCORE_KEYS if gt is not None else CM.NR_KEYS)
            if list(measured)!=keys or not all(math.isfinite(v) for v in measured.values()):
                raise RuntimeError('Metric scalar keys/nonfinite result')
            row = {**expected,'metrics':dict(measured),'nr_per_image':CM.nr_contrib(measured),
                   'seconds':time.perf_counter()-started,'seed_scope':'split_seed1234_continuous_rng'}
            with (out/'measurements.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row)
            status.update(status='scoring',completed=len(rows),scalar_metrics_completed=sum(len(r['metrics']) for r in rows),last_split=expected['split'],last_case=expected['case'])
            atomic(out/'status.json',status)
            return measured
        CM.create_metrics, CM.load_img, CM.measure = audited_create, audited_load, logged_measure
        old_cwd, old_argv = Path.cwd(),sys.argv
        argv = [str(tool/'score_dir.py'),'--val_dir',f'{args.label}={render/"val"}',
                '--test_dir',f'{args.label}={render/"test"}','--n','100','--csv','--out',str(out/'result.json')]
        sidecar['original_argv']=argv
        atomic(out/'protocol.json',sidecar)
        try:
            os.chdir(out);sys.argv=argv
            score_dir.main()  # Original loop, split seeds, aggregation, CSV and JSON.
        finally:
            os.chdir(old_cwd);sys.argv=old_argv
            CM.create_metrics, CM.load_img, CM.measure = original_create, original_load, original_measure
        if len(rows)!=103 or sum(len(r['metrics']) for r in rows)!=418 or load_index!=len(loads):
            raise RuntimeError('Incomplete103/418/original load sequence')
        for path,expected in loads:
            if sha(path)!=expected:
                raise RuntimeError(f'Image changed during scoring: {path}')
        for path,expected in source.items():
            if sha(path)!=expected:
                raise RuntimeError(f'Source changed during scoring: {path}')
        raw=json.loads((out/'result.json').read_text())[args.label]
        if raw['test']['n']!=100:
            raise RuntimeError('Original test count mismatch')
        proof={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir() if p.is_file() and p.name not in {'status.json','artifact_hashes.json'}}
        atomic(out/'artifact_hashes.json',proof)
        status.update(status='complete',completed=103,scalar_metrics_completed=418,S_proxy=raw['S_proxy'],fresh_original_tool=True)
        atomic(out/'status.json',status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
    except BaseException as e:
        status.update(status='failed',completed=len(rows),error_type=type(e).__name__,error=str(e))
        atomic(out/'status.json',status)
        raise

if __name__=='__main__':
    main()
