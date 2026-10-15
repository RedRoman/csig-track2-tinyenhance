"""Read-only CPU audit of initial stage5 migration/source/first RNG draws."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = ''
CACHE = {}


def need(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path); stat = path.stat(); token = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    if path not in CACHE or CACHE[path][0] != token:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for b in iter(lambda: stream.read(8 * 1024**2), b''): h.update(b)
        need((path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino) == token, 'File changed during SHA')
        CACHE[path] = token, h.hexdigest()
    return CACHE[path][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args(); plan = read(args.plan); root = Path(plan['root']); pp = plan['parent']
    import torch
    from PIL import Image
    need(not torch.cuda.is_initialized(), 'CUDA context initialized')
    torch.set_num_threads(2)
    pd = root / 'adaptation/runs' / pp['run']; resume = pd / pp['resume_filename']; raw = pd / pp['checkpoint_filename']
    need(sha(resume) == pp['resume_sha256'] and sha(raw) == pp['checkpoint_sha256']
         and sha(pd / 'config.json') == pp['config_sha256'], 'Sealed parent SHA differs')
    parent_cfg = read(pd / 'config.json'); parent = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
    need(parent['global_step'] == 7250 and parent['local_step'] == 750 and parent['optimizer_step'] == 1250
         and parent['protocol_id'] == pp['protocol_id'] == parent_cfg['protocol_id'], 'Sealed parent counters/protocol differ')
    raw_parent = torch.load(raw, map_location='cpu', weights_only=True, mmap=True)
    need(len(raw_parent) == len(parent['lora']) == 514 and list(raw_parent) == list(parent['lora']) and
         all(v.dtype == torch.float32 and torch.equal(v, parent['lora'][k]) for k,v in raw_parent.items()),
         'Parent raw differs from exact FP32 resume')
    rng = random.Random(0); rng.setstate(parent['python_rng']); case = rng.randint(1,100)
    with Image.open(parent_cfg['protocol']['inputs'][case-1]['path']) as im: w,h = im.size
    crop = [rng.randrange(w-511),rng.randrange(h-511)]
    report = dict(schema='cuda_nr_stage5_initial_CPU_audit_v1',status='PASS',CPU_only=True,
        audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA256=sha(args.plan),
        parent_raw_SHA256=sha(raw),parent_resume_SHA256=sha(resume),parent_config_SHA256=sha(pd/'config.json'),
        parent_raw_exact_equal_immutable_resume_FP32=True,first_case=case,first_crop_xy=crop,branches=[])
    for b in plan['branches']:
        run = root / 'adaptation/runs' / b['run']; cfg = read(run/'config.json'); p = cfg['protocol']
        need(hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() == cfg['protocol_id'], 'Protocol ID differs')
        need(sha(b['CLI_config_path']) == b['CLI_config_sha256'], 'Original CLI config SHA differs')
        for field,expected in b['expected_protocol_fields'].items():
            value=p
            for part in field.split('.'): value=value[part]
            need(value==expected, f"Independent expectation differs: {b['run']}/{field}")
        for field in ('lr','reward_weight','clip_weight','musiq_weight','gradient_anchor'):
            need(p[field]==b[field], 'Loss coefficient differs: '+field)
        need(all(sha(path)==expected for path,expected in p['source_sha256'].items()), 'Actual frozen source SHA differs')
        anchor=p['anchor_checkpoint'];need(sha(anchor)==b['anchor']['resume_sha256'], 'Teacher resume SHA differs')
        migration=read(run/'parent_migration.json'); lineage=p['parent_lineage']
        need(migration['protocol_id']==cfg['protocol_id'] and migration['parent_lineage']==lineage, 'Migration lineage differs')
        need(migration['FP32_LoRA_exact_equal_before_update'] and migration['FP32_LoRA_tensors']==514 and
             migration['FP32_Adam_moments_exact_equal_before_update'] and migration['Adam_parameter_states']==514,
             'Exact FP32 master/Adam restoration proof missing')
        need(migration['parent_raw_exact_equal_immutable_resume_FP32'] and
             migration['parent_checkpoint_sha256']==pp['checkpoint_sha256'] and
             migration['restored_global_step']==7250 and migration['restored_optimizer_step']==1250,
             'Parent counters/raw restoration proof differs')
        need(migration['teacher_FP32_exact_sealed_resume'] and migration['teacher_frozen'] and
             migration['anchor_resume_sha256']==b['anchor']['resume_sha256'], 'Frozen exact6500 teacher proof differs')
        first=json.loads((run/'train.jsonl').read_text().splitlines()[0]); pre=read(run/'gradient_preflight.json')
        need((first['case'],first['crop_xy'])==(case,crop), 'Independent parent first RNG crop differs')
        need((first['local_step'],first['global_step'],first['optimizer_step'])==(1,7251,1251), 'First counters differ')
        need(first['nonzero_lora_gradients']==first['finite_lora_gradients']==514 and first['lr']==2e-7,
             'First LoRA gradient/LR evidence differs')
        need(math.isfinite(first['fidelity']) and first['fidelity']>=0 and not p['fresh_stage_initial_fidelity_expected_zero'],
             'Old6500 teacher must not assert zero7250 fidelity')
        expected_reward=b['musiq_weight']*first['scores']['musiq']+first['scores']['maniqa']+b['clip_weight']*first['scores']['clipiqa']
        need(abs(first['reward']-expected_reward)<5e-6 and
             abs(first['loss']-(first['fidelity']-b['reward_weight']*first['reward']))<5e-7, 'First reward/loss differs')
        need(pre['status']=='passed' and pre['frozen_metrics'] and pre['frozen_same_latent_anchor'] and
             all(v['input_gradient_finite'] and v['input_gradient_norm']>0 for v in pre['components'].values()),
             'Real metric input-gradient preflight absent')
        report['branches'].append(dict(run=b['run'],protocol_id=cfg['protocol_id'],first_row=first,
            exact_FP32_LoRA_and_Adam_initial_migration=True,teacher6500_exact_frozen=True,
            actual_source_and_CLI_SHA_verified=True,real_metric_input_gradients=pre['components']))
    reference=report['branches'][0]['first_row']['scores']
    report['initial_pre_update_metric_scores_exact_equal']=all(x['first_row']['scores']==reference for x in report['branches'])
    need(report['initial_pre_update_metric_scores_exact_equal'], 'Three initial student metric scores differ')
    need(not torch.cuda.is_initialized(), 'CUDA context initialized at end')
    report['CUDA_context_initialized']=False
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
