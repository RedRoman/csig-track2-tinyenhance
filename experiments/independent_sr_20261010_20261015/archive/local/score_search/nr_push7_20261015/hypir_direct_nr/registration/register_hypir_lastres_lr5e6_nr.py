"""Register actual last-residual-and-output HYPIR milestones; CPU plan unless --execute.

The raw contains all514 FP32 LoRA tensors. Adam updates only six final-resnet/conv_out
tensors; the other508 must still equal A7750 exactly. No GPU or checkpoint copy.
"""
import argparse
from datetime import datetime, timezone
import hashlib, json, os, random, re, shutil, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
R = Path('/cache/independent_sr_cuda_20261013')
RUN = R/'adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100'
BASE = R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
INITIAL = R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
A_SHA = '589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
RUNNER_SHA = 'dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
CONFIG_SHA = '8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8'
TRAINER_SHA = '5f99fcebc6cb50ed5c3ae18ae8df0cdf10cb2ab3b62e124c5097b0c81cf5cd9b'
MODULE_SHA = 'b215dc4247ecb06cd3040160b7671e8e3c956b090bef4909d3020e90b5719738'
PARITY_SHA = 'ddd34d2ea7e1ab834fa8928c6e39da417d3be40ce2f7307ae9c1172743e44950'
NIQE_SHA = 'fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d'
SCHEMA = 'hypir_push7_lastres_nr_niqe768_optimizer6_full_gaussian_fused_bf16_v3'
SELECTED = ('up_blocks.3.resnets.2.conv1.lora_A.default.weight', 'up_blocks.3.resnets.2.conv1.lora_B.default.weight',
            'up_blocks.3.resnets.2.conv2.lora_A.default.weight', 'up_blocks.3.resnets.2.conv2.lora_B.default.weight',
            'conv_out.lora_A.default.weight', 'conv_out.lora_B.default.weight')
SHAPES = ((256, 640, 3, 3), (320, 256, 1, 1), (256, 320, 3, 3), (320, 256, 1, 1), (256, 320, 3, 3), (4, 256, 1, 1))
SEEN = {}

def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def stamp(p):
    s = Path(p).stat(); return (s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev)

def sha(p):
    p = Path(p).resolve(); before = stamp(p)
    if p in SEEN:
        need(SEEN[p][0] == before, 'Source changed: '+str(p)); return SEEN[p][1]
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    need(stamp(p) == before, 'Source changed during read: '+str(p))
    SEEN[p] = (before, h.hexdigest()); return h.hexdigest()

def read(p):
    sha(p); return json.loads(Path(p).read_text())

def ref(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))

def main():
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--step', type=int, choices=[7800, 7850], required=True)
    q.add_argument('--label'); q.add_argument('--execute', action='store_true'); a = q.parse_args()
    local = a.step-7750; label = a.label or f'push7_a7750_lastres_lr5e6_nr_s{a.step}'
    need(re.fullmatch(r'push7_a7750_lastres_lr5e6_nr_[A-Za-z0-9_]+', label), 'New lastres label required')
    dest = R/'submissions'/f'hypir_{label}'; marker = RUN/f'checkpoint_step{a.step}.json'
    need(not dest.exists(), 'Immutable candidate already exists; never overwrite')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY', run=str(RUN), step=a.step, local_step=local,
            optimizer_step=local, optimizer_tensor_count=6, autograd_LoRA_tensor_count=514,
            expected_changed_parameter_keys=list(SELECTED), unselected508_must_equal_A=True,
            marker=str(marker), model_dir=str(dest/'model_dir'), label=label,
            runner_sha256=RUNNER_SHA, initial_A_sha256=A_SHA,
            requires_actual_completed_marker_raw_resume=True, GPU=False))); return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import numpy as np
    torch.set_num_threads(4); need(not torch.cuda.is_initialized(), 'CPU registrar initialized CUDA')
    need(marker.is_file(), 'Actual completed milestone marker absent')
    m = read(marker); cfgpath = RUN/'config.json'; cfg = read(cfgpath); p = cfg['protocol']
    need(cfg['protocol_id'] == hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest(), 'Protocol digest mismatch')
    need(p['schema'] == SCHEMA and m['protocol_id'] == cfg['protocol_id'] and
         m['global_step'] == a.step and m['local_step'] == m['optimizer_step'] == local, 'Protocol/counters mismatch')
    for v in [p, m]:
        need(v['optimizer_tensor_count'] == 6 and v['requires_grad_LoRA_tensor_count'] == 514 and
             v['optimizer_parameter_keys'] == list(SELECTED) and v['optimizer_parameter_elements'] == 3113984,
             'Actual autograd514/optimizer6 schema differs')
    need(m['all514_FP32_finite'] is True and m['all4_RNG_saved'] is True and
         m['unselected508_exact_A'] is True and m['unselected508_exact_A_count'] == 508 and
         m['changed_FP32_LoRA_tensors'] == m['changed_selected_FP32_LoRA_tensors'] == m['optimizer_state_entries'] == 6,
         'Milestone does not declare actual selected6 update and frozen508')
    need(m['initial_A_checkpoint_sha256'] == A_SHA and m['fresh_Adam_initial_step'] == 0 and
         m['no_teacher_outputs'] is True and m['no_val_GT'] is True and
         p['parent_raw_sha256'] == A_SHA and p['new_Adam_initial_step'] == 0 and
         p['teacher_output_loss'] is False and p['distillation'] is False and p['validation_GT_training'] is False,
         'Fresh A initialization/no distillation/noGT lineage differs')
    need(p['unselected_LoRA_tensor_count'] == 508 and p['training_crop_size'] == 768 and
         p['training_crop_origin_alignment'] == 256 and p['deployment_patch_size'] == 512 and p['deployment_stride'] == 256,
         'Original HYPIR crop/deployment geometry differs')
    objective = dict(lr=5e-6, reward_weight=.1, musiq_weight=1.5, clip_weight=.1,
                     input_l1=.02, input_lpips=.01, input_gradient=.05, parameter_anchor=1., niqe_weight=.5)
    need(p['objective'] == objective and p['NIQE_training']['weight'] == .5 and
         p['NIQE_training']['parameters_sha256'] == NIQE_SHA and sha(p['NIQE_training']['parameters_path']) == NIQE_SHA,
         'Fixed bounded objective/original NIQE parameters differ')
    need(p['deployment_aligned_module_sha256'] == MODULE_SHA and sha(p['deployment_aligned_module']) == MODULE_SHA and
         p['deployment_parity_proof_sha256'] == PARITY_SHA and sha(p['deployment_parity_proof']) == PARITY_SHA,
         'Original deployment-aligned fusion/parity changed')
    need(m['source_sha256'] == p['source_sha256'] and TRAINER_SHA in p['source_sha256'].values(), 'Actual trainer source differs')
    for path, digest in m['source_sha256'].items(): need(sha(path) == digest, 'Actual training source changed: '+path)
    catalog = Path(m['input_catalog'])
    need(catalog.resolve() == (RUN/'input_catalog.json').resolve() and sha(catalog) == m['input_catalog_sha256'], 'Input catalog differs')
    inputs = read(catalog)
    need(inputs == p['inputs'] and [x['case'] for x in inputs] == list(range(1, 101)), 'Test100 input order differs')
    for row in inputs:
        src = R/'source_payload/assets/input'/f'case{row["case"]}.jpg'
        need(Path(row['path']).resolve() == src.resolve() and sha(src) == row['sha256'], 'Actual original input changed')
        need(not ({'teacher', 'teacher_sha256', 'gt', 'gt_sha256', 'target'} & set(row)), 'Teacher/GT data in input catalog')
    raw = Path(m['raw_checkpoint']).resolve(); resume = Path(m['resume_checkpoint']).resolve()
    need(raw == RUN/f'step{a.step}.pth' and resume == RUN/f'resume_step{a.step}.pt', 'Unexpected raw/resume path')
    need(sha(raw) == m['raw_checkpoint_sha256'] and sha(resume) == m['resume_checkpoint_sha256'] and sha(INITIAL) == A_SHA,
         'Actual raw/resume/initialA SHA differs')
    state = torch.load(raw, map_location='cpu', weights_only=True, mmap=True)
    saved = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
    initial = torch.load(INITIAL, map_location='cpu', weights_only=True, mmap=True)
    need(list(state) == list(initial) == list(saved['lora']) and len(state) == 514 and
         sum(v.numel() for v in state.values()) == 259474432 and
         all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()), 'Raw514 finite FP32 schema differs')
    need(all(saved['lora'][n].dtype == torch.float32 and saved['lora'][n].shape == v.shape and
             bool(torch.isfinite(saved['lora'][n]).all()) and torch.equal(v, saved['lora'][n])
             for n, v in state.items()), 'Resume LoRA must be finite FP32 and exactly equal raw')
    changed = [n for n, v in state.items() if not torch.equal(v, initial[n])]
    need(tuple(changed) == SELECTED and tuple(tuple(state[n].shape) for n in SELECTED) == SHAPES,
         'Actual updated tensor keys/shapes differ; all other508 must equal A')
    need(saved['protocol_id'] == m['protocol_id'] and saved['local_step'] == saved['optimizer_step'] == local and
         saved['global_step'] == a.step and saved['optimizer_tensor_count'] == 6 and
         saved['optimizer_parameter_keys'] == list(SELECTED) and saved['requires_grad_LoRA_tensor_count'] == 514 and
         saved['unselected508_exact_A'] is True, 'Resume lineage/counters differ')
    opt = saved['optimizer']; need(len(opt['param_groups']) == 1, 'One Adam group required')
    group = opt['param_groups'][0]; ids = group['params']
    need(len(ids) == len(set(ids)) == 6 and set(ids) == set(opt['state']), 'Adam must have exactly six unique state entries')
    need(group['lr'] == 5e-6 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8 and
         group['weight_decay'] == 0 and group['foreach'] is False and group.get('maximize', False) is False,
         'Actual Adam hyperparameters differ')
    for ident, key in zip(ids, SELECTED):
        v = opt['state'][ident]; step = v['step']
        need(torch.is_tensor(step) and step.numel() == 1 and bool(torch.isfinite(step).all()) and float(step) == local,
             'Actual fresh Adam local counter differs')
        for name in ['exp_avg', 'exp_avg_sq']:
            x = v[name]
            need(x.dtype == torch.float32 and x.shape == state[key].shape and bool(torch.isfinite(x).all()), 'Adam moment schema/nonfinite')
        need(bool((v['exp_avg_sq'] >= 0).all()), 'Negative Adam second moment')
    random.Random().setstate(saved['python_rng']); np.random.RandomState().set_state(saved['numpy_rng'])
    torch.Generator(device='cpu').set_state(saved['torch_rng'])
    need(saved['cuda_rng'].device.type == 'cpu' and saved['cuda_rng'].dtype == torch.uint8 and saved['cuda_rng'].numel() > 0,
         'Saved CUDA RNG byte state missing')
    need(sha(BASE/'runner.py') == RUNNER_SHA and sha(BASE/'config.json') == CONFIG_SHA, 'Frozen runtime changed')
    runtime = read(BASE/'config.json'); runtime.update(label=label, weight_path=str(raw))
    manifest = dict(schema='hypir_lastres_nr_candidate_v1', model_type='HYPIR_SD2_rank256_native512',
        training_run=str(RUN), label=label, model_dir=str(dest/'model_dir'), checkpoint=str(raw),
        checkpoint_sha256=m['raw_checkpoint_sha256'], initial_checkpoint=str(INITIAL), initial_checkpoint_sha256=A_SHA,
        runner_sha256=RUNNER_SHA, global_step=a.step, local_step=local, optimizer_step=local,
        autograd_LoRA_tensor_count=514, optimizer_tensor_count=6, optimizer_parameter_keys=list(SELECTED), optimizer_parameter_elements=3113984,
        actual_changed_parameter_keys=changed, actual_unselected508_equal_A=True,
        no_teacher_outputs=True, no_val_GT_training=True, distillation=False,
        training_provenance=dict(config=ref(cfgpath), marker=ref(marker), input_catalog=ref(catalog), resume=ref(resume),
            parity_proof=ref(p['deployment_parity_proof']), sources=[dict(path=k, sha256=v) for k, v in m['source_sha256'].items()]),
        actual_raw_and_resume_lora_equal=True, raw_FP32_tensor_count=514, new_Adam_initial_step=0,
        actual_Adam_state_count=6, actual_Adam_moments_finite=True, four_training_RNG_states_saved=True,
        quality_reuse_claim=False, registrar_source_sha256=sha(__file__), registered_utc=datetime.now(timezone.utc).isoformat(),
        CPU_only=True, CUDA_initialized=False)
    runtime_bytes = (json.dumps(runtime, indent=2)+'\n').encode()
    manifest['config_sha256'] = hashlib.sha256(runtime_bytes).hexdigest()
    for path, (before, _) in SEEN.items(): need(stamp(path) == before, 'Source mutated before registration: '+str(path))
    temporary = Path(tempfile.mkdtemp(prefix='.hypir_lastres_nr_', dir=dest.parent))
    try:
        md = temporary/'model_dir'; md.mkdir(); shutil.copyfile(BASE/'runner.py', md/'runner.py')
        (md/'config.json').write_bytes(runtime_bytes)
        (temporary/'candidate_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
        need(sha(md/'runner.py') == RUNNER_SHA and sha(md/'config.json') == manifest['config_sha256'], 'Candidate byte-copy failed')
        os.rename(temporary, dest)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    need(not torch.cuda.is_initialized(), 'Registrar initialized CUDA')
    print(json.dumps(dict(status='REGISTERED_ACTUAL_IMMUTABLE_HYPIR_LASTRES_CHECKPOINT', model_dir=str(dest/'model_dir'),
        checkpoint_sha256=m['raw_checkpoint_sha256'], candidate_manifest_sha256=sha(dest/'candidate_manifest.json'),
        runner_sha256=RUNNER_SHA, config_sha256=manifest['config_sha256'], global_step=a.step, local_step=local,
        optimizer_step=local, optimizer_tensor_count=6, actual_unselected508_equal_A=True)))

if __name__ == '__main__': main()
