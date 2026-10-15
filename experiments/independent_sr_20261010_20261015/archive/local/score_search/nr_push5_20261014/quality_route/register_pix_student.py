"""CPU-only registration of an actual trained Pix checkpoint; no GPU launch.

Default dry-run. --execute creates one new deployment directory using the
unchanged supplied G4 inference implementation, exact trained raw weights, and
metadata-only adapter. Original G4/HYPIR files are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

R=Path('/cache/independent_sr_cuda_20261013')
RUN='pix_push5_g4_nr250'
TRAIN_SHA='3a389e55e898be8a64cdda7637339126b1caf94d0b9e7fb19c9f66abe61e8a16'
CLI_SHA='d0b1d8560b9bdf43e628fe6fa0b8191b24a85cadc4c4f27fe9964823a7cd4b1c'
G4_SHA='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
RUNNER_SHA='6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'
PIX_SHA='5b1ba75929d25c2663ec5aba5f7cd4d4a56b79f214dd8e7a49e48c34eca50f5f'

WRAPPER='''"""Metadata adapter for a genuinely trained G4-initialized pixel student."""
from pathlib import Path
import hashlib
import importlib.util
import json
import random
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Runner:
    def __init__(self, modeldir):
        import torch
        root=Path(modeldir).resolve()
        binding=json.loads((root/'student_binding.json').read_text())
        assert sha(root/'model.pth')==binding['raw_sha256']
        assert sha(root/'reference_runner.py')==binding['reference_runner_sha256']
        assert sha(root/'pix_model.py')==binding['pix_model_sha256']
        random.seed(231);torch.manual_seed(231);torch.cuda.manual_seed_all(231)
        name='push5_trained_pix_student_impl'
        spec=importlib.util.spec_from_file_location(name,root/'reference_runner.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        self.impl=module.Runner(str(root))
        self.device=self.impl.device;self.weight_dtype=self.impl.weight_dtype;self._path=self.impl._path
        self.protocol=dict(schema='push5_trained_pix_student_adapter_v1',label=binding['label'],
            provenance='New real pixel-model fine-tuning from supplied trained G4, with offline native A7750 pseudo-labels; not reference reproduction',
            runner_path=self._path,checkpoint_sha256=binding['raw_sha256'],weight_path=binding['raw_checkpoint'],
            runner_sha256=sha(root/'runner.py'),repo_path=str(root),
            source_sha256={p.name:sha(p) for p in [root/'reference_runner.py',root/'pix_model.py',root/'student_binding.json']},
            initial_reference_sha256=binding['initial_reference_sha256'],new_training_step=binding['local_step'],
            original_reference_training_step=30000,training_source_sha256=binding['training_source_sha256'],
            resume_sha256=binding['resume_sha256'],checkpoint_marker_sha256=binding['checkpoint_marker_sha256'],
            teacher_raw_sha256=binding['teacher_raw_sha256'],teacher_manifest_sha256=binding['teacher_manifest_sha256'],
            input='Native CUDA FP16 [1,3,H,W] [-1,1]',
            output='Native CUDA FP16 [1,3,H,W] [-1,1]; unchanged G4 fold + channels_last + optional compiled graph',
            adapter='Only seed and provenance metadata; infer delegates unchanged original supplied Runner',
            seed=231,stochastic_inference=False)

    def infer(self,x):
        return self.impl.infer(x)
'''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def need(ok,message):
    if not ok:raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,choices=[125,250],default=250)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();step=a.step
    source=R/'calibration/g4/model_dir';run=R/'adaptation/runs'/RUN
    raw=run/f'step{step}.pth';resume=run/f'resume_step{step}.pt';marker_path=run/f'checkpoint_step{step}.json'
    need(raw.exists() and resume.exists() and marker_path.exists(), 'Actual immutable trained checkpoint/marker must exist first')
    marker=json.loads(marker_path.read_text());runtime=json.loads((run/'config.json').read_text())
    need(marker['local_step']==marker['optimizer_step']==step and Path(marker['raw_checkpoint'])==raw and
         Path(marker['resume_checkpoint'])==resume,'Saved marker paths/counters differ')
    need(sha(raw)==marker['raw_sha256'] and sha(resume)==marker['resume_sha256'],'Actual trained raw/resume SHA differs')
    need((resume.stat().st_size,resume.stat().st_mtime_ns,resume.stat().st_ino)==
         (marker['resume_bytes'],marker['resume_mtime_ns'],marker['resume_inode']),'Immutable resume metadata changed')
    need(sha(R/'code/push5/train_pix_g4_nr_stage1.py')==TRAIN_SHA and
         sha(R/'code/push5/g4_stage1_conservative250.json')==CLI_SHA,'Frozen training source/CLI changed')
    need(sha(source/'model.pth')==G4_SHA and sha(source/'runner.py')==RUNNER_SHA and
         sha(source/'pix_model.py')==PIX_SHA,'Original supplied G4 artifacts changed')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    ck=torch.load(raw,map_location='cpu',weights_only=False,mmap=True)
    rs=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(source/'model.pth',map_location='cpu',weights_only=False,mmap=True)
    need(ck['step']==rs['local_step']==rs['optimizer_step']==step and ck['reference_step']==30000,'Saved tensor counters differ')
    need(ck['protocol_id']==rs['protocol_id']==runtime['protocol_id'],'Saved protocol binding differs')
    need(ck['cfg']==rs['cfg']==initial['cfg'] and len(ck['state'])==len(rs['state'])==100,'Pixel cfg/state schema differs')
    need(list(ck['state'])==list(rs['state'])==list(initial['state']) and sum(v.numel() for v in ck['state'].values())==3163404,'Pixel parameter order/count differs')
    for key,value in ck['state'].items():
        need(value.shape==initial['state'][key].shape and value.dtype==rs['state'][key].dtype==torch.float32 and bool(torch.isfinite(value).all()) and
             torch.equal(value,rs['state'][key]) and not torch.equal(value,initial['state'][key]),
             'Actual saved FP32/raw-resume equality or real update missing: '+key)
    states=rs['optimizer']['state']
    need(len(states)==100 and {int(v['step']) for v in states.values()}=={step},'Actual saved Adam counters differ')
    need(all(v[k].dtype==torch.float32 and bool(torch.isfinite(v[k]).all()) for v in states.values() for k in ['exp_avg','exp_avg_sq']),
         'Actual saved Adam moments differ')
    need({'python_rng','numpy_rng','torch_rng','cuda_rng'}<=set(rs),'Four saved RNG states required')
    need(not torch.cuda.is_initialized(),'CPU registration must never initialize CUDA')
    protocol=runtime['protocol'];target=R/f'submissions/{RUN}_s{step}/model_dir'
    label=f'push5_pix_g4_distill{step}_full'
    binding=dict(schema='push5_actual_pix_student_binding_v1',label=label,run=RUN,local_step=step,optimizer_step=step,
        raw_checkpoint=str(raw),raw_sha256=marker['raw_sha256'],resume_checkpoint=str(resume),resume_sha256=marker['resume_sha256'],
        checkpoint_marker=str(marker_path),checkpoint_marker_sha256=sha(marker_path),initial_reference_sha256=G4_SHA,
        training_source_sha256=TRAIN_SHA,CLI_config_sha256=CLI_SHA,runtime_config_sha256=sha(run/'config.json'),
        reference_runner_sha256=RUNNER_SHA,pix_model_sha256=PIX_SHA,
        teacher_raw_sha256=protocol['teacher_raw_sha256'],teacher_manifest_sha256=protocol['teacher_manifest_sha256'],
        no_validation_GT_training=True,all100_actual_FP32_parameters_changed_and_raw_equal_resume=True,
        Adam100_FP32_finite_same_step=True,all4_RNG_saved=True,registration_CPU_only=True,CUDA_context_initialized=False)
    proof=dict(status='ready_actual_checkpoint_validated',dry_run=not a.execute,model_dir=str(target),label=label,
               binding=binding,wrapper_sha256=hashlib.sha256(WRAPPER.encode()).hexdigest())
    if a.execute:
        need(not target.exists(),'Never overwrite any existing student/reference deployment')
        target.mkdir(parents=True)
        for old,new in [('runner.py','reference_runner.py'),('pix_model.py','pix_model.py'),('config.yaml','config.yaml')]:
            os.link(source/old,target/new)
        os.link(raw,target/'model.pth')
        (target/'runner.py').write_text(WRAPPER)
        (target/'student_binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n')
        need(sha(target/'model.pth')==marker['raw_sha256'],'Registered student bytes differ')
        proof['status']='registered_actual_trained_student'
        p=R/'nr_push5_20261014/preparation'/f'pix_student_step{step}_registration.json';p.parent.mkdir(parents=True,exist_ok=True)
        need(not p.exists(),'Never overwrite prior registration proof')
        p.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
