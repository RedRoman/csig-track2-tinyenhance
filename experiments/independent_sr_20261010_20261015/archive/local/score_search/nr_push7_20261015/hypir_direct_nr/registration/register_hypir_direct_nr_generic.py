"""Register real local50/100 HYPIR directNR milestones from explicit allowed runs; default CPU plan.

Copies the frozen Push6 native512 runner byte-for-byte. Config differs only in
label and real new raw path. No checkpoint copying, GPU use, or teacher output.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import re,shutil,sys,tempfile
sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013')
RUN_NAMES=('hypir_push7_A7750_direct_nr100','hypir_push7_A7750_nrdominant100')
BASE=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
INITIAL=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
RUNNER_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
CONFIG_SHA='8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8'
SEEN={}

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def stamp(p):
    s=Path(p).stat();return (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)

def sha(p):
    p=Path(p).resolve();before=stamp(p)
    if p in SEEN:
        need(SEEN[p][0]==before,'Source changed: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    need(stamp(p)==before,'Source changed during read: '+str(p));SEEN[p]=(before,h.hexdigest());return h.hexdigest()

def read(p):
    sha(p);return json.loads(Path(p).read_text())

def ref(p):
    return dict(path=str(Path(p).resolve()),sha256=sha(p))

def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--step',type=int,choices=[7800,7850],required=True)
    q.add_argument('--run',choices=RUN_NAMES,required=True);q.add_argument('--label');q.add_argument('--execute',action='store_true');a=q.parse_args()
    RUN=R/'adaptation/runs'/a.run
    branch='nrdominant' if a.run.endswith('_nrdominant100') else 'direct_nr'
    label=a.label or f'push7_a7750_{branch}_s{a.step}'
    need(re.fullmatch(r'push7_(?:a7750|hypir)_(?:direct_nr|nrdominant)_[A-Za-z0-9_]+',label),'New directNR label required')
    dest=R/'submissions'/f'hypir_{label}'
    marker=RUN/f'checkpoint_step{a.step}.json';local=a.step-7750
    need(not dest.exists(),'Immutable candidate already exists; use new label')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',run=str(RUN),step=a.step,local_step=local,optimizer_step=local,marker=str(marker),
            model_dir=str(dest/'model_dir'),label=label,runner_sha256=RUNNER_SHA,initial_A_sha256=A_SHA,
            requires_actual_completed_marker_raw_resume=True,GPU=False)));return
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CPU registrar initialized CUDA')
    need(marker.is_file(),'Actual completed milestone marker absent')
    m=read(marker);cfgpath=RUN/'config.json';cfg=read(cfgpath);p=cfg['protocol']
    need(m['protocol_id']==cfg['protocol_id'] and m['global_step']==a.step and m['local_step']==m['optimizer_step']==local,
         'Actual checkpoint/config counters mismatch')
    need(m['initial_A_checkpoint_sha256']==A_SHA and m['fresh_Adam_initial_step']==0 and m['no_teacher_outputs'] is True and
         m['no_val_GT'] is True and m['all514_FP32_finite'] is True and m['all4_RNG_saved'] is True and m['changed_FP32_LoRA_tensors']>0,
         'Not a real changed directNR checkpoint with fresh Adam/no teacher/noGT')
    need(p['parent_raw_sha256']==A_SHA and p['new_Adam_initial_step']==0 and p['teacher_output_loss'] is False and
         p['distillation'] is False and p['validation_GT_training'] is False,'Training config lineage differs')
    need(m['source_sha256']==p['source_sha256'],'Training source bindings differ')
    for path,digest in m['source_sha256'].items():need(sha(path)==digest,'Actual training source changed: '+path)
    catalog=Path(m['input_catalog']);need(catalog.resolve()==(RUN/'input_catalog.json').resolve() and sha(catalog)==m['input_catalog_sha256'],'Actual input catalog differs')
    inputs=read(catalog);need(inputs==p['inputs'] and [x['case'] for x in inputs]==list(range(1,101)),'Actual directNR test100 inputs/order changed')
    for row in inputs:
        src=R/'source_payload/assets/input'/f'case{row["case"]}.jpg'
        need(Path(row['path']).resolve()==src.resolve() and sha(src)==row['sha256'],'Actual original input changed')
        need(not({'teacher','teacher_sha256','gt','gt_sha256','target'}&set(row)),'Teacher/GT data in directNR catalog')
    raw=Path(m['raw_checkpoint']).resolve();resume=Path(m['resume_checkpoint']).resolve()
    need(raw==RUN/f'step{a.step}.pth' and resume==RUN/f'resume_step{a.step}.pt','Unexpected immutable raw/resume path')
    need(sha(raw)==m['raw_checkpoint_sha256'] and sha(resume)==m['resume_checkpoint_sha256'] and sha(INITIAL)==A_SHA,'Actual raw/resume/initialA SHA differs')
    state=torch.load(raw,map_location='cpu',weights_only=True,mmap=True)
    saved=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
    need(len(state)==514 and sum(x.numel() for x in state.values())==259474432 and
         all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in state.values()),'Raw must be514 finite FP32 original HYPIR LoRA tensors')
    need(saved['protocol_id']==m['protocol_id'] and saved['local_step']==saved['optimizer_step']==local and saved['global_step']==a.step,
         'Actual resume counters/protocol differ')
    need(list(state)==list(saved['lora']) and all(torch.equal(v,saved['lora'][k]) for k,v in state.items()),'Actual immutable resume LoRA != raw')
    need(len(saved['optimizer']['state'])==514 and {int(v['step']) for v in saved['optimizer']['state'].values()}=={local},'Actual fresh Adam counters differ')
    need(sha(BASE/'runner.py')==RUNNER_SHA and sha(BASE/'config.json')==CONFIG_SHA,'Frozen original runtime changed')
    runtime=read(BASE/'config.json');runtime.update(label=label,weight_path=str(raw))
    manifest=dict(schema='hypir_direct_nr_candidate_v1',model_type='HYPIR_SD2_rank256_native512',training_run=str(RUN),label=label,model_dir=str(dest/'model_dir'),
        checkpoint=str(raw),checkpoint_sha256=m['raw_checkpoint_sha256'],initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=A_SHA,
        runner_sha256=RUNNER_SHA,global_step=a.step,local_step=local,optimizer_step=local,no_teacher_outputs=True,no_val_GT_training=True,
        training_provenance=dict(config=ref(cfgpath),marker=ref(marker),input_catalog=ref(catalog),resume=ref(resume),
            sources=[dict(path=k,sha256=v) for k,v in m['source_sha256'].items()]),
        actual_raw_and_resume_lora_equal=True,raw_FP32_tensor_count=514,new_Adam_initial_step=0,quality_reuse_claim=False,
        registrar_source_sha256=sha(__file__),registered_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,CUDA_initialized=False)
    runtime_bytes=(json.dumps(runtime,indent=2)+'\n').encode();manifest['config_sha256']=hashlib.sha256(runtime_bytes).hexdigest()
    for path,(before,digest) in SEEN.items():need(stamp(path)==before,'Source mutated before registration: '+str(path))
    temporary=Path(tempfile.mkdtemp(prefix='.hypir_direct_nr_',dir=dest.parent))
    try:
        md=temporary/'model_dir';md.mkdir();shutil.copyfile(BASE/'runner.py',md/'runner.py');(md/'config.json').write_bytes(runtime_bytes)
        (temporary/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        need(sha(md/'runner.py')==RUNNER_SHA and sha(md/'config.json')==manifest['config_sha256'],'Candidate byte-copy failed')
        os.rename(temporary,dest)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True);raise
    need(not torch.cuda.is_initialized(),'Registrar initialized CUDA')
    print(json.dumps(dict(status='REGISTERED_ACTUAL_IMMUTABLE_HYPIR_CHECKPOINT',model_dir=str(dest/'model_dir'),
        checkpoint_sha256=m['raw_checkpoint_sha256'],candidate_manifest_sha256=sha(dest/'candidate_manifest.json'),
        runner_sha256=RUNNER_SHA,config_sha256=manifest['config_sha256'],global_step=a.step,local_step=local,optimizer_step=local)))

if __name__=='__main__':main()
