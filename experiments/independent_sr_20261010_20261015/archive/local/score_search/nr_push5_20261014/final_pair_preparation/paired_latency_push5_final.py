"""Plan fresh G4 reference -> HYPIR A7750 -> newly trained Pix student. Default never launches.

Only the root scheduler should pass --execute after CUDA physical GPU0 is idle.
The unchanged benchmark runs once per fresh subprocess, warm30/n200/FP16 512.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys

ROOT = Path('/cache/independent_sr_cuda_20261013')
BENCHMARK_SHA = '3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
G4_RUNNER_SHA = '6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'


def need(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def physical_gpu():
    value = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=index,uuid,pci.bus_id,name',
                                     '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(io.StringIO(value)))
    need(len(rows) == 1 and len(rows[0]) == 4, 'Cannot identify exactly one physical GPU0')
    index, uuid, pci, name = (v.strip() for v in rows[0])
    need(index == '0' and uuid.startswith('GPU-'), 'Unexpected physical GPU identity')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader,nounits'], text=True)
    active = []
    for row in csv.reader(io.StringIO(apps)):
        if len(row) >= 3 and row[0].strip() == uuid:
            active.append({'pid': row[1].strip(), 'name': ','.join(row[2:]).strip()})
    return {'index': 0, 'uuid': uuid, 'pci_bus_id': pci, 'name': name, 'active_compute_processes': active}


def evaluation_identity(directory):
    directory = directory.resolve(strict=True)
    need(directory.is_relative_to(ROOT), 'Evaluation outside experiment root')
    protocol = read(directory/'render_protocol.json')['protocol']
    result = read(directory/'scores/result.json')
    need(result['status']=='complete' and result['completed']==result['expected']==103
         and result['official_full_count'] is True, 'Only actual completed full103 candidates allowed')
    runner = protocol['runner']; model = Path(protocol['model_dir']).resolve()
    checkpoint = Path(runner['weight_path']).resolve()
    need(sha(checkpoint)==runner['checkpoint_sha256'], 'Actual quality checkpoint changed')
    need(sha(model/'runner.py')==runner['runner_sha256'], 'Actual quality runner changed')
    return {'label':directory.name,'evaluation_dir':str(directory),'model_dir':str(model),
            'checkpoint':str(checkpoint),'checkpoint_sha256':runner['checkpoint_sha256'],
            'runner_sha256':runner['runner_sha256']}

def marker_identity(path, step, student):
    path=path.resolve(strict=True); marker=read(path)
    need(step>0 and marker.get('local_step',marker.get('step'))==step
         and marker['optimizer_step']==step, 'Actual new-training marker counters differ')
    raw=Path(marker['raw_checkpoint']);raw=(raw if raw.is_absolute() else path.parent/raw).resolve()
    resume=Path(marker['resume_checkpoint']);resume=(resume if resume.is_absolute() else path.parent/resume).resolve()
    need(raw.parent==path.parent and raw.name==f'step{step}.pth', 'Actual trained checkpoint path differs')
    need(sha(raw)==marker['raw_sha256']==student['checkpoint_sha256'], 'Student is not the completed trained raw checkpoint')
    need(sha(resume)==marker['resume_sha256'], 'Actual trained resume changed')
    config=path.parent/'config.json';cfg=read(config);initial=path.parent/'initial_fp32_proof.json'
    need(cfg['initial_optimizer_step']==0 and cfg['original_reference_training_step']==30000
         and cfg['target_stage_steps']>=step,'New-training configuration counters differ')
    need(sha(Path(cfg['protocol']['initial_reference_checkpoint']))==cfg['protocol']['initial_reference_sha256']
         =='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83','Supplied G4 initialization identity differs')
    return {'config':str(config),'config_sha256':sha(config),'protocol_id':cfg['protocol_id'],
            'initial_FP32_proof':str(initial),'initial_FP32_proof_sha256':sha(initial),
            'marker':str(path),'marker_sha256':sha(path),'local_step':step,'optimizer_step':step,
            'raw_checkpoint':str(raw),'raw_sha256':marker['raw_sha256'],
            'resume_checkpoint':str(resume),'resume_sha256':marker['resume_sha256'],
            'provenance':'New Pix training from supplied G4 initialization; not the supplied-reference reproduction'}

def make_plan(student_evaluation, student_marker, student_step):
    benchmark=ROOT/'code/benchmark_cuda.py'; need(sha(benchmark)==BENCHMARK_SHA,'Original benchmark changed')
    hypir_selection=ROOT/'results/nr_push4_selection.json'
    need(sha(hypir_selection)=='b3148b21f466854f9fe8f81c7eca1348da5bf793141799573ff3be8b20e25c3f','Frozen A7750 selection changed')
    h=read(hypir_selection); hypir=evaluation_identity(Path(h['quality_dir']))
    need(hypir['checkpoint_sha256']==h['checkpoint_sha256'],'Frozen HYPIR checkpoint differs')
    g4=evaluation_identity(ROOT/'nr_push5_full_evaluations/push5_g4_reference_full')
    need(g4['checkpoint_sha256']=='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83','Supplied G4 reference changed')
    need(g4['runner_sha256']=='276da6fe32a8e67f595f6ecf8a95b4b3ad2d87ec2beb0ca9b8fb1822777ff972','G4 metadata adapter changed')
    need(sha(Path(g4['model_dir'])/'reference_runner.py')==G4_RUNNER_SHA,'Original G4 inference changed')
    student=evaluation_identity(student_evaluation);training=marker_identity(student_marker,student_step,student)
    need(student['checkpoint_sha256']!=g4['checkpoint_sha256'],'Do not label supplied-reference weights as new training')
    stages=[]
    for key,row in [('g4',g4),('hypir',hypir),('student',student)]:
        out=ROOT/f'calibration/push5_final_{key}_latency.json'
        argv=[str(ROOT/'env/bin/python'),str(benchmark),'--model-dir',row['model_dir'],'--out',str(out),
              '--device','0','--warm','30','--n','200','--seed','231','--hash-file',row['checkpoint']]
        if key!='g4':argv+=['--ref-ms','<FRESH_G4_EVENT_P10>']
        stages.append({'key':key,**row,'out':str(out),'argv':argv,
                       'reviewable_shell_command':'CUDA_VISIBLE_DEVICES=0 '+shlex.join(argv)})
    return {'schema':'push5_final_fresh_paired_latency_plan_v1','status':'planned_not_launched',
            'created_utc':now(),'benchmark_sha256':BENCHMARK_SHA,'launcher_sha256':sha(__file__),
            'cuda_visible_devices':'0','g4_reference':g4,'hypir':hypir,'student':student,
            'student_training':training,'hypir_selection_sha256':sha(hypir_selection),
            'stages':stages,'no_historical_latency_reuse':True,
            'G4_rule':'Exactly one fixed fresh G4 run, warm30/n200/seed231/original p10; no target-driven sample/run selection',
            'same_physical_card_proof':'NVML GPU0 UUID and PCI before/after every separate process'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--student-evaluation',type=Path,required=True)
    ap.add_argument('--student-marker',type=Path,required=True)
    ap.add_argument('--student-step',type=int,required=True)
    ap.add_argument('--plan-out', type=Path)
    ap.add_argument('--execute', action='store_true', help='ROOT scheduler only: run fresh CUDA subprocesses; omitted means plan only')
    args = ap.parse_args(); plan = make_plan(args.student_evaluation,args.student_marker,args.student_step)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.plan_out:write(args.plan_out, plan)
    if not args.execute:return
    need(args.student_evaluation.resolve().parent==ROOT/'nr_push5_full_evaluations','Execute requires an actual Push5 full evaluation')
    # Never launch from an inherited nonzero CUDA selection or overwrite a prior pairing.
    need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '0'), 'Root must select CUDA physical GPU0')
    provenance = ROOT / 'calibration/push5_final_latency_launch.json'
    need(not provenance.exists(), 'Fresh paired launch already exists; preserve its evidence rather than overwrite')
    for stage in plan['stages']:need(not Path(stage['out']).exists(), 'Fresh output name already exists')
    initial = physical_gpu(); need(not initial['active_compute_processes'], 'Physical GPU0 is occupied')
    record = dict(plan, schema='push5_final_fresh_paired_latency_launch_v1', status='running',
                  initial_physical_gpu=initial, runs=[])
    write(provenance, record)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    ref = None
    for stage in plan['stages']:
        before = physical_gpu()
        need((before['uuid'], before['pci_bus_id']) == (initial['uuid'], initial['pci_bus_id']), 'Physical GPU changed before run')
        need(not before['active_compute_processes'], 'GPU0 must be idle between fresh benchmark processes')
        argv = [str(ref) if item == '<FRESH_G4_EVENT_P10>' else item for item in stage['argv']]
        start = now(); code = subprocess.run(argv, env=env).returncode; end = now()
        after = physical_gpu()
        need(code == 0, f'Fresh benchmark failed: {stage["key"]}')
        need((after['uuid'], after['pci_bus_id']) == (initial['uuid'], initial['pci_bus_id']), 'Physical GPU changed after run')
        result = read(Path(stage['out'])); need(result['status'] == 'complete', 'Benchmark did not complete')
        if stage['key'] == 'g4':ref = result['latency_512']['event_p10']
        record['runs'].append({'key': stage['key'], 'argv': argv, 'cuda_visible_devices': '0',
                               'started_utc': start, 'ended_utc': end, 'physical_before': before,
                               'physical_after': after, 'returncode': code, 'out': stage['out'],
                               'out_sha256': sha(stage['out'])})
        write(provenance, record)
    record.update(status='complete', finished_utc=now(), same_UUID_PCI_all_three=True)
    write(provenance, record)
    print(json.dumps({'paired_launch_complete': True, 'provenance': str(provenance),
                      'provenance_sha256': sha(provenance), 'G4_event_p10_ms': ref}), flush=True)


if __name__ == '__main__':main()
