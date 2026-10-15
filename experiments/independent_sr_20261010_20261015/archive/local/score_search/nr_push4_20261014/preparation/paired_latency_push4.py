"""Plan fresh G4 -> parent7250 -> selected Push4 latency. Default never launches.

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
PARENT_SHA = 'fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e'
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


def locate(label):
    need(label and Path(label).name == label, 'Safe candidate label required')
    for base in ['nr_push4_full_evaluations', 'nr_push4_pilot_evaluations']:
        p = ROOT / base / label / 'render_protocol.json'
        if p.is_file():return p
    raise RuntimeError('Candidate render_protocol.json missing; use an existing render label')


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


def make_plan(selection_path, candidate_label):
    benchmark = ROOT / 'code/benchmark_cuda.py'
    need(sha(benchmark) == BENCHMARK_SHA, 'Original benchmark source SHA differs')
    parent = read(ROOT / 'results/nr_push3_selection.json')
    need(parent['global_step'] == 7250 and parent['checkpoint_sha256'] == PARENT_SHA, 'Sealed best7250 parent required')
    if candidate_label:
        protocol_path = locate(candidate_label)
        protocol = read(protocol_path)['protocol']
        candidate = {'label': candidate_label, 'model_dir': protocol['model_dir'],
                     'checkpoint': protocol['runner']['weight_path'],
                     'checkpoint_sha256': protocol['runner']['checkpoint_sha256']}
        selection_sha = None
    else:
        need(selection_path.is_file(), 'Final selection missing; --candidate-label can prepare provisional timing')
        candidate = read(selection_path)
        need(candidate['status'] == 'selected_after_independent_full_audit', 'Final audited selection required')
        selection_sha = sha(selection_path)
    for row in [parent, candidate]:
        need(sha(row['checkpoint']) == row['checkpoint_sha256'], 'Checkpoint source SHA differs')
        model = Path(row['model_dir']).resolve()
        need(model.is_relative_to(ROOT / 'submissions'), 'Model dir outside experiment submissions')
        config = read(model / 'config.json')
        need(Path(config['weight_path']).resolve() == Path(row['checkpoint']).resolve(), 'Model config checkpoint differs')
    g4 = ROOT / 'calibration/g4/model_dir'
    need(sha(g4 / 'runner.py') == G4_RUNNER_SHA, 'Original G4 runner changed')
    stages = []
    for key, model, checkpoint in [('g4', g4, None), ('parent7250', Path(parent['model_dir']), Path(parent['checkpoint'])),
                                    ('selected', Path(candidate['model_dir']), Path(candidate['checkpoint']))]:
        out = ROOT / f'calibration/push4_{key}_latency.json'
        args = [str(ROOT / 'env/bin/python'), str(benchmark), '--model-dir', str(model), '--out', str(out),
                '--device', '0', '--warm', '30', '--n', '200', '--seed', '231']
        if checkpoint is not None:args += ['--hash-file', str(checkpoint), '--ref-ms', '<FRESH_G4_EVENT_P10>']
        stages.append({'key': key, 'model_dir': str(model), 'checkpoint': str(checkpoint) if checkpoint else None,
                       'out': str(out), 'argv': args,
                       'reviewable_shell_command': 'CUDA_VISIBLE_DEVICES=0 ' + shlex.join(args)})
    return {'schema': 'push4_fresh_paired_latency_plan_v1', 'status': 'planned_not_launched',
            'created_utc': now(), 'selection_path': str(selection_path), 'selection_sha256': selection_sha,
            'candidate': candidate, 'parent_selection_sha256': sha(ROOT / 'results/nr_push3_selection.json'), 'parent7250': {k: parent[k] for k in ['label', 'model_dir', 'checkpoint', 'checkpoint_sha256']},
            'benchmark_sha256': BENCHMARK_SHA, 'launcher_sha256': sha(__file__), 'cuda_visible_devices': '0',
            'same_physical_card_proof': 'NVML GPU0 UUID and PCI bus recorded before/after every independent subprocess',
            'stages': stages, 'no_historical_latency_reuse': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection-json', type=Path, default=ROOT / 'results/nr_push4_selection.json')
    ap.add_argument('--candidate-label', help='Optional existing full/pilot render label when final selection is not written')
    ap.add_argument('--plan-out', type=Path)
    ap.add_argument('--execute', action='store_true', help='ROOT scheduler only: run fresh CUDA subprocesses; omitted means plan only')
    args = ap.parse_args(); plan = make_plan(args.selection_json, args.candidate_label)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.plan_out:write(args.plan_out, plan)
    if not args.execute:return
    need(args.candidate_label is None and args.selection_json.resolve() == ROOT / 'results/nr_push4_selection.json',
         'Execute requires the formal final Push4 selection; candidate labels and self-smoke plans are planning only')
    # Never launch from an inherited nonzero CUDA selection or overwrite a prior pairing.
    need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '0'), 'Root must select CUDA physical GPU0')
    provenance = ROOT / 'calibration/push4_latency_launch.json'
    need(not provenance.exists(), 'Fresh paired launch already exists; preserve its evidence rather than overwrite')
    for stage in plan['stages']:need(not Path(stage['out']).exists(), 'Fresh output name already exists')
    initial = physical_gpu(); need(not initial['active_compute_processes'], 'Physical GPU0 is occupied')
    record = dict(plan, schema='push4_fresh_paired_latency_launch_v1', status='running',
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
