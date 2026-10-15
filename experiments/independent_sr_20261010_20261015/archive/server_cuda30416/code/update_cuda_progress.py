"""Collect CUDA render/scoring progress once; never start a compute worker."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re

ROOT = Path('/cache/independent_sr_cuda_20261013')
LOG_MAP = {
    's8000_mt200_ct200_cuda_all': 'render_s8000_cuda.log',
    's6000_mt200_ct175_cuda_all': 'render_s6000_ct175_cuda.log',
    's6000_mt200_ct175_compiled_cuda_all': 'render_s6000_ct175_compiled_cuda.log',
}


def read_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_processes():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [x.decode(errors='replace') for x in
                    (proc / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
            if not any(str(ROOT) in x for x in args):
                continue
            scripts = [Path(x).name for x in args if x.endswith('.py')]
            if not any(re.search(r'(render|score|eval|benchmark)', x) for x in scripts):
                continue
            output = None
            for name in ('--out-dir', '--output-dir', '--output', '--out'):
                if name in args and args.index(name) + 1 < len(args):
                    output = args[args.index(name) + 1]
                    break
            flags = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            visible = next((x.split(b'=', 1)[1].decode() for x in
                            (proc / 'environ').read_bytes().split(b'\0')
                            if x.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
            found.append(dict(pid=int(proc.name), command=args, output=output,
                              cuda_visible_devices=visible, state=flags[0],
                              starttime_ticks=flags[19]))
        except (OSError, ProcessLookupError):
            continue
    return found


def collect():
    processes = active_processes()
    candidates = []
    for directory in sorted((ROOT / 'evaluations').iterdir()):
        if not directory.is_dir():
            continue
        render = read_json(directory / 'render_status.json') or {}
        controller = read_json(directory / 'status.json') or {}
        render_protocol = read_json(directory / 'render_protocol.json') or {}
        protocol = render_protocol.get('protocol', {})
        runner = protocol.get('runner', {})
        relevant = [p for p in processes if
                    (p['output'] and (p['output'] == str(directory) or
                                      p['output'].startswith(str(directory) + '/'))) or
                    any(x == str(directory) or str(directory) + '/' in x for x in p['command'])]
        render_active = [p for p in relevant if any('render_cuda_submission.py' in x for x in p['command'])]
        score_status = {}
        for name in ('stream_status.json', 'status.json', 'score_status.json'):
            value = read_json(directory / 'scores' / name)
            if value is not None:
                score_status = value
                break
        scored, measurements = 0, None
        for name in ('measurements.jsonl', 'metrics.jsonl'):
            path = directory / 'scores' / name
            if path.exists():
                scored = len([x for x in path.read_text().splitlines() if x.strip()])
                measurements = str(path)
                break
        result = None
        result_path = directory / 'scores/result.json'
        if result_path.is_file():
            result = read_json(result_path)
        complete_render = bool((directory / 'render_complete').is_file() and
                               (directory / 'render_manifest.json').is_file() and
                               render.get('completed') == render.get('total', 103))
        stage = 'rendering' if not complete_render else 'waiting_for_scoring'
        if score_status:
            stage = score_status.get('stage', score_status.get('status', 'scoring'))
        if result and result.get('status') == 'complete':
            stage = 'complete'
        elif not complete_render and not render_active:
            stage = 'needs_attention_render_incomplete_no_active_process'
        errors = []
        logs = list((directory / 'logs').glob('*.log'))
        mapped = LOG_MAP.get(directory.name)
        if mapped:
            logs.append(ROOT / 'logs' / mapped)
        for path in logs:
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 16000))
                tail = stream.read().decode(errors='replace')
            messages = [line[-600:] for line in tail.splitlines() if any(x in line for x in
                        ('Traceback (most recent call last)', 'RuntimeError:',
                         'AssertionError:', 'FileNotFoundError:', 'ValueError:', 'ERROR:'))]
            if messages:
                errors.append(dict(log=str(path), lines=messages[-4:]))
        entry = dict(label=directory.name, stage=stage, total=render.get('total', 103),
                     render_completed=render.get('completed', 0),
                     png_count=len(list((directory / 'render').rglob('*.png'))),
                     render_complete=complete_render, scored=scored,
                     render_status=render, score_status=score_status,
                     controller_status=controller, measurements=measurements,
                     active_processes=relevant, errors=errors,
                     render_protocol_id=render_protocol.get('protocol_id'),
                     render_protocol_file_sha256=sha(directory / 'render_protocol.json') if render_protocol else None,
                     renderer_source_sha256=protocol.get('source_sha256'),
                     runner_sha256=runner.get('runner_sha256'),
                     checkpoint_sha256=runner.get('checkpoint_sha256'),
                     runner_internal_precision=runner.get('internal_precision'))
        if complete_render:
            entry['render_manifest_sha256'] = sha(directory / 'render_manifest.json')
        if result and result.get('status') == 'complete':
            entry['complete_result'] = result
            entry['result_file_sha256'] = sha(result_path)
        candidates.append(entry)
    value = dict(schema='cuda_search_status_v1',
                 time_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                 root=str(ROOT), candidates=candidates, active_processes=processes,
                 note='CUDA quality and latency are separate; only complete result files are reported.')
    target = ROOT / 'results/status_cuda_search.json'
    target.parent.mkdir(exist_ok=True)
    temp = target.with_name(target.name + '.tmp.' + str(os.getpid()))
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp.write_text(body)
    temp.replace(target)
    print(body)


if __name__ == '__main__':
    collect()
