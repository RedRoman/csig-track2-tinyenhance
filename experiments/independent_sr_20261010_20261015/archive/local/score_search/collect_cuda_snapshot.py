"""Run the one-shot remote collector and preserve completed metadata stages."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

LOCAL = Path(__file__).resolve().parent
REMOTE = '/cache/independent_sr_cuda_20261013'
SSH = ['ssh', '-p', '30416', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       'huawei-modelarts-31514']


def run(command, source=None):
    return subprocess.run(SSH + [command], input=source, text=True,
                          capture_output=True, check=True).stdout


def atomic_local(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(body)
    temp.replace(path)


def main():
    body = run('python3 ' + REMOTE + '/code/update_cuda_progress.py')
    state = json.loads(body)
    atomic_local(LOCAL / 'results/status_cuda_search.json', body.encode())
    index_path = LOCAL / 'audits/cuda_completed_stage_downloads.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    downloaded = []
    for candidate in state['candidates']:
        stages = []
        if candidate['render_complete']:
            stages.append('render_complete')
        if candidate['stage'] == 'complete' and candidate.get('complete_result'):
            stages.append('score_complete')
        for stage in stages:
            key = candidate['label'] + ':' + stage
            if key in index:
                continue
            remote_directory = REMOTE + '/evaluations/' + candidate['label']
            source = ('from pathlib import Path\nimport json,hashlib\n'
                      'D=Path(' + repr(remote_directory) + ')\n'
                      'files=[p for p in D.rglob("*") if p.is_file() and '
                      'p.suffix in (".json",".jsonl",".csv") and "render" not in '
                      'p.relative_to(D).parts]\n'
                      'print(json.dumps({str(p.relative_to(D)):{"text":p.read_bytes().decode("utf-8"),'
                      '"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}))\n')
            bundle = json.loads(run('python3 -', source))
            proof = []
            for relative, value in bundle.items():
                path = Path(relative)
                assert not path.is_absolute() and '..' not in path.parts
                target = LOCAL / 'results' / candidate['label'] / path
                encoded = value['text'].encode()
                assert hashlib.sha256(encoded).hexdigest() == value['sha256']
                atomic_local(target, encoded)
                proof.append(dict(remote=remote_directory + '/' + relative,
                                  local=str(target), sha256=value['sha256']))
            index[key] = dict(downloaded_bjt=datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                source_port=30416, stage=stage, render_protocol_id=candidate['render_protocol_id'],
                files=proof)
            downloaded.append(key)
    atomic_local(index_path, json.dumps(index, ensure_ascii=False, indent=2).encode())
    print(json.dumps(dict(time_bjt=state['time_bjt'], downloaded_stages=downloaded,
        candidates=[dict(label=c['label'], stage=c['stage'], render=c['render_completed'],
                         scored=c['scored'], errors=c['errors']) for c in state['candidates']]),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
