"""Fetch exact selected full-render PNGs. CPU-only; never reads pilot renders."""
from pathlib import Path
from PIL import Image
import datetime
import hashlib
import json
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = '/cache/independent_sr_cuda_20261013'
CASES = [7, 16, 41, 83]
PARENT = 'nr_conservative_s6250_compiled_all'
LABELS = ['nr_stage3_anchor6250_m1_s6500_compiled_all',
          'nr_stage3_anchor6250_m15_s6500_compiled_all']
REMOTE_CODE = r'''
from pathlib import Path
import os,json,hashlib,io
os.environ['CUDA_VISIBLE_DEVICES']=''
r=Path('/cache/independent_sr_cuda_20261013')
cases=[7,16,41,83]
result={'labels':{}}
for label in ['nr_stage3_anchor6250_m1_s6500_compiled_all','nr_stage3_anchor6250_m15_s6500_compiled_all']:
 p=r/'nr_push2_full_evaluations'/label
 item={'remote_directory':str(p),'records':[],'source_kind':'not_ready'}
 if (p/'render_protocol.json').exists():
  rp=json.loads((p/'render_protocol.json').read_text())
  assert rp['protocol']['cases']=='all'
  assert len(rp['protocol']['tasks'])==103
  assert [(t['split'],t['case']) for t in rp['protocol']['tasks']]==[('val',n) for n in range(1,4)]+[('test',n) for n in range(1,101)]
  mode='m15' if '_m15_' in label else 'm1'
  assert rp['protocol']['runner']['weight_path']==str(r/'adaptation/runs'/f'cuda_nr_stage3_s6250_anchor6250_{mode}_250'/'step6500.pth')
  assert rp['protocol']['runner']['model_t']==200 and rp['protocol']['runner']['coeff_t']==175
  item['render_protocol']=rp
  if (p/'render_manifest.json').exists():
   raw=(p/'render_manifest.json').read_bytes();m=json.loads(raw)
   assert m['scope']=='test100+validation3' and len(m['records'])==103
   item['source_kind']='final_full_render_manifest';item['final_manifest']=m
  elif (p/'render_progress.pt').exists():
   import torch
   raw=(p/'render_progress.pt').read_bytes()
   m=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
   assert m['checkpoint_sha256']==rp['protocol']['runner']['checkpoint_sha256']
   item['source_kind']='committed_full_render_progress_cpu'
  else:
   m={'records':[],'protocol_id':rp['protocol_id']};raw=b''
  assert m['protocol_id']==rp['protocol_id']
  item['protocol_id']=m['protocol_id'];item['state_file_sha256']=hashlib.sha256(raw).hexdigest()
  item['committed_count']=len(m['records'])
  for rec in m['records']:
   if rec['split']=='test' and rec['case'] in cases:
    path=p/rec['output']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
    item['records'].append(rec)
  metrics=p/'scores/measurements.jsonl'
  if metrics.exists():
   rows=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
   item['available_per_image_metrics']=[row for row in rows if row['split']=='test' and row['case'] in cases]
 result['labels'][label]=item
print(json.dumps(result))
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path, record):
    assert sha(path) == record['sha256'], str(path)
    with Image.open(path) as image:
        assert list(image.size) == record['size'], str(path)


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output = subprocess.run(['ssh', '-q', '-p', '30416', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                             'huawei-modelarts-31514', f'{ROOT}/env/bin/python -'],
                            input=REMOTE_CODE, text=True, capture_output=True, check=True)
    snapshot = json.loads(output.stdout)
    (OUT / 'provenance').mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (OUT / 'provenance' / f'remote_full_snapshot_{timestamp}.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    state = {'updated_utc': now, 'cases': CASES, 'sources': {},
             'limitation': 'Full-render PNG provenance only; no partial/full average NR is inferred.'}
    parent_manifest = BASE / 'nr_recheck_visual_20261014/provenance/nr_s6250_render_manifest.json'
    m = json.loads(parent_manifest.read_text())
    records = {row['case']: row for row in m['records'] if row['split'] == 'test'}
    parent_tasks = {row['case']: row for row in m['protocol']['tasks'] if row['split'] == 'test'}
    for case in CASES:
        record = records[case]
        original = BASE / 'nr_cuda_outputs' / PARENT / record['output']
        verify(original, record)
        target = OUT / PARENT / record['output']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(original, target)
        verify(target, record)
        state['sources'][f'{PARENT}/{case}'] = {
            'case': case, 'label': PARENT, 'local_path': str(target),
            'original_local_path': str(original),
            'remote_path': f'{ROOT}/nr_full_evaluations/{PARENT}/{record["output"]}',
            'sha256': record['sha256'], 'size_wh': record['size'],
            'source_kind': 'final_full_render_manifest', 'protocol_id': m['protocol_id'],
            'manifest_path': str(parent_manifest), 'final_manifest_verified': True,
        }
    summaries = []
    for label, item in snapshot['labels'].items():
        if 'render_protocol' in item:
            tasks = {row['case']: row for row in item['render_protocol']['protocol']['tasks'] if row['split'] == 'test'}
            for case in CASES:
                assert tasks[case]['input_sha256'] == parent_tasks[case]['input_sha256']
            (OUT / 'provenance' / f'{label}_render_protocol.json').write_text(json.dumps(item['render_protocol'], indent=2) + '\n')
        final_path = OUT / 'provenance' / f'{label}_final_render_manifest.json'
        if 'final_manifest' in item:
            if not final_path.exists() or sha(final_path) != item['state_file_sha256']:
                part = final_path.with_suffix('.json.part')
                subprocess.run(['scp', '-q', '-O', '-P30416',
                                f'huawei-modelarts-31514:{item["remote_directory"]}/render_manifest.json', str(part)], check=True)
                assert sha(part) == item['state_file_sha256']
                part.replace(final_path)
            assert json.loads(final_path.read_text()) == item['final_manifest']
        for record in item['records']:
            target = OUT / label / record['output']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                part = target.with_suffix('.png.part')
                subprocess.run(['scp', '-q', '-O', '-P30416',
                                f'huawei-modelarts-31514:{item["remote_directory"]}/{record["output"]}', str(part)], check=True)
                verify(part, record)
                part.replace(target)
            verify(target, record)
            state['sources'][f'{label}/{record["case"]}'] = {
                'case': record['case'], 'label': label, 'local_path': str(target),
                'remote_path': f'{item["remote_directory"]}/{record["output"]}',
                'sha256': record['sha256'], 'size_wh': record['size'],
                'source_kind': item['source_kind'], 'protocol_id': item['protocol_id'],
                'state_file_sha256': item['state_file_sha256'],
                'final_manifest_verified': 'final_manifest' in item,
                'manifest_path': str(final_path) if 'final_manifest' in item else None,
            }
        summaries.append({'label': label, 'committed_full_render_count': item.get('committed_count', 0),
                          'selected_cases_available': [row['case'] for row in item['records']],
                          'source_kind': item['source_kind']})
    state['remote_summaries'] = summaries
    state['per_image_metrics'] = {label: item.get('available_per_image_metrics', [])
                                  for label, item in snapshot['labels'].items()}
    state['all_4_cases_downloaded'] = all(f'{label}/{case}' in state['sources'] for label in LABELS for case in CASES)
    state['all_final_manifests_verified'] = state['all_4_cases_downloaded'] and all(
        source['final_manifest_verified'] for source in state['sources'].values())
    (OUT / 'download_state.json').write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'summaries': summaries, 'all_4_cases_downloaded': state['all_4_cases_downloaded'],
                      'all_final_manifests_verified': state['all_final_manifests_verified']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
