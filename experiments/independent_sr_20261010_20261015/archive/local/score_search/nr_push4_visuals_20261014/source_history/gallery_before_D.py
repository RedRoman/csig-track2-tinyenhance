"""CPU-only Push4 full-output gallery, with explicit fetch and pixel audit.

init reuses four existing parent7250 PNGs as hardlinks. fetch reads only an
explicit native full103 candidate directory and downloads at most four PNGs.
Default fetch requires a final full manifest; --allow-progress permits committed
full-render records, with a mandatory final-manifest audit before final delivery.
No full mean is calculated from selected cases. No original pixels are modified.
"""
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont
from html.parser import HTMLParser
import argparse
import datetime
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = '/cache/independent_sr_cuda_20261013'
HOST = 'huawei-modelarts-31514'
PARENT = 'push3_c7250_compiled_all'
PARENT_SHA = 'fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e'
CASES = [7, 16, 41, 83]
BOXES = {7: (1856, 1408, 2496, 1856), 16: (1280, 1216, 1920, 1728),
         41: (768, 1664, 1408, 2176), 83: (1216, 1920, 1856, 2432)}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def protocol_identity(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def verify_full_protocol(protocol, expected_sha):
    p = protocol['protocol']
    expected = [('val', n) for n in range(1, 4)] + [('test', n) for n in range(1, 101)]
    need(p['cases'] == 'all' and len(p['tasks']) == 103, 'Native full103 protocol required; pilot forbidden')
    need([(x['split'], x['case']) for x in p['tasks']] == expected, 'Native full task order differs')
    need(protocol_identity(p) == protocol['protocol_id'], 'Render protocol ID differs')
    need(p['runner']['checkpoint_sha256'] == expected_sha, 'Explicit checkpoint SHA differs')
    return p


def verify_image(path, record):
    need(sha(path) == record['sha256'], f'Original PNG SHA mismatch: {path}')
    with Image.open(path) as image:
        need(image.mode == 'RGB' and image.format == 'PNG', f'Native RGB PNG required: {path}')
        need(list(image.size) == record['size'] and sorted(image.size) == [3072, 4096], f'Image size differs: {path}')


def full_quality(result, rows):
    if result.get('status') != 'complete':
        return None
    need(result['official_full_count'] is True and result['completed'] == result['expected'] == 103,
         'Complete full quality must contain103 rows')
    need(result['S_proxy_scope'] == 'full100_test_NR_plus3val_FR', 'Full score scope differs')
    order = [('validation', n) for n in range(1, 4)] + [('test', n) for n in range(1, 101)]
    need([(r['split'], r['case']) for r in rows] == order, 'Full metrics order differs')
    means = {}
    for split, count, keys in [('validation', 3, ['lpips', 'dists', 'niqe', 'musiq', 'maniqa', 'clipiqa']),
                               ('test', 100, ['niqe', 'musiq', 'maniqa', 'clipiqa'])]:
        selected = [r for r in rows if r['split'] == split]
        need(len(selected) == count and all(set(r['metrics']) == set(keys) for r in selected), 'Metric count/key set differs')
        means[split] = {}
        for key in keys:
            values = [r['metrics'][key] for r in selected]
            need(all(math.isfinite(x) for x in values), 'Nonfinite full metric')
            means[split][key] = sum(values) / count
            need(abs(means[split][key] - result[split][key]) <= 1e-12, 'Published full metric mean differs')
    t, v = means['test'], means['validation']
    nr = t['clipiqa'] + t['maniqa'] + t['musiq'] / 100 + max(0, (10 - t['niqe']) / 10)
    fr = 2 - v['lpips'] - v['dists']
    need(abs(nr - result['test']['nr']) <= 1e-12 and abs(fr - result['validation']['fr']) <= 1e-12
         and abs(nr + fr - result['S_proxy']) <= 1e-12, 'Full NR/FR/S formula differs')
    return {'NR': nr, 'FR': fr, 'S_proxy': nr + fr, 'component_means': means,
            'scope': 'Original full100 test NR + validation3 FR; never inferred from four selected cases'}


def init():
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / 'sources.json'
    if state_path.exists():
        state = read(state_path)
        need(state['models'][0]['label'] == PARENT, 'Existing gallery parent differs')
        return state
    previous = BASE / 'nr_push3_visuals_20261014'
    old_state = read(previous / 'sources.json')
    parents = [m for m in old_state['models'] if m['label'] == PARENT]
    need(len(parents) == 1, 'Prior best7250 source must exist exactly once')
    prior = parents[0]
    need(prior['checkpoint_sha256'] == PARENT_SHA and prior['final_manifest'], 'Prior parent checkpoint/final identity differs')
    manifest_path = Path(prior['manifest_path']); manifest = read(manifest_path)
    need(sha(manifest_path) == prior['manifest_sha256'], 'Prior full manifest bytes changed')
    p = verify_full_protocol({'protocol_id': manifest['protocol_id'], 'protocol': manifest['protocol']}, PARENT_SHA)
    need(manifest['scope'] == 'test100+validation3' and len(manifest['records']) == 103, 'Parent final full manifest required')
    provenance = OUT / 'provenance' / PARENT
    provenance.mkdir(parents=True, exist_ok=True)
    old_provenance = previous / 'provenance' / PARENT
    for name in ('render_manifest.json', 'render_protocol.json', 'result.json', 'measurements.jsonl'):
        shutil.copyfile(old_provenance / name, provenance / name)
    result = read(provenance / 'result.json')
    rows = [json.loads(line) for line in (provenance / 'measurements.jsonl').read_text().splitlines() if line.strip()]
    model = {'label': PARENT, 'name': 'Parent7250 / best Push3',
             'remote_directory': ROOT + '/nr_push3_full_evaluations/' + PARENT,
             'checkpoint_sha256': PARENT_SHA, 'manifest_path': str(provenance / 'render_manifest.json'),
             'manifest_sha256': sha(provenance / 'render_manifest.json'), 'final_manifest': True,
             'full_quality': full_quality(result, rows), 'sources': {}}
    need(model['full_quality'] == prior['full_quality'], 'Parent full103/418 means differ from prior proof')
    records = {r['case']: r for r in manifest['records'] if r['split'] == 'test'}
    for case in CASES:
        record = records[case]; original = Path(prior['sources'][str(case)]['local_path'])
        verify_image(original, record)
        target = OUT / PARENT / record['output']; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(original, target)
        verify_image(target, record)
        need(target.stat().st_ino == original.stat().st_ino, 'Parent must reuse original bytes through a hardlink')
        metric = next(row for row in rows if row['split'] == 'test' and row['case'] == case)
        need(metric['sr_sha256'] == record['sha256'], 'Parent metric PNG SHA differs')
        model['sources'][str(case)] = dict(case=case, local_path=str(target), original_local_path=str(original),
            remote_path=model['remote_directory'] + '/' + record['output'], sha256=record['sha256'],
            size_wh=record['size'], record=record, parent_hardlink=True, metric=metric)
    state = {'schema': 'push4_full_gallery_sources_v1', 'updated_utc': now(), 'cases': CASES,
             'roi_boxes_xyxy_exclusive': BOXES, 'models': [model],
             'scope': 'Native full outputs only; no pilot PNGs; four examples never define a full mean'}
    write(state_path, state)
    return state


REMOTE_SNAPSHOT = r'''
from pathlib import Path
import hashlib,io,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
q=json.loads(REQUEST_JSON)
d=Path(q['directory']).resolve();r=Path('/cache/independent_sr_cuda_20261013')
assert d.is_relative_to(r/'nr_push4_full_evaluations') and d.name==q['label']
rp=json.loads((d/'render_protocol.json').read_text());p=rp['protocol']
order=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
assert p['cases']=='all' and [(t['split'],t['case']) for t in p['tasks']]==order
assert p['runner']['checkpoint_sha256']==q['checkpoint_sha256']
if (d/'render_manifest.json').exists():
 raw=(d/'render_manifest.json').read_bytes();m=json.loads(raw);final=True
 assert m['scope']=='test100+validation3' and len(m['records'])==103 and m['protocol']==p
elif q['allow_progress']:
 import torch
 raw=(d/'render_progress.pt').read_bytes();m=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);final=False
 assert m['checkpoint_sha256']==q['checkpoint_sha256'] and not torch.cuda.is_initialized()
else:
 raise RuntimeError('Final full manifest unavailable; explicit --allow-progress required for committed full preview')
assert m['protocol_id']==rp['protocol_id']
records=m['records'];assert [(x['split'],x['case']) for x in records]==order[:len(records)]
selected=[x for x in records if x['split']=='test' and x['case'] in q['cases']]
for rec in selected:
 path=(d/rec['output']).resolve();assert path.is_relative_to(d)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
score=d/'scores/result.json';metrics=d/'scores/measurements.jsonl'
rows=[json.loads(x) for x in metrics.read_text().splitlines() if x.strip()] if metrics.exists() else []
print(json.dumps(dict(directory=str(d),protocol=rp,final_manifest=final,
 manifest=m if final else None,state_file_sha256=hashlib.sha256(raw).hexdigest(),
 records=selected,committed_full_render_count=len(records),result=json.loads(score.read_text()) if score.exists() else {},metrics=rows)))
'''


def fetch(args):
    state = init()
    need(re.fullmatch(r'[A-Za-z0-9_-]+', args.label), 'Safe candidate label required')
    need(re.fullmatch(r'[0-9a-f]{64}', args.checkpoint_sha256), 'Explicit candidate checkpoint SHA256 required')
    need(args.label != PARENT, 'Candidate cannot overwrite parent')
    need(args.label in state.get('expected_candidates', []), 'Configure root-chosen full labels before fetching; pilot labels are forbidden')
    old = next((m for m in state['models'] if m['label'] == args.label), None)
    need(old is not None or len(state['models']) < 3, 'Gallery supports parent plus at most two full candidates')
    request = dict(directory=args.directory, label=args.label, checkpoint_sha256=args.checkpoint_sha256,
                   allow_progress=args.allow_progress, cases=CASES)
    code = 'REQUEST_JSON=' + repr(json.dumps(request)) + '\n' + REMOTE_SNAPSHOT
    call = subprocess.run(['ssh', '-q', '-p', '30416', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                           HOST, ROOT + '/env/bin/python -B -'], input=code, text=True, capture_output=True, check=True)
    snapshot = json.loads(call.stdout)
    p = verify_full_protocol(snapshot['protocol'], args.checkpoint_sha256)
    parent = state['models'][0]
    parent_manifest = read(parent['manifest_path'])
    parent_tasks = {(t['split'], t['case']): t for t in parent_manifest['protocol']['tasks']}
    need(all(t['input_sha256'] == parent_tasks[(t['split'], t['case'])]['input_sha256'] for t in p['tasks']),
         'Candidate and parent input hashes/order differ')
    provenance = OUT / 'provenance' / args.label; provenance.mkdir(parents=True, exist_ok=True)
    write(provenance / 'snapshot.json', snapshot)
    write(provenance / 'render_protocol.json', snapshot['protocol'])
    if snapshot['final_manifest']:
        final_path = provenance / 'render_manifest.json'
        if not final_path.exists() or sha(final_path) != snapshot['state_file_sha256']:
            part = final_path.with_suffix('.json.part')
            subprocess.run(['scp', '-q', '-O', '-P', '30416',
                            HOST + ':' + snapshot['directory'] + '/render_manifest.json', str(part)], check=True)
            need(sha(part) == snapshot['state_file_sha256'], 'Remote final manifest byte SHA differs')
            need(read(part) == snapshot['manifest'], 'Final manifest content differs from snapshot')
            part.replace(final_path)
        need(read(final_path) == snapshot['manifest'], 'Final full manifest content differs')
    write(provenance / 'result.json', snapshot['result'])
    (provenance / 'measurements.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in snapshot['metrics']))
    model = {'label': args.label, 'name': args.name or args.label, 'remote_directory': snapshot['directory'],
             'checkpoint_sha256': args.checkpoint_sha256, 'final_manifest': snapshot['final_manifest'],
             'manifest_path': str(provenance / 'render_manifest.json') if snapshot['final_manifest'] else None,
             'manifest_sha256': sha(provenance / 'render_manifest.json') if snapshot['final_manifest'] else None,
             'remote_state_file_sha256': snapshot['state_file_sha256'],
             'committed_full_render_count': snapshot['committed_full_render_count'],
             'full_quality': full_quality(snapshot['result'], snapshot['metrics']), 'sources': {}}
    for record in snapshot['records']:
        case = record['case']; target = OUT / args.label / record['output']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            part = target.with_suffix('.png.part')
            subprocess.run(['scp', '-q', '-O', '-P', '30416',
                            HOST + ':' + snapshot['directory'] + '/' + record['output'], str(part)], check=True)
            verify_image(part, record); part.replace(target)
        verify_image(target, record)
        metric = next((r for r in snapshot['metrics'] if r['split'] == 'test' and r['case'] == case), None)
        if metric:
            need(metric['sr_sha256'] == record['sha256'], 'Selected per-image metric SHA differs')
        model['sources'][str(case)] = dict(case=case, local_path=str(target),
            remote_path=snapshot['directory'] + '/' + record['output'], sha256=record['sha256'],
            size_wh=record['size'], record=record, parent_hardlink=False, metric=metric)
    if old:
        state['models'][state['models'].index(old)] = model
    else:
        state['models'].append(model)
    state['updated_utc'] = now(); write(OUT / 'sources.json', state)
    return {'label': args.label, 'selected_cases': list(model['sources']),
            'final_manifest': model['final_manifest'], 'full_quality_complete': model['full_quality'] is not None}


def configure(args):
    state = init(); labels = args.candidate_label
    need(1 <= len(labels) <= 2 and len(set(labels)) == len(labels), 'Choose one or two unique full candidate labels')
    need(all(re.fullmatch(r'[A-Za-z0-9_-]+', label) and label != PARENT for label in labels), 'Safe new candidate labels required')
    need(all('pilot' not in label.lower() and label.endswith('_all') for label in labels), 'Full candidate labels only')
    need(all(model['label'] in [PARENT] + labels for model in state['models']), 'Configured labels cannot hide already fetched candidates')
    if args.selected_label:need(args.selected_label in [PARENT] + labels, 'Selected label must be parent or a configured full candidate')
    state['expected_candidates'] = labels
    state['expected_selected_label'] = args.selected_label
    state['updated_utc'] = now();write(OUT / 'sources.json', state)
    return {'expected_full_candidates': labels, 'expected_selected_label': args.selected_label, 'candidate_downloads': 0}


def select(args):
    state = init(); label = args.selected_label
    need(label in [PARENT] + state.get('expected_candidates', []), 'Selected label must be parent or configured')
    if state.get('expected_selected_label'):
        need(label == state['expected_selected_label'], 'Selection differs from configured winner')
    path = OUT / 'provenance/final_selection.json';path.parent.mkdir(parents=True, exist_ok=True)
    if args.selection_json:
        shutil.copyfile(args.selection_json, path)
    else:
        remote = ROOT + '/results/nr_push4_selection.json'
        code = 'from pathlib import Path;import hashlib;print(hashlib.sha256(Path(' + repr(remote) + ').read_bytes()).hexdigest())'
        expected = subprocess.check_output(['ssh', '-q', '-p', '30416', '-o', 'BatchMode=yes', HOST,
            ROOT + '/env/bin/python -B -c ' + __import__('shlex').quote(code)], text=True).strip()
        subprocess.run(['scp', '-q', '-O', '-P', '30416', HOST + ':' + remote, str(path)], check=True)
        need(sha(path) == expected, 'Final selection original byte SHA differs')
    selected = read(path);model = next((m for m in state['models'] if m['label'] == label), None)
    need(model is not None and model['final_manifest'] and model['full_quality'] is not None, 'Fetch selected full103 completion before binding winner')
    need(selected['status'] == 'selected_after_independent_full_audit' and selected['label'] == label
         and selected['checkpoint_sha256'] == model['checkpoint_sha256'], 'Final selected full identity differs')
    need(all(abs(selected[k] - model['full_quality'][k]) <= 1e-12 for k in ('NR', 'FR', 'S_proxy')), 'Final selected full103/418 score differs')
    state['selection'] = {'label': label, 'path': str(path), 'sha256': sha(path), 'checkpoint_sha256': model['checkpoint_sha256']}
    for row in state['models']:
        row['name'] = row['name'].removesuffix(' | SELECTED BEST') + (' | SELECTED BEST' if row['label'] == label else '')
    state['expected_selected_label'] = label;state['updated_utc'] = now()
    write(OUT / 'sources.json', state)
    return state['selection']


def font(size):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def image(path):
    with Image.open(path) as im:
        return im.copy()


def link(path, label):
    return '<a href="' + html.escape(os.path.relpath(path, OUT), quote=True) + '">' + html.escape(label) + '</a>'


def build():
    state = read(OUT / 'sources.json'); models = state['models']
    need(2 <= len(models) <= 3, 'Fetch at least one explicitly selected full candidate before building')
    cases = [c for c in CASES if all(str(c) in m['sources'] for m in models)]
    need(cases, 'No common committed full cases yet')
    f, fs, ft = font(20), font(16), font(27)
    gap, cell, rowheight = 16, 512, 560
    outputs = []; panes = []; source_files = []
    for model in models:
        for case in cases:
            source = model['sources'][str(case)]; verify_image(source['local_path'], source['record'])
            source_files.append(dict(id=f"{model['label']}/{case}", label=model['label'], **source))
    overview_path = OUT / 'overview_full_comparison.png'
    overview = Image.new('RGB', (gap + len(models) * (cell + gap), 112 + rowheight * len(cases)), '#f2f5fa')
    d = ImageDraw.Draw(overview)
    d.text((16, 12), 'Full-render originals | overview thumbnails at exactly 1/8', font=ft, fill='#172133')
    d.text((16, 50), 'Parent7250 vs new full candidates. Native pixel crops below; no original pixel edits.', font=fs, fill='#172133')
    for col, model in enumerate(models):
        d.text((gap + col * (cell + gap), 80), model['name'], font=f, fill='#172133')
    for row, case in enumerate(cases):
        y = 112 + row * rowheight
        d.text((16, y), f'case{case} | full-render originals', font=f, fill='#172133')
        for col, model in enumerate(models):
            source = model['sources'][str(case)]; im = image(source['local_path'])
            size = (im.width // 8, im.height // 8); thumb = im.resize(size, Image.Resampling.LANCZOS)
            xy = (gap + col * (cell + gap) + (cell - size[0]) // 2, y + 32 + (cell - size[1]) // 2)
            overview.paste(thumb, xy)
            panes.append(dict(type='overview_thumbnail', case=case, source=f"{model['label']}/{case}",
                output_path=str(overview_path), output_size_wh=list(size), paste_xy=list(xy),
                scale_xy=[.125, .125], resampling='Pillow Lanczos'))
    overview.save(overview_path); outputs.append(overview_path)
    for case in cases:
        box = BOXES[case]; width, height = box[2] - box[0], box[3] - box[1]
        roi_path = OUT / f'roi_case{case}_native_1to1.png'
        canvas = Image.new('RGB', (gap + len(models) * (width + gap), height + 110), '#f2f5fa')
        dd = ImageDraw.Draw(canvas)
        dd.text((16, 10), f'case{case} | native pixels 1:1 | crop {box}', font=f, fill='#172133')
        dd.text((16, 40), 'Same integer coordinates. No ROI resize, sharpening or retouching.', font=fs, fill='#172133')
        for col, model in enumerate(models):
            x = gap + col * (width + gap); dd.text((x, 72), model['name'], font=f, fill='#172133')
            canvas.paste(image(model['sources'][str(case)]['local_path']).crop(box), (x, 104))
            panes.append(dict(type='native_roi', case=case, source=f"{model['label']}/{case}",
                output_path=str(roi_path), box_xyxy_exclusive=list(box), paste_xy=[x, 104], scale_xy=[1, 1], resampling=None))
        canvas.save(roi_path); outputs.append(roi_path)
    selected = [dict(case=c, label=m['label'], metric=m['sources'][str(c)]['metric'],
                     scope='Individual rows from full100 sequence only; never a selected-case mean')
                for c in cases for m in models]
    write(OUT / 'selected_per_image_metrics.json', selected)
    preview = dict(schema='push4_scientific_full_visuals_v1', created_utc=now(),
        final_selection=state.get('selection'),
        models=[dict(label=m['label'], name=m['name'], checkpoint_sha256=m['checkpoint_sha256'],
                     final_manifest=m['final_manifest'], manifest_path=m['manifest_path'],
                     manifest_sha256=m['manifest_sha256'], full_quality=m['full_quality']) for m in models],
        cases=cases, all_four_cases_available=(cases == CASES), all_final_manifests=all(m['final_manifest'] for m in models),
        source_files=source_files, coordinate_convention='integer [left,top,right,bottom), top-left source origin',
        previews=panes, pillow_version=Image.__version__,
        outputs=[dict(path=str(p), sha256=sha(p), size_wh=list(image(p).size)) for p in outputs],
        limitation='No test GT; metric gain does not prove true detail. No full mean inferred from four examples.')
    write(OUT / 'preview_manifest.json', preview)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Push4 full 原像素比较</title><style>body{font:16px/1.6 system-ui;max-width:1600px;margin:24px auto;padding:0 20px;background:#f5f7fb;color:#172133}table{border-collapse:collapse;background:white}th,td{border:1px solid #ccd4df;padding:8px}a{color:#1759a9}figure{margin:24px 0;padding:16px;background:white;border:1px solid #dbe1eb}.overview{width:100%;height:auto}.native{overflow:auto}.native img{max-width:none}.note{white-space:pre-wrap}</style>',
        '<h1>父7250 与第四轮候选 · full 原图比较</h1>',
        '<p>总览为原图1/8缩略，ROI为相同整数坐标的1:1原像素。仅使用full推理输出，测试参考真值不可用；指标提高不能直接证明细节真实恢复。</p>',
        '<p>' + link(OUT / 'preview_manifest.json', '源SHA、坐标与比例') + ' · ' + link(OUT / 'sources.json', '完整来源状态') + ' · ' + link(OUT / 'selected_per_image_metrics.json', '逐图记录') + '</p>']
    notes = OUT / 'visual_observations_zh.txt'
    if notes.exists():
        parts.append('<p class="note">' + html.escape(notes.read_text()) + '</p>')
    validation_index = BASE / 'nr_push4_val_visuals_20261014/index.html'
    if validation_index.exists():
        parts.append('<p>' + link(validation_index, '验证GT负例：NR收益与FR下降的逐图诊断') + '</p>')
    parts.append('<table><tr><th>完整100测试+3验证</th><th>NR</th><th>FR</th><th>S</th><th>CLIPIQA</th><th>MANIQA</th><th>MUSIQ</th><th>NIQE</th></tr>')
    for m in models:
        q = m['full_quality']
        if q:
            t = q['component_means']['test']; values = [q['NR'], q['FR'], q['S_proxy'], t['clipiqa'], t['maniqa'], t['musiq'], t['niqe']]
            cells = ''.join('<td>' + f'{v:.8f}' + '</td>' for v in values)
        else:
            cells = '<td colspan="7">完整评分尚未完成，不从示例图推断均分</td>'
        parts.append('<tr><td>' + html.escape(m['name']) + '</td>' + cells + '</tr>')
    parts.append('</table><p>以下原PNG与逐图NR来自同一full100推理/评分序列；四张示例不构成全量平均。</p><table><tr><th>病例</th>')
    parts.extend('<th>' + html.escape(m['name']) + '</th>' for m in models)
    parts.append('</tr>')
    for case in cases:
        parts.append('<tr><td>' + str(case) + '</td>')
        for m in models:
            s = m['sources'][str(case)]; metric = s['metric']
            value = f" · 逐图NR {metric['nr_per_image']:.8f}" if metric else ' · 待评分'
            parts.append('<td>' + link(s['local_path'], '原PNG') + value + '</td>')
        parts.append('</tr>')
    parts.extend(['</table>', '<figure><figcaption>全图总览 · 1/8缩略</figcaption><a href="overview_full_comparison.png"><img class="overview" src="overview_full_comparison.png"></a></figure>'])
    for case in cases:
        name = f'roi_case{case}_native_1to1.png'
        parts.append(f'<figure><figcaption>case{case} · 1:1原像素ROI · {BOXES[case]}</figcaption><div class="native"><img src="{name}"></div></figure>')
    parts.append('</html>'); (OUT / 'index.html').write_text('\n'.join(parts))
    return {'models': len(models), 'cases': cases, 'index': str(OUT / 'index.html')}


def audit(final=False):
    state = read(OUT / 'sources.json'); preview = read(OUT / 'preview_manifest.json')
    if final:
        need(preview['all_four_cases_available'] and preview['all_final_manifests'], 'Final gallery needs all4 cases and final full manifests')
        need(all(m['full_quality'] is not None for m in state['models']), 'Final gallery needs complete full103 quality per model')
        need(state.get('selection') is not None, 'Final gallery requires explicit final-selection binding')
        need([m['label'] for m in state['models'][1:]] == state.get('expected_candidates'), 'Final candidate coverage/order differs from root configuration')
    models = {m['label']: m for m in state['models']}
    selection_proof = state.get('selection')
    if selection_proof:
        need(sha(selection_proof['path']) == selection_proof['sha256'], 'Final selected model JSON SHA differs')
        selection = read(selection_proof['path']); selected_model = models[selection_proof['label']]
        need(selection['status'] == 'selected_after_independent_full_audit' and
             selection['label'] == selected_model['label'] and
             selection['checkpoint_sha256'] == selected_model['checkpoint_sha256'], 'Final selected model differs')
        need(all(abs(selection[k] - selected_model['full_quality'][k]) <= 1e-12 for k in ('NR', 'FR', 'S_proxy')),
             'Final selection full quality differs from gallery complete rows')
        need(preview.get('final_selection') == selection_proof, 'Preview selection proof differs')
    sources = {s['id']: s for s in preview['source_files']}; originals = []; roi_checks = []
    expected_sources = {f'{label}/{case}' for label in models for case in preview['cases']}
    need(set(sources) == expected_sources and len(preview['source_files']) == len(expected_sources), 'Unique source coverage differs')
    for kind in ('native_roi', 'overview_thumbnail'):
        need([p['source'] for p in preview['previews'] if p['type'] == kind] and
             len([p for p in preview['previews'] if p['type'] == kind]) == len(expected_sources) and
             {p['source'] for p in preview['previews'] if p['type'] == kind} == expected_sources,
             'Unique preview pane coverage differs: ' + kind)
    for model in models.values():
        provenance = OUT / 'provenance' / model['label']
        result = read(provenance / 'result.json')
        rows = [json.loads(line) for line in (provenance / 'measurements.jsonl').read_text().splitlines() if line.strip()]
        need(full_quality(result, rows) == model['full_quality'], 'Saved full quality differs from complete original rows')
        for case in preview['cases']:
            source = model['sources'][str(case)]
            metric = next((r for r in rows if r['split'] == 'test' and r['case'] == case), None)
            need(metric == source['metric'], 'Saved selected metric row differs')
            if metric:
                need(metric['sr_sha256'] == source['sha256'], 'Selected metric row PNG SHA differs')
    for key, source in sources.items():
        model = models[source['label']]; verify_image(source['local_path'], source['record'])
        if model['final_manifest']:
            need(sha(model['manifest_path']) == model['manifest_sha256'], 'Final manifest bytes changed')
            if 'remote_state_file_sha256' in model:
                need(model['manifest_sha256'] == model['remote_state_file_sha256'], 'Final manifest original remote byte SHA differs')
            m = read(model['manifest_path'])
            need(m['scope'] == 'test100+validation3' and len(m['records']) == 103, 'Final full manifest scope differs')
            rec = next(r for r in m['records'] if r['split'] == 'test' and r['case'] == source['case'])
            need(rec == source['record'], 'Source record differs from final full manifest')
        originals.append(dict(source_id=key, actual_sha256=sha(source['local_path']),
                              final_manifest_verified=model['final_manifest'], original_sha_unchanged=True))
    for pane in preview['previews']:
        expected = image(sources[pane['source']]['local_path'])
        if pane['type'] == 'native_roi':
            need(pane['scale_xy'] == [1, 1] and pane['resampling'] is None, 'ROI resampling forbidden')
            need(pane['box_xyxy_exclusive'] == list(BOXES[pane['case']]), 'Shared integer ROI coordinates differ')
            expected = expected.crop(pane['box_xyxy_exclusive'])
        else:
            need(pane['scale_xy'] == [.125, .125], 'Overview must be exactly1/8')
            expected = expected.resize(pane['output_size_wh'], Image.Resampling.LANCZOS)
        x, y = pane['paste_xy']; actual = image(pane['output_path']).crop((x, y, x + expected.width, y + expected.height))
        need(ImageChops.difference(expected, actual).getbbox() is None, 'Panel pixels differ from declared source operation')
        if pane['type'] == 'native_roi':
            roi_checks.append(dict(source_id=pane['source'], exact_source_pixel_match=True,
                                   box_xyxy_exclusive=pane['box_xyxy_exclusive'], paste_xy=pane['paste_xy']))
    for output in preview['outputs']:
        need(sha(output['path']) == output['sha256'], 'Preview output SHA changed')
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('href', 'src'):
                    need((OUT / value).exists(), 'Broken HTML link: ' + value)
    Links().feed((OUT / 'index.html').read_text())
    report = dict(schema='push4_native_full_visual_CPU_audit_v1', status='PASS', created_utc=now(), CPU_only=True,
        final_delivery=final, models=len(models), cases=preview['cases'], original_png_count=len(originals),
        original_png_checks=originals, native_roi_exact_pixel_checks=roi_checks,
        all_original_png_sha_unchanged=True, all_html_links_exist=True, full_outputs_only=True,
        overview_scale_xy=[.125, .125], overview_resampling='Pillow Lanczos thumbnails only',
        no_full_mean_inferred_from_examples=True, preview_manifest_sha256=sha(OUT / 'preview_manifest.json'),
        final_selection=selection_proof,
        gallery_source_sha256=sha(__file__))
    write(OUT / ('final_visual_audit.json' if final else 'preview_visual_audit.json'), report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    c = sub.add_parser('configure');c.add_argument('--candidate-label', action='append', required=True)
    c.add_argument('--selected-label')
    f = sub.add_parser('fetch'); f.add_argument('--label', required=True); f.add_argument('--directory', required=True)
    f.add_argument('--checkpoint-sha256', required=True); f.add_argument('--name'); f.add_argument('--allow-progress', action='store_true')
    sub.add_parser('build'); a = sub.add_parser('audit'); a.add_argument('--final', action='store_true')
    s = sub.add_parser('select');s.add_argument('--selected-label', required=True);s.add_argument('--selection-json', type=Path)
    args = ap.parse_args()
    if args.command == 'init':
        state = init(); result = {'parent_hardlinks': len(state['models'][0]['sources']), 'candidate_downloads': 0}
    elif args.command == 'fetch': result = fetch(args)
    elif args.command == 'configure': result = configure(args)
    elif args.command == 'select': result = select(args)
    elif args.command == 'build': result = build()
    else: result = audit(args.final)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
