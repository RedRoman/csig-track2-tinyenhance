"""CPU-only worst-validation FR diagnostic; no full test means or image edits."""
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont
import datetime
import hashlib
import html
import json
import os
import subprocess
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = '/cache/independent_sr_cuda_20261013'
HOST = 'huawei-modelarts-31514'
PARENT = 'nr_stage3_anchor6250_m15_s6500_compiled_all'
LABELS = [PARENT, 'push3_c7000_compiled_all', 'push3_c7250_compiled_all']
NAMES = ['Parent 6500 / M1.5', 'C7000 NR-strong', 'C7250 NR-strong']
SHA = ['cfcc3013fa87aa721f346d6cc285e70b6d6ff2c88315ce9f1465b07ffe020e68',
       '122c362902c204d60c4cd8433019b9b8ebcde4ffccc8e789094b4c71ef022df0',
       'fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def image(path):
    with Image.open(path) as im:
        return im.convert('RGB')


REMOTE = r'''
from pathlib import Path
import hashlib,io,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
r=Path('/cache/independent_sr_cuda_20261013');q=json.loads(REQUEST_JSON)
models=[]
for index,(label,checkpoint) in enumerate(zip(q['labels'],q['sha'])):
 d=r/('nr_push2_full_evaluations' if index==0 else 'nr_push3_full_evaluations')/label
 rp=json.loads((d/'render_protocol.json').read_text());p=rp['protocol']
 assert p['cases']=='all' and len(p['tasks'])==103 and p['runner']['checkpoint_sha256']==checkpoint
 assert hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()==rp['protocol_id']
 if (d/'render_manifest.json').exists():
  raw=(d/'render_manifest.json').read_bytes();m=json.loads(raw);final=True
  assert m['scope']=='test100+validation3' and len(m['records'])==103 and m['protocol']==p
 else:
  raw=(d/'render_progress.pt').read_bytes();m=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);final=False
  assert m['checkpoint_sha256']==checkpoint
 assert m['protocol_id']==rp['protocol_id']
 rows=[json.loads(x) for x in (d/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
 val=[row for row in rows if row['split']=='validation'];assert [x['case'] for x in val]==[1,2,3]
 score=torch.load(d/'scores/rng_checkpoint.pt',map_location='cpu',weights_only=False)
 assert score['rows'][:3]==val
 records=[]
 for case in [1,2,3]:
  rec=next(x for x in m['records'] if x['split']=='val' and x['case']==case);row=val[case-1]
  assert rec['sha256']==row['sr_sha256'] and (d/rec['output']).resolve()==Path(row['sr_path']).resolve()
  gt=r/'scoring_tool/data/验证集'/('case3_gt.jpg' if case==3 else f'case{case}_gt.png')
  actual_gt=hashlib.sha256(gt.read_bytes()).hexdigest();assert actual_gt==row['gt_sha256']
  task=p['tasks'][case-1];assert hashlib.sha256(Path(task['input']).read_bytes()).hexdigest()==task['input_sha256']
  records.append(dict(record=rec,row=row,GT_path=str(gt),GT_sha256=actual_gt,input_sha256=task['input_sha256']))
 models.append(dict(label=label,directory=str(d),checkpoint_sha256=checkpoint,protocol_id=m['protocol_id'],
                    final_manifest=final,state_SHA256=hashlib.sha256(raw).hexdigest(),
                    full_render_committed=len(m['records']),validation=records))
table=[]
for case in [1,2,3]:
 values=[2-m['validation'][case-1]['row']['metrics']['lpips']-m['validation'][case-1]['row']['metrics']['dists'] for m in models]
 delta=[x-values[0] for x in values[1:]]
 table.append(dict(case=case,FR=values,delta_vs_parent=delta,mean_candidate_FR_delta=sum(delta)/2))
chosen=min(table,key=lambda x:x['mean_candidate_FR_delta'])['case']
for m in models:
 rec=m['validation'][chosen-1];path=Path(m['directory'])/rec['record']['output']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['record']['sha256']
assert not torch.cuda.is_initialized()
print(json.dumps(dict(models=models,per_validation_table=table,selected_case=chosen,
 criterion='Most negative mean FR delta of C7000/C7250 versus parent among all3 validation cases; FR=2-LPIPS-DISTS',
 CPU_only=True,CUDA_context_initialized=False)))
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    request = dict(labels=LABELS, sha=SHA)
    code = 'REQUEST_JSON=' + repr(json.dumps(request)) + '\n' + REMOTE
    call = subprocess.run(['ssh', '-q', '-p', '30416', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                           HOST, ROOT + '/env/bin/python -B -'], input=code, text=True, capture_output=True, check=True)
    snapshot = json.loads(call.stdout); write(OUT / 'provenance.json', snapshot)
    case = snapshot['selected_case']; sources = []
    gt_record = snapshot['models'][0]['validation'][case - 1]
    gt_name = Path(gt_record['GT_path']).name
    original_gt = BASE.parent / 'data/验证集' / gt_name
    assert sha(original_gt) == gt_record['GT_sha256']
    gt_path = OUT / gt_name
    if not gt_path.exists(): os.link(original_gt, gt_path)
    assert sha(gt_path) == gt_record['GT_sha256'] and gt_path.stat().st_ino == original_gt.stat().st_ino
    sources.append(dict(label='GT', name='Validation GT', path=str(gt_path), sha256=sha(gt_path),
                        remote_path=gt_record['GT_path'], hardlink=True))
    for name, model in zip(NAMES, snapshot['models']):
        selected = model['validation'][case - 1]; rec = selected['record']
        assert selected['GT_sha256'] == gt_record['GT_sha256'] and selected['input_sha256'] == gt_record['input_sha256']
        target = OUT / model['label'] / f'case{case}.png';target.parent.mkdir(parents=True, exist_ok=True)
        remote = model['directory'] + '/' + rec['output']
        if not target.exists():
            part = target.with_suffix('.png.part')
            subprocess.run(['scp', '-q', '-O', '-P', '30416', HOST + ':' + remote, str(part)], check=True)
            assert sha(part) == rec['sha256'];part.replace(target)
        assert sha(target) == rec['sha256']
        sources.append(dict(label=model['label'], name=name, path=str(target), sha256=rec['sha256'], remote_path=remote,
                            final_manifest=model['final_manifest'], row=selected['row'], render_record=rec))
    sizes = [image(s['path']).size for s in sources];assert len(set(sizes)) == 1
    # Explicitly choose the most changed640x512 region on a128-pixel grid.
    # Selection is based on candidate-versus-parent RGB difference, never GT.
    parent = np.asarray(image(sources[1]['path']), dtype=np.float32)
    diff = sum(np.abs(np.asarray(image(s['path']), dtype=np.float32)-parent).mean(axis=2) for s in sources[2:])/2
    integral = np.pad(diff.astype(np.float64).cumsum(0).cumsum(1), ((1,0),(1,0)))
    width,height=sizes[0];rw,rh=640,512
    xs=sorted(set(list(range(0,width-rw+1,128))+[width-rw]));ys=sorted(set(list(range(0,height-rh+1,128))+[height-rh]))
    candidates=[]
    for y in ys:
        for x in xs:
            value=(integral[y+rh,x+rw]-integral[y,x+rw]-integral[y+rh,x]+integral[y,x])/(rw*rh)
            candidates.append((float(value),x,y))
    value,x,y=max(candidates);box=(x,y,x+rw,y+rh)
    fontpath='/System/Library/Fonts/Supplemental/Arial.ttf'
    f=ImageFont.truetype(fontpath,20);fs=ImageFont.truetype(fontpath,16)
    gap=16;canvas=Image.new('RGB',(4*rw+5*gap,rh+110),'#f2f5fa');d=ImageDraw.Draw(canvas)
    d.text((16,10),f'Validation case{case} | native pixels1:1 | ROI {box}',font=f,fill='#172133')
    d.text((16,40),'Chosen by max candidate-vs-parent RGB change; GT not used to select ROI. No pixel filtering.',font=fs,fill='#172133')
    panes=[]
    for col,source in enumerate(sources):
        px=gap+col*(rw+gap);d.text((px,72),source['name'],font=f,fill='#172133')
        canvas.paste(image(source['path']).crop(box),(px,104));panes.append(dict(source=source['label'],paste_xy=[px,104]))
    roi=OUT/f'val_case{case}_roi_native_1to1.png';canvas.save(roi)
    overview=Image.new('RGB',(4*512+5*gap,610),'#f2f5fa');dd=ImageDraw.Draw(overview)
    dd.text((16,10),f'Validation case{case} | full-frame overview, exactly1/8 thumbnails',font=f,fill='#172133')
    for col,source in enumerate(sources):
        im=image(source['path']);size=(im.width//8,im.height//8);px=gap+col*(512+gap)
        dd.text((px,44),source['name'],font=f,fill='#172133');overview.paste(im.resize(size,Image.Resampling.LANCZOS),(px+(512-size[0])//2,80+(512-size[1])//2))
    overview_path=OUT/f'val_case{case}_overview_1over8.png';overview.save(overview_path)
    checks=[]
    for source,pane in zip(sources,panes):
        expected=image(source['path']).crop(box);px,py=pane['paste_xy'];actual=image(roi).crop((px,py,px+rw,py+rh))
        assert ImageChops.difference(expected,actual).getbbox() is None and sha(source['path'])==source['sha256']
        checks.append(dict(source=source['label'],original_sha256=source['sha256'],exact_native_ROI_pixels=True))
    manifest=dict(schema='push3_worst_validation_diagnostic_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      selected_case=case,validation_selection=snapshot['criterion'],per_validation_table=snapshot['per_validation_table'],
      ROI_criterion='Maximum mean absolute candidate-versus-parent RGB difference averaged over C7000/C7250;640x512 window on128-pixel top-left grid, plus last image boundary; GT excluded from ROI selection',
      ROI_box_xyxy_exclusive=box,ROI_selection_mean_RGB_difference_in_uint8_levels=value,
      ROI_scale_xy=[1,1],overview_scale_xy=[.125,.125],overview_resampling='Pillow Lanczos only',
      sources=sources,panes=panes,ROI_path=str(roi),ROI_SHA256=sha(roi),overview_path=str(overview_path),overview_SHA256=sha(overview_path),
      exact_native_pixel_checks=checks,CPU_only=True,all_original_SHA_unchanged=True,
      scope='Per-validation-case FR diagnostic only; no full test mean inferred; selected maximum-change crop is not an unbiased visual average')
    write(OUT/'diagnostic_manifest.json',manifest)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Push3 验证负例诊断</title>',
      '<style>body{font:16px/1.6 system-ui;margin:24px;background:#f5f7fb;color:#172133}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}.native{overflow:auto}.native img{max-width:none}.overview{max-width:100%}</style>',
      f'<h1>验证case{case} · FR下降病例诊断</h1>',
      '<p>按全部三张验证图中两候选相对父版的平均FR降幅选病例；本页仅作逐验证图诊断。ROI按候选相对父版RGB变化最大区域选择，未用GT选位置，因而不是无偏视觉平均。</p>',
      '<table><tr><th>验证病例</th><th>父6500 FR</th><th>C7000 FR / Δ</th><th>C7250 FR / Δ</th></tr>']
    for row in snapshot['per_validation_table']:
        p,c1,c2=row['FR'];d1,d2=row['delta_vs_parent'];parts.append(f'<tr><td>{row["case"]}</td><td>{p:.9f}</td><td>{c1:.9f} / {d1:+.9f}</td><td>{c2:.9f} / {d2:+.9f}</td></tr>')
    parts.append('</table><p>FR=2−LPIPS−DISTS，来自正式full序列已提交的验证评分；没有从这张图推断测试均分。固定原像素ROI不缩放、不锐化；全图总览单独标为1/8缩略。</p>')
    notes=OUT/'visual_observations_zh.txt'
    if notes.exists():parts.append('<p style="white-space:pre-wrap">'+html.escape(notes.read_text())+'</p>')
    parts.append('<p><a href="diagnostic_manifest.json">源SHA、坐标与诊断依据</a> · <a href="provenance.json">已提交渲染/评分/GT证明</a></p>')
    parts.append('<p>'+ ' · '.join('<a href="'+html.escape(os.path.relpath(s['path'],OUT),quote=True)+'">'+html.escape(s['name'])+'原图</a>' for s in sources)+'</p>')
    parts.append(f'<h2>1/8全图总览</h2><img class="overview" src="{overview_path.name}"><h2>1:1原像素ROI · {box}</h2><div class="native"><img src="{roi.name}"></div></html>')
    (OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps({'selected_case':case,'ROI_box':box,'source_pngs':len(sources),'exact_native_pixel_checks':len(checks),'full_render_committed':[m['full_render_committed'] for m in snapshot['models']],'index':str(OUT/'index.html')}))


if __name__=='__main__':main()
