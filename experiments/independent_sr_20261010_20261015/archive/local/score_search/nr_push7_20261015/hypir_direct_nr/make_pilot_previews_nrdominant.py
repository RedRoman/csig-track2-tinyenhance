"""CPU actual-pixel ROIs from the SAME pilot23 parent/NR50/NR100 queues.

No formal-A/native4 diagnostic outputs, resampling, filtering, or score claims.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
from PIL import Image,ImageDraw,ImageFont
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/cache/independent_sr_cuda_20261013')
PARENT=R/'nr_push7_pilot_evaluations/push7_a7750_parent_pilot'
CANDIDATES=R/'nr_push7_hypir_pilot_evaluations'
RUNNER_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
BOXES={7:(1856,1408,2496,1856),16:(1280,1216,1920,1728),41:(768,1664,1408,2176)}
LABELS={'push7_a7750_parent_pilot':'A7750 parent / same pilot queue',
        'push7_a7750_direct_nr_s7800_pilot':'HYPIR direct NR / +50 / global7800',
        'push7_a7750_direct_nr_s7850_pilot':'HYPIR direct NR / +100 / global7850',
        'push7_a7750_nrdominant_s7800_pilot':'HYPIR NR-dominant / +50 / global7800',
        'push7_a7750_nrdominant_s7850_pilot':'HYPIR NR-dominant / +100 / global7850'}


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def snapshot(directory):
    need(directory==PARENT or directory.parent==CANDIDATES,'Only the matched Push7 pilot queues allowed')
    need(directory.name in LABELS,'Unregistered pilot label')
    wrapper=json.loads((directory/'render_protocol.json').read_text());p=wrapper['protocol'];rr=p['runner']
    need(wrapper['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'Protocol hash differs')
    need(p['cases']=='pilot' and len(p['tasks'])==23 and p['warmup'] is False,'Not original pilot23')
    model=Path(p['model_dir']);config=json.loads((model/'config.json').read_text())
    need(rr['runner_sha256']==sha(model/'runner.py')==RUNNER_SHA,'Not the original native512 Push6 runner')
    need(rr['model_t']==200 and rr['coeff_t']==175 and rr['patch']==512 and rr['stride']==256 and rr['scale']==1,'Changed HYPIR protocol')
    need(rr['seed']=='231 before initialization; continuous inference RNG','Changed initialization RNG')
    need(sha(model/'config.json')==p['model_dir_sha256']['config.json'],'Config changed')
    need(Path(config['weight_path']).resolve()==Path(rr['weight_path']).resolve(),'Config/runner checkpoint differs')
    need(sha(rr['weight_path'])==rr['checkpoint_sha256'],'Actual checkpoint changed')
    need((rr['checkpoint_sha256']==A_SHA)==(directory==PARENT),'InitialA/new checkpoint provenance confused')
    complete=(directory/'render_complete').is_file()
    if complete:
        path=directory/'render_manifest.json';raw=path.read_bytes();state=json.loads(raw)
        need(state['protocol']==p and state['protocol_id']==wrapper['protocol_id'],'Final pilot manifest differs')
    else:
        import torch
        need(not torch.cuda.is_initialized(),'Preview must be CPU-only')
        path=directory/'render_progress.pt';raw=path.read_bytes();state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        need(state['protocol_id']==wrapper['protocol_id'] and state['checkpoint_sha256']==rr['checkpoint_sha256'],'Committed progress belongs to another run')
    records=state['records'];need([(q['split'],q['case']) for q in records]==[(q['split'],q['case']) for q in p['tasks'][:len(records)]],'Committed task order differs')
    selected={}
    for case,box in BOXES.items():
        row=next((q for q in records if q['split']=='test' and q['case']==case),None)
        need(row is not None,f'{directory.name}: case{case} not yet committed')
        task=next(q for q in p['tasks'] if q['split']=='test' and q['case']==case);image=directory/row['output']
        need(row['output']==f'render/test/case{case}.png' and sha(image)==row['sha256'],'Actual committed PNG differs')
        need(sha(task['input'])==task['input_sha256'],'Actual pilot input differs')
        with Image.open(image) as im:
            need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==row['size']==task['size'],'Not canonical native RGB PNG')
            need(0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height,'ROI outside native image')
        selected[case]=dict(path=str(image),sha256=row['sha256'],input_sha256=task['input_sha256'],size=row['size'],task_index=row['task_index'])
    return dict(directory=str(directory),label=directory.name,display=LABELS[directory.name],protocol=p,protocol_id=wrapper['protocol_id'],checkpoint_sha256=rr['checkpoint_sha256'],config_sha256=sha(model/'config.json'),snapshot_path=str(path),snapshot_sha256=hashlib.sha256(raw).hexdigest(),render_complete=complete,committed_images=len(records),selected=selected)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate-dir',type=Path,action='append',required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
    out=a.out_dir.resolve();need(out.is_relative_to(R/'analysis/push7') and not out.exists(),'Use a new isolated preview output')
    dirs=[PARENT,*(p.resolve() for p in a.candidate_dir)]
    need(1<=len(a.candidate_dir)<=2 and len(set(dirs))==len(dirs),'One or two distinct new checkpoints')
    data=[snapshot(p) for p in dirs]
    for q in data[1:]:
        need(q['protocol']['tasks']==data[0]['protocol']['tasks'],'Not the SAME pilot input/order/native-size sequence')
        need(q['protocol']['inference_rng']==data[0]['protocol']['inference_rng'],'Pilot RNG policy differs')
    out.mkdir(parents=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    outputs=[];comparisons=[]
    for case,box in BOXES.items():
        tiles=[]
        for q in data:
            with Image.open(q['selected'][case]['path']) as im:tiles.append(im.crop(box))
        w,h=tiles[0].size;canvas=Image.new('RGB',(16+len(tiles)*(w+16),h+132),'#f3f5f8');draw=ImageDraw.Draw(canvas)
        draw.text((16,10),f'case{case} / SAME pilot23 queues / native ROI {box} / stored1:1 pixels',font=font,fill='#172033')
        draw.text((16,36),'Original HYPIR architecture. Direct NR LoRA adaptation. No teacher-output distillation.',font=font,fill='#172033')
        draw.text((16,62),'Actual committed PNGs, same inputs/order/seed policy. No fullA or native4 outputs mixed.',font=font,fill='#172033')
        for col,(tile,q) in enumerate(zip(tiles,data)):
            left=16+col*(w+16);draw.text((left,92),q['display'],font=font,fill='#172033');canvas.paste(tile,(left,116))
        path=out/f'case{case}_hypir_direct_nr_pilot_roi_1to1.png';canvas.save(path,compress_level=1)
        with Image.open(path) as actual:
            for col,tile in enumerate(tiles):
                left=16+col*(w+16);need(np.array_equal(np.asarray(actual.crop((left,116,left+w,116+h))),np.asarray(tile)),'Saved panel pixels changed')
        base=np.asarray(tiles[0],dtype=np.int16)
        for q,tile in zip(data[1:],tiles[1:]):
            delta=np.abs(np.asarray(tile,dtype=np.int16)-base)
            comparisons.append(dict(case=case,candidate=q['label'],ROI_RGB_mean_abs=float(delta.mean()),ROI_RGB_max_abs=int(delta.max()),ROI_any_channel_changed_pixel_fraction=float(np.any(delta>0,axis=2).mean()),scope='Descriptive ROI difference only, not quality or full metric'))
        outputs.append(dict(name=path.name,sha256=sha(path),bytes=path.stat().st_size,case=case,ROI=list(box),panel_source_pixels_exact=True))
    html='<!doctype html><meta charset="utf-8"><title>HYPIR direct NR matched pilot</title><style>body{font:16px system-ui;margin:24px;color:#172033}figure{margin:28px 0;overflow:auto}img{max-width:none}a{color:#245abd}</style><h1>HYPIR NR-dominant微调：同pilot队列局部对比</h1><p>列为A7750、NR-dominant50步、NR-dominant100步（按实际提供候选）。固定ROI原像素1:1，无缩放、锐化或修图；每张来自自己的新pilot23连续推理队列。没有混入完整103或native4诊断队列，局部案例不代表全量指标。</p>'
    for q in outputs:html+=f'<figure><figcaption>case{q["case"]}，ROI {q["ROI"]}；<a href="{q["name"]}">打开原像素面板</a></figcaption><img src="{q["name"]}"></figure>'
    html+='<p><a href="preview_provenance.json">实际源SHA与逐像素排版证明</a></p>'
    (out/'index.html').write_text(html)
    proof=dict(status='PASS',CPU_only=True,GPU_used=False,scope='Actual matched pilot23 ROI7/16/41 only; no full quality/score claim',same_inputs_and_task_order=True,same_native512_runner_and_seed_policy=True,post_fourRNG_equality_not_claimed=True,no_teacher_output_distillation=True,source_SHA_verified=True,all_ROI_pixels_exact=True,snapshots=data,outputs=outputs,ROI_comparisons=comparisons,HTML_sha256=sha(out/'index.html'),script_sha256=sha(__file__))
    with (out/'preview_provenance.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',out_dir=str(out),cases=list(BOXES),proof_sha256=sha(out/'preview_provenance.json'))),flush=True)


if __name__=='__main__':main()
