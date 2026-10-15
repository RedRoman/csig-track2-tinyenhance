"""CPU fixed ROI7/16 from matched NEW Push7 pilot queues, never formal/native4."""
from pathlib import Path
import argparse
import hashlib
import io
import json
from PIL import Image,ImageDraw,ImageFont
import numpy as np

R=Path('/cache/independent_sr_cuda_20261013');P=R/'nr_push7_pilot_evaluations'
BOXES={7:(1856,1408,2496,1856),16:(1280,1216,1920,1728)}


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def snapshot(directory):
    need(directory.resolve().parent==P,'Only the new matched pilot queues permitted')
    p=json.loads((directory/'render_protocol.json').read_text());proto=p['protocol']
    need(p['protocol_id']==hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest() and proto['cases']=='pilot' and proto['warmup'] is False,'Wrong pilot protocol')
    need(proto['runner']['checkpoint_sha256']=='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d','Wrong A7750 checkpoint')
    if (directory/'render_complete').exists():
        m=json.loads((directory/'render_manifest.json').read_text());need(m['protocol']==proto,'Pilot manifest/protocol differ');rows=m['records'];state_sha=sha(directory/'render_manifest.json')
    else:
        import torch
        need(not torch.cuda.is_initialized(),'CPU-only preview required');data=(directory/'render_progress.pt').read_bytes();s=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False);need(s['protocol_id']==p['protocol_id'],'Pilot committed snapshot differs');rows=s['records'];state_sha=hashlib.sha256(data).hexdigest()
    need([(r['split'],r['case']) for r in rows]==[(t['split'],t['case']) for t in proto['tasks'][:len(rows)]],'Committed pilot order differs')
    selected={}
    for case in BOXES:
        row=next((r for r in rows if r['split']=='test' and r['case']==case),None);need(row is not None,'Selected pilot case not yet committed')
        task=next(t for t in proto['tasks'] if t['split']=='test' and t['case']==case);path=directory/row['output'];need(row['output']==f'render/test/case{case}.png' and sha(path)==row['sha256'] and sha(task['input'])==task['input_sha256'],'Actual committed pilot image/input differs')
        selected[case]={**row,'path':str(path),'input_sha256':task['input_sha256']}
    return dict(directory=str(directory),protocol=proto,protocol_id=p['protocol_id'],commit_snapshot_sha256=state_sha,committed_images=len(rows),selected=selected)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate-dir',type=Path,action='append',required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args();d=a.out_dir.resolve();need(d.is_relative_to(R/'analysis/push7') and not (d/'preview_provenance.json').exists(),'New Push7 output required')
    dirs=[P/'push7_a7750_parent_pilot',*(x.resolve() for x in a.candidate_dir)];data=[snapshot(x) for x in dirs]
    need(data[0]['protocol']['runner']['runner_sha256']=='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b','Not the approved Push6 baseline')
    need(all(x['protocol']['tasks']==data[0]['protocol']['tasks'] for x in data),'Not the SAME pilot input/order sequence')
    d.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',19)
    except OSError:font=ImageFont.load_default()
    sources=[];outputs=[]
    for case,box in BOXES.items():
        tiles=[]
        for obj in data:
            rec=obj['selected'][case]
            with Image.open(rec['path']) as im:need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==rec['size'],'Not native pilot PNG');tiles.append(im.crop(box))
            sources.append(dict(case=case,pilot_dir=obj['directory'],path=rec['path'],sha256=rec['sha256'],input_sha256=rec['input_sha256']))
        w,h=tiles[0].size;canvas=Image.new('RGB',(len(tiles)*(w+16)+16,h+136),'#f2f4f7');draw=ImageDraw.Draw(canvas)
        draw.text((16,10),f'case{case} / MATCHED NEW PILOT QUEUES / ROI1:1 {box}',font=font,fill='#172033')
        draw.text((16,36),'Same pilot inputs/order; each algorithm continuous RNG. No formal/native4 sequence mixed.',font=font,fill='#172033')
        draw.text((16,62),'Committed native PNG pixels. No filtering/retouching. No partial mean or full-score claim.',font=font,fill='#172033')
        for col,(tile,obj) in enumerate(zip(tiles,data)):
            x=16+col*(w+16);draw.text((x,94),Path(obj['directory']).name,font=font,fill='#172033');canvas.paste(tile,(x,120))
        p=d/f'case{case}_matched_pilot_roi_1to1.png';need(not p.exists(),'ROI already exists');canvas.save(p,compress_level=1)
        with Image.open(p) as check:
            for col,tile in enumerate(tiles):
                x=16+col*(w+16);need(np.array_equal(np.asarray(check.crop((x,120,x+w,120+h))),np.asarray(tile)),'Panel pixels differ')
        outputs.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    proof=dict(status='PASS',CPU_only=True,GPU_used=False,scope='Fixed ROI from matched NEW pilot queues only; not native4 or official full103',same_inputs_and_task_order=True,post_RNG_equality_not_claimed=True,actual_committed_source_SHA_all_PASS=True,all_ROI_pixels_exact_equal_to_source=True,snapshots=data,sources=sources,outputs=outputs,script_sha256=sha(__file__))
    with (d/'preview_provenance.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','out_dir':str(d),'cases':list(BOXES),'proof_sha256':sha(d/'preview_provenance.json')}))


if __name__=='__main__':main()
