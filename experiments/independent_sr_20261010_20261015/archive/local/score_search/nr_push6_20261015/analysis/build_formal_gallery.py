"""CPU 1:1 ROI gallery from frozen-parent and committed formal candidate PNGs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
from PIL import Image,ImageDraw,ImageFont
import numpy as np

BOXES={7:(1856,1408,2496,1856),16:(1280,1216,1920,1728),41:(768,1664,1408,2176),83:(1216,1920,1856,2432)}


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--snapshot-json',type=Path,required=True);ap.add_argument('--parent-local-dir',type=Path,required=True);ap.add_argument('--candidate-local-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--allow-progress',action='store_true');a=ap.parse_args()
    j=json.loads(a.snapshot_json.read_text());d=a.out_dir.resolve();need(j['status']=='PASS' and not (d/'preview_provenance.json').exists(),'New complete preview proof required')
    need(j['candidate_formal_complete103'] or a.allow_progress,'Progress preview must be explicitly labeled')
    cases=sorted(int(c) for c in j['candidate_records']);need(cases and (j['candidate_formal_complete103'] is False or cases==[7,16,41,83]),'Four final committed cases required')
    try:font=ImageFont.truetype('DejaVuSans.ttf',19)
    except OSError:font=ImageFont.load_default()
    d.mkdir(parents=True,exist_ok=True);sources=[];panes=[];outputs=[];body=[]
    for case in cases:
        imgs=[]
        for role,base,key in [('A7750 quality baseline',a.parent_local_dir,'parent_records'),('Same-weight RO+wavelet inference',a.candidate_local_dir,'candidate_records')]:
            rec=j[key][str(case)];path=base/f'case{case}.png';need(path.is_file() and sha(path)==rec['sha256'] and path.stat().st_size==rec['bytes'],'Local formal PNG bytes differ')
            with Image.open(path) as im:need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==rec['size'],'Not native formal RGB PNG');im.load();imgs.append(im.copy())
            link=d/'sources'/key/f'case{case}.png';link.parent.mkdir(parents=True,exist_ok=True)
            if link.exists():need(sha(link)==rec['sha256'],'Existing source link differs')
            else:os.link(path,link)
            sources.append(dict(case=case,role=role,formal_source_path=rec['path'],local_actual_path=str(path.resolve()),gallery_link=str(link.relative_to(d)),sha256=rec['sha256'],bytes=rec['bytes'],input_sha256=rec['input_sha256'],native_size=rec['size']))
        box=BOXES[case];tiles=[im.crop(box) for im in imgs];w,h=tiles[0].size;canvas=Image.new('RGB',(2*w+48,h+112),'#f2f4f7');draw=ImageDraw.Draw(canvas)
        draw.text((16,12),f'case{case} / FORMAL QUEUE / FIXED ROI 1:1 xyxy={box}',font=font,fill='#172033')
        draw.text((16,40),'Same trained weights. Original pixels; no resizing, filtering, sharpening or retouching.',font=font,fill='#172033')
        for col,(tile,role) in enumerate(zip(tiles,['A7750 quality baseline','Same-weight RO+wavelet inference'])):
            x=16+col*(w+16);draw.text((x,68),role,font=font,fill='#172033');canvas.paste(tile,(x,96));panes.append(dict(case=case,column=col,ROI_xyxy_exclusive=list(box),paste_xy=[x,96],size=[w,h],scale=1,resampling=None))
        output=d/f'case{case}_formal_roi_1to1.png';need(not output.exists(),'ROI already exists');canvas.save(output,compress_level=1)
        with Image.open(output) as check:
            for col,tile in enumerate(tiles):
                x=16+col*(w+16);need(np.array_equal(np.asarray(check.crop((x,96,x+w,96+h))),np.asarray(tile)),'Panel pixels differ')
        exact=j['parent_records'][str(case)]['sha256']==j['candidate_records'][str(case)]['sha256'];outputs.append(dict(path=str(output),sha256=sha(output),bytes=output.stat().st_size,full_native_source_SHA_equal=exact))
        body.append(f'<section><h2>case{case} · 1:1 ROI {box}</h2><p>Full native PNG SHA equal: {exact}. <a href="sources/parent_records/case{case}.png">Open native A7750 PNG</a> / <a href="sources/candidate_records/case{case}.png">Open native accelerated PNG</a></p><div class="native"><img src="{output.name}"></div></section>')
    stage='FINAL FORMAL103' if j['candidate_formal_complete103'] else f'PROGRESS ONLY: {j["committed_images"]}/103 committed; no complete score inferred'
    page='<!doctype html><meta charset="utf-8"><title>A7750 same-weight inference gallery</title><style>body{font:16px system-ui;margin:24px;color:#172033}section{margin:32px 0}.native{overflow:auto;max-width:100%;border:1px solid #ddd}.native img{display:block;max-width:none}a{color:#1758a7}</style><h1>A7750 baseline vs same-weight accelerated inference</h1><p>'+stage+'</p><p>Both columns use formal val3→test100 seed231 outputs. No diagnostic-sequence images, no new training, and no mean score inferred from selected cases. ROI geometry is retained from earlier inspection; all image bytes come from the indicated formal cohorts.</p>'+''.join(body)
    (d/'index.html').write_text(page)
    proof=dict(status='PASS',CPU_only=True,GPU_used=False,stage=stage,snapshot_json_sha256=sha(a.snapshot_json),snapshot=j,source_images=sources,ROIs=panes,outputs=outputs,all_ROI_panels_exact_equal_source_crops=True,native_originals_unmodified=True,script_sha256=sha(__file__))
    with (d/'preview_provenance.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','gallery':str(d/'index.html'),'cases':cases,'stage':stage,'provenance_sha256':sha(d/'preview_provenance.json')}))


if __name__=='__main__':main()
