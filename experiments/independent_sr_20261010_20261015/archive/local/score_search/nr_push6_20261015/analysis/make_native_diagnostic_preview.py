"""CPU layout of actual same-RNG diagnostic PNGs. No model inference or filters."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BOXES={7:(1856,1408,2496,1856),41:(768,1664,1408,2176)}


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnostic-json',type=Path,required=True)
    ap.add_argument('--source-dir',type=Path,help='Optional local PNG copies; exact source SHA remains required')
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args();j=json.loads(a.diagnostic_json.read_text());d=a.out_dir.resolve()
    need(not (d/'preview_provenance.json').exists(),'Immutable preview already completed')
    need(j['status']=='complete' and j['phase']=='candidate' and j['suite']=='native7_41' and j['cases']==[7,41],'Only the same complete native7_41 diagnostic sequence')
    need(j['constructor_four_RNG_equal_reference'] is True and j['output_ownership_preserved'] is True and all(row['post_four_rng_equal'] is True for row in j['rows']),'Same four-RNG/ownership proof required')
    need(j['reference_dir']!=j['model_dir'],'Invalid source identity')
    d.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',19)
    except OSError:font=ImageFont.load_default()
    sources=[];panes=[];outputs=[];body=[]
    for row in j['rows']:
        case=row['case'];need(case in BOXES,'Unexpected fixed case')
        images=[];source_records=[]
        for role,rec in [('A7750 same-RNG reference',row['reference_png']),('Candidate '+Path(j['model_dir']).parent.name,row['png'])]:
            original=Path(rec['path']);p=a.source_dir/original.name if a.source_dir else original
            need(p.is_file() and sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes'],'Actual diagnostic PNG differs')
            with Image.open(p) as im:need(im.format=='PNG' and im.mode=='RGB','Original canonical RGB PNG required');im.load();image=im.copy()
            images.append(image);source_records.append(dict(case=case,role=role,original_path=str(original),actual_path=str(p.resolve()),sha256=rec['sha256'],bytes=rec['bytes'],size=list(image.size)))
        need(images[0].size==images[1].size==tuple(reversed(row['shape'][2:])),'Native size differs')
        box=BOXES[case];need(0<=box[0]<box[2]<=images[0].width and 0<=box[1]<box[3]<=images[0].height,'Fixed ROI outside native image')
        sources.extend(source_records)
        for kind in ['native_full','native_roi']:
            tiles=images if kind=='native_full' else [im.crop(box) for im in images]
            w,h=tiles[0].size;canvas=Image.new('RGB',(2*w+48,h+112),'#f2f4f7');draw=ImageDraw.Draw(canvas)
            title=f'case{case} / '+('FULL NATIVE 1:1' if kind=='native_full' else f'FIXED ROI 1:1 xyxy={box}')
            draw.text((16,12),title,font=font,fill='#172033')
            draw.text((16,40),'Actual same-sequence diagnostic PNGs. No resizing, filtering, sharpening or retouching.',font=font,fill='#172033')
            for col,(tile,rec) in enumerate(zip(tiles,source_records)):
                x=16+col*(w+16);y=96;draw.text((x,68),rec['role'],font=font,fill='#172033');canvas.paste(tile,(x,y))
                panes.append(dict(case=case,type=kind,source_sha256=rec['sha256'],ROI_xyxy_exclusive=list(box) if kind=='native_roi' else [0,0,images[0].width,images[0].height],paste_xy=[x,y],size=[w,h],scale=1,resampling=None,output=f'case{case}_{kind}_1to1.png'))
            path=d/f'case{case}_{kind}_1to1.png';need(not path.exists(),'Preview output already exists');canvas.save(path,compress_level=1)
            with Image.open(path) as check:
                for col,tile in enumerate(tiles):
                    x=16+col*(w+16);need(np.array_equal(np.asarray(check.crop((x,96,x+w,96+h))),np.asarray(tile)),'Layout changed source pixels')
            outputs.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,size=list(canvas.size)))
        full=f'case{case}_native_full_1to1.png';roi=f'case{case}_native_roi_1to1.png'
        body.append(f'<section><h2>case{case}</h2><p>Same constructor and per-call four-RNG state. Float max delta {row["max_abs"]:.9g}; mean delta {row["mean_abs"]:.9g}. No full-quality conclusion follows from these two cases.</p><p>Fixed ROI: {box}. Source pixels below are 1:1, with horizontal scrolling.</p><div class="native"><img src="{roi}" alt="fixed native ROI"></div><p><a href="{full}">Open full native 1:1 side-by-side</a></p></section>')
    proof=dict(status='PASS',CPU_only=True,GPU_used=False,scope='Same-RNG diagnostic visualization only; never mix previous official full PNGs with this sequence',diagnostic_json=str(a.diagnostic_json.resolve()),diagnostic_json_sha256=sha(a.diagnostic_json),diagnostic_model_dir=j['model_dir'],reference_dir=j['reference_dir'],reference_suite=j['suite'],fixed_ROI_xyxy_exclusive={str(k):list(v) for k,v in BOXES.items()},ROI_geometry_source='Coordinates retained from previous visual inspection; only geometry reused, no previous image bytes',all_four_RNG_equal=True,all_output_ownership_preserved=True,source_files=sources,panes=panes,outputs=outputs,all_native_full_and_ROI_panel_pixels_exact_equal_to_sources=True,source_float_and_RNG_records=j['rows'],reused_reference_files_SHA256=j['reused_reference_files_SHA256'],script_sha256=sha(__file__))
    page='<!doctype html><meta charset="utf-8"><title>A7750 same-RNG diagnostics</title><style>body{font:16px system-ui;margin:24px;color:#172033}section{margin:32px 0}.native{overflow:auto;max-width:100%;border:1px solid #ddd}.native img{display:block;max-width:none}a{color:#1758a7}</style><h1>A7750 same-RNG diagnostic images</h1><p>Both columns use the actual diagnostic sequence. Previous official full-cohort images are excluded. PNG sources and exact panel pixel checks are recorded in preview_provenance.json.</p>'+''.join(body)
    (d/'index.html').write_text(page)
    with (d/'preview_provenance.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','out_dir':str(d),'source_images':len(sources),'exact_native_panels':len(outputs),'provenance_sha256':sha(d/'preview_provenance.json')}))


if __name__=='__main__':main()
