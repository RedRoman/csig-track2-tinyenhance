"""CPU-only 1:1 fixed-ROI layout of new Push7 native4 diagnostic PNGs.

Common constructor RNG is required; changed post-call RNG is recorded rather
than forced equal. No formal/past diagnostic images or image filters are used.
"""
from pathlib import Path
import argparse
import hashlib
import json
from PIL import Image,ImageDraw,ImageFont
import numpy as np

R=Path('/cache/independent_sr_cuda_20261013')
BASELINE=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
BASE_SOURCE='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
BOXES={7:(1856,1408,2496,1856),16:(1280,1216,1920,1728),41:(768,1664,1408,2176),83:(1216,1920,1856,2432)}


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate-json',type=Path,action='append',required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
    d=a.out_dir.resolve();need(d.is_relative_to(R/'analysis/push7') and not (d/'preview_provenance.json').exists(),'New Push7 preview directory required')
    reports=[json.loads(p.read_text()) for p in a.candidate_json];need(1<=len(reports)<=3,'At most three declared scale candidates')
    for j in reports:
        need(j['status']=='complete' and j['phase']=='candidate' and j['cases']==[7,16,41,83] and j['constructor_four_RNG_equal_reference'] is True and j['output_ownership_preserved'] is True,'Incomplete native4/constructor/ownership proof')
        need(j['baseline_model_dir']==str(BASELINE) and j['baseline_runner_sha256']==BASE_SOURCE and j['post_inference_RNG_equality_required'] is False,'Wrong reference/RNG policy')
        need(Path(j['reference_dir']).is_relative_to(R/'analysis/push7'),'No past reference sequence may be mixed')
    d.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',19)
    except OSError:font=ImageFont.load_default()
    sources=[];panes=[];outputs=[];body=[]
    for case,box in BOXES.items():
        rows=[next(r for r in j['rows'] if r['case']==case) for j in reports]
        reference=rows[0]['reference_png'];need(all(r['reference_png']['sha256']==reference['sha256'] and r['input_float_sha256']==rows[0]['input_float_sha256'] for r in rows),'Reference sequence/input differs across candidates')
        records=[('Push6 A7750 baseline',reference,None)]+[(Path(j['model_dir']).parent.name,r['png'],r['post_four_rng_equal']) for j,r in zip(reports,rows)]
        tiles=[]
        for label,rec,post_equal in records:
            p=Path(rec['path']);need(p.is_relative_to(R/'analysis/push7') and p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],'Actual diagnostic PNG SHA/size differs')
            with Image.open(p) as im:
                need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==list(reversed(rows[0]['shape'][2:])),'Not native RGB PNG')
                need(0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height,'Fixed ROI outside source');tiles.append(im.crop(box))
            sources.append(dict(case=case,label=label,path=str(p),sha256=rec['sha256'],bytes=rec['bytes'],post_four_RNG_equal_to_baseline=post_equal))
        w,h=tiles[0].size;canvas=Image.new('RGB',(len(tiles)*(w+16)+16,h+136),'#f2f4f7');draw=ImageDraw.Draw(canvas)
        draw.text((16,10),f'case{case} / NEW NATIVE4 DIAGNOSTIC / FIXED ROI 1:1 xyxy={box}',font=font,fill='#172033')
        draw.text((16,36),'Common constructor seed; each runner continuous. Scale changes post-call noise consumption.',font=font,fill='#172033')
        draw.text((16,62),'Actual PNG pixels, no resizing/filtering/retouching. These cases do not imply a full mean score.',font=font,fill='#172033')
        for col,(tile,(label,_,post_equal)) in enumerate(zip(tiles,records)):
            x=16+col*(w+16);draw.text((x,94),label,font=font,fill='#172033');canvas.paste(tile,(x,120));panes.append(dict(case=case,column=col,label=label,ROI_xyxy_exclusive=list(box),paste_xy=[x,120],scale=1,resampling=None))
        path=d/f'case{case}_scaled_native4_roi_1to1.png';need(not path.exists(),'Preview output already exists');canvas.save(path,compress_level=1)
        with Image.open(path) as check:
            for col,tile in enumerate(tiles):
                x=16+col*(w+16);need(np.array_equal(np.asarray(check.crop((x,120,x+w,120+h))),np.asarray(tile)),'Layout changed source pixels')
        outputs.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
        details='; '.join(f'{Path(j["model_dir"]).parent.name}: post-RNG equal={r["post_four_rng_equal"]}, max float delta={r["max_abs"]:.7g}, mean={r["mean_abs"]:.7g}' for j,r in zip(reports,rows))
        body.append(f'<section><h2>case{case}: native 1:1 ROI {box}</h2><p>{details}</p><div class="native"><img src="{path.name}"></div></section>')
    page='<!doctype html><meta charset="utf-8"><title>Push7 native4 visual diagnostic</title><style>body{font:16px system-ui;margin:24px;color:#172033}section{margin:32px 0}.native{overflow:auto;max-width:100%;border:1px solid #ddd}.native img{display:block;max-width:none}</style><h1>Push7 scaled residual vs A7750 baseline</h1><p>Fresh native4 diagnostic sequence only. Same constructor seed; each runner continues its own RNG. Scaling changes stochastic draw shapes, so post-call RNG differences are expected and explicitly reported. No old formal/diagnostic image is mixed. No full-score claim follows from four selected cases.</p>'+''.join(body)
    (d/'index.html').write_text(page)
    proof=dict(status='PASS',CPU_only=True,GPU_used=False,scope='New native4 visual diagnostic; not full-cohort IQA or latency',common_constructor_RNG=True,post_four_RNG_equality_required=False,output_ownership_all_PASS=True,input_reports=[dict(path=str(p),sha256=sha(p)) for p in a.candidate_json],reports=reports,sources=sources,ROIs=panes,outputs=outputs,all_ROI_pixels_exact_equal_to_actual_PNG_crops=True,no_past_formal_or_diagnostic_PNG_reuse=True,script_sha256=sha(__file__))
    with (d/'preview_provenance.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','gallery':str(d/'index.html'),'cases':list(BOXES),'candidate_columns':len(reports),'proof_sha256':sha(d/'preview_provenance.json')}))


if __name__=='__main__':main()
