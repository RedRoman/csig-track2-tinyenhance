"""CPU native ROI progress comparison; every column is an actual full103 run.

Order: earlier distilled reference, two real new checkpoints, fixed HYPIR.
No packaged JPEG, metric computation, selection, resampling or enhancement.
"""
from pathlib import Path
import argparse
import hashlib
import html
import importlib.util
import json
from PIL import Image, ImageDraw, ImageFont

BASE_SHA='c5a16137df0ef8c84196103fffc53bfdf3a6af0a728998487f3781b97b04f160'


def need(v,m):
    if not v:raise RuntimeError(m)


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--base-helper',type=Path,default=Path(__file__).with_name('make_fresh_distill_gallery.py'))
    q.add_argument('--spec',type=Path,required=True);q.add_argument('--out-dir',type=Path,required=True)
    q.add_argument('--execute',action='store_true');a=q.parse_args()
    need(hashlib.sha256(a.base_helper.read_bytes()).hexdigest()==BASE_SHA,'Frozen full-gallery source differs')
    spec=importlib.util.spec_from_file_location('_verified_full_gallery_base',a.base_helper)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=m.read(a.spec);cols=cfg['columns']
    need([c['role'] for c in cols]==['distilled_pix_reference','distilled_pix_candidate','distilled_pix_candidate','fixed_HYPIR_teacher'],
         'Required explicit earlier reference/new candidates/fixed HYPIR order')
    need(len({c['label'] for c in cols})==4 and len({c['checkpoint_sha256'] for c in cols[:3]})==3,'Distinct actual Pix checkpoints required')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',columns=cols,out=str(a.out_dir))));return
    out=a.out_dir.resolve();need(not out.exists(),'New gallery output required')
    Image.MAX_IMAGE_PIXELS=None;sources=[m.snapshot(c) for c in cols]
    for src in sources[1:]:
        for k in ['tasks','input_encoding','output_encoding','warmup','inference_rng','source_sha256','torch']:
            need(src['protocol'][k]==sources[0]['protocol'][k],'Unmatched full input/encoding/order: '+k)
    out.mkdir(parents=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    outputs=[]
    for case,box in m.BOXES.items():
        tiles=[]
        for src in sources:
            row=next(x for x in src['images'] if x['split']=='test' and x['case']==case)
            with Image.open(row['PNG']['path']) as im:
                need(im.mode=='RGB' and im.format=='PNG' and list(im.size)==row['native_size'],'Noncanonical native source')
                need(box[2]<=im.width and box[3]<=im.height,'Fixed ROI out of bounds');tiles.append(im.crop(box))
        need(len({t.size for t in tiles})==1,'Unmatched native ROI shapes')
        w,h=tiles[0].size;panel=Image.new('RGB',(16+len(cols)*(w+16),h+86),'#f4f4f4');draw=ImageDraw.Draw(panel)
        draw.text((16,10),f'case{case} | native 1:1 ROI',font=font,fill='black')
        for i,(c,t) in enumerate(zip(cols,tiles)):
            x=16+i*(w+16);draw.text((x,42),c['title'],font=font,fill='black');panel.paste(t,(x,70))
        path=out/f'case{case}_structure_progress_ROI_1to1.png';panel.save(path,compress_level=1)
        with Image.open(path) as saved:
            for i,t in enumerate(tiles):
                x=16+i*(w+16);need(saved.crop((x,70,x+w,70+h)).tobytes()==t.tobytes(),'Stored crop pixels differ')
        outputs.append(dict(case=case,ROI=list(box),crop_pixels_exact=True,**m.record(path)))
    page='<!doctype html><meta charset="utf-8"><title>PixG5i distillation progress</title><style>body{font:16px system-ui;margin:24px}figure{overflow:auto}img{max-width:none}</style><h1>PixG5i distillation: texture and structure</h1><p>Real full103 native output crops, identical ordered inputs. All image pixels are shown at fixed native 1:1 geometry; no resizing, filtering or enhancement. Reduced repeated texture or smoother surfaces alone do not prove restoration of characters, thin bars or leaf veins. No score or model selection is inferred from these four examples.</p>'
    for r in outputs:page+=f'<figure><figcaption>case{r["case"]} | {html.escape(str(r["ROI"]))}</figcaption><img src="{Path(r["path"]).name}"></figure>'
    page+='<details><summary>Actual PNG and input provenance</summary><a href="proof.json">proof.json</a></details>';(out/'index.html').write_text(page)
    for p in list(m.SEEN):m.sha(p)
    proof=dict(status='PASS_STRUCTURE_PROGRESS_FRESH_FULL_NATIVE_ROI4',CPU_only=True,GPU_used=False,spec=m.record(a.spec),
               helper=m.record(__file__),frozen_base_helper=m.record(a.base_helper),sources=sources,outputs=outputs,
               no_resize_filter_or_enhancement=True,packaged_JPEG_used=False,inference_postRNG_equality_claim=False,
               no_score_or_selection_from_cases=True,Pix_model_dir_files_including_model_pth_rehashed=True,
               external_HYPIR_raw_and_engine_bytes_rehashed=False,HTML=m.record(out/'index.html'))
    (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),proof_sha256=m.sha(out/'proof.json'))))


if __name__=='__main__':main()
