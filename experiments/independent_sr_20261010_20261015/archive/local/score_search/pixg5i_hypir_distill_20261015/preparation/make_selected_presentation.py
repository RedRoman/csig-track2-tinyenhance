"""Re-layout verified native ROI pixels after root supplies an actual selection.

CPU/Pillow only. Source is an already verified fresh full gallery, not packaged
JPEGs or model inference. Nothing is selected automatically and no score is
inferred. This permits a concise 2/3-column view without re-reading full PNGs
during formal timing. Default is a read-only plan.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
from PIL import Image, ImageDraw, ImageFont


def need(v,m):
    if not v:raise RuntimeError(m)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--gallery-dir',type=Path,required=True)
    q.add_argument('--source-proof-sha256',required=True)
    q.add_argument('--selected-label',required=True)
    q.add_argument('--selected-checkpoint-sha256',required=True)
    q.add_argument('--layout',choices=['two','three'],default='three')
    q.add_argument('--out-dir',type=Path,required=True)
    q.add_argument('--execute',action='store_true');a=q.parse_args()
    src=a.gallery_dir.resolve();p=src/'proof.json'
    need(sha(p)==a.source_proof_sha256,'Actual full-gallery proof changed')
    d=json.loads(p.read_text());s=d['sources']
    need(d['status']=='PASS_FRESH_FULL_NATIVE_ROI4' and d['packaged_JPEG_used'] is False and
         d['no_resize_filter_or_enhancement'] is True and len(d['outputs'])==4,'Verified four-case fresh gallery required')
    bi=[i for i,x in enumerate(s) if x['role']=='original_pix_baseline']
    ti=[i for i,x in enumerate(s) if x['role']=='fixed_HYPIR_teacher']
    ci=[i for i,x in enumerate(s) if x['role']=='distilled_pix_candidate' and x['label']==a.selected_label and
        x['checkpoint_sha256']==a.selected_checkpoint_sha256]
    need(len(bi)==len(ti)==len(ci)==1,'Explicit actual selected identity must match one gallery column')
    indices=bi+ci+(ti if a.layout=='three' else [])
    titles=['Original PixG5i','Distilled PixG5i']+(['Fixed HYPIR teacher'] if a.layout=='three' else [])
    need(s[ci[0]]['checkpoint_sha256']!=s[bi[0]]['checkpoint_sha256'],'Selected candidate is unchanged baseline')
    need(all(x['actual103_PNG_and_input_SHA_verified'] is True for x in s),'Full input and PNG provenance required')
    for r in d['outputs']:
        f=src/Path(r['path']).name
        need(sha(f)==r['sha256'] and r['crop_pixels_exact'] is True,'Native panel source changed')
    if not a.execute:
        print(json.dumps(dict(status='PLAN_ONLY',selected_label=a.selected_label,selected_checkpoint_sha256=a.selected_checkpoint_sha256,
                              columns=[s[i]['label'] for i in indices],out=str(a.out_dir))));return
    out=a.out_dir.resolve();need(not out.exists(),'New isolated presentation output required');out.mkdir(parents=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:
        try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
        except OSError:font=ImageFont.load_default()
    outputs=[]
    for r in d['outputs']:
        roi=r['ROI'];w=roi[2]-roi[0];h=roi[3]-roi[1];f=src/Path(r['path']).name
        with Image.open(f) as parent:
            need(parent.mode=='RGB' and parent.format=='PNG' and parent.size==(16+len(s)*(w+16),h+86),'Unexpected verified panel geometry')
            tiles=[parent.crop((16+i*(w+16),70,16+i*(w+16)+w,70+h)) for i in indices]
        panel=Image.new('RGB',(16+len(indices)*(w+16),h+86),'#f4f4f4');draw=ImageDraw.Draw(panel)
        draw.text((16,10),f'case{r["case"]} | native 1:1 ROI',font=font,fill='black')
        for j,(title,tile) in enumerate(zip(titles,tiles)):
            x=16+j*(w+16);draw.text((x,42),title,font=font,fill='black');panel.paste(tile,(x,70))
        path=out/f'case{r["case"]}_selected_{a.layout}_column_ROI.png';panel.save(path,compress_level=1)
        with Image.open(path) as saved:
            for j,tile in enumerate(tiles):
                x=16+j*(w+16);need(saved.crop((x,70,x+w,70+h)).tobytes()==tile.tobytes(),'Stored native presentation pixels changed')
        need(sha(f)==r['sha256'],'Source panel changed during layout')
        outputs.append(dict(case=r['case'],ROI=roi,source_panel=record(f),source_column_indices=indices,
                            pixel_SHA256=[hashlib.sha256(t.tobytes()).hexdigest() for t in tiles],crop_pixels_exact=True,**record(path)))
    page='<!doctype html><meta charset="utf-8"><title>Selected PixG5i native ROI</title><style>body{font:16px system-ui;margin:24px}figure{overflow:auto}img{max-width:none}</style><h1>Original and distilled PixG5i</h1><p>Actual fresh full-queue output pixels at fixed native 1:1 ROIs. Layout only; no filtering, resizing, enhancement or inference. Root supplied the selected checkpoint identity. These four examples alone establish neither dataset score nor truth recovery.</p>'
    for r in outputs:page+=f'<figure><figcaption>case{r["case"]} | {html.escape(str(r["ROI"]))}</figcaption><img src="{Path(r["path"]).name}"></figure>'
    page+='<details><summary>Actual pixel provenance</summary><a href="proof.json">proof.json</a></details>';(out/'index.html').write_text(page)
    need(sha(p)==a.source_proof_sha256,'Source full gallery proof changed')
    proof=dict(status='PASS_SELECTED_PRESENTATION_NATIVE_ROI4',CPU_only=True,GPU_used=False,root_supplied_selected_label=a.selected_label,
               selected_checkpoint_sha256=a.selected_checkpoint_sha256,source_full_gallery=record(p),source_columns=[s[i] for i in indices],
               no_resampling_filter_enhancement=True,no_new_model_output_generation=True,no_score_inferred=True,outputs=outputs,
               HTML=record(out/'index.html'),helper=record(__file__))
    (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),proof_sha256=sha(out/'proof.json'))))


if __name__=='__main__':main()
