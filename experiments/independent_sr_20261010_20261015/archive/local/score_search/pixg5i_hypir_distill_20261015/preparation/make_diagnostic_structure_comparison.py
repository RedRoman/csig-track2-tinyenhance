"""CPU layout of actual diagnostic4 crops and verified full-reference crop cache.

Diagnostic candidate is not claimed to use the full103 RNG sequence. No score,
truth recovery, model selection, resizing, filtering or enhancement is implied.
"""
from pathlib import Path
import argparse
import hashlib
import json
from PIL import Image,ImageDraw,ImageFont

def need(v,m):
    if not v:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)

def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--crop-dir',type=Path,required=True);q.add_argument('--crop-proof-sha256',required=True)
    q.add_argument('--candidate-checkpoint-sha256',required=True);q.add_argument('--candidate-title',required=True)
    q.add_argument('--reference-gallery',type=Path,required=True);q.add_argument('--reference-proof-sha256',required=True)
    q.add_argument('--out-dir',type=Path,required=True);a=q.parse_args()
    cp=a.crop_dir/'proof.json';rp=a.reference_gallery/'proof.json'
    need(sha(cp)==a.crop_proof_sha256 and sha(rp)==a.reference_proof_sha256,'Actual diagnostic/reference proofs changed')
    d=json.loads(cp.read_text());r=json.loads(rp.read_text())
    need(d['status']=='PASS_ACTUAL_DIAGNOSTIC4_NATIVE_ROI_ONLY' and d['checkpoint_sha256']==a.candidate_checkpoint_sha256 and
         [x['case'] for x in d['outputs']]==[7,16,41,83],'Real explicit diagnostic4 checkpoint required')
    need(r['status']=='PASS_STRUCTURE_PROGRESS_FRESH_FULL_NATIVE_ROI4' and r['packaged_JPEG_used'] is False,'Verified full reference gallery required')
    refs=r['sources'];mi=[i for i,s in enumerate(refs) if s['checkpoint_sha256']=='45c7f6bb414567f31b906fc996c037f31df7ce6aebb615c19183e02886950fab']
    ti=[i for i,s in enumerate(refs) if s['role']=='fixed_HYPIR_teacher' and s['checkpoint_sha256']=='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c']
    need(len(mi)==len(ti)==1,'Actual fixed medium600 and HYPIR reference columns required')
    out=a.out_dir.resolve();need(not out.exists(),'New diagnostic presentation only');out.mkdir(parents=True)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    except OSError:font=ImageFont.load_default()
    outputs=[]
    for row in d['outputs']:
        case=row['case'];box=row['ROI'];w=box[2]-box[0];h=box[3]-box[1]
        source=a.crop_dir/Path(row['output']['path']).name;need(sha(source)==row['output']['sha256'],'Downloaded diagnostic crop differs')
        with Image.open(source) as im:need(im.mode=='RGB' and im.size==(w,h),'Wrong native diagnostic crop');candidate=im.copy()
        panelrec=next(x for x in r['outputs'] if x['case']==case);panelpath=a.reference_gallery/Path(panelrec['path']).name
        need(sha(panelpath)==panelrec['sha256'] and panelrec['ROI']==box,'Verified reference panel changed')
        with Image.open(panelpath) as full:
            need(full.mode=='RGB' and full.size==(16+len(refs)*(w+16),h+86),'Wrong full-reference panel geometry')
            cached=[full.crop((16+i*(w+16),70,16+i*(w+16)+w,70+h)) for i in [mi[0],ti[0]]]
        originals=[]
        for i in [mi[0],ti[0]]:
            actual=next(x for x in refs[i]['images'] if x['split']=='test' and x['case']==case)
            need(actual['input_sha256']==row['input_sha256'] and actual['native_size']==row['size'],'Diagnostic/reference native input differs');originals.append(actual)
        tiles=[cached[0],candidate,cached[1]];titles=['Medium 600 (full reference)',a.candidate_title,'Fixed HYPIR teacher']
        panel=Image.new('RGB',(16+3*(w+16),h+86),'#f4f4f4');draw=ImageDraw.Draw(panel)
        draw.text((16,10),f'case{case} | DIAGNOSTIC comparison | native 1:1 ROI',font=font,fill='black')
        for i,(title,tile) in enumerate(zip(titles,tiles)):
            x=16+i*(w+16);draw.text((x,42),title,font=font,fill='black');panel.paste(tile,(x,70))
        path=out/f'case{case}_diagnostic_structure_ROI_1to1.png';panel.save(path,compress_level=1)
        with Image.open(path) as check:
            for i,tile in enumerate(tiles):
                x=16+i*(w+16);need(check.crop((x,70,x+w,70+h)).tobytes()==tile.tobytes(),'Stored native pixels changed')
        outputs.append(dict(case=case,ROI=box,diagnostic_source=row,full_reference_sources=originals,
                            cached_panel=record(panelpath),crop_pixels_exact=True,**record(path)))
    page='<!doctype html><meta charset="utf-8"><title>Structure diagnostic4 only</title><style>body{font:16px system-ui;margin:24px}figure{overflow:auto}img{max-width:none}</style><h1>Structure-stage diagnostic: not a full score</h1><p>Middle column: real candidate native diagnostic4 sequence (cases7/16/41/83). Left and right: verified medium600 and fixed HYPIR full103 reference crop caches. Actual inputs and native geometry agree, but this is not a matched full-render RNG sequence or formal quality evaluation. Fixed 1:1 pixels, layout only; no resize, filter, enhancement or regenerated pixels. Smoothness alone does not prove character, bar or leaf structure recovery.</p>'
    for o in outputs:page+=f'<figure><img src="{Path(o["path"]).name}"></figure>'
    page+='<details><summary>Actual PNG/input provenance and cache scope</summary><a href="proof.json">proof.json</a></details>';(out/'index.html').write_text(page)
    need(sha(cp)==a.crop_proof_sha256 and sha(rp)==a.reference_proof_sha256,'Source proof changed')
    proof=dict(status='PASS_DIAGNOSTIC4_VS_FULL_REFERENCE_NATIVE_ROI',CPU_only=True,GPU_used=False,
               diagnostic_crop_proof=record(cp),full_reference_gallery_proof=record(rp),candidate_checkpoint_sha256=a.candidate_checkpoint_sha256,
               reference_pixels_scope='Content-addressed crop cache from actual SHA/pixel-verified full103 PNGs',outputs=outputs,
               same_input_SHA_and_geometry=True,matched_full_RNG_or_score_claim=False,no_resampling_filter_enhancement=True,
               HTML=record(out/'index.html'),helper=record(__file__))
    (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),proof_sha256=sha(out/'proof.json'))))

if __name__=='__main__':main()
