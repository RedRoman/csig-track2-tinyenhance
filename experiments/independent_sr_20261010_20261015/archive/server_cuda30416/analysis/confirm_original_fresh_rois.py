"""CPU check of real original PixG5i full103 and four native RGB ROI crops."""
from pathlib import Path
import hashlib
import json
import argparse
from PIL import Image

R = Path('/cache/independent_sr_cuda_20261013')
D = R/'nr_pixg5i_distill_full_evaluations/pixg5i_original_20261015_all'
BOXES = {7: (1856,1408,2496,1856), 16: (1280,1216,1920,1728),
         41: (768,1664,1408,2176), 83: (1216,1920,1856,2432)}
RAW = 'f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
ADAPTER = '70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
CM = 'f47a1b5318be533c221d612a2baf99d90ee49267d4728e50bfb8e42b395bfbff'
SEEN = {}

def need(v,m):
    if not v: raise RuntimeError(m)

def sha(p):
    p=Path(p).resolve();s=p.stat();stamp=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
    if p in SEEN:
        need(SEEN[p][0]==stamp,'Changed source: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    s=p.stat();need(stamp==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Unstable read')
    SEEN[p]=(stamp,h.hexdigest());return h.hexdigest()

def rec(p): return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def read(p):sha(p);return json.loads(Path(p).read_text())

def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--out-dir',type=Path,required=True);a=q.parse_args()
    out=a.out_dir.resolve();need(out.is_relative_to(R/'analysis') and not out.exists(),'New analysis output only')
    need((D/'render_complete').is_file(),'Real full render not final')
    w=read(D/'render_protocol.json');p=w['protocol'];rr=p['runner'];md=Path(p['model_dir'])
    need(w['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'Protocol content hash')
    need(rr['checkpoint_sha256']==RAW and rr['runner_sha256']==ADAPTER and rr['stage']=='original_baseline','Wrong original model')
    need(rr['model_type']=='supplied_pixG5i_pixel_UNet' and rr['used_as_G4_reference'] is False,'Wrong family')
    need(sha(md/'runner.py')==ADAPTER and sha(md/'model.pth')==RAW,'Actual source/raw changed')
    need(sha(md.parent/'candidate_manifest.json')==CM,'Actual registered original manifest changed')
    for n,v in p['model_dir_sha256'].items():need(sha(md/n)==v,'Actual modeldir differs')
    order=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
    need(p['cases']=='all' and p['warmup'] is False and [(t['split'],t['case']) for t in p['tasks']]==order,'Wrong full103 queue')
    m=read(D/'render_manifest.json');need(m['protocol']==p and m['protocol_id']==w['protocol_id'],'Final protocol differs')
    rows=m['records'];need([(r['split'],r['case']) for r in rows]==order,'Incomplete final103')
    actual=[]
    for task,row in zip(p['tasks'],rows):
        rel=f'render/{row["split"]}/case{row["case"]}.png';f=D/rel
        need(row['output']==rel and not f.is_symlink() and sha(f)==row['sha256'],'Actual committed PNG differs')
        need(sha(task['input'])==task['input_sha256'],'Actual source input differs')
        actual.append(dict(split=row['split'],case=row['case'],PNG=rec(f),input_sha256=task['input_sha256'],size=row['size']))
    need({str(f.relative_to(D)) for f in (D/'render').rglob('*') if f.is_file()}=={r['output'] for r in rows},'Extra native render files')
    out.mkdir(parents=True);outputs=[]
    for case,box in BOXES.items():
        row=next(r for r in actual if r['split']=='test' and r['case']==case)
        with Image.open(row['PNG']['path']) as im:
            need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==row['size'],'Not canonical RGB PNG')
            crop=im.crop(box)
        f=out/f'case{case}_original_fresh_native_ROI.png';crop.save(f,compress_level=1)
        with Image.open(f) as check:need(check.tobytes()==crop.tobytes(),'Stored native crop changed')
        outputs.append(dict(case=case,ROI=list(box),source=row,output=rec(f),crop_pixels_exact=True))
    for f in list(SEEN):sha(f)
    proof=dict(status='PASS_ACTUAL_ORIGINAL_FRESH103_ROI4',CPU_only=True,GPU_used=False,packaged_JPEG_used=False,
               source=rec(__file__),evaluation_dir=str(D),protocol=rec(D/'render_protocol.json'),final_manifest=rec(D/'render_manifest.json'),
               checkpoint_sha256=RAW,adapter_sha256=ADAPTER,candidate_manifest=rec(md.parent/'candidate_manifest.json'),
               actual103PNG_and_input_SHA_verified=True,actual_images=actual,outputs=outputs,
               no_score_or_RNG_equivalence_claim=True,no_resize_filter_or_enhancement=True)
    (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status=proof['status'],out_dir=str(out),proof_sha256=sha(out/'proof.json'))))

if __name__=='__main__':main()
