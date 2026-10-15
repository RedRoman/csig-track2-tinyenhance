"""CPU-only receipt and native ROI crops from a completed real diagnostic4."""
from pathlib import Path
import argparse
import hashlib
import json
from PIL import Image

R=Path('/cache/independent_sr_cuda_20261013')
PRODUCER='cee4ceb8df95cfc5a7ebbc2a00b1ce0566f1866cbf5cb8e4ee7726fb306fdca9'
ADAPTER='70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
BOXES={7:(1856,1408,2496,1856),16:(1280,1216,1920,1728),41:(768,1664,1408,2176),83:(1216,1920,1856,2432)}

def need(v,m):
    if not v:raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def record(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)

def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--diag-dir',type=Path,required=True);q.add_argument('--candidate-manifest-sha256',required=True)
    q.add_argument('--checkpoint-sha256',required=True);q.add_argument('--out-dir',type=Path,required=True);a=q.parse_args()
    d=a.diag_dir.resolve();out=a.out_dir.resolve();need(out.is_relative_to(R/'analysis') and not out.exists(),'New CPU crop output required')
    need((d/'diagnostic_complete').is_file(),'Real diagnostic4 incomplete')
    f=d/'diagnostic_manifest.json';fp=record(f);m=json.loads(f.read_text());p=m['protocol'];rr=p['runner']
    need(m['status']=='COMPLETE_DIAGNOSTIC4_NOT_FULL_OR_SCORE' and m['actual_native_count']==4 and m['quality_or_speed_claim'] is False,
         'Not a completed diagnostic4-only receipt')
    need(m['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'Diagnostic protocol hash differs')
    need(p['scope']=='DIAGNOSTIC_NATIVE4_ONLY_NOT_FORMAL_OR_SCORE' and p['cases']==list(BOXES) and p['seed']==231 and
         p['no_val_or_GT'] is True and p['source']['sha256']==PRODUCER==sha(p['source']['path']),'Diagnostic source/seed/scope differs')
    md=Path(p['model_dir']);cm=md.parent/'candidate_manifest.json'
    need(p['candidate_manifest']['sha256']==a.candidate_manifest_sha256==sha(cm) and
         rr['checkpoint_sha256']==p['checkpoint_sha256']==a.checkpoint_sha256==sha(md/'model.pth') and
         rr['runner_sha256']==ADAPTER==sha(md/'runner.py'),'Actual diagnostic model identity differs')
    for name,digest in p['model_dir_sha256'].items():need(sha(md/name)==digest,'Actual source changed')
    tasks=p['tasks'];rows=m['records']
    need([x['case'] for x in tasks]==[x['case'] for x in rows]==list(BOXES) and len(tasks)==len(rows)==4,'Wrong diagnostic four-image order')
    outputs=[];out.mkdir(parents=True)
    for i,(t,r) in enumerate(zip(tasks,rows)):
        c=t['case'];box=BOXES[c];png=d/f'render/test/case{c}.png'
        need(r['task_index']==i and r['split']==t['split']=='test' and r['output']==str(png.relative_to(d)) and
             Path(r['PNG']['path']).resolve()==png and sha(png)==r['PNG']['sha256'] and
             r['input_sha256']==t['input_sha256']==sha(t['input']),'Actual diagnostic PNG/input differs')
        with Image.open(png) as im:
            need(im.mode=='RGB' and im.format=='PNG' and list(im.size)==r['size']==t['size'],'Native diagnostic PNG geometry differs');crop=im.crop(box)
        target=out/f'case{c}_structure_v2_diagnostic_native_ROI.png';crop.save(target,compress_level=1)
        with Image.open(target) as saved:need(saved.tobytes()==crop.tobytes(),'Stored native ROI pixels changed')
        outputs.append(dict(case=c,ROI=list(box),native_source=record(png),input_sha256=t['input_sha256'],
                            size=t['size'],before_fourRNG_sha256=r['before_fourRNG_sha256'],after_fourRNG_sha256=r['after_fourRNG_sha256'],
                            output=record(target),crop_pixels_exact=True))
    need(sha(f)==fp['sha256'] and sha(cm)==a.candidate_manifest_sha256,'Source changed during cropping')
    proof=dict(status='PASS_ACTUAL_DIAGNOSTIC4_NATIVE_ROI_ONLY',CPU_only=True,GPU_used=False,
               diagnostic_manifest=fp,diagnostic_protocol=record(d/'diagnostic_protocol.json'),protocol=p,
               candidate_manifest=record(cm),checkpoint_sha256=a.checkpoint_sha256,outputs=outputs,
               no_full_score_or_matched_full_RNG_claim=True,no_resize_filter_enhancement=True,helper=record(__file__))
    (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),proof_sha256=sha(out/'proof.json'))))

if __name__=='__main__':main()
