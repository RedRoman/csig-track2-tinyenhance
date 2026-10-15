from pathlib import Path
import json,hashlib
from PIL import Image
R=Path('/cache/independent_sr_cuda_20261013');ev=R/'nr_pixg5i_distill_full_evaluations/pixg5i_hypir_coverage_v4_s200_all';dg=R/'analysis/pixg5i_diag4/coverage_v4_s200_diagnostic'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
mp=ev/'render_manifest.json';j=json.loads(mp.read_text());assert len(j['records'])==103 and j['scope']=='test100+validation3'
assert [(x['split'],x['case']) for x in j['records']]==[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
assert j['protocol']['runner']['checkpoint_sha256']=='503074449893247332df973a85f9be1f348d9f130f31a28455b62790fb50c929'
rows=[]
for case,box in [(7,[1856,1408,2496,1856]),(16,[1280,1216,1920,1728]),(41,[768,1664,1408,2176]),(83,[1216,1920,1856,2432])]:
 r=next(x for x in j['records'] if x['split']=='test' and x['case']==case);a=ev/r['output'];b=dg/'render/test'/f'case{case}.png';sa=sha(a);sb=sha(b);assert sa==sb==r['sha256']
 task=next(x for x in j['protocol']['tasks'] if x['split']=='test' and x['case']==case)
 with Image.open(a) as ia, Image.open(b) as ib:
  assert ia.mode==ib.mode=='RGB' and ia.size==ib.size==tuple(r['size']);ca=ia.crop(box);cb=ib.crop(box);assert ca.tobytes()==cb.tobytes();ch=hashlib.sha256(ca.tobytes()).hexdigest()
 rows.append(dict(case=case,ROI=box,full_PNG=dict(path=str(a),sha256=sa,bytes=a.stat().st_size),diagnostic_PNG=dict(path=str(b),sha256=sb,bytes=b.stat().st_size),native_size=r['size'],input_sha256=task['input_sha256'],whole_PNG_bytes_equal=True,actual_ROI_RGB_pixels_equal=True,ROI_RGB_pixels_sha256=ch))
print(json.dumps(dict(status='PASS_ACTUAL_FULL4_DIAGNOSTIC4_BYTES_AND_ROI_PIXELS_EXACT',CPU_only=True,GPU_used=False,scope='Four fixed visual cases only; not full103 image/quality audit',checkpoint_sha256=j['protocol']['runner']['checkpoint_sha256'],full_manifest=dict(path=str(mp),sha256=sha(mp),records=103,scope=j['scope']),records=rows,no_model_selection_or_quality_score_claim=True)))
