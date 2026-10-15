"""Bind the four ACTUAL full render queues to a native ROI gallery spec."""
from pathlib import Path
import hashlib
import json
import argparse
R=Path('/cache/independent_sr_cuda_20261013')
P=R/'nr_pixg5i_distill_full_evaluations'
TEACHER=R/'nr_push7_recovered_full_evaluations/push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all'
IDENTITIES=[
    ('original_pix_baseline','Original PixG5i',P/'pixg5i_original_20261015_all',
     'f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487',
     '70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a',
     'f47a1b5318be533c221d612a2baf99d90ee49267d4728e50bfb8e42b395bfbff'),
    ('fixed_HYPIR_teacher','Fixed HYPIR teacher',TEACHER,
     '23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c',
     'c98c406884cf07bc5541e315393b98e11351165b35c94d33f926d5443f9563c3',
     'd7e03918355aa9cb459c1d37f8ffafcee89856bb8b23cdf0346b5e440b3046cb'),
    ('distilled_pix_candidate','PixG5i Mild 100',P/'pixg5i_hypir_mild_s100_all',
     '6a2a7e69c24b8af8287ce91d7f8ec451149818fde312c6919d0eb42d714d1e2a',
     '70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a',
     'f1d5ca308f00f118a2d03bb7c87ee612c182410df1437f9c4ac52df082f5c739'),
    ('distilled_pix_candidate','PixG5i Medium 100',P/'pixg5i_hypir_medium_s100_all',
     'afa0b9a5702615be5314868e345a703f4014c29b40462cf38c9adc0e707008de',
     '70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a',
     'd69d5c87ab7a6cabb5ca4fa6b0f91b1d4db3ae958d0afe76ac91e06516b25ce7')]

def need(v,m):
    if not v:raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    pending=[str(d) for _,_,d,_,_,_ in IDENTITIES if not(d/'render_complete').is_file() or not(d/'render_manifest.json').is_file()]
    if pending:
        print(json.dumps(dict(status='PENDING_FINAL_RENDER103',pending=pending)));return
    columns=[]
    for role,title,d,raw,runner,cmsha in IDENTITIES:
        p=json.loads((d/'render_protocol.json').read_text())['protocol'];md=Path(p['model_dir']);rr=p['runner']
        cm=md.parent/'candidate_manifest.json'
        need(sha(cm)==cmsha and rr['checkpoint_sha256']==raw and rr['runner_sha256']==runner,'Actual registered identity differs')
        bindings=[dict(path=str(cm),sha256=cmsha)]
        if role!='fixed_HYPIR_teacher':
            b=md/'pixg5i_binding.json';bindings.append(dict(path=str(b),sha256=sha(b)))
        columns.append(dict(role=role,title=title,label=d.name,evaluation_dir=str(d),checkpoint_sha256=raw,runner_sha256=runner,
                            render_protocol_sha256=sha(d/'render_protocol.json'),render_manifest_sha256=sha(d/'render_manifest.json'),binding_files=bindings))
    need(not a.out.exists(),'Spec output must be new/immutable')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(schema='fresh_pixg5i_HYPIR_gallery_actual_s100_v1',columns=columns),indent=2)+'\n')
    print(json.dumps(dict(status='ACTUAL_FINAL103_SPEC_READY',path=str(a.out),sha256=sha(a.out))))

if __name__=='__main__':main()
