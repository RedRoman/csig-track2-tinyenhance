"""Actual medium300 / mild600 / medium600 / HYPIR full-queue identities only."""
from pathlib import Path
import argparse
import hashlib
import json
R=Path('/cache/independent_sr_cuda_20261013');P=R/'nr_pixg5i_distill_full_evaluations'
ADAPTER='70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
IDENTITIES=[
 ('distilled_pix_reference','PixG5i Medium 300',P/'pixg5i_hypir_medium_s300_all','005d13cb5b003d54e576678c52908181016cb9fa1746a9455465f5f2f619c458',ADAPTER,'654f287813048468d994e9729de8d14cf68f126830a9d7967f49669e86de351b'),
 ('distilled_pix_candidate','PixG5i Mild 600',P/'pixg5i_hypir_mild_s600_all','51dfbd335e4076009409ae7efc727db99f98bb6dc1cc76d1161dba9b75e3ffc5',ADAPTER,'f34003092be152c6f7723ed2a3232fcaa9a8ea4b8b279cfbfb74136cc762320d'),
 ('distilled_pix_candidate','PixG5i Medium 600',P/'pixg5i_hypir_medium_s600_all','45c7f6bb414567f31b906fc996c037f31df7ce6aebb615c19183e02886950fab',ADAPTER,'1c5a7813c254ace9415486dd95fec8bf4412860f8ed7286e6989f1ab83b11b35'),
 ('fixed_HYPIR_teacher','Fixed HYPIR teacher',R/'nr_push7_recovered_full_evaluations/push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all','23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c','c98c406884cf07bc5541e315393b98e11351165b35c94d33f926d5443f9563c3','d7e03918355aa9cb459c1d37f8ffafcee89856bb8b23cdf0346b5e440b3046cb')]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
    if not v:raise RuntimeError(m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    pending=[str(d) for _,_,d,_,_,_ in IDENTITIES if not(d/'render_complete').is_file() or not(d/'render_manifest.json').is_file()]
    if pending:print(json.dumps(dict(status='PENDING_FINAL_RENDER103',pending=pending)));return
    cols=[]
    for role,title,d,raw,runner,cmsha in IDENTITIES:
        protocol=json.loads((d/'render_protocol.json').read_text())['protocol'];md=Path(protocol['model_dir']);rr=protocol['runner'];cm=md.parent/'candidate_manifest.json'
        need(sha(cm)==cmsha and rr['checkpoint_sha256']==raw and rr['runner_sha256']==runner,'Actual registered identity differs')
        binds=[dict(path=str(cm),sha256=cmsha)]
        if role!='fixed_HYPIR_teacher':
            b=md/'pixg5i_binding.json';binds.append(dict(path=str(b),sha256=sha(b)))
        cols.append(dict(role=role,title=title,label=d.name,evaluation_dir=str(d),checkpoint_sha256=raw,runner_sha256=runner,
                         render_protocol_sha256=sha(d/'render_protocol.json'),render_manifest_sha256=sha(d/'render_manifest.json'),binding_files=binds))
    need(not a.out.exists(),'New immutable actual spec required');a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(schema='medium300_mild600_medium600_fixed_teacher_actual_v1',columns=cols),indent=2)+'\n')
    print(json.dumps(dict(status='ACTUAL_FINAL103_SPEC_READY',path=str(a.out),sha256=sha(a.out))))


if __name__=='__main__':main()
