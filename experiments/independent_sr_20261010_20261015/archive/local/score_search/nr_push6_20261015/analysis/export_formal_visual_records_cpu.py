"""CPU-only committed formal preview snapshot, optionally before full103 EOF."""
from pathlib import Path
import argparse
import hashlib
import io
import json

R=Path('/cache/independent_sr_cuda_20261013')
P=R/'nr_push4_full_evaluations/push4_a7750_compiled_all'
C=R/'nr_push6_full_evaluations/push6_a7750_ro_wavelet_all'
SEL='b3148b21f466854f9fe8f81c7eca1348da5bf793141799573ff3be8b20e25c3f'
RUNNER='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
RAW='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
CASES=[7,16,41,83]


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--allow-progress',action='store_true');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    need(a.out.resolve().is_relative_to(R/'analysis/push6') and not a.out.exists(),'New push6 snapshot path required')
    selection=R/'results/nr_push4_selection.json';need(sha(selection)==SEL,'Frozen selection changed');s=read(selection)
    audit_path=Path(s['full_audit']);need(sha(audit_path)==s['full_audit_sha256'],'Frozen parent audit changed');audit=read(audit_path)
    pm=read(P/'render_manifest.json');pp=read(C/'render_protocol.json');protocol=pp['protocol'];pid=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    need(pp['protocol_id']==pid and protocol['cases']=='all' and protocol['warmup'] is False and protocol['runner']['runner_sha256']==RUNNER and protocol['runner']['checkpoint_sha256']==RAW,'Not the sealed formal RO-wavelet queue')
    need(protocol['inference_rng']=='seed231 before model initialization; continuous ordered val3 then tests','Wrong formal RNG policy')
    complete=(C/'render_complete').is_file()
    if complete:
        m=read(C/'render_manifest.json');need(m['protocol']==protocol and m['protocol_id']==pid and len(m['records'])==103,'Incomplete final manifest');rows=m['records'];state_sha=sha(C/'render_manifest.json')
    else:
        need(a.allow_progress,'Full103 not complete; explicitly allow a progress-only preview')
        import torch
        need(not torch.cuda.is_initialized(),'CPU-only snapshot required')
        # Atomic renderer replacement is allowed; this open/read preserves one
        # committed inode snapshot, never the writer's next partial update.
        data=(C/'render_progress.pt').read_bytes();state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
        need(state['protocol_id']==pid and state['checkpoint_sha256']==RAW,'Committed progress identity differs');rows=state['records'];state_sha=hashlib.sha256(data).hexdigest()
    order=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
    need([(x['split'],x['case']) for x in rows]==order[:len(rows)] and [(x['split'],x['case']) for x in protocol['tasks']]==order,'Formal task order changed')
    parent={};candidate={}
    for case in CASES:
        index=case+2;old=pm['records'][index];frozen=audit['records'][index];task=protocol['tasks'][index]
        need((old['split'],old['case'])==('test',case) and frozen['split']=='test' and frozen['case']==case and old['sha256']==frozen['actual_png_sha256'] and task['input_sha256']==frozen['actual_input_sha256'],'Parent does not match frozen formal audit')
        path=P/old['output'];need(sha(path)==old['sha256'],'Actual formal parent PNG changed')
        parent[str(case)]={**old,'path':str(path),'bytes':path.stat().st_size,'input_sha256':task['input_sha256']}
        if index<len(rows):
            row=rows[index];need(row['task_index']==index and row['output']==f'render/test/case{case}.png' and row['size']==old['size'],'Wrong committed candidate path/size')
            path=C/row['output'];need(not path.is_symlink() and sha(path)==row['sha256'] and sha(task['input'])==task['input_sha256'],'Actual committed PNG/input changed')
            candidate[str(case)]={**row,'path':str(path),'bytes':path.stat().st_size,'input_sha256':task['input_sha256']}
    out=dict(schema='push6_formal_visual_committed_snapshot_v1',status='PASS',CPU_only=True,GPU_used=False,candidate_formal_complete103=complete,committed_images=len(rows),formal_candidate_dir=str(C),formal_parent_dir=str(P),scope='Same A7750 weights, accelerated inference; no new fine-tuning. Progress is never a full-quality result.',candidate_runner_sha256=RUNNER,checkpoint_sha256=RAW,candidate_protocol_id=pid,candidate_protocol_file_sha256=sha(C/'render_protocol.json'),committed_state_or_final_manifest_sha256=state_sha,parent_selection_sha256=SEL,parent_full_audit_sha256=s['full_audit_sha256'],parent_manifest_sha256=sha(P/'render_manifest.json'),parent_records=parent,candidate_records=candidate,script_sha256=sha(__file__))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','snapshot':str(a.out),'sha256':sha(a.out),'committed_images':len(rows),'available_cases':list(candidate),'complete103':complete}))


if __name__=='__main__':main()
