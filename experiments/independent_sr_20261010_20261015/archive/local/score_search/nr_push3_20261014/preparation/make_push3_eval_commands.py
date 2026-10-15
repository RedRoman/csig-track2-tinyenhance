"""Emit reviewable frozen-renderer/original-score commands; never launches GPU.

Use --label multiple times to emit a sequential render shell and one interleaved
multi-context scorer. The root scheduler sets each CUDA_VISIBLE_DEVICES value.
"""
from pathlib import Path
import argparse, hashlib, json, re, shlex
ROOT=Path('/cache/independent_sr_cuda_20261013')
CASES=[1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',action='append',required=True,help='Registration label, excluding _pilot/_all suffix')
    p.add_argument('--scope',choices=['pilot','full'],required=True)
    p.add_argument('--write-dir',type=Path,default=None,help='Optional fresh shell output directory; refuses overwrite')
    a=p.parse_args();assert len(a.label)==len(set(a.label))
    scope=a.scope;folder='nr_push3_'+scope+'_evaluations';suffix='pilot' if scope=='pilot' else 'all'
    renders=[];score=[str(ROOT/'env/bin/python'),str(ROOT/'code/score_cuda_stream.py'),'--tool-dir',str(ROOT/'scoring_tool'),'--out-root',str(ROOT/folder)]
    checks=[]
    for label in a.label:
        assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,150}',label)
        submit=ROOT/'submissions'/('hypir_'+label);reg=json.loads((submit/'registration.json').read_text())
        assert reg['label']==label and reg['mode']=='execute'
        for item in reg['registered_files']:assert sha(item['path'])==item['sha256']
        assert sha(reg['checkpoint']['path'])==reg['checkpoint']['sha256']
        for item in reg['sources']:assert sha(item['path'])==item['sha256']
        commands=reg['commands'][scope]
        renders.append(commands['render'])
        score+=['--evaluation',commands['label']+'='+str(ROOT/folder/commands['label']/'render')]
        checks.append({'registration':str(submit/'registration.json'),'registration_sha256':sha(submit/'registration.json'),'checkpoint_sha256':reg['checkpoint']['sha256'],'evaluation_label':commands['label']})
    if scope=='pilot':score+=['--test-cases',','.join(map(str,CASES))]
    header='#!/usr/bin/env bash\nset -euo pipefail\n: "${CUDA_VISIBLE_DEVICES:?Root scheduler must assign GPU explicitly}"\n'
    plan={'scope':scope,'cpu_only':True,'registration_checks':checks,'render_commands':renders,'score_command':shlex.join(score),'warning':'Pilot scores are S_subset, never full S_proxy. Commands require distinct immutable output labels.'}
    if a.write_dir:
        a.write_dir.mkdir(parents=True,exist_ok=False)
        (a.write_dir/'render_sequential.sh').write_text(header+'\n'.join(renders)+'\n')
        (a.write_dir/'score_contexts.sh').write_text(header+plan['score_command']+'\n')
        (a.write_dir/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
