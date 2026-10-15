"""Read only independently PASSed complete pilots; no torch, CUDA or image I/O."""
import argparse
import hashlib
import json
import math
from pathlib import Path

CASES = [1, 6, 7, 11, 16, 18, 21, 23, 25, 28, 31, 36, 41, 46, 51, 61, 71, 81, 91, 100]
ORDER = [('validation', n) for n in range(1, 4)] + [('test', n) for n in CASES]
KEYS = ['clipiqa', 'maniqa', 'musiq', 'niqe']


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def means(rows, split):
    rr = [r for r in rows if r['split'] == split]
    keys = KEYS + (['lpips', 'dists'] if split == 'validation' else [])
    return {k: sum(r['metrics'][k] for r in rr) / len(rr) for k in keys}


def contributions(test, val):
    d = {'CLIPIQA': test['clipiqa'], 'MANIQA': test['maniqa'],
         'MUSIQ/100': test['musiq'] / 100, 'NIQE_term': max(0, (10-test['niqe']) / 10),
         'LPIPS_FR_term': -val['lpips'], 'DISTS_FR_term': -val['dists']}
    d['NR'] = sum(d[k] for k in ['CLIPIQA', 'MANIQA', 'MUSIQ/100', 'NIQE_term'])
    d['FR'] = 2-val['lpips']-val['dists']
    d['S_subset'] = d['NR']+d['FR']
    return d


def load(audit_path, rows_path, marker_path=None):
    a = json.loads(audit_path.read_text())
    need(a['status'] == 'PASS' and a['image_rows'] == 23 and a['scalar_metrics'] == 98,
         f'Not complete independently PASSed23/98: {audit_path}')
    need(a['actual_png_manifest_scored_sha'] == 'PASS23' and a['actual_source_code_sha'] == 'PASS',
         'Actual image/source verification absent')
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    need([(r['split'], r['case']) for r in rows] == ORDER, 'Fixed pilot ordering differs')
    ar = {(r['split'], r['case']): r for r in a['records']}
    need(len(ar) == 23, 'Audit record count/uniqueness differs')
    for r in rows:
        need(r['sr_sha256'] == ar[(r['split'], r['case'])]['actual_png_sha256'], 'Row SHA differs from actual audit')
        need(all(isinstance(v, (int, float)) and math.isfinite(v) for v in r['metrics'].values()), 'Nonfinite metric')
    test, val = means(rows, 'test'), means(rows, 'validation')
    c = contributions(test, val)
    for key, value in c.items():
        if key in ('NR', 'FR', 'S_subset'):
            need(abs(value-a[key]) <= 1e-12, f'Actual row means differ from audit {key}')
    for split, m in [('test', test), ('validation', val)]:
        need(all(abs(v-a['component_means'][split][k]) <= 1e-12 for k,v in m.items()), 'Component mean differs')
    if marker_path:
        m = json.loads(marker_path.read_text())
        need(m['raw_checkpoint_sha256'] == a['actual_checkpoint_sha256'], 'Immutable marker checkpoint differs')
    return {'audit': a, 'rows': rows, 'test': test, 'validation': val, 'contributions': c,
            'proofs': {'audit': {'path': str(audit_path), 'sha256': digest(audit_path)},
                       'rows': {'path': str(rows_path), 'sha256': digest(rows_path)},
                       'checkpoint_sha256': a['actual_checkpoint_sha256']}}


def compare(candidate, parent):
    ca, pa = candidate['audit'], parent['audit']
    for field in ['current_source_sha256', 'actual_metric_weights_sha256', 'actual_base_assets_sha256',
                  'test_cases', 'metric_order', 'render_post_rng_hashes', 'score_post_rng_hashes']:
        need(ca[field] == pa[field], f'Incompatible pilot {field}')
    for cr, pr in zip(ca['records'], pa['records']):
        need(all(cr[k] == pr[k] for k in ['split', 'case', 'actual_input_sha256', 'actual_gt_sha256']),
             'Original input/GT mismatch')
    cc, pc = candidate['contributions'], parent['contributions']
    ds = {k: cc[k]-pc[k] for k in cc}
    paired_test = []
    paired_val = []
    for cr, pr in zip(candidate['rows'], parent['rows']):
        cm, pm = cr['metrics'], pr['metrics']
        if cr['split'] == 'test':
            # The original formula clamps the mean NIQE term. All compared means
            # must stay below10 for this linear per-image decomposition.
            need(candidate['test']['niqe'] < 10 and parent['test']['niqe'] < 10, 'NIQE clamp regime changed')
            d = {'CLIPIQA': cm['clipiqa']-pm['clipiqa'], 'MANIQA': cm['maniqa']-pm['maniqa'],
                 'MUSIQ/100': (cm['musiq']-pm['musiq'])/100, 'NIQE_term': -(cm['niqe']-pm['niqe'])/10}
            paired_test.append({'case': cr['case'], 'delta_NR': sum(d.values()), 'delta_contributions': d})
        else:
            paired_val.append({'case': cr['case'], 'delta_FR': -(cm['lpips']-pm['lpips'])-(cm['dists']-pm['dists']),
                               'delta_lpips': cm['lpips']-pm['lpips'], 'delta_dists': cm['dists']-pm['dists']})
    need(abs(sum(r['delta_NR'] for r in paired_test)/20-ds['NR']) < 1e-12, 'Paired NR mean differs')
    need(abs(sum(r['delta_FR'] for r in paired_val)/3-ds['FR']) < 1e-12, 'Paired FR mean differs')
    return {'delta_normalized_contributions': ds,
            'test_better_count_NR': sum(r['delta_NR'] > 0 for r in paired_test),
            'test_better_counts_by_contribution': {k: sum(r['delta_contributions'][k] > 0 for r in paired_test) for k in KEYS_MAP},
            'test20_paired': paired_test, 'validation3_paired': paired_val,
            'scope': 'Complete fixed20-test pilot plus all3validation only; no full100 score inference; counts descriptive, no statistical significance claim.'}


KEYS_MAP = ['CLIPIQA', 'MANIQA', 'MUSIQ/100', 'NIQE_term']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args();base=args.base.resolve()
    parent=load(base/'nr_push4_20261014/preparation/parent7250_pilot_cpu_audit.json',
                base/'nr_push3_20261014/evaluation_audits/push3_c7250_compiled_pilot/proof/scores/measurements.jsonl')
    candidates={};pending=[]
    for branch in 'abc':
        for step in [7500,7750]:
            label=f'push4_{branch}{step}_compiled_pilot';d=base/'nr_push4_20261014/evaluation_audits'/label
            if not (d/'audit.json').is_file(): pending.append(label);continue
            a=json.loads((d/'audit.json').read_text())
            if a.get('status') != 'PASS':pending.append(label);continue
            candidates[label]=load(d/'audit.json',d/'proof/scores/measurements.jsonl',d/'proof/checkpoint_marker.json')
    comparisons={k:compare(v,parent) for k,v in candidates.items()}
    minimal={}
    for step in [7500,7750]:
        for c,p in [('b','a'),('c','b')]:
            cl=f'push4_{c}{step}_compiled_pilot';pl=f'push4_{p}{step}_compiled_pilot'
            if cl in candidates and pl in candidates:minimal[cl+'__minus__'+pl]=compare(candidates[cl],candidates[pl])
    for b in 'abc':
        cl=f'push4_{b}7750_compiled_pilot';pl=f'push4_{b}7500_compiled_pilot'
        if cl in candidates and pl in candidates:minimal[cl+'__minus__'+pl]=compare(candidates[cl],candidates[pl])
    rank=sorted(candidates,key=lambda k:candidates[k]['contributions']['S_subset'],reverse=True)
    report={'schema':'push4_complete_pilot_platform_readonly_CPU_analysis_v1','CPU_only':True,'GPU_used':False,
            'scope':'fixed20test+all3val pilot23/98; not full100 score; completed PASS only',
            'complete_cohort':len(candidates)==6,'complete_count':len(candidates),'pending':pending,
            'parent':{'label':parent['audit']['label'],'contributions':parent['contributions'],'proofs':parent['proofs']},
            'pilot_rank_by_S_subset_only':rank,
            'candidates':{k:{'contributions':v['contributions'],'component_means':{'test':v['test'],'validation':v['validation']},
                             'proofs':v['proofs'],'vs_parent7250':comparisons[k]} for k,v in candidates.items()},
            'minimal_contrasts':minimal}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'complete_count':len(candidates),'complete_cohort':len(candidates)==6,'rank':rank,
                      'deltas':{k:comparisons[k]['delta_normalized_contributions'] for k in rank}},indent=2))


if __name__=='__main__':main()
