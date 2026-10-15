"""Summarize completed, independently audited full103 evaluations; no model execution."""
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE/'nr_push2_full_evaluations'
LABELS = [
    'nr_stage2_cont_s6500_compiled_all',
    'nr_stage3_anchor6250_m1_s6500_compiled_all',
    'nr_stage3_anchor6250_m15_s6500_compiled_all',
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parent_path=BASE/'audits/nr_cuda_full_s6250_audit.json'
    parent=json.loads(parent_path.read_text())
    training_runs={LABELS[0]:('stage2','cuda_nr_stage2_s6250_cont250'), LABELS[1]:('stage3','cuda_nr_stage3_s6250_anchor6250_m1_250'), LABELS[2]:('stage3','cuda_nr_stage3_s6250_anchor6250_m15_250')}
    assert parent['status']=='PASS'
    entries=[]
    downloaded=[]
    for label in LABELS:
        ap=OUT/'audits'/f'{label}_audit.json'
        if not ap.exists():
            continue
        a=json.loads(ap.read_text())
        assert a['status']=='PASS' and a['image_rows']==103 and a['scalar_metrics']==418
        assert not a['cuda_initialized'] and a['cpu_only']
        rp=OUT/'audits'/f'{label}_download_remote_sha.json'
        hashes=json.loads(rp.read_text())
        assert len(hashes)==7
        for name,rec in hashes.items():
            path=OUT/'results'/label/name
            actual=sha(path)
            assert actual==rec['sha256'] and path.stat().st_size==rec['bytes']
            downloaded.append({'path':str(path.relative_to(OUT)), 'bytes':rec['bytes'],
                               'sha256':actual, 'remote_sha256':rec['sha256']})
        stage,run=training_runs[label]
        config_path=BASE/'cuda_training'/stage/'runs'/run/'config.json'
        config=json.loads(config_path.read_text())
        training=config['protocol']
        tm=a['component_means']['test']; pm=parent['component_means']['test']
        contributions={'niqe_clamp':max(0,(10-tm['niqe'])/10)-max(0,(10-pm['niqe'])/10), 'musiq_over100':(tm['musiq']-pm['musiq'])/100, 'maniqa':tm['maniqa']-pm['maniqa'], 'clipiqa':tm['clipiqa']-pm['clipiqa']}
        assert abs(sum(contributions.values())-(a['NR']-parent['NR']))<1e-12
        identity_fields=['split','case','actual_input_sha256','actual_gt_sha256']
        assert [[r[k] for k in identity_fields] for r in a['records']]==[[r[k] for k in identity_fields] for r in parent['records']]
        entries.append({
            'label':label,'status':'PASS','NR':a['NR'],'FR':a['FR'],'S_proxy':a['S_proxy'],
            'component_means':a['component_means'],'image_rows':103,'scalar_metrics':418,
            'training':{'run':run,'local_steps':config['target_stage_steps'],'global_step':config['target_global_step'],'Adam_step':config['target_optimizer_step'], 'anchor':training['anchor'], 'musiq_reward_weight':training.get('musiq_weight',1.0),'nr_round_ste':training['nr_round_ste'],'config_sha256':sha(config_path)},
            'delta_NR_contributions':contributions,
            'same_order_input_GT_as_parent6250':True,
            'same_source_hashes_as_parent6250':a['current_source_sha256']==parent['current_source_sha256'],
            'same_render_post_rng_as_parent6250':a['render_post_rng_hashes']==parent['render_post_rng_hashes'],
            'same_score_post_rng_as_parent6250':a['score_post_rng_hashes']==parent['score_post_rng_hashes'],
            'actual_checkpoint_sha256':a['actual_checkpoint_sha256'],
            'render_protocol_id':a['render_protocol_id'],
            'delta_vs_parent6250_full':{k:a[k]-parent[k] for k in ['NR','FR','S_proxy']},
            'test_component_delta_vs_parent6250_full':{
                k:v-parent['component_means']['test'][k] for k,v in a['component_means']['test'].items()},
            'max_mean_formula_delta':a['mean_formula_max_delta'],
            'sha_checks':{'actual_png_manifest_scored':103,'actual_source_inputs':103,'actual_GT':3,
                          'metric_weights':8,'base_assets':13,'downloaded_small_files':7,
                          'source_and_protocol_fingerprint':'PASS'},
            'audit_path':str(ap.resolve()),'audit_sha256':sha(ap),
            'evaluation_dir':a['evaluation_dir'],
        })
    pending=[label for label in LABELS if label not in {e['label'] for e in entries}]
    summary={
        'status':'PASS' if not pending else 'AUDITED_COMPLETED_SUBSET',
        'updated_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'scope':'Full test100_NR + validation3_FR local proxy; no latency/official total; pilots excluded',
        'reference':{'label':parent['label'], **{k:parent[k] for k in ['NR','FR','S_proxy','component_means']},
                     'audit_path':str(parent_path),'audit_sha256':sha(parent_path)},
        'evaluations':entries,'pending':pending,
        'auditor_source_sha256':sha(BASE/'audits/audit_push2_full_cpu.py'),
        'notes':[
            'All compared values are full103 values. Pilot subsets are not compared with full test NR.',
            'Eight actual metric weights and 13 base assets hashed. Actual PNG/input/GT/source/checkpoint hashes verified.',
            'Each split was seeded once1234, and committed RNG checkpoints match their protocol/context and rows.',
            'CPU audit did not rerun models or score metrics, and did not initialize CUDA.',
            'One fixed test set is used for training/model selection; small gains are empirical local proxy differences, not statistical/official leaderboard or unseen-data guarantees.',
            '418 scalars are not independent samples; FR uses only3 validation images; one seed gives no cross-seed confidence interval.',
            'Training MUSIQ reward weights do not change the formal scoring coefficient1/100.',
        ],
    }
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'download_sha_manifest.json').write_text(json.dumps({'status':'PASS','count':len(downloaded),'files':downloaded},ensure_ascii=False,indent=2)+'\n')
    rows=[('父6250',parent,None)]+[(e['label'],e,e['delta_vs_parent6250_full']) for e in entries]
    lines=['# NR后续full103独立CPU审计','',
           '各项均先逐图求均值，再按原公式计算NR/FR/S_proxy；完整测试100图、验证3图。', '',
           '|版本|NR ↑|FR ↑|S_proxy ↑|相对父6250 ΔS|','|---|---:|---:|---:|---:|']
    for name,item,delta in rows:
        lines.append(f"|{name}|{item['NR']:.12f}|{item['FR']:.12f}|{item['S_proxy']:.12f}|{delta['S_proxy']:+.9f}|" if delta else
                     f"|{name}|{item['NR']:.12f}|{item['FR']:.12f}|{item['S_proxy']:.12f}|—|")
    lines+=['','|版本|NIQE ↓|MUSIQ ↑|MANIQA ↑|CLIPIQA ↑|','|---|---:|---:|---:|---:|']
    for name,item,_ in rows:
        m=item['component_means']['test']
        lines.append(f"|{name}|{m['niqe']:.9f}|{m['musiq']:.9f}|{m['maniqa']:.9f}|{m['clipiqa']:.9f}|")
    lines+=['','|版本|val LPIPS ↓|val DISTS ↓|ΔNR: NIQE|ΔNR: MUSIQ|ΔNR: MANIQA|ΔNR: CLIPIQA|','|---|---:|---:|---:|---:|---:|---:|']
    for e in entries:
        v=e['component_means']['validation'];c=e['delta_NR_contributions']
        lines.append(f"|{e['label']}|{v['lpips']:.9f}|{v['dists']:.9f}|{c['niqe_clamp']:+.9f}|{c['musiq_over100']:+.9f}|{c['maniqa']:+.9f}|{c['clipiqa']:+.9f}|")
    lines+=['','已通过每版本103行/418指标、数字顺序、PNG/input/GT真实SHA、源码/模型/8项指标权重/13项基础资产SHA和保存RNG状态关联审核。',
            f"已下载并核对{len(downloaded)}个小记录；GPU未初始化。完整证据见audits/和download_sha_manifest.json。", '',
            '本地质量代理分，不含速度项，不代表官方榜单分。pilot单独记录；同一测试集被用于训练和选模，微小差异不能据此声称统计显著或跨数据泛化。FR仅3图，418项不是独立样本。训练MUSIQ权重不改变正式评分的1/100系数。']
    if pending:
        lines+=['','尚待full完成审核：'+', '.join(pending)]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'audited':len(entries),'downloaded':len(downloaded),'pending':pending}))

if __name__=='__main__':
    main()
