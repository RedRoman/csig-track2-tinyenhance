"""CPU-only actual23/98 matched HYPIR FP32 parameter interpolation pilot audit; no full score or latency claim."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
CORE=R/'code/push7/audit_and_join_cpu.py'
CORE_SHA='a47b8ee9c9b05ffbddd69a9b88293332bfa83f429e89db4f2133d1dd89336d1d'
PARENT=R/'nr_push7_pilot_evaluations/push7_a7750_parent_pilot'
PARENT_MODEL=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
RUNNER='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
A='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
def need(x,m):
    if not x: raise RuntimeError(m)
def h(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
    return d.hexdigest()
def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--evaluation-dir',type=Path,action='append',required=True)
    q.add_argument('--out',type=Path,required=True);q.add_argument('--execute',action='store_true');a=q.parse_args()
    out=a.out.resolve();need(out.is_relative_to(R/'audits/push7') and not out.exists(),'Use a new Push7 immutable proof')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',evaluations=[str(x) for x in a.evaluation_dir],parent=str(PARENT),scope='actual23/98 fixed pilot only',GPU=False)));return
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CUDA initialized')
    need(h(CORE)==CORE_SHA,'Frozen common core changed')
    spec=importlib.util.spec_from_file_location('frozen_push7_audit_core',CORE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.torch=torch;m.np=np
    def read(p):return m.read(p)
    def ref(p):return dict(path=str(Path(p).resolve()),sha256=m.sha(p))
    def run(d):
        d=d.resolve()
        for name in ['render_protocol.json','render_manifest.json','render_status.json','render_progress.pt','scores/status.json','scores/result.json','scores/protocol.json','scores/measurements.jsonl','scores/rng_checkpoint.pt']:m.sha(d/name)
        need((d/'render_complete').is_file(),'Actual render incomplete')
        core=m.audit_core(d.name,d.parent)
        need(core['image_rows']==23 and core['scalar_metrics']==98 and core['official_full_count'] is False,'Require complete actual23/98 pilot')
        rp=read(d/'render_protocol.json')['protocol'];model=Path(rp['model_dir']).resolve()
        cfg=read(model/'config.json');rr=rp['runner'];mf=read(d/'render_manifest.json')
        need(m.sha(model/'runner.py')==rr['runner_sha256']==RUNNER,'Original native HYPIR runner required')
        need(rr['model_t']==200 and rr['lora_rank']==256 and rr['patch']==512 and rr['stride']==256 and rr['scale']==1 and rr['internal_precision']=='BF16','Original HYPIR native geometry changed')
        need(rr['coeff_t']==cfg['coeff_t'] and rr['checkpoint_sha256']==m.sha(cfg['weight_path']),'Config/checkpoint binding changed')
        need(rp['inference_rng']=='seed231 before model initialization; continuous ordered val3 then tests' and rp['warmup'] is False,'Render seed/order/warmup changed')
        need({str(p.relative_to(d)) for p in (d/'render').rglob('*') if p.is_file()}=={r['output'] for r in mf['records']},'Extra/missing actual render PNG')
        need(all(not (d/r['output']).is_symlink() for r in mf['records']),'Canonical PNG must be a real file')
        rows=[json.loads(x) for x in (d/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
        for row in rows:
            need(row['seed_scope']=='split_seed1234_continuous_rng' and abs(m.nr(row['metrics'])-row['nr_per_image'])<1e-12,'Per-image NR/score RNG metadata mismatch')
        core.update(model_dir=str(model),config=ref(model/'config.json'),render_manifest=ref(d/'render_manifest.json'))
        return core,rp,read(d/'scores/protocol.json')
    parent,pr,ps=run(PARENT)
    need(parent['config']['sha256']=='8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8','Fixed coeff175 parent config required')
    need(parent['actual_checkpoint_sha256']==A and Path(pr['model_dir']).resolve()==PARENT_MODEL,'Wrong fixed parent model')
    table=[];audits=[]
    def contributions(core):
        t=core['component_means']['test'];v=core['component_means']['validation']
        return dict(CLIPIQA=t['clipiqa'],MANIQA=t['maniqa'],MUSIQ_div100=t['musiq']/100,NIQE_clamped=max(0,(10-t['niqe'])/10),negative_LPIPS=-v['lpips'],negative_DISTS=-v['dists'])
    pc=contributions(parent)
    for evaluation in a.evaluation_dir:
        core,rp,sp=run(evaluation)
        need([(x['split'],x['case'],x['actual_input_sha256'],x['actual_gt_sha256']) for x in core['records']]==
             [(x['split'],x['case'],x['actual_input_sha256'],x['actual_gt_sha256']) for x in parent['records']],'Pilot inputs/GT/order differ')
        for k in ['schema','cases','input_encoding','output_encoding','warmup','inference_rng','source_sha256','torch']:
            need(rp[k]==pr[k],'Original renderer metadata changed: '+k)
        for k in ['metric_source','original_tool_deployment_sha256','original_sources','script_sha256','environment_protocol_sha256','environment_versions','numpy_legacy_compat','seed','seed_scope','metric_order','image_policy','resume_policy','interleaving_policy','sequence']:
            need(sp[k]==ps[k],'Original metric protocol changed: '+k)
        proofs=[json.loads(json.dumps(p['maniqa_offline_init_proof'])) for p in [sp,ps]]
        for p in proofs:
            for row in p['before_init_records']:row.pop('before_original_checkpoint_load',None)
        need(proofs[0]==proofs[1] and core['score_post_rng_hashes']==parent['score_post_rng_hashes'],'Formal MANIQA100/continuous score4RNG differ')
        model=Path(core['model_dir']);manifest=read(model.parent/'candidate_manifest.json')
        need(manifest['runner_sha256']==RUNNER and manifest['config_sha256']==core['config']['sha256'] and manifest['checkpoint_sha256']==core['actual_checkpoint_sha256'],'Actual registered manifest mismatch')
        need(Path(manifest['model_dir']).resolve()==model,'Wrong manifest model dir')
        need(manifest['schema']=='HYPIR_FP32_parameter_interpolation_v1' and manifest['alpha'] in [.25,.5] and
             manifest['new_training'] is False and manifest['distillation'] is False and manifest['lightweight_model'] is False and
             manifest['no_val_GT'] is True and manifest['no_teacher_outputs'] is True and manifest['quality_reuse_claim'] is False,'Not explicit HYPIR parameter interpolation')
        need(manifest['initial_checkpoint_sha256']==A and manifest['finetuned_checkpoint_sha256']=='763950727fe29a009245cd9cd855743a99803fa9fbf6824a62d38994fa8e0650' and
             manifest['recipe']=='torch.lerp(initial_A_FP32,NIQE100_FP32,alpha) on CPU per514 ordered tensors' and
             manifest['optimizer_state']=='not applicable: derived parameters, no new Adam/step/resume','Derived optimizer/recipe lineage mislabeled')
        need(read(model/'config.json')['coeff_t']==175,'Original coeff175 protocol required')
        rawproof_path=R/'audits/push7/parameter_blends/independent514_cpu.json';rawproof=read(rawproof_path)
        need(rawproof['status']=='PASS_EXACT_FP32_LERP514_BOTH' and rawproof['auditor']['sha256']==m.sha(rawproof['auditor']['path'])=='1286bac0cf9e70a90a63c10b5a727a24cacec71976c416f6148c1fa3c683e20b', 'Actual independent514 formula proof missing')
        matched=[x for x in rawproof['candidates'] if x['checkpoint']['sha256']==core['actual_checkpoint_sha256']]
        need(len(matched)==1,'Raw formula proof is for another checkpoint');derived=matched[0]
        need(derived['alpha']==manifest['alpha'] and derived['exact_FP32_lerp_tensors']==derived['changed_tensors_from_A']==514 and
             derived['native_candidate_manifest']['sha256']==m.sha(model.parent/'candidate_manifest.json') and derived['derived_optimizer_or_resume'] is False, 'Actual native manifest/raw formula differs')
        need(m.sha(rawproof['producer']['path'])==rawproof['producer']['sha256']==manifest['derivation_source_sha256']=='2bb83af6f7cd988b7cbf210aa241f7be48f93337c126ca0813ec4a02e802a2d3','Interpolation producer source changed')
        provenance=dict(alpha=manifest['alpha'],actual_raw_formula_proof=ref(rawproof_path),actual_derived_record=derived,new_training=False,no_derived_optimizer_or_RNG=True)
        cc=contributions(core);delta={k:cc[k]-pc[k] for k in pc}
        table.append(dict(label=core['label'],NR=core['NR'],FR=core['FR'],S_subset=core['S_subset'],
            delta_NR=core['NR']-parent['NR'],delta_FR=core['FR']-parent['FR'],delta_S_subset=core['S_subset']-parent['S_subset'],
            component_mean_delta={split:{k:v-parent['component_means'][split][k] for k,v in means.items()} for split,means in core['component_means'].items()},
            additive_score_contribution_delta=delta,checkpoint_sha256=core['actual_checkpoint_sha256']))
        core.update(candidate_manifest=ref(model.parent/'candidate_manifest.json'),registered_parameter_derivation=provenance)
        audits.append(core)
    m.stable_files()
    proof=dict(schema='push7_actual_matched_hypir_parameter_blends_pilot_v1',status='PASS_MATCHED23_98',cpu_only=True,CUDA_initialized=False,
        scope='fixed20 test plus3validation; S_subset only; no full quality or tool prediction',quality_reused=False,
        parent=parent,candidates=audits,table=table,common_core=ref(CORE),auditor=ref(__file__),
        score_4RNG_equal_to_parent=True,render_4RNG_equality_required=False,no_latency_claim=True,no_65_claim=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),sha256=h(out),table=table)))
if __name__=='__main__':main()

