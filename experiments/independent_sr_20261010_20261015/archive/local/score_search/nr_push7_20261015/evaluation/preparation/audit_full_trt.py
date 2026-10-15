"""CPU-only actual103/418 HYPIR TensorRT-VAE full quality audit; no latency or official total claim."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
CORE=R/'code/push7/audit_and_join_cpu.py'
CORE_SHA='a47b8ee9c9b05ffbddd69a9b88293332bfa83f429e89db4f2133d1dd89336d1d'
PARENT=R/'nr_push4_full_evaluations/push4_a7750_compiled_all'
PARENT_RENDER=R/'nr_push6_full_evaluations/push6_a7750_ro_wavelet_all'
PARENT_AUDIT=R/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json'
PARENT_AUDIT_SHA='ae5b2234f5e110240507fbec515139d1b2a23bf8f664e7477e34c377e09c507b'
BRIDGE=R/'audits/push6/formal_png103/ro_wavelet_exact103_v2.json'
BRIDGE_SHA='bc5dff12c258069bb77e5f3e0b6982737a9d74ea705782bb230ef244b1850e16'
PARENT_CONFIG_SHA='8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8'
PARENT_MODEL=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
RUNNER='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
TRT_RUNNER='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9'
TRT_RUNTIME='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4'
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
        print(json.dumps(dict(status='CPU_PLAN_ONLY',evaluations=[str(x) for x in a.evaluation_dir],parent=str(PARENT),scope='fresh actual103/418 full quality; parent full418 plus verified Push6 exact103 renderer',GPU=False)));return
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
        need(core['image_rows']==103 and core['scalar_metrics']==418 and core['official_full_count'] is True,'Require complete actual103/418 full quality')
        rp=read(d/'render_protocol.json')['protocol'];model=Path(rp['model_dir']).resolve()
        cfg=read(model/'config.json');rr=rp['runner'];mf=read(d/'render_manifest.json')
        expected=RUNNER if model==PARENT_MODEL else TRT_RUNNER
        need(m.sha(model/'runner.py')==rr['runner_sha256']==expected,'Explicit parent or TensorRT hybrid source required')
        need(rr['model_t']==200 and rr['lora_rank']==256 and rr['patch']==512 and rr['stride']==256 and rr['scale']==1,'Original HYPIR native geometry changed')
        need(rr['internal_precision']==('BF16' if model==PARENT_MODEL else 'UNet original BF16; TensorRT VAE FP16 IO/weights, original posterior/sample and tile merges BF16; native wavelet/output FP32'),'Actual declared precision changed')
        need(rr['coeff_t']==cfg['coeff_t'] and rr['checkpoint_sha256']==m.sha(cfg['weight_path']),'Config/checkpoint binding changed')
        need(rp['inference_rng']=='seed231 before model initialization; continuous ordered val3 then tests' and rp['warmup'] is False,'Render seed/order/warmup changed')
        need({str(p.relative_to(d)) for p in (d/'render').rglob('*') if p.is_file()}=={r['output'] for r in mf['records']},'Extra/missing actual render PNG')
        need(all(not (d/r['output']).is_symlink() for r in mf['records']),'Canonical PNG must be a real file')
        rows=[json.loads(x) for x in (d/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
        for row in rows:
            need(row['seed_scope']=='split_seed1234_continuous_rng' and abs(m.nr(row['metrics'])-row['nr_per_image'])<1e-12,'Per-image NR/score RNG metadata mismatch')
        core.update(model_dir=str(model),config=ref(model/'config.json'),render_manifest=ref(d/'render_manifest.json'))
        return core,rp,read(d/'scores/protocol.json')
    need(m.sha(PARENT_AUDIT)==PARENT_AUDIT_SHA and m.sha(BRIDGE)==BRIDGE_SHA,'Frozen full A quality/Push6 exact bridge changed')
    frozen=read(PARENT_AUDIT);bridge=read(BRIDGE)
    parent=m.audit_core(PARENT.name,PARENT.parent)
    need(parent['image_rows']==103 and parent['scalar_metrics']==418 and parent['official_full_count'] is True and
         parent['actual_checkpoint_sha256']==A and parent['published_result']==frozen['published_result'] and
         parent['records']==frozen['records'],'Actual full parent source/418/PNG differ from frozen A audit')
    for key in ['NR','FR','S_proxy']:need(abs(parent[key]-frozen[key])<1e-12,'Parent full mean drifted: '+key)
    need(bridge['status']=='PASS_EXACT103' and bridge['PNG_exact_count']==103 and bridge['final_fourRNG_equal'] is True and
         Path(bridge['parent_dir']).resolve()==PARENT and Path(bridge['candidate_dir']).resolve()==PARENT_RENDER,'Wrong frozen parent render bridge')
    for key,digest in [('parent_render_manifest_sha256',m.sha(PARENT/'render_manifest.json')),('candidate_render_manifest_sha256',m.sha(PARENT_RENDER/'render_manifest.json'))]:
        need(bridge[key]==digest,'Actual parent render manifest changed: '+key)
    render_record=read(PARENT_RENDER/'render_protocol.json');pr=render_record['protocol'];pm=read(PARENT_RENDER/'render_manifest.json')
    need(hashlib.sha256(json.dumps(pr,sort_keys=True).encode()).hexdigest()==render_record['protocol_id']==pm['protocol_id'] and pm['protocol']==pr,'Push6 render protocol changed')
    need(Path(pr['model_dir']).resolve()==PARENT_MODEL and pr['runner']['checkpoint_sha256']==A and pr['runner']['runner_sha256']==RUNNER and
         m.sha(PARENT_MODEL/'runner.py')==RUNNER and m.sha(PARENT_MODEL/'config.json')==PARENT_CONFIG_SHA and pr['runner']['coeff_t']==175 and pr['cases']=='all',
         'Actual Push6 parent is not frozen A/native512/coeff175')
    need((PARENT_RENDER/'render_complete').is_file() and read(PARENT_RENDER/'render_status.json')['completed']==103 and len(pm['records'])==len(pr['tasks'])==103,'Push6 actual formal103 incomplete')
    need({str(p.relative_to(PARENT_RENDER)) for p in (PARENT_RENDER/'render').rglob('*') if p.is_file()}=={r['output'] for r in pm['records']},'Extra/missing Push6 parent actual PNG')
    for i,(actual,rec,task,old) in enumerate(zip(parent['records'],pm['records'],pr['tasks'],bridge['rows'])):
        split=actual['split'].replace('validation','val');case=actual['case'];relative=f'render/{split}/case{case}.png'
        need((rec['task_index'],rec['split'],rec['case'],rec['output'])==(i,split,case,relative) and (task['split'],task['case'])==(split,case),'Push6 formal order changed')
        png=PARENT_RENDER/relative
        need(not png.is_symlink() and m.sha(png)==rec['sha256']==actual['actual_png_sha256']==old['parent_sha256']==old['candidate_sha256'] and old['exact'] is True,'Push6 actual canonical PNG no longer byte-exact A')
        need(m.sha(task['input'])==task['input_sha256']==actual['actual_input_sha256'],'Push6 actual input differs from A full')
    saved=torch.load(PARENT_RENDER/'render_progress.pt',map_location='cpu',weights_only=False)
    need(saved['records']==pm['records'] and saved['protocol_id']==pm['protocol_id'] and saved['checkpoint_sha256']==A,'Push6 terminal state differs')
    need({k:m.rng_sha(saved[k]) for k in ['python_rng','numpy_rng','torch_rng','device_rng']}==bridge['candidate_final_fourRNG']==parent['render_post_rng_hashes'],'Push6 terminal fourRNG changed')
    ps=read(PARENT/'scores/protocol.json')
    parent.update(verified_Push6_renderer_dir=str(PARENT_RENDER),frozen_quality_audit=ref(PARENT_AUDIT),exact103_bridge=ref(BRIDGE),parent_render_rehashed_actual103=True)

    table=[];audits=[]
    def contributions(core):
        t=core['component_means']['test'];v=core['component_means']['validation']
        return dict(CLIPIQA=t['clipiqa'],MANIQA=t['maniqa'],MUSIQ_div100=t['musiq']/100,NIQE_clamped=max(0,(10-t['niqe'])/10),negative_LPIPS=-v['lpips'],negative_DISTS=-v['dists'])
    pc=contributions(parent)
    for evaluation in a.evaluation_dir:
        core,rp,sp=run(evaluation)
        need([(x['split'],x['case'],x['actual_input_sha256'],x['actual_gt_sha256']) for x in core['records']]==
             [(x['split'],x['case'],x['actual_input_sha256'],x['actual_gt_sha256']) for x in parent['records']],'Formal full inputs/GT/order differ')
        for k in ['schema','cases','input_encoding','output_encoding','warmup','inference_rng','source_sha256','torch']:
            need(rp[k]==pr[k],'Original renderer metadata changed: '+k)
        for k in ['metric_source','original_tool_deployment_sha256','original_sources','script_sha256','environment_protocol_sha256','environment_versions','numpy_legacy_compat','seed','seed_scope','metric_order','image_policy','resume_policy','interleaving_policy','sequence']:
            need(sp[k]==ps[k],'Original metric protocol changed: '+k)
        proofs=[json.loads(json.dumps(p['maniqa_offline_init_proof'])) for p in [sp,ps]]
        for p in proofs:
            for row in p['before_init_records']:row.pop('before_original_checkpoint_load',None)
        need(proofs[0]==proofs[1] and core['score_post_rng_hashes']==parent['score_post_rng_hashes'],'Formal MANIQA100/continuous score4RNG differ')
        model=Path(core['model_dir']);manifest=read(model.parent/'candidate_manifest.json')
        need(manifest['runner_sha256']==TRT_RUNNER and manifest['config_sha256']==core['config']['sha256'] and manifest['checkpoint_sha256']==core['actual_checkpoint_sha256'],'Actual registered manifest mismatch')
        need(Path(manifest['model_dir']).resolve()==model,'Wrong manifest model dir')
        need(manifest['schema']=='hypir_trt_vae_hybrid_candidate_v1' and manifest['model_type']=='HYPIR_SD2_rank256_TRT_VAE_native512' and
             manifest['no_training'] is True and manifest['no_distillation'] is True and manifest['quality_reuse_claim'] is False,'Only sealed HYPIR TensorRT hybrid candidate allowed')
        rr=rp['runner'];cfg=read(model/'config.json')
        need(rr['schema']=='hypir_trt_vae_hybrid_runner_v1' and rr['model_dir_config_sha256']==manifest['config_sha256'] and
             manifest['TRT_runtime_sha256']==rr['TRT']['runtime_sha256']==TRT_RUNTIME and rr['coeff_t']==175,'Runtime/config/backend binding changed')
        for source,digest in rr['implementation_sources'].items():need(m.sha(source)==digest,'Backend implementation source changed')
        need(m.sha(model/'reference_runner.py')==RUNNER and m.sha(model/'trt_vae_runtime_v2.py')==TRT_RUNTIME,'Parent/runtime bytes changed')
        engines={}
        for name,record in manifest['actual_engines'].items():
            c=cfg['TRT_engines'][name];actual=rr['TRT']['engines'][name]
            for k in ['path','sha256','manifest','manifest_sha256']:need(c[k]==record[k]==actual[k],'Engine provenance mismatch: '+name+'/'+k)
            need(m.sha(record['path'])==record['sha256'] and Path(record['path']).stat().st_size==record['bytes'] and
                 m.sha(record['manifest'])==record['manifest_sha256'],'Actual engine or sealed engine manifest changed')
            engines[name]=dict(record)
        need(set(engines)=={'encoder','decoder'} and m.sha(cfg['base_vae_path'])==manifest['original_VAE_sha256']==rr['TRT']['original_VAE_sha256'],'Original VAE identity changed')
        provenance={'actual_TRT_engines':engines,'runtime':ref(model/'trt_vae_runtime_v2.py'),'reference_runner':ref(model/'reference_runner.py')}
        if 'training_provenance' in manifest:
            t=manifest['training_provenance']
            for k in ['config','marker','input_catalog','resume']:
                need(m.sha(t[k]['path'])==t[k]['sha256'],'Training provenance SHA changed: '+k);provenance[k]=dict(t[k])
            marker=read(t['marker']['path'])
            need(marker['raw_checkpoint_sha256']==core['actual_checkpoint_sha256'] and marker['initial_A_checkpoint_sha256']==A and
                 marker['no_teacher_outputs'] is True and marker['no_val_GT'] is True and marker['global_step']==manifest['global_step'] and
                 marker['local_step']==marker['optimizer_step']==manifest['local_step'],'Actual training marker lineage/counters changed')
        else:
            need(core['actual_checkpoint_sha256']==A and manifest['no_training'] is True,'Unexpected untrained control provenance')
        cc=contributions(core);delta={k:cc[k]-pc[k] for k in pc}
        table.append(dict(label=core['label'],NR=core['NR'],FR=core['FR'],S_proxy=core['S_proxy'],
            delta_NR=core['NR']-parent['NR'],delta_FR=core['FR']-parent['FR'],delta_S_proxy=core['S_proxy']-parent['S_proxy'],
            component_mean_delta={split:{k:v-parent['component_means'][split][k] for k,v in means.items()} for split,means in core['component_means'].items()},
            additive_score_contribution_delta=delta,checkpoint_sha256=core['actual_checkpoint_sha256']))
        core.update(candidate_manifest=ref(model.parent/'candidate_manifest.json'),registered_backend_provenance=provenance)
        audits.append(core)
    m.stable_files()
    proof=dict(schema='push7_actual_native_hypir_trt_full_v1',status='PASS_FULL103_418',cpu_only=True,CUDA_initialized=False,
        scope='full100 test NR plus3validation FR, fresh103/418; local quality proxy, no latency/tool total',quality_reused=False,
        parent=parent,candidates=audits,table=table,common_core=ref(CORE),auditor=ref(__file__),
        score_4RNG_equal_to_parent=True,render_4RNG_equality_required=False,no_latency_claim=True,no_65_claim=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),sha256=h(out),table=table)))
if __name__=='__main__':main()

