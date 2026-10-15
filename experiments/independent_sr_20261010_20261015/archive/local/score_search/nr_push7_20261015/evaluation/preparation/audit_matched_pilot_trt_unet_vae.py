"""CPU-only actual23/98 matched HYPIR TensorRT-UNet-VAE pilot audit; no full score or latency claim."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
CORE=R/'code/push7/audit_and_join_cpu.py'
CORE_SHA='a47b8ee9c9b05ffbddd69a9b88293332bfa83f429e89db4f2133d1dd89336d1d'
PARENT=R/'nr_push7_pilot_evaluations/push7_a7750_parent_pilot'
PARENT_MODEL=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
RUNNER='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
TRT_RUNNER='270d2434179d17658e9f1df228c8ef99124a2695141c92f0b5fecfa73d8d3080'
TRT_CONFIG='63e227319f2c2fb327257a810f6709b4f1d738732571f6c100e22b75174694e5'
VAE_HYBRID='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9'
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
        expected=RUNNER if model==PARENT_MODEL else TRT_RUNNER
        need(m.sha(model/'runner.py')==rr['runner_sha256']==expected,'Explicit parent or TensorRT hybrid source required')
        need(rr['model_t']==200 and rr['lora_rank']==256 and rr['patch']==512 and rr['stride']==256 and rr['scale']==1,'Original HYPIR native geometry changed')
        need(rr['internal_precision']==('BF16' if model==PARENT_MODEL else 'FP16 TensorRT UNet+VAE, BF16 eps/latent/DiagonalGaussianDistribution/scheduler/tiling, FP32 wavelet/output'),'Actual declared precision changed')
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
    need(parent['config']['sha256']=='8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8','Original coeff175 parent config required')
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
        need(manifest['runner_sha256']==TRT_RUNNER and manifest['config_sha256']==core['config']['sha256'] and manifest['checkpoint_sha256']==core['actual_checkpoint_sha256'],'Actual registered manifest mismatch')
        need(Path(manifest['model_dir']).resolve()==model,'Wrong manifest model dir')
        need(manifest['schema']=='hypir_trt_unet_vae_hybrid_candidate_v1' and manifest['model_type']=='HYPIR_SD2_rank256_TRT_UNet_VAE_native512' and
             manifest['no_training'] is True and manifest['no_distillation'] is True and manifest['quality_reuse_claim'] is False,'Only sealed HYPIR TensorRT hybrid candidate allowed')
        rr=rp['runner'];cfg=read(model/'config.json')
        need(rr['schema']=='hypir_trt_unet_vae_hybrid_runner_v1' and rr['model_dir_config_sha256']==manifest['config_sha256']==TRT_CONFIG and
             rr['TRT']['runtime_sha256']==TRT_RUNTIME and rr['coeff_t']==175,'Runtime/config/backend binding changed')
        for source,digest in rr['implementation_sources'].items():need(m.sha(source)==digest,'Backend implementation source changed')
        need(m.sha(model/'reference_runner.py')==RUNNER and m.sha(model/'trt_vae_runtime_v2.py')==TRT_RUNTIME,'Parent/runtime bytes changed')
        need(m.sha(model/'vae_hybrid_runner.py')==VAE_HYBRID,'Original VAE hybrid implementation changed')
        engines={}
        for name,record in manifest['actual_engines'].items():
            if name=='UNet':
                c=cfg['TRT_UNet_engine'];u=rr['UNet_TRT']
                actual=dict(path=u['engine_path'],sha256=u['engine_sha256'],manifest=u['manifest'],manifest_sha256=u['manifest_sha256'])
            else:
                c=cfg['TRT_engines'][name];actual=rr['TRT']['engines'][name]
            for k in ['path','sha256','manifest','manifest_sha256']:need(c[k]==record[k]==actual[k],'Engine provenance mismatch: '+name+'/'+k)
            need(m.sha(record['path'])==record['sha256'] and Path(record['path']).stat().st_size>0 and
                 m.sha(record['manifest'])==record['manifest_sha256'],'Actual engine or sealed engine manifest changed')
            engines[name]=dict(record,bytes=Path(record['path']).stat().st_size)
        need(set(engines)=={'encoder','decoder','UNet'} and m.sha(cfg['base_vae_path'])==manifest['original_VAE_sha256']==rr['TRT']['original_VAE_sha256'],'Original VAE identity changed')
        u=rr['UNet_TRT'];um=read(engines['UNet']['manifest'])
        builder=R/'code/push7/trt_unet/build_hypir_unet_trt.py';builder_sha='4281d9be015142919b51b84d946ad8d23b8bd49451ad54adb9ada8f1d42cdcca'
        need(engines['UNet']['sha256']=='08fca16f286c604614e9322ded93f848b0c1d3dd40de7ca793f10b98f286aec8' and
             engines['UNet']['manifest_sha256']=='627f1e314b1a6d6a467fd1eca2188b9ed840d54ba9bc9b80ec9ae27a84c68db6' and
             um['status']=='BUILT_COMPONENT_PROBED' and um['engine_sha256']==u['engine_sha256'] and um['engine_bytes']==engines['UNet']['bytes'], 'Actual UNet engine/manifest changed')
        need(u['producer_sha256']==m.sha(builder)==um['source_sha256'][str(builder)]==builder_sha and um['raw_sha256']==u['initial_A_sha256']==A,'Actual UNet builder/A identity changed')
        for source,digest in um['source_sha256'].items():need(m.sha(source)==digest,'Actual UNet component producer dependency changed')
        need(manifest['BF16_original_fused_G_state']==um['BF16_original_fused_G_state']==u['BF16_fused_G_digest_from_builder'] and
             manifest['fixed_conditioning']==um['fixed_conditioning'] and
             u['fixed_original_BF16_conditioning_sha256']==manifest['fixed_conditioning']['BF16_sha256']['sha256']=='f65bb78e0c2376a4df09f17d8671ba7dfe163375d3349a1db49f12c5bbd9d0a9', 'Original fused BF16 G/fixed conditioning provenance changed')
        need(u['construction_all_four_RNG_restored'] is True and u['forward_RNG_reset'] is False and u['scheduler_DDPM_original_BF16_outside_engine'] is True,'UNet engine RNG/scheduler policy changed')
        need(m.sha(Path(rr['base_model_path'])/'unet/diffusion_pytorch_model.safetensors')==manifest['actual_base_UNet_sha256']==um['actual_base_UNet_sha256'], 'Original UNet base file changed')
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
        table.append(dict(label=core['label'],NR=core['NR'],FR=core['FR'],S_subset=core['S_subset'],
            delta_NR=core['NR']-parent['NR'],delta_FR=core['FR']-parent['FR'],delta_S_subset=core['S_subset']-parent['S_subset'],
            component_mean_delta={split:{k:v-parent['component_means'][split][k] for k,v in means.items()} for split,means in core['component_means'].items()},
            additive_score_contribution_delta=delta,checkpoint_sha256=core['actual_checkpoint_sha256']))
        core.update(candidate_manifest=ref(model.parent/'candidate_manifest.json'),registered_backend_provenance=provenance)
        audits.append(core)
    m.stable_files()
    proof=dict(schema='push7_actual_matched_native_hypir_trt_unet_vae_pilot_v1',status='PASS_MATCHED23_98',cpu_only=True,CUDA_initialized=False,
        scope='fixed20 test plus3validation; S_subset only; no full quality or tool prediction',quality_reused=False,
        parent=parent,candidates=audits,table=table,common_core=ref(CORE),auditor=ref(__file__),
        score_4RNG_equal_to_parent=True,render_4RNG_equality_required=False,no_latency_claim=True,no_65_claim=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),sha256=h(out),table=table)))
if __name__=='__main__':main()

