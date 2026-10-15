"""CPU metadata binding for the four explicitly allowed registered HYPIR2/6-update runs."""
from pathlib import Path
import hashlib, json
R=Path('/cache/independent_sr_cuda_20261013')
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
DFD_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
FUSION_SHA='b215dc4247ecb06cd3040160b7671e8e3c956b090bef4909d3020e90b5719738'
CONV=['conv_out.lora_A.default.weight','conv_out.lora_B.default.weight']
RES=['up_blocks.3.resnets.2.conv1.lora_A.default.weight','up_blocks.3.resnets.2.conv1.lora_B.default.weight',
     'up_blocks.3.resnets.2.conv2.lora_A.default.weight','up_blocks.3.resnets.2.conv2.lora_B.default.weight']+CONV
ALLOWED={
 'hypir_push7_A7750_lastconv_nr100':('hypir_lastconv_nr_candidate_v1','hypir_push7_lastconv_nr_niqe768_optimizer2_full_gaussian_fused_bf16_v1',
   'b108f74548629d9877e6bb2998bee2474726e8c36b506c0539dcd35c15ba014c','24dd2608888bd240b310d3a7fde678e376483229fe35c0cd7e71ab61e87f1082',CONV,738304,2e-7),
 'hypir_push7_A7750_lastconv_lr1e6_nr100':('hypir_lastconv_nr_candidate_v1','hypir_push7_lastconv_nr_niqe768_optimizer2_full_gaussian_fused_bf16_v1',
   'c7bb01c0d9b0cc28fd972e1b4bfd60858116baef1337b0f9a936f680ee986a32','8e8d5e1fcac90504d029c6df436172d9380d46ddc5b888dda01fd6b6528a25f7',CONV,738304,1e-6),
 'hypir_push7_A7750_lastres_nr100':('hypir_lastres_nr_candidate_v1','hypir_push7_lastres_nr_niqe768_optimizer6_full_gaussian_fused_bf16_v3',
   'be97885d1073e94f920990c72518777cdaef4c516bc9aad454c7d24f41272d83','18178cdbc4b8af36505fce6b6286d990049e13100e497007e6ba2890e47950de',RES,3113984,1e-6),
 'hypir_push7_A7750_lastres_lr5e6_nr100':('hypir_lastres_nr_candidate_v1','hypir_push7_lastres_nr_niqe768_optimizer6_full_gaussian_fused_bf16_v3',
   '216c5faa11450a7e77f7c42f89e24d5a5d0a98e29338a9002d3721494cee8e14','5f99fcebc6cb50ed5c3ae18ae8df0cdf10cb2ab3b62e124c5097b0c81cf5cd9b',RES,3113984,5e-6)}

def need(v,message):
    if not v:raise RuntimeError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def bind_native(path,expected_sha,hash_raw=False):
    path=Path(path).resolve();need(sha(path)==expected_sha,'Native candidate manifest SHA differs')
    c=json.loads(path.read_text());run=Path(c['training_run']);need(run.parent==R/'adaptation/runs' and run.name in ALLOWED,'Unreviewed HYPIR training run')
    schema,pschema,registrar,trainer,keys,elements,lr=ALLOWED[run.name];count=len(keys);frozen=514-count
    md=Path(c['model_dir']).resolve();step=c['global_step'];local=c['local_step'];raw=Path(c['checkpoint']).resolve()
    need(path==md.parent/'candidate_manifest.json' and c['schema']==schema and c['model_type']=='HYPIR_SD2_rank256_native512' and
         c['registrar_source_sha256']==registrar and local in [50,100] and step==7750+local and c['optimizer_step']==local and
         raw==run/f'step{step}.pth' and c['initial_checkpoint_sha256']==A_SHA and c['checkpoint_sha256']!=A_SHA,'Native lineage/schema/counters differ')
    need(c['runner_sha256']==DFD_SHA and sha(md/'runner.py')==DFD_SHA and sha(md/'config.json')==c['config_sha256'],'Native runtime sources changed')
    need(c['optimizer_tensor_count']==count and c['autograd_LoRA_tensor_count']==514 and c['optimizer_parameter_keys']==c['actual_changed_parameter_keys']==keys and
         c['optimizer_parameter_elements']==elements and c[f'actual_unselected{frozen}_equal_A'] is True and
         c['actual_raw_and_resume_lora_equal'] is True and c['raw_FP32_tensor_count']==514 and c['new_Adam_initial_step']==0 and
         c['no_teacher_outputs'] is True and c['no_val_GT_training'] is True and c['distillation'] is False and c['quality_reuse_claim'] is False,'Not a registered HYPIR2/6 fine-tune')
    prov=c['training_provenance']
    for key in ['config','marker','input_catalog','parity_proof']:
        x=prov[key];need(sha(x['path'])==x['sha256'],'Changed native provenance: '+key)
    need(Path(prov['config']['path'])==run/'config.json' and Path(prov['marker']['path'])==run/f'checkpoint_step{step}.json','Unexpected actual training config/marker path')
    m=json.loads(Path(prov['marker']['path']).read_text());cfg=json.loads(Path(prov['config']['path']).read_text());p=cfg['protocol']
    need(cfg['protocol_id']==m['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() and
         p['schema']==pschema and p['parent_raw_sha256']==A_SHA and p['new_Adam_initial_step']==0 and p['objective']['lr']==lr and p['objective']['niqe_weight']==.5 and
         p['optimizer_parameter_keys']==m['optimizer_parameter_keys']==keys and p['optimizer_parameter_elements']==m['optimizer_parameter_elements']==elements and
         p['optimizer_tensor_count']==m['optimizer_tensor_count']==count and p['unselected_LoRA_tensor_count']==frozen and
         m[f'unselected{frozen}_exact_A'] is True and m[f'unselected{frozen}_exact_A_count']==frozen and m['changed_FP32_LoRA_tensors']==count,
         'Actual marker/config selected parameter lineage differs')
    need(m['raw_checkpoint_sha256']==c['checkpoint_sha256'] and Path(m['raw_checkpoint']).resolve()==raw and
         m['local_step']==m['optimizer_step']==local and m['global_step']==step and m['resume_checkpoint_sha256']==prov['resume']['sha256'] and
         Path(m['resume_checkpoint']).resolve()==Path(prov['resume']['path']).resolve() and
         m['source_sha256']==p['source_sha256'] and trainer in p['source_sha256'].values() and
         p['deployment_aligned_module_sha256']==FUSION_SHA and p['teacher_output_loss'] is False and p['distillation'] is False and p['validation_GT_training'] is False,
         'Actual raw/marker/source/noGT/no-distillation provenance differs')
    for src,h in m['source_sha256'].items():need(sha(src)==h,'Actual training source changed')
    inp=json.loads(Path(prov['input_catalog']['path']).read_text());need(inp==p['inputs'] and [x['case'] for x in inp]==list(range(1,101)),'Expected test100 LQ-only catalog')
    need(p['training_crop_size']==768 and p['training_crop_origin_alignment']==256 and p['deployment_patch_size']==512 and p['deployment_stride']==256,'Native HYPIR geometry differs')
    runtime=json.loads((md/'config.json').read_text());need(runtime['weight_path']==str(raw),'Native config weight path differs')
    if hash_raw:need(sha(raw)==c['checkpoint_sha256'],'Actual trained checkpoint bytes differ')
    return dict(native_candidate_manifest=ref(path),label=c['label'],model_dir=str(md),checkpoint=str(raw),checkpoint_sha256=c['checkpoint_sha256'],
        initial_A_sha256=A_SHA,training_run=str(run),local_step=local,global_step=step,optimizer_step=local,optimizer_tensor_count=count,
        optimizer_parameter_keys=keys,optimizer_parameter_elements=elements,unselected_exact_A_count=frozen,training_provenance=prov,
        actual_raw_SHA_verified_here=hash_raw,raw_resume_tensors_not_reopened_here=True,source_registrar_sha256=registrar)
