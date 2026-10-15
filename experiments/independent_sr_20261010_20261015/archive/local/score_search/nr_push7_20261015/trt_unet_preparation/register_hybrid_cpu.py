"""Register actual sealed A7750 UNet+VAE TRT hybrid. CPU/std-library only."""
from pathlib import Path
import argparse, ast, hashlib, json, os, shutil, tempfile
R=Path('/cache/independent_sr_cuda_20261013')
BASE=R/'submissions/hypir_push7_a7750_trtvae_fp16/model_dir'
DEST=R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16'
LABEL='push7_a7750_trt_unet_vae_fp16'
MANIFEST=R/'backends/push7_trt/unet_fp16_A7750/component_manifest.json'
MH='627f1e314b1a6d6a467fd1eca2188b9ed840d54ba9bc9b80ec9ae27a84c68db6'
EH='08fca16f286c604614e9322ded93f848b0c1d3dd40de7ca793f10b98f286aec8'
TEMPLATE_SHA='270d2434179d17658e9f1df228c8ef99124a2695141c92f0b5fecfa73d8d3080'

def need(v,msg):
    if not v:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    if not a.execute:print(json.dumps(dict(status='CPU_PLAN_ONLY',destination=str(DEST),actual_manifest=str(MANIFEST),expected_manifest_sha256=MH,expected_engine_sha256=EH,GPU=False)));return
    need(not DEST.exists(),'Never overwrite a registered candidate')
    need(sha(MANIFEST)==MH,'Actual component manifest changed');m=json.loads(MANIFEST.read_text());engine=Path(m['engine_path'])
    need(m['status']=='BUILT_COMPONENT_PROBED' and sha(engine)==m['engine_sha256']==EH and engine.stat().st_size==m['engine_bytes'],'Actual completed engine differs')
    template=Path(__file__).with_name('runner_template.py');need(sha(template)==TEMPLATE_SHA,'Reviewed wrapper template changed');ast.parse(template.read_text())
    c=json.loads((BASE/'config.json').read_text());need(sha(BASE/'config.json')=='661bf9bd6a7ecb2ae5b475dda9dcd855700bb52be87ab0f2b6d97740f32308f4','Original VAE hybrid configuration differs')
    need(c['checkpoint_sha256']==m['raw_sha256']=='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d' and sha(c['weight_path'])==m['raw_sha256'],'Actual A7750 weight differs')
    need(sha(Path(c['base_model_path'])/'unet/diffusion_pytorch_model.safetensors')==m['actual_base_UNet_sha256'],'Actual UNet base differs')
    files={'runner.py':(template,TEMPLATE_SHA),'vae_hybrid_runner.py':(BASE/'runner.py','2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9'),
           'reference_runner.py':(BASE/'reference_runner.py','dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'),
           'trt_vae_runtime_v2.py':(BASE/'trt_vae_runtime_v2.py','dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4')}
    for p,h in files.values():need(sha(p)==h,'Source bytes differ: '+str(p))
    for name,v in c['TRT_engines'].items():need(sha(v['path'])==v['sha256'] and sha(v['manifest'])==v['manifest_sha256'],'Actual VAE engine provenance differs: '+name)
    c.update(label=LABEL,UNet_TRT_backend='A7750_FP16_TRT_fixed64_empty_t200',TRT_UNet_engine=dict(path=str(engine),sha256=EH,manifest=str(MANIFEST),manifest_sha256=MH))
    raw=(json.dumps(c,indent=2)+'\n').encode();cfgsha=hashlib.sha256(raw).hexdigest()
    candidate=dict(schema='hypir_trt_unet_vae_hybrid_candidate_v1',status='CPU_REGISTERED_ACTUAL_ENGINE_ARTIFACTS',label=LABEL,model_dir=str(DEST/'model_dir'),
        model_type='HYPIR_SD2_rank256_TRT_UNet_VAE_native512',checkpoint=c['weight_path'],checkpoint_sha256=c['checkpoint_sha256'],runner_sha256=TEMPLATE_SHA,config_sha256=cfgsha,
        implementation_sources={name:h for name,(_,h) in files.items()},actual_engines=dict(c['TRT_engines'],UNet=c['TRT_UNet_engine']),
        actual_base_UNet_sha256=m['actual_base_UNet_sha256'],BF16_original_fused_G_state=m['BF16_original_fused_G_state'],fixed_conditioning=m['fixed_conditioning'],
        constructor_fixed_conditioning_and_four_RNG='Strict runtime guards; not executed by CPU registrar',
        component_diagnostics_not_tool_score=True,no_training=True,no_distillation=True,quality_reuse_claim=False,requires_new_actual_full103_418=True,CPU_registrar_sha256=sha(__file__))
    temp=Path(tempfile.mkdtemp(prefix='.register_trt_unet_',dir=DEST.parent))
    try:
        md=temp/'model_dir';md.mkdir()
        for name,(p,h) in files.items():shutil.copyfile(p,md/name);need(sha(md/name)==h,'Byte-copy failed')
        (md/'config.json').write_bytes(raw);(temp/'candidate_manifest.json').write_text(json.dumps(candidate,indent=2)+'\n')
        need(sha(md/'config.json')==cfgsha,'Config write failed');os.rename(temp,DEST)
    except BaseException:shutil.rmtree(temp,ignore_errors=True);raise
    print(json.dumps(dict(status=candidate['status'],model_dir=str(DEST/'model_dir'),candidate_manifest_sha256=sha(DEST/'candidate_manifest.json'),runner_sha256=TEMPLATE_SHA,config_sha256=cfgsha,UNet_engine_sha256=EH,GPU=False)))

if __name__=='__main__':main()
