"""CPU actual-pixel ROIs: formal full103 original A7750 vs TRT FP16 UNet+VAE backend.

No pilot/native4 diagnostic outputs, resampling, filtering, or score claims.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
from PIL import Image,ImageDraw,ImageFont
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/cache/independent_sr_cuda_20261013')
PARENT=R/'nr_push4_full_evaluations/push4_a7750_compiled_all'
PARENT_AUDIT=R/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json'
PARENT_AUDIT_SHA='ae5b2234f5e110240507fbec515139d1b2a23bf8f664e7477e34c377e09c507b'
PARENT_FULL_RUNNER_SHA='396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae'
CANDIDATES=R/'nr_push7_hypir_full_evaluations'
RUNNER_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
BOXES={7:(1856,1408,2496,1856),16:(1280,1216,1920,1728),41:(768,1664,1408,2176),83:(1216,1920,1856,2432)}
TRT_RUNNER_SHA='270d2434179d17658e9f1df228c8ef99124a2695141c92f0b5fecfa73d8d3080'
VAE_HYBRID_SHA='2bc74f65029c5c95c9a8c285bf7db1c63b03a15fb9fa06cd56d93cf6b1fe2fc9'
UNET_ENGINE_SHA='08fca16f286c604614e9322ded93f848b0c1d3dd40de7ca793f10b98f286aec8'
UNET_MANIFEST_SHA='627f1e314b1a6d6a467fd1eca2188b9ed840d54ba9bc9b80ec9ae27a84c68db6'
TRT_CONFIG_SHA='63e227319f2c2fb327257a810f6709b4f1d738732571f6c100e22b75174694e5'
TRT_RUNTIME_SHA='dc2597edfe8a7d71526535bd4e7540bd9c7623de00b6639c61139e08837aa9a4'
TRT_ENGINES={'encoder':('14c206896a7caa6b6491dae33133d149283025cfb8e15b6cf129d8854176ba55','21d3781a5cc092062d4ef87844e74ce916c461ea5b9939658917333d74a2f0ae'),
             'decoder':('fe81087086d76aa8d0111cb80070b65b897e0438ebb2a8a020ef0ced8bc6bf30','1a3a349641e65ca5ad5d92301b25b3b1eae214e760645b5a7c294460f0ac08fd')}
LABELS={'push4_a7750_compiled_all':'A7750 / original BF16 VAE / FULL queue',
        'push7_a7750_trt_unet_vae_fp16_all':'A7750 / TRT FP16 UNet+VAE / FULL queue'}


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def snapshot(directory,allow_progress=False):
    need(directory==PARENT or directory.parent==CANDIDATES,'Only the formal full103 queues allowed')
    need(directory.name in LABELS,'Unregistered full103 label')
    wrapper=json.loads((directory/'render_protocol.json').read_text());p=wrapper['protocol'];rr=p['runner']
    need(wrapper['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'Protocol hash differs')
    need(p['cases']=='all' and len(p['tasks'])==103 and p['warmup'] is False,'Not original full103 sequence')
    model=Path(p['model_dir']);config=json.loads((model/'config.json').read_text())
    actual_runner=sha(model/'runner.py');need(rr['runner_sha256']==actual_runner,'Actual runner/protocol source differs')
    if directory==PARENT:
        need(actual_runner==PARENT_FULL_RUNNER_SHA,'Not frozen original full-A runner')
        need(sha(PARENT_AUDIT)==PARENT_AUDIT_SHA,'Frozen A full audit changed')
    else:
        need(actual_runner==TRT_RUNNER_SHA and sha(model/'config.json')==TRT_CONFIG_SHA,'Unreviewed TRT hybrid source/config')
        need(sha(model/'reference_runner.py')==RUNNER_SHA and sha(model/'trt_vae_runtime_v2.py')==TRT_RUNTIME_SHA,'TRT original wrapper/runtime changed')
        need(rr['TRT']['initialization_all_four_RNG_restored'] is True and rr['TRT']['forward_RNG_reset'] is False,'TRT RNG declaration differs')
        need(sha(model/'vae_hybrid_runner.py')==VAE_HYBRID_SHA,'Original VAE wrapper source differs')
        spec=config['TRT_UNet_engine']
        need(spec['sha256']==UNET_ENGINE_SHA and spec['manifest_sha256']==UNET_MANIFEST_SHA and sha(spec['path'])==UNET_ENGINE_SHA and sha(spec['manifest'])==UNET_MANIFEST_SHA,'Actual sealed UNet engine/manifest differs')
        need(rr['UNet_TRT']['engine_sha256']==UNET_ENGINE_SHA and rr['UNet_TRT']['manifest_sha256']==UNET_MANIFEST_SHA and rr['UNet_TRT']['construction_all_four_RNG_restored'] is True and rr['UNet_TRT']['forward_RNG_reset'] is False and rr['UNet_TRT']['scheduler_DDPM_original_BF16_outside_engine'] is True,'UNet backend/RNG/scheduler declaration differs')
        for name,(eh,mh) in TRT_ENGINES.items():
            spec=config['TRT_engines'][name]
            need(spec['sha256']==eh and spec['manifest_sha256']==mh and sha(spec['path'])==eh and sha(spec['manifest'])==mh,'Actual sealed engine/manifest changed')
    need(rr['model_t']==200 and rr['coeff_t']==175 and rr['patch']==512 and rr['stride']==256 and rr['scale']==1,'Changed HYPIR protocol')
    need(rr['seed']=='231 before initialization; continuous inference RNG','Changed initialization RNG')
    need(sha(model/'config.json')==p['model_dir_sha256']['config.json'],'Config changed')
    need(Path(config['weight_path']).resolve()==Path(rr['weight_path']).resolve(),'Config/runner checkpoint differs')
    need(sha(rr['weight_path'])==rr['checkpoint_sha256'],'Actual checkpoint changed')
    need(rr['checkpoint_sha256']==A_SHA,'Both backend variants must retain the same actual A7750 raw')
    complete=(directory/'render_complete').is_file()
    if complete:
        path=directory/'render_manifest.json';raw=path.read_bytes();state=json.loads(raw)
        need(state['protocol']==p and state['protocol_id']==wrapper['protocol_id'],'Final full manifest differs')
    else:
        need(allow_progress,'Full render is not complete; explicit --allow-progress needed for committed-case previews')
        import torch
        need(not torch.cuda.is_initialized(),'Preview must be CPU-only')
        path=directory/'render_progress.pt';raw=path.read_bytes();state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        need(state['protocol_id']==wrapper['protocol_id'] and state['checkpoint_sha256']==rr['checkpoint_sha256'],'Committed progress belongs to another run')
    records=state['records'];need([(q['split'],q['case']) for q in records]==[(q['split'],q['case']) for q in p['tasks'][:len(records)]],'Committed task order differs')
    selected={}
    for case,box in BOXES.items():
        row=next((q for q in records if q['split']=='test' and q['case']==case),None)
        need(row is not None,f'{directory.name}: case{case} not yet committed')
        task=next(q for q in p['tasks'] if q['split']=='test' and q['case']==case);image=directory/row['output']
        need(row['output']==f'render/test/case{case}.png' and sha(image)==row['sha256'],'Actual committed PNG differs')
        need(sha(task['input'])==task['input_sha256'],'Actual full input differs')
        with Image.open(image) as im:
            need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==row['size']==task['size'],'Not canonical native RGB PNG')
            need(0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height,'ROI outside native image')
        if directory==PARENT:
            frozen=next(q for q in json.loads(PARENT_AUDIT.read_text())['records'] if q['split']=='test' and q['case']==case)
            need(row['sha256']==frozen['actual_png_sha256'] and task['input_sha256']==frozen['actual_input_sha256'],'Current original-A PNG/input differs from frozen full audit')
        selected[case]=dict(path=str(image),sha256=row['sha256'],input_sha256=task['input_sha256'],size=row['size'],task_index=row['task_index'])
    return dict(directory=str(directory),label=directory.name,display=LABELS[directory.name],protocol=p,protocol_id=wrapper['protocol_id'],checkpoint_sha256=rr['checkpoint_sha256'],config_sha256=sha(model/'config.json'),snapshot_path=str(path),snapshot_sha256=hashlib.sha256(raw).hexdigest(),render_complete=complete,committed_images=len(records),selected=selected)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate-dir',type=Path,action='append',required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--allow-progress',action='store_true');a=ap.parse_args()
    out=a.out_dir.resolve();need(out.is_relative_to(R/'analysis/push7') and not out.exists(),'Use a new isolated preview output')
    dirs=[PARENT,*(p.resolve() for p in a.candidate_dir)]
    need(len(a.candidate_dir)==1 and len(set(dirs))==len(dirs),'One explicit TRT backend variant')
    data=[snapshot(p,a.allow_progress) for p in dirs]
    for q in data[1:]:
        need(q['protocol']['tasks']==data[0]['protocol']['tasks'],'Not the SAME full103 input/order/native-size sequence')
        need(q['protocol']['inference_rng']==data[0]['protocol']['inference_rng'],'Formal RNG policy differs')
    out.mkdir(parents=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    outputs=[];comparisons=[]
    for case,box in BOXES.items():
        tiles=[]
        for q in data:
            with Image.open(q['selected'][case]['path']) as im:tiles.append(im.crop(box))
        w,h=tiles[0].size;canvas=Image.new('RGB',(16+len(tiles)*(w+16),h+132),'#f3f5f8');draw=ImageDraw.Draw(canvas)
        draw.text((16,10),f'case{case} / SAME full103 task sequence / native ROI {box} / stored1:1 pixels',font=font,fill='#172033')
        draw.text((16,36),'Same original HYPIR A7750 checkpoint. UNet+VAE backend/FP16 arithmetic changed; no training.',font=font,fill='#172033')
        draw.text((16,62),'Actual committed PNGs, same inputs/order/seed policy. No pilot or native4 diagnostic sequence mixed.',font=font,fill='#172033')
        for col,(tile,q) in enumerate(zip(tiles,data)):
            left=16+col*(w+16);draw.text((left,92),q['display'],font=font,fill='#172033');canvas.paste(tile,(left,116))
        path=out/f'case{case}_hypir_trtunetvae_full_roi_1to1.png';canvas.save(path,compress_level=1)
        with Image.open(path) as actual:
            for col,tile in enumerate(tiles):
                left=16+col*(w+16);need(np.array_equal(np.asarray(actual.crop((left,116,left+w,116+h))),np.asarray(tile)),'Saved panel pixels changed')
        base=np.asarray(tiles[0],dtype=np.int16)
        for q,tile in zip(data[1:],tiles[1:]):
            delta=np.abs(np.asarray(tile,dtype=np.int16)-base)
            comparisons.append(dict(case=case,candidate=q['label'],ROI_RGB_mean_abs=float(delta.mean()),ROI_RGB_max_abs=int(delta.max()),ROI_any_channel_changed_pixel_fraction=float(np.any(delta>0,axis=2).mean()),scope='Descriptive ROI difference only, not quality or full metric'))
        outputs.append(dict(name=path.name,sha256=sha(path),bytes=path.stat().st_size,case=case,ROI=list(box),panel_source_pixels_exact=True))
    html='<!doctype html><meta charset="utf-8"><title>HYPIR A7750 TensorRT UNet+VAE formal full outputs</title><style>body{font:16px system-ui;margin:24px;color:#172033}figure{margin:28px 0;overflow:auto}img{max-width:none}a{color:#245abd}</style><h1>HYPIR A7750：原后端与TensorRT FP16 UNet+VAE完整任务序列对比</h1><p>两列为同一A7750权重，原BF16后端与TensorRT FP16 UNet+VAE；没有新训练或缩放。固定ROI原像素1:1，无缩放、锐化或修图；每张来自自己的原生完整103任务序列（取图时可仍在生成，仅使用已提交图）。没有混入pilot23或native4诊断队列，局部案例不代表全量指标。</p>'
    for q in outputs:html+=f'<figure><figcaption>case{q["case"]}，ROI {q["ROI"]}；<a href="{q["name"]}">打开原像素面板</a></figcaption><img src="{q["name"]}"></figure>'
    html+='<p><a href="preview_provenance.json">实际源SHA与逐像素排版证明</a></p>'
    (out/'index.html').write_text(html)
    proof=dict(status='PASS',CPU_only=True,GPU_used=False,scope='Actual formal full103 sequence ROI7/16/41/83 only; no score inferred from selected cases',same_inputs_and_task_order=True,same_native512_HYPIR_geometry_and_seed_policy=True,same_A7750_LoRA_checkpoint=True,UNet_and_VAE_backend_precision_changed=True,post_fourRNG_equality_not_claimed=True,no_teacher_output_distillation=True,source_SHA_verified=True,frozen_original_full_A_selected_PNG_SHA_bound=True,parent_full_audit=dict(path=str(PARENT_AUDIT),sha256=PARENT_AUDIT_SHA),explicit_progress_allowed=a.allow_progress,all_ROI_pixels_exact=True,snapshots=data,outputs=outputs,ROI_comparisons=comparisons,HTML_sha256=sha(out/'index.html'),script_sha256=sha(__file__))
    with (out/'preview_provenance.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',out_dir=str(out),cases=list(BOXES),proof_sha256=sha(out/'preview_provenance.json'))),flush=True)


if __name__=='__main__':main()
