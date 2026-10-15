"""CPU actual-pixel ROIs from the SAME pilot23 parent/NR50/NR100 queues.

No formal-A/native4 diagnostic outputs, resampling, filtering, or score claims.
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
PARENT=R/'nr_push7_lastconv_pilot_evaluations/push7_recovered_a7750_native_pilot'
CANDIDATES=R/'nr_push7_lastconv_pilot_evaluations'
RUNNER_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
BOXES={7:(1856,1408,2496,1856),16:(1280,1216,1920,1728),41:(768,1664,1408,2176)}
LABELS={'push7_recovered_a7750_native_pilot':'A7750 reference',
        'push7_a7750_lastconv_lr1e6_nr_s7800':'NR 50',
        'push7_a7750_lastconv_lr1e6_nr_s7850':'NR 100'}
TRAIN_SOURCE=R/'code/push7/hypir_direct_nr/train_hypir_lastconv_nr_lr1e6_v2.py'
TRAIN_SHA='8e8d5e1fcac90504d029c6df436172d9380d46ddc5b888dda01fd6b6528a25f7'
SELECTED_KEYS=['conv_out.lora_A.default.weight','conv_out.lora_B.default.weight']



def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def snapshot(directory):
    need(directory==PARENT or directory.parent==CANDIDATES,'Only the matched Push7 pilot queues allowed')
    need(directory.name in LABELS,'Unregistered pilot label')
    wrapper=json.loads((directory/'render_protocol.json').read_text());p=wrapper['protocol'];rr=p['runner']
    need(wrapper['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'Protocol hash differs')
    need(p['cases']=='pilot' and len(p['tasks'])==23 and p['warmup'] is False,'Not original pilot23')
    model=Path(p['model_dir']);config=json.loads((model/'config.json').read_text())
    need(sha(R/'code/render_cuda_submission.py')==p['source_sha256']=='1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5','Actual original renderer changed')
    for rel,digest in rr['source_sha256'].items():need(sha(Path(rr['repo_path'])/rel)==digest,'Actual HYPIR source changed: '+rel)
    need(rr['runner_sha256']==sha(model/'runner.py')==RUNNER_SHA,'Not the original native512 Push6 runner')
    need(rr['model_t']==200 and rr['coeff_t']==175 and rr['patch']==512 and rr['stride']==256 and rr['scale']==1,'Changed HYPIR protocol')
    need(rr['seed']=='231 before initialization; continuous inference RNG','Changed initialization RNG')
    need(sha(model/'config.json')==p['model_dir_sha256']['config.json'],'Config changed')
    need(Path(config['weight_path']).resolve()==Path(rr['weight_path']).resolve(),'Config/runner checkpoint differs')
    need(sha(rr['weight_path'])==rr['checkpoint_sha256'],'Actual checkpoint changed')
    need((rr['checkpoint_sha256']==A_SHA)==(directory==PARENT),'InitialA/new checkpoint provenance confused')
    training_binding=None
    if directory==PARENT:
        need(sha(model/'config.json')=='8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8','Parent config differs')
    else:
        manifest_path=model.parent/'candidate_manifest.json';manifest=json.loads(manifest_path.read_text())
        need(manifest['schema']=='hypir_lastconv_nr_candidate_v1' and manifest['checkpoint_sha256']==rr['checkpoint_sha256'] and manifest['config_sha256']==sha(model/'config.json') and manifest['runner_sha256']==RUNNER_SHA,'Candidate registry differs')
        need(manifest['optimizer_tensor_count']==2 and manifest['optimizer_parameter_keys']==manifest['actual_changed_parameter_keys']==SELECTED_KEYS and manifest['actual_unselected512_equal_A'] is True,'Only the final conv_out A/B may change')
        need(manifest['initial_checkpoint_sha256']==A_SHA and manifest['no_teacher_outputs'] is True and manifest['no_val_GT_training'] is True and manifest['distillation'] is False,'Only direct HYPIR NR training lineage allowed')
        refs=manifest['training_provenance']
        for name in ['config','marker','input_catalog']:
            need(sha(refs[name]['path'])==refs[name]['sha256'],'Actual training provenance changed: '+name)
        tc=json.loads(Path(refs['config']['path']).read_text());tp=tc['protocol'];mk=json.loads(Path(refs['marker']['path']).read_text())
        need(tc['protocol_id']==hashlib.sha256(json.dumps(tp,sort_keys=True).encode()).hexdigest()==mk['protocol_id'],'Training protocol hash differs')
        need(tp['schema']=='hypir_push7_lastconv_nr_niqe768_optimizer2_full_gaussian_fused_bf16_v1' and tp['training_crop_size']==768 and tp['training_crop_origin_alignment']==256 and tp['deployment_patch_size']==512 and tp['deployment_stride']==256,'Lastconv training/deployment geometry differs')
        need(tp['objective']['lr']==1e-6 and tp['objective']['niqe_weight']==.5,'This gallery requires the higher-LR branch')
        need(tp['teacher_output_loss'] is False and tp['distillation'] is False and tp['validation_GT_training'] is False,'Forbidden output-teacher/GT training')
        need(tp['source_sha256'][str(TRAIN_SOURCE)]==sha(TRAIN_SOURCE)==TRAIN_SHA,'Actual frozen lastconv training source differs')
        need(mk['raw_checkpoint_sha256']==rr['checkpoint_sha256'] and mk['local_step']==mk['optimizer_step']==manifest['local_step'] and mk['global_step']==7750+mk['local_step']==manifest['global_step'] and mk['local_step'] in [50,100],'Milestone/raw/counter differs')
        need(directory.name==f'push7_a7750_lastconv_lr1e6_nr_s{mk["global_step"]}','Candidate label/counter differs')
        training_binding=dict(candidate_manifest_path=str(manifest_path),candidate_manifest_sha256=sha(manifest_path),source_sha256=TRAIN_SHA,protocol_id=tc['protocol_id'],local_step=mk['local_step'],global_step=mk['global_step'],config=refs['config'],marker=refs['marker'],input_catalog=refs['input_catalog'])
    complete=(directory/'render_complete').is_file()
    need(complete,'All three FINAL render manifests required before gallery generation')
    path=directory/'render_manifest.json';raw=path.read_bytes();state=json.loads(raw)
    need(state['protocol']==p and state['protocol_id']==wrapper['protocol_id'],'Final pilot manifest differs')
    records=state['records'];need([(q['split'],q['case']) for q in records]==[(q['split'],q['case']) for q in p['tasks'][:len(records)]],'Committed task order differs')
    need(len(records)==23,'Final manifest must commit exactly23 images')
    actual_images=[]
    for task,row in zip(p['tasks'],records):
        image=directory/row['output']
        need(row['output']==f'render/{row["split"]}/case{row["case"]}.png' and image.is_file() and not image.is_symlink() and sha(image)==row['sha256'],'Actual committed pilot PNG differs')
        need(sha(task['input'])==task['input_sha256'],'Actual ordered input changed')
        actual_images.append(dict(split=row['split'],case=row['case'],path=str(image),sha256=row['sha256'],input_sha256=task['input_sha256']))
    need({str(q.relative_to(directory)) for q in (directory/'render').rglob('*') if q.is_file()}=={q['output'] for q in records},'Extra/missing canonical pilot render files')
    selected={}
    for case,box in BOXES.items():
        row=next((q for q in records if q['split']=='test' and q['case']==case),None)
        need(row is not None,f'{directory.name}: case{case} not yet committed')
        task=next(q for q in p['tasks'] if q['split']=='test' and q['case']==case);image=directory/row['output']
        need(row['output']==f'render/test/case{case}.png' and sha(image)==row['sha256'],'Actual committed PNG differs')
        need(sha(task['input'])==task['input_sha256'],'Actual pilot input differs')
        with Image.open(image) as im:
            need(im.format=='PNG' and im.mode=='RGB' and list(im.size)==row['size']==task['size'],'Not canonical native RGB PNG')
            need(0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height,'ROI outside native image')
        selected[case]=dict(path=str(image),sha256=row['sha256'],input_sha256=task['input_sha256'],size=row['size'],task_index=row['task_index'])
    return dict(directory=str(directory),label=directory.name,display=LABELS[directory.name],protocol=p,protocol_id=wrapper['protocol_id'],checkpoint_sha256=rr['checkpoint_sha256'],config_sha256=sha(model/'config.json'),training_binding=training_binding,all23_actual_images_SHA_verified=actual_images,snapshot_path=str(path),snapshot_sha256=hashlib.sha256(raw).hexdigest(),render_complete=complete,committed_images=len(records),selected=selected)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate-dir',type=Path,action='append',required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
    out=a.out_dir.resolve();need(out.is_relative_to(R/'analysis/push7') and not out.exists(),'Use a new isolated preview output')
    dirs=[PARENT,*(p.resolve() for p in a.candidate_dir)]
    need(len(a.candidate_dir)==2 and len(set(dirs))==3 and [p.name for p in dirs]==list(LABELS),'Exact order: recovered original, lastconv50, lastconv100')
    data=[snapshot(p) for p in dirs]
    for q in data[1:]:
        need(q['protocol']['tasks']==data[0]['protocol']['tasks'],'Not the SAME pilot input/order/native-size sequence')
        for key in ['inference_rng','input_encoding','output_encoding','source_sha256','torch']:
            need(q['protocol'][key]==data[0]['protocol'][key],'Recovered pilot protocol differs: '+key)
        need(q['protocol']['runner']['versions']==data[0]['protocol']['runner']['versions'],'Recovered runtime versions differ')
    out.mkdir(parents=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    outputs=[];comparisons=[]
    for case,box in BOXES.items():
        tiles=[]
        for q in data:
            with Image.open(q['selected'][case]['path']) as im:tiles.append(im.crop(box))
        w,h=tiles[0].size;canvas=Image.new('RGB',(16+len(tiles)*(w+16),h+86),'#f3f5f8');draw=ImageDraw.Draw(canvas)
        draw.text((16,10),f'case{case} | 1:1 ROI {box}',font=font,fill='#172033')
        for col,(tile,q) in enumerate(zip(tiles,data)):
            left=16+col*(w+16);draw.text((left,44),q['display'],font=font,fill='#172033');canvas.paste(tile,(left,70))
        path=out/f'case{case}_hypir_lastconv_lr1e6_50_100_pilot_roi_1to1.png';canvas.save(path,compress_level=1)
        with Image.open(path) as actual:
            for col,tile in enumerate(tiles):
                left=16+col*(w+16);need(np.array_equal(np.asarray(actual.crop((left,70,left+w,70+h))),np.asarray(tile)),'Saved panel pixels changed')
        base=np.asarray(tiles[0],dtype=np.int16)
        for q,tile in zip(data[1:],tiles[1:]):
            delta=np.abs(np.asarray(tile,dtype=np.int16)-base)
            comparisons.append(dict(case=case,candidate=q['label'],ROI_RGB_mean_abs=float(delta.mean()),ROI_RGB_max_abs=int(delta.max()),ROI_any_channel_changed_pixel_fraction=float(np.any(delta>0,axis=2).mean()),scope='Descriptive ROI difference only, not quality or full metric'))
        outputs.append(dict(name=path.name,sha256=sha(path),bytes=path.stat().st_size,case=case,ROI=list(box),panel_source_pixels_exact=True))
    html='<!doctype html><meta charset="utf-8"><title>HYPIR lastconv matched pilot</title><style>body{font:16px system-ui;margin:24px;color:#172033}figure{margin:28px 0;overflow:auto}img{max-width:none}a{color:#245abd}</style><h1>HYPIR末层NR微调 LR1e-6：同一新pilot队列</h1><p>三列为 A7750 reference / NR 50 / NR 100；本轮学习率1e-6，仅末层两张LoRA权重更新，其余512与原A相同，实际推理仍tile512/stride256。固定ROI原像素1:1，无缩放、锐化或修图；每张来自自己的新pilot23连续推理队列。没有混入完整103或native4诊断队列，局部案例不代表全量指标。</p>'
    for q in outputs:html+=f'<figure><figcaption>case{q["case"]}，ROI {q["ROI"]}；<a href="{q["name"]}">打开原像素面板</a></figcaption><img src="{q["name"]}"></figure>'
    html+='<p><a href="preview_provenance.json">实际源SHA与逐像素排版证明</a></p>'
    (out/'index.html').write_text(html)
    proof=dict(status='PASS',CPU_only=True,GPU_used=False,scope='Actual matched pilot23 ROI7/16/41 only; no full quality/score claim',same_inputs_and_task_order=True,same_native512_runner_and_seed_policy=True,post_fourRNG_equality_not_claimed=True,no_teacher_output_distillation=True,source_SHA_verified=True,all69_actual_committed_PNG_SHA_verified=True,all_final_manifests=True,all_ROI_pixels_exact=True,snapshots=data,outputs=outputs,ROI_comparisons=comparisons,HTML_sha256=sha(out/'index.html'),script_sha256=sha(__file__))
    with (out/'preview_provenance.json').open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',out_dir=str(out),cases=list(BOXES),proof_sha256=sha(out/'preview_provenance.json'))),flush=True)


if __name__=='__main__':main()
