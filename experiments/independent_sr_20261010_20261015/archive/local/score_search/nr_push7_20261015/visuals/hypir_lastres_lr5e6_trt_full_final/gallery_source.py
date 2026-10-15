"""CPU-only, fresh recovered full103 A7750 versus a real fine-tuned TRT HYPIR.

The supplied backend manifest SHA must come from actual registration, not a plan.
Only real native PNG crops are displayed. No model image is resized or filtered.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
R = Path('/cache/independent_sr_cuda_20261013')
FULL = R / 'nr_push7_recovered_full_evaluations'
LABELS = ['push7_recovered_a7750_trtunetvae_all', 'push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all']
COLUMNS = ['A7750 reference', 'HYPIR NR candidate']
RAW = ['589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d',
       '23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c']
RUNNERS = ['270d2434179d17658e9f1df228c8ef99124a2695141c92f0b5fecfa73d8d3080',
           'c98c406884cf07bc5541e315393b98e11351165b35c94d33f926d5443f9563c3']
NATIVE_MF_SHA = '0477903b0bc478064ad2f03a5d429eb37459097ac6a0014ec76d92efef7fa4c0'
DFD = 'dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
RENDER = '1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
BOXES = {7: (1856, 1408, 2496, 1856), 16: (1280, 1216, 1920, 1728),
         41: (768, 1664, 1408, 2176), 83: (1216, 1920, 1856, 2432)}
SEEN = {}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    st = path.stat()
    stamp = (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev)
    if path in SEEN:
        need(SEEN[path][0] == stamp, 'Source changed during preview: ' + str(path))
        return SEEN[path][1]
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    now = path.stat()
    need(stamp == (now.st_size, now.st_mtime_ns, now.st_ino, now.st_dev), 'Unstable source read')
    SEEN[path] = (stamp, h.hexdigest())
    return h.hexdigest()


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path), bytes=Path(path).stat().st_size)


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def snapshot(index, candidate_manifest_sha):
    directory = FULL / LABELS[index]
    need((directory / 'render_complete').is_file(), 'Both fresh full103 final manifests are required')
    wrapper = read(directory / 'render_protocol.json')
    p = wrapper['protocol']
    need(wrapper['protocol_id'] == hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest(), 'Protocol hash differs')
    need(p['cases'] == 'all' and len(p['tasks']) == 103 and p['warmup'] is False, 'Not an original full103 queue')
    expected_order = [('val', n) for n in range(1, 4)] + [('test', n) for n in range(1, 101)]
    need([(t['split'], t['case']) for t in p['tasks']] == expected_order, 'Not val3 then test100')
    need(sha(R / 'code/render_cuda_submission.py') == p['source_sha256'] == RENDER, 'Original renderer changed')
    md = Path(p['model_dir'])
    rr = p['runner']
    cfg = read(md / 'config.json')
    need(sha(md / 'runner.py') == rr['runner_sha256'] == RUNNERS[index], 'Wrong actual full runner')
    need(sha(md / 'config.json') == p['model_dir_sha256']['config.json'] == rr['model_dir_config_sha256'], 'Actual config differs from render')
    need(rr['checkpoint_sha256'] == RAW[index] and cfg['checkpoint_sha256'] == RAW[index] and cfg['weight_path'] == rr['weight_path'], 'Wrong actual checkpoint declaration')
    need(rr['model_t'] == 200 and rr['coeff_t'] == 175 and rr['lora_rank'] == 256 and rr['patch'] == 512 and rr['stride'] == 256 and rr['scale'] == 1, 'Changed native HYPIR geometry')
    need(rr['seed'] == '231 before initialization; continuous inference RNG', 'Wrong inference RNG policy')
    for rel, digest in rr['source_sha256'].items():
        need(sha(Path(rr['repo_path']) / rel) == digest, 'Actual original HYPIR source changed')
    for path, digest in rr['implementation_sources'].items():
        need(sha(path) == digest, 'Actual backend implementation changed')
    need(sha(md / 'reference_runner.py') == DFD, 'Wrong original reference implementation')
    need(rr['TRT']['initialization_all_four_RNG_restored'] is True and rr['TRT']['forward_RNG_reset'] is False and
         rr['UNet_TRT']['construction_all_four_RNG_restored'] is True and rr['UNet_TRT']['forward_RNG_reset'] is False, 'Backend RNG declaration differs')
    engines = {}
    for name, spec in dict(rr['TRT']['engines'], UNet=dict(path=rr['UNet_TRT']['engine_path'], sha256=rr['UNet_TRT']['engine_sha256'],
                           manifest=rr['UNet_TRT']['manifest'], manifest_sha256=rr['UNet_TRT']['manifest_sha256'])).items():
        need(sha(spec['manifest']) == spec['manifest_sha256'], 'Component manifest changed')
        component = read(spec['manifest'])
        need(component['engine_sha256'] == spec['sha256'], 'Component engine declaration differs')
        engines[name] = dict(path=spec['path'], declared_sha256=spec['sha256'], component_manifest=record(spec['manifest']))
    backend_manifest = record(md.parent / 'candidate_manifest.json')
    if index:
        need(backend_manifest['sha256'] == candidate_manifest_sha, 'Actual newly registered backend manifest differs')
        mf = read(md.parent / 'candidate_manifest.json')
        need(mf['checkpoint_sha256'] == RAW[index] and mf['runner_sha256'] == RUNNERS[index] and mf['config_sha256'] == sha(md / 'config.json'), 'Registered trained backend source/weight differs')
        native = cfg['native_candidate_manifest']
        need(sha(native['path']) == native['sha256'] == NATIVE_MF_SHA, 'Wrong original registered fine-tune')
        need(rr['UNet_TRT']['checkpoint_sha256'] == RAW[index] and rr['UNet_TRT']['native_candidate_manifest']['sha256'] == NATIVE_MF_SHA, 'Engine is not bound to this trained weight')
        need(rr['UNet_TRT']['engine_sha256'] != '08fca16f286c604614e9322ded93f848b0c1d3dd40de7ca793f10b98f286aec8', 'Old A-only engine is forbidden for trained weights')
    state = read(directory / 'render_manifest.json')
    need(state['protocol'] == p and state['protocol_id'] == wrapper['protocol_id'], 'Final manifest protocol differs')
    rows = state['records']
    need(len(rows) == 103 and [(t['split'], t['case']) for t in rows] == expected_order, 'Incomplete final native image sequence')
    images = []
    for task, row in zip(p['tasks'], rows):
        png = directory / row['output']
        need(row['output'] == f'render/{row["split"]}/case{row["case"]}.png' and not png.is_symlink() and sha(png) == row['sha256'], 'Actual committed native PNG differs')
        need(sha(task['input']) == task['input_sha256'], 'Actual ordered input differs')
        images.append(dict(split=row['split'], case=row['case'], PNG=record(png), input_sha256=task['input_sha256'], size=row['size']))
    need({str(q.relative_to(directory)) for q in (directory / 'render').rglob('*') if q.is_file()} == {r['output'] for r in rows}, 'Extra or missing native full PNG')
    return dict(label=LABELS[index], column=COLUMNS[index], directory=str(directory), protocol=p, protocol_id=wrapper['protocol_id'],
                final_manifest=record(directory / 'render_manifest.json'), backend_manifest=backend_manifest, engines=engines,
                image_count=103, all103_actual_PNG_SHA_verified=True, images=images)


def main():
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--candidate-manifest-sha256', required=True)
    q.add_argument('--out-dir', type=Path, required=True)
    a = q.parse_args()
    need(re.fullmatch(r'[0-9a-f]{64}', a.candidate_manifest_sha256), 'Actual registration SHA is required')
    out = a.out_dir.resolve()
    need(out.is_relative_to(R / 'analysis/push7') and not out.exists(), 'New isolated output only')
    Image.MAX_IMAGE_PIXELS = None
    sources = [snapshot(i, a.candidate_manifest_sha256) for i in range(2)]
    for key in ['tasks', 'input_encoding', 'output_encoding', 'warmup', 'inference_rng', 'source_sha256', 'torch']:
        need(sources[0]['protocol'][key] == sources[1]['protocol'][key], 'Fresh full input/order/protocol differs: ' + key)
    need(sources[0]['protocol']['runner']['versions'] == sources[1]['protocol']['runner']['versions'], 'Runtime versions differ')
    out.mkdir(parents=True)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    outputs = []
    for case, box in BOXES.items():
        tiles = []
        for source in sources:
            row = next(r for r in source['images'] if r['split'] == 'test' and r['case'] == case)
            with Image.open(row['PNG']['path']) as im:
                need(im.mode == 'RGB' and im.format == 'PNG' and list(im.size) == row['size'], 'Not a canonical RGB native PNG')
                need(0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height, 'ROI exceeds source bounds')
                tiles.append(im.crop(box))
        w, h = tiles[0].size
        canvas = Image.new('RGB', (16 + 2 * (w + 16), h + 86), '#f3f5f8')
        d = ImageDraw.Draw(canvas)
        d.text((16, 10), f'case{case} | 1:1 ROI {box}', font=font, fill='#172033')
        for col, tile in enumerate(tiles):
            x = 16 + col * (w + 16)
            d.text((x, 44), COLUMNS[col], font=font, fill='#172033')
            canvas.paste(tile, (x, 70))
        path = out / f'case{case}_hypir_lastres_lr5e6_trt_full_roi_1to1.png'
        canvas.save(path, compress_level=1)
        with Image.open(path) as panel:
            for col, tile in enumerate(tiles):
                x = 16 + col * (w + 16)
                need(np.array_equal(np.asarray(panel.crop((x, 70, x + w, 70 + h))), np.asarray(tile)), 'Stored panel crop pixels changed')
        delta = np.abs(np.asarray(tiles[1], dtype=np.int16) - np.asarray(tiles[0], dtype=np.int16))
        outputs.append(dict(**record(path), name=path.name, case=case, ROI=list(box), crop_pixels_exact=True,
                            descriptive_ROI_RGB_mean_abs=float(delta.mean()), descriptive_ROI_RGB_max_abs=int(delta.max())))
    html = '<!doctype html><meta charset="utf-8"><title>Fresh full HYPIR comparison</title><style>body{font:16px system-ui;margin:24px}figure{overflow:auto;margin:28px 0}img{max-width:none}</style><h1>HYPIR：原A与本体NR微调，全新完整103队列</h1><p>两列为A7750 reference和HYPIR NR candidate。双方均来自本轮恢复环境后独立完整103任务序列，同val3+100test输入、顺序和seed策略；没有混用pilot或重启前输出。固定ROI原尺寸1:1，没有缩放、滤镜、增强或AI修图。候选使用自身新建UNet引擎，局部案例不代表全量质量或分数。</p>'
    for row in outputs:
        html += f'<figure><figcaption>case{row["case"]} | ROI {row["ROI"]}</figcaption><a href="{row["name"]}"><img src="{row["name"]}"></a></figure>'
    html += '<details><summary>来源与验证范围</summary><p>全部206张实际已提交PNG及输入SHA、实际源码和组件manifest已核验。大模型权重和engine字节不在本图库重复读取，完整质量审计独立进行。记录constructor RNG恢复声明，不声称逐call或终点四RNG已在本图库独立比较。</p><a href="preview_provenance.json">完整来源、manifest及逐像素裁剪证明</a></details>'
    (out / 'index.html').write_text(html)
    for path, (stamp, _) in list(SEEN.items()):
        st = path.stat()
        need(stamp == (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev), 'Source changed before gallery publication')
    proof = dict(status='PASS_FRESH_FULL103_NATIVE_ROI4', CPU_only=True, GPU_used=False, all_final_manifests=True,
                 actual_PNG_count=206, fixed_ROIs=BOXES, no_image_resampling_or_filtering=True, all_crop_pixels_exact=True,
                 big_model_and_engine_bytes_reread_in_this_gallery=False, source_engine_identity_scope='Actual render protocol and component manifest bindings; separate full quality audit owns complete asset verification',
                 post_fourRNG_equality_not_claimed=True, no_score_inferred_from_cases=True, sources=sources, outputs=outputs,
                 HTML=record(out / 'index.html'), script=record(__file__))
    with (out / 'preview_provenance.json').open('x') as f:
        json.dump(proof, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=proof['status'], out_dir=str(out), proof_sha256=sha(out / 'preview_provenance.json'))), flush=True)


if __name__ == '__main__':
    main()
