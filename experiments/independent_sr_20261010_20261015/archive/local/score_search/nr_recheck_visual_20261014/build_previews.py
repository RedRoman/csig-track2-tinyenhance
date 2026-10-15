from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import html
import json
import os

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CASES = [7, 16, 23, 41, 83]
REMOTE = '/cache/independent_sr_cuda_20261013'
CONFIG = {
    'baseline': {
        'label': 'Baseline s6000',
        'local': 'cuda_outputs/s6000_mt200_ct175_compiled_cuda_all',
        'remote': 'evaluations/s6000_mt200_ct175_compiled_cuda_all',
        'manifest': 'provenance/baseline_s6000_render_manifest.json',
        'metrics': 'results/s6000_mt200_ct175_compiled_cuda_all/scores/measurements.jsonl',
    },
    'latest': {
        'label': 'NR conservative s6250',
        'local': 'nr_cuda_outputs/nr_conservative_s6250_compiled_all',
        'remote': 'nr_full_evaluations/nr_conservative_s6250_compiled_all',
        'manifest': 'provenance/nr_s6250_render_manifest.json',
        'metrics': 'nr_cuda_evaluations/nr_conservative_s6250_compiled_all/scores/measurements.jsonl',
    },
}
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
F = ImageFont.truetype(FONT, 20)
FS = ImageFont.truetype(FONT, 16)
FT = ImageFont.truetype(FONT, 28)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return os.path.relpath(path, OUT)


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def text(draw, xy, value, font=F):
    draw.text(xy, value, fill='#172133', font=font)


sources = {}
metrics = {}
for key, cfg in CONFIG.items():
    manifest_path = OUT / cfg['manifest']
    m = json.loads(manifest_path.read_text())
    rows = {r['case']: r for r in m['records'] if r['split'] == 'test'}
    tasks = {r['case']: r for r in m['protocol']['tasks'] if r['split'] == 'test'}
    measurements = [json.loads(x) for x in (BASE / cfg['metrics']).read_text().splitlines() if x.strip()]
    metrics[key] = {r['case']: r for r in measurements if r['split'] == 'test'}
    for case in CASES:
        row = rows[case]
        path = BASE / cfg['local'] / row['output']
        actual_sha = sha(path)
        with Image.open(path) as im:
            size = list(im.size)
        assert actual_sha == row['sha256']
        assert size == row['size']
        assert metrics[key][case]['sr_sha256'] == actual_sha
        sources[f'{case}/{key}'] = {
            'id': f'{case}/{key}',
            'case': case, 'variant': key, 'label': cfg['label'],
            'local_path': str(path), 'remote_path': f"{REMOTE}/{cfg['remote']}/{row['output']}",
            'sha256': actual_sha, 'manifest_sha256': row['sha256'],
            'size_wh': size, 'manifest_path': str(manifest_path),
            'manifest_file_sha256': sha(manifest_path), 'protocol_id': m['protocol_id'],
            'metric_jsonl': str(BASE / cfg['metrics']),
            'verification': 'SHA-256, size, metric sr_sha256 all match',
        }
        if key == 'latest':
            path = BASE / 'visual_checks/input' / f'case{case}.jpg'
            actual_sha = sha(path)
            with Image.open(path) as im:
                size = list(im.size)
            assert actual_sha == tasks[case]['input_sha256']
            assert size == tasks[case]['size']
            sources[f'{case}/input'] = {
                'id': f'{case}/input',
                'case': case, 'variant': 'input', 'label': 'Input LQ',
                'local_path': str(path), 'remote_path': tasks[case]['input'],
                'sha256': actual_sha, 'manifest_sha256': tasks[case]['input_sha256'],
                'size_wh': size, 'manifest_path': str(manifest_path),
                'verification': 'SHA-256 and size match manifest input task',
            }

variants = ['input', 'baseline', 'latest']
labels = ['Input LQ', 'Baseline s6000', 'NR conservative s6250']
overview_scale = 1 / 8
cellw, cellh, gap = 512, 512, 16
rowh = 570
overview = Image.new('RGB', (3 * cellw + 4 * gap, 112 + len(CASES) * rowh), '#f2f5fa')
d = ImageDraw.Draw(overview)
text(d, (16, 12), 'Five test cases | full-frame thumbnails at 1/8 scale', FT)
text(d, (16, 48), 'Only overview images are resampled (Lanczos). See native 1:1 ROIs for pixel inspection.', FS)
for col, label in enumerate(labels):
    text(d, (gap + col * (cellw + gap), 80), label)
previews = []
for row, case in enumerate(CASES):
    y = 112 + row * rowh
    a, b = metrics['baseline'][case]['nr_per_image'], metrics['latest'][case]['nr_per_image']
    text(d, (gap, y), f'case{case}   NR: {a:.6f} -> {b:.6f}   delta {b-a:+.6f}', F)
    for col, key in enumerate(variants):
        src = sources[f'{case}/{key}']
        im = load_rgb(src['local_path'])
        size = (im.width // 8, im.height // 8)
        im = im.resize(size, Image.Resampling.LANCZOS)
        x = gap + col * (cellw + gap)
        paste_xy = (x + (cellw - im.width) // 2, y + 36 + (cellh - im.height) // 2)
        overview.paste(im, paste_xy)
        previews.append({'type': 'overview_thumbnail', 'case': case, 'variant': key,
                         'output_path': str(OUT / 'overview_5cases_input_baseline_s6000_latest_s6250.png'),
                         'source': f'{case}/{key}', 'scale_xy': [overview_scale, overview_scale],
                         'resampling': 'Pillow Lanczos', 'output_size_wh': list(size),
                         'overview_paste_xy': list(paste_xy)})
overview.save(OUT / 'overview_5cases_input_baseline_s6000_latest_s6250.png')

# Integer source coordinates, same crop for all three images, no resize or filtering.
rois = [
    (7, (1856, 1408, 2496, 1856), 'scaffold banner and window edges'),
    (16, (1280, 1216, 1920, 1728), 'window bars and facade'),
    (41, (768, 1664, 1408, 2176), 'flower petals and dense leaves'),
    (83, (1216, 1920, 1856, 2432), 'rose, leaves, and textured wall; negative NR case'),
]
roi_links = []
for case, box, topic in rois:
    w, h = box[2] - box[0], box[3] - box[1]
    canvas = Image.new('RGB', (3 * w + 4 * gap, h + 110), '#f2f5fa')
    dd = ImageDraw.Draw(canvas)
    text(dd, (gap, 10), f'case{case} | native pixels 1:1 | crop {box}', F)
    text(dd, (gap, 40), topic, FS)
    for col, key in enumerate(variants):
        x = gap + col * (w + gap)
        text(dd, (x, 72), labels[col])
        im = load_rgb(sources[f'{case}/{key}']['local_path'])
        crop = im.crop(box)
        assert crop.size == (w, h)
        canvas.paste(crop, (x, 104))
        previews.append({'type': 'native_roi', 'case': case, 'variant': key,
                         'output_path': str(OUT / f'roi_case{case}_native_1to1.png'),
                         'source': f'{case}/{key}', 'box_xyxy_exclusive': list(box),
                         'scale_xy': [1, 1], 'resampling': None,
                         'operations': ['integer crop', 'paste unchanged RGB samples', 'labels outside image'],
                         'roi_paste_xy': [x, 104]})
    filename = f'roi_case{case}_native_1to1.png'
    canvas.save(OUT / filename)
    roi_links.append((case, filename, box, topic))

metrics_out = []
for case in CASES:
    metrics_out.append({'case': case,
                        'baseline': metrics['baseline'][case],
                        'latest': metrics['latest'][case],
                        'nr_delta': metrics['latest'][case]['nr_per_image'] - metrics['baseline'][case]['nr_per_image']})
(OUT / 'selected_case_metrics.json').write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False) + '\n')

preview_manifest = {
    'schema': 'scientific_visual_inspection_v1',
    'created_local_date': '2026-10-14',
    'cases': CASES,
    'purpose': 'Local visual inspection of exact full-render PNGs; originals unmodified.',
    'limitations': 'No test GT available here. Visible sharpening, generated texture or NR gains do not prove faithful detail recovery.',
    'source_files': list(sources.values()),
    'overview': {'path': str(OUT / 'overview_5cases_input_baseline_s6000_latest_s6250.png'),
                 'scale_xy': [0.125, 0.125], 'resampling': 'Pillow Lanczos', 'padding': 'neutral light background'},
    'roi_coordinate_convention': 'integer [left, top, right, bottom); origin top-left; source pixel coordinates; no resampling',
    'previews': previews,
    'outputs': [{'path': str(p), 'sha256': sha(p)} for p in [OUT / 'overview_5cases_input_baseline_s6000_latest_s6250.png'] + [OUT / r[1] for r in roi_links]],
}
(OUT / 'preview_manifest.json').write_text(json.dumps(preview_manifest, indent=2, ensure_ascii=False) + '\n')

def link(path, label):
    return f'<a href="{html.escape(rel(path), quote=True)}">{html.escape(label)}</a>'

parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>6250 与 6000 原图比较 · 2026-10-14</title>',
         '<style>body{font:16px/1.6 system-ui,sans-serif;margin:28px auto;max-width:1500px;padding:0 24px;color:#182338;background:#f5f7fb}h1{font-size:26px}table{border-collapse:collapse;background:white}td,th{border:1px solid #cbd3dd;padding:8px 12px;text-align:left}img.overview{width:100%;height:auto}figure{margin:22px 0;padding:16px;background:white;border:1px solid #dbe1eb}a{color:#1759a9}.native{overflow:auto}.native img{max-width:none}.caption{color:#49566c}</style>',
         '<h1>最新 NR conservative s6250 与 baseline s6000</h1>',
         '<p>5 例测试图：7、16、23、41，以及 NR 分数退步最大的 83。10 张输出 PNG 和 5 张输入 JPG 均经远端完整渲染 manifest 的 SHA-256 与尺寸核验；所有原图保持原样。点击链接可打开 4K 原图。</p>',
         '<p>总览统一为原图的 1/8（Lanczos 缩略）；下面 4 组 ROI 仅整数裁剪与拼接，保留 1:1 原像素，可横向滚动。测试图未提供参考真值；锐化、纹理增加或 NR 指标提高，均不能证明恢复了真实细节。</p>',
         '<p><strong>目测结论：</strong>两版都比输入更锐利，6250 相对 6000 的可见变化较小。case7 字缘及 case16 窗格仍有过强边缘、细碎亮点和不自然重复纹理；case41 花瓣偏平滑、叶片边界较硬。负例 case83 的 NR 下降 0.13855，但两版肉眼接近，叶脉、花瓣折边等纹理仍不能确认真实。case23 只列缩略总览，不据此判断细微差异。</p>',
         '<p>' + link(OUT / 'preview_manifest.json', '预览来源、SHA与ROI坐标') + ' · ' + link(OUT / 'selected_case_metrics.json', '5例完整指标') + ' · ' + link(BASE / CONFIG['latest']['metrics'], '最新逐图指标 JSONL') + ' · ' + link(BASE / CONFIG['baseline']['metrics'], 'baseline 逐图指标 JSONL') + '</p>',
         '<table><thead><tr><th>病例</th><th>原输入</th><th>Baseline 4K PNG</th><th>最新 4K PNG</th><th>NR baseline</th><th>NR latest</th><th>差值</th></tr></thead><tbody>']
for row in metrics_out:
    case = row['case']
    cells = [str(case), link(sources[f'{case}/input']['local_path'], '输入 JPG'),
             link(sources[f'{case}/baseline']['local_path'], 's6000 PNG'),
             link(sources[f'{case}/latest']['local_path'], 's6250 PNG'),
             f"{row['baseline']['nr_per_image']:.8f}", f"{row['latest']['nr_per_image']:.8f}", f"{row['nr_delta']:+.8f}"]
    parts.append('<tr>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>')
parts.extend(['</tbody></table>', '<figure><figcaption>全图缩略总览 · 1/8</figcaption><a href="overview_5cases_input_baseline_s6000_latest_s6250.png"><img class="overview" src="overview_5cases_input_baseline_s6000_latest_s6250.png"></a></figure>'])
for case, filename, box, topic in roi_links:
    parts.append(f'<figure><figcaption>case{case} · 1:1 原像素 ROI · 坐标 {box} · {html.escape(topic)}</figcaption><div class="native"><img src="{filename}"></div></figure>')
parts.append('<p class="caption">本页为本地图片索引。源 PNG 未重新压缩、修饰或覆盖；对比图仅用于检视。</p></html>')
(OUT / 'index.html').write_text('\n'.join(parts))
(OUT / 'visual_observations_zh.txt').write_text(
    '2026-10-14 原图目测记录（5例概览、4组原像素ROI）\n'
    '两版均明显锐化输入；6250相对6000的可见变化较小，不能声称肉眼显著提升。\n'
    'case7：字缘清晰但边缘偏强；两版均有细粒纹理。case16：窗格更清楚，但仍有细碎亮点、重复线条及竖边不自然纹理。\n'
    'case41：花瓣较平滑、叶片边界偏硬，6250与6000很接近。case83：NR下降0.13855，肉眼两版仍很接近；光滑叶脉及花瓣折边不能确认是真实细节。\n'
    'case23只看1/8全图总览，不能用于细微差异判断。无测试参考真值；NR提高、清晰感和生成纹理不等同真实细节恢复。\n'
    '10张PNG、5张输入JPG已匹配远端full manifest SHA-256及原尺寸；原图未处理。\n'
)
print(json.dumps({'output': str(OUT), 'verified_original_pngs': 10, 'verified_input_jpgs': 5,
                  'overview': preview_manifest['overview']['path'], 'rois': [r[1] for r in roi_links]}, ensure_ascii=False))
