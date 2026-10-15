"""Scientific comparison: overview thumbnails and unchanged native-pixel crops."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import datetime
import hashlib
import html
import json
import os

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
LABELS = ['nr_conservative_s6250_compiled_all',
          'nr_stage3_anchor6250_m1_s6500_compiled_all',
          'nr_stage3_anchor6250_m15_s6500_compiled_all']
NAMES = ['Parent s6250', 'Stage3 m1 s6500', 'Stage3 m15 s6500']
BOXES = {7: (1856, 1408, 2496, 1856), 16: (1280, 1216, 1920, 1728),
         41: (768, 1664, 1408, 2176), 83: (1216, 1920, 1856, 2432)}
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
F = ImageFont.truetype(FONT, 20)
FS = ImageFont.truetype(FONT, 16)
FT = ImageFont.truetype(FONT, 28)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def text(draw, xy, value, font=F):
    draw.text(xy, value, font=font, fill='#172133')


def link(path, label):
    return '<a href="' + html.escape(os.path.relpath(path, OUT), quote=True) + '">' + html.escape(label) + '</a>'


def main():
    state = json.loads((OUT / 'download_state.json').read_text())
    sources = state['sources']
    cases = [case for case in [7, 16, 41, 83] if all(f'{label}/{case}' in sources for label in LABELS)]
    if not cases:
        print('No common committed full cases yet')
        return
    for source in sources.values():
        assert sha(source['local_path']) == source['sha256']
    previews = []
    outputs = []
    cell, gap, rowheight = 512, 16, 560
    overview = Image.new('RGB', (1600, 112 + rowheight * len(cases)), '#f2f5fa')
    d = ImageDraw.Draw(overview)
    text(d, (16, 12), 'Full-render comparison | whole-frame thumbnails at 1/8 scale', FT)
    text(d, (16, 50), 'Parent s6250 vs teacher-anchor6250 stage3 m1/m15. Native 1:1 crops below; no pixel edits.', FS)
    for col, name in enumerate(NAMES):
        text(d, (gap + col * (cell + gap), 80), name)
    overview_path = OUT / 'overview_full_comparison.png'
    for row, case in enumerate(cases):
        y = 112 + row * rowheight
        text(d, (16, y), f'case{case} | full-render originals')
        for col, label in enumerate(LABELS):
            key = f'{label}/{case}'
            im = load(sources[key]['local_path'])
            size = (im.width // 8, im.height // 8)
            im = im.resize(size, Image.Resampling.LANCZOS)
            xy = (gap + col * (cell + gap) + (cell - im.width) // 2,
                  y + 32 + (cell - im.height) // 2)
            overview.paste(im, xy)
            previews.append({'type': 'overview_thumbnail', 'case': case, 'source': key,
                             'output_path': str(overview_path), 'scale_xy': [0.125, 0.125],
                             'resampling': 'Pillow Lanczos', 'output_size_wh': list(size),
                             'paste_xy': list(xy)})
    overview.save(overview_path)
    outputs.append(overview_path)
    for case in cases:
        box = BOXES[case]
        width, height = box[2] - box[0], box[3] - box[1]
        roi_path = OUT / f'roi_case{case}_native_1to1.png'
        canvas = Image.new('RGB', (3 * width + 4 * gap, height + 110), '#f2f5fa')
        dd = ImageDraw.Draw(canvas)
        text(dd, (16, 10), f'case{case} | native pixels 1:1 | crop {box}')
        text(dd, (16, 40), 'Same integer coordinates in full outputs. No ROI resize, sharpening or retouching.', FS)
        for col, label in enumerate(LABELS):
            x = gap + col * (width + gap)
            text(dd, (x, 72), NAMES[col])
            key = f'{label}/{case}'
            canvas.paste(load(sources[key]['local_path']).crop(box), (x, 104))
            previews.append({'type': 'native_roi', 'case': case, 'source': key,
                             'output_path': str(roi_path), 'box_xyxy_exclusive': list(box),
                             'scale_xy': [1, 1], 'resampling': None, 'paste_xy': [x, 104],
                             'operations': ['integer crop', 'paste unchanged RGB samples', 'labels outside image']})
        canvas.save(roi_path)
        outputs.append(roi_path)
    parent_metrics_path = BASE / 'nr_cuda_evaluations/nr_conservative_s6250_compiled_all/scores/measurements.jsonl'
    parent_rows = [json.loads(line) for line in parent_metrics_path.read_text().splitlines() if line.strip()]
    metrics = {LABELS[0]: {row['case']: row for row in parent_rows if row['split'] == 'test'}}
    for label in LABELS[1:]:
        metrics[label] = {row['case']: row for row in state.get('per_image_metrics', {}).get(label, [])}
    selected_metrics = []
    for case in cases:
        for label in LABELS:
            if case in metrics[label]:
                row = metrics[label][case]
                assert row['sr_sha256'] == sources[f'{label}/{case}']['sha256']
                selected_metrics.append({'case': case, 'label': label, 'scope': 'per_image_only_from_full_render_sequence',
                                         'row': row, 'not_a_full_average': True})
    (OUT / 'selected_per_image_metrics.json').write_text(json.dumps(selected_metrics, indent=2) + '\n')
    manifest = {'schema': 'full_scientific_visual_comparison_v1',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'cases_available_in_all_models': cases,
                'all_four_cases_downloaded': state['all_4_cases_downloaded'],
                'all_final_manifests_verified': state['all_final_manifests_verified'],
                'source_files': [{'id': key, **source} for key, source in sources.items()],
                'coordinate_convention': 'integer [left,top,right,bottom); top-left origin in source pixels',
                'overview_policy': 'Only overview resampled at exactly1/8 with Lanczos; ROI scale1 unchanged',
                'limitation': 'No test GT. Sharpening or NR gains do not establish true recovered detail. No full mean NR is inferred from selected cases.',
                'previews': previews,
                'outputs': [{'path': str(path), 'sha256': sha(path), 'size_wh': list(Image.open(path).size)} for path in outputs]}
    (OUT / 'preview_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    provenance = '全部所选 full 原图已再次匹配最终103图 manifest。' if state['all_final_manifests_verified'] else '新图已匹配 full 渲染的已提交 progress 记录；最终 manifest 尚待复核。'
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Stage3 full 原像素比较 · 2026-10-14</title>',
             '<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1500px;margin:26px auto;padding:0 24px;background:#f5f7fb;color:#172133}h1{font-size:25px}table{border-collapse:collapse;background:white}td,th{border:1px solid #cbd3dd;padding:8px 10px}a{color:#1759a9}figure{margin:22px 0;padding:16px;border:1px solid #dbe1eb;background:white}.overview{width:100%;height:auto}.native{overflow:auto}.native img{max-width:none}.note{white-space:pre-wrap;color:#49566c}</style>',
             '<h1>父6250 与 Stage3 m1 / m15 · full 原图比较</h1>',
             f'<p>当前共同可用病例：{", ".join(map(str, cases))}。{provenance} 本页只使用 full 渲染 PNG。</p>',
             '<p>总览为原图1/8缩略；ROI保留1:1原像素，仅整数裁剪和拼接。测试参考真值不可用，清晰感、生成纹理或NR提高均不能证明真实细节恢复。</p>',
             '<p>' + link(OUT / 'preview_manifest.json', '源SHA、坐标与缩略比例') + ' · ' + link(OUT / 'download_state.json', '下载与full来源状态') + ' · ' + link(OUT / 'selected_per_image_metrics.json', '已有逐图指标（不是full均值）') + '</p>']
    notes = OUT / 'visual_observations_zh.txt'
    if notes.exists():
        parts.append('<p class="note">' + html.escape(notes.read_text()) + '</p>')
    parts.append('<table><tr><th>病例</th><th>原输入</th><th>父6250 原PNG</th><th>m1 原PNG</th><th>m15 原PNG</th></tr>')
    for case in cases:
        cells = [str(case), link(BASE / 'visual_checks/input' / f'case{case}.jpg', '输入JPG')]
        cells += [link(sources[f'{label}/{case}']['local_path'], name) for label, name in zip(LABELS, NAMES)]
        parts.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    parts.extend(['</table>', '<p>下表只列对应 full 推理序列的逐图 NR，不计算或推断全量平均；“待评分”表示本次快照尚无该逐图记录。</p>', '<table><tr><th>病例</th><th>父6250 逐图NR</th><th>m1 逐图NR</th><th>m15 逐图NR</th></tr>'])
    for case in cases:
        values = [f"{metrics[label][case]['nr_per_image']:.8f}" if case in metrics[label] else '待评分' for label in LABELS]
        parts.append('<tr><td>' + str(case) + '</td>' + ''.join(f'<td>{value}</td>' for value in values) + '</tr>')
    parts.extend(['</table>', '<figure><figcaption>全图总览 · 1/8 缩略</figcaption><a href="overview_full_comparison.png"><img class="overview" src="overview_full_comparison.png"></a></figure>'])
    for case in cases:
        filename = f'roi_case{case}_native_1to1.png'
        parts.append(f'<figure><figcaption>case{case} · 1:1原像素ROI · {BOXES[case]}</figcaption><div class="native"><img src="{filename}"></div></figure>')
    parts.append('</html>')
    (OUT / 'index.html').write_text('\n'.join(parts))
    print(json.dumps({'cases': cases, 'final_manifests_verified': state['all_final_manifests_verified'],
                      'index': str(OUT / 'index.html'), 'native_rois': len(cases)}))


if __name__ == '__main__':
    main()
