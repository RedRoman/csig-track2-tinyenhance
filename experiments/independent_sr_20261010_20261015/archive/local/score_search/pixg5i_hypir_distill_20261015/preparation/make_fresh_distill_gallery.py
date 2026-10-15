"""CPU-only native ROI gallery for actual fresh PixG5i/HYPIR evaluations.

Provide an explicit JSON spec after real labels and hashes exist. This helper
does not register models, execute inference, or infer scores from chosen cases.
No packaged JPEG can be used as a fresh evaluation source.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import re
from PIL import Image, ImageDraw, ImageFont

os.environ['CUDA_VISIBLE_DEVICES'] = ''
BOXES = {7: (1856, 1408, 2496, 1856), 16: (1280, 1216, 1920, 1728),
         41: (768, 1664, 1408, 2176), 83: (1216, 1920, 1856, 2432)}
ORDER = [('val', n) for n in range(1, 4)] + [('test', n) for n in range(1, 101)]
SEEN = {}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    st = path.stat()
    stamp = (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev)
    if path in SEEN:
        need(SEEN[path][0] == stamp, 'Changed source: ' + str(path))
        return SEEN[path][1]
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    st = path.stat()
    need(stamp == (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev), 'Unstable read')
    SEEN[path] = (stamp, h.hexdigest())
    return h.hexdigest()


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path), bytes=Path(path).stat().st_size)


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot(column):
    directory = Path(column['evaluation_dir']).resolve()
    need(directory.name == column['label'], 'Explicit actual label/path differs')
    need((directory / 'render_complete').is_file(), 'Full render is not complete')
    pfile = directory / 'render_protocol.json'
    mfile = directory / 'render_manifest.json'
    need(sha(pfile) == column['render_protocol_sha256'], 'Actual protocol differs from supplied hash')
    need(sha(mfile) == column['render_manifest_sha256'], 'Actual final manifest differs from supplied hash')
    wrapper = read(pfile)
    p = wrapper['protocol']
    need(wrapper['protocol_id'] == digest(p), 'Protocol content hash differs')
    need(p['cases'] == 'all' and p['warmup'] is False, 'Fresh full103 queue required')
    need([(r['split'], r['case']) for r in p['tasks']] == ORDER, 'Required val3 then test100 order differs')
    rr = p['runner']
    need(rr['checkpoint_sha256'] == column['checkpoint_sha256'], 'Wrong explicit checkpoint')
    need(rr['runner_sha256'] == column['runner_sha256'], 'Wrong explicit renderer runner')
    if column['role'] == 'fixed_HYPIR_teacher':
        need(rr['model_type'].startswith('HYPIR_'), 'Teacher must be the actual fixed HYPIR')
    else:
        need(rr['model_type'] == 'supplied_pixG5i_pixel_UNet' and rr['used_as_G4_reference'] is False,
             'Actual supplied PixG5i family required; this is not G4')
        stage = 'original_baseline' if column['role'] == 'original_pix_baseline' else 'distilled_checkpoint'
        need(rr['stage'] == stage, 'Actual original/distilled PixG5i stage differs')
    md = Path(p['model_dir'])
    need(sha(md / 'runner.py') == column['runner_sha256'], 'Current inference source changed')
    # Each actual model-family binding/provenance file is supplied explicitly;
    # this covers PixG5i binding or HYPIR component manifest without pretending
    # that a metadata record independently verifies an engine's large bytes.
    bindings = []
    need(len(column['binding_files']) >= 1, 'Actual family binding/provenance is required')
    for b in column['binding_files']:
        need(sha(b['path']) == b['sha256'], 'Actual family binding changed')
        bindings.append(record(b['path']))
    for name, value in p['model_dir_sha256'].items():
        path = md / name
        need(path.is_file() and sha(path) == value, 'Render model-directory file changed: ' + name)
    manifest = read(mfile)
    need(manifest['protocol'] == p and manifest['protocol_id'] == wrapper['protocol_id'], 'Final/protocol mismatch')
    rows = manifest['records']
    need([(r['split'], r['case']) for r in rows] == ORDER, 'Incomplete final103 committed records')
    images = []
    for task, row in zip(p['tasks'], rows):
        rel = f'render/{row["split"]}/case{row["case"]}.png'
        need(row['output'] == rel, 'Noncanonical native PNG output')
        png = directory / rel
        need(not png.is_symlink() and sha(png) == row['sha256'], 'Actual committed PNG differs')
        need(sha(task['input']) == task['input_sha256'], 'Actual ordered native input differs')
        images.append(dict(split=row['split'], case=row['case'], PNG=record(png),
                           native_size=row['size'], input_sha256=task['input_sha256']))
    need({str(q.relative_to(directory)) for q in (directory / 'render').rglob('*') if q.is_file()}
         == {r['output'] for r in rows}, 'Extra or missing full render file')
    return dict(label=column['label'], title=column['title'], role=column['role'],
                checkpoint_sha256=rr['checkpoint_sha256'], runner_sha256=rr['runner_sha256'],
                protocol=p, protocol_id=wrapper['protocol_id'], protocol_file=record(pfile),
                final_manifest=record(mfile), bindings=bindings, images=images,
                actual103_PNG_and_input_SHA_verified=True)


def main():
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--spec', type=Path, required=True,
                   help='Actual columns: role/title/label/evaluation_dir/checkpoint_sha256/runner_sha256/'
                        'render_protocol_sha256/render_manifest_sha256/binding_files[{path,sha256}]')
    q.add_argument('--out-dir', type=Path, required=True)
    q.add_argument('--execute', action='store_true')
    a = q.parse_args()
    spec = read(a.spec)
    cols = spec['columns']
    need(len(cols) in (3, 4), 'Baseline + fixed teacher + one/two candidates required')
    need([c['role'] for c in cols[:2]] == ['original_pix_baseline', 'fixed_HYPIR_teacher'], 'Column roles/order differs')
    need(all(c['role'] == 'distilled_pix_candidate' for c in cols[2:]), 'Candidate must be the actual distilled PixG5i')
    for c in cols:
        need(0 < len(c['title']) <= 42, 'Use a concise image column title')
        for key in ['checkpoint_sha256', 'runner_sha256', 'render_protocol_sha256', 'render_manifest_sha256']:
            need(re.fullmatch('[0-9a-f]{64}', c[key]), 'An actual hash is required: ' + key)
    need(cols[0]['checkpoint_sha256'] == 'f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487',
         'Baseline must be the actual supplied original PixG5i checkpoint')
    need(all(cols[0]['checkpoint_sha256'] != c['checkpoint_sha256'] for c in cols[2:]),
         'Baseline weights cannot masquerade as a candidate')
    need(len({c['label'] for c in cols}) == len(cols), 'Evaluation labels must be independent')
    if not a.execute:
        print(json.dumps(dict(status='PLAN_ONLY', columns=cols, cases=list(BOXES),
                              no_inference_or_scoring=True, out=str(a.out_dir))))
        return
    out = a.out_dir.resolve()
    need(not out.exists(), 'New isolated gallery directory required')
    Image.MAX_IMAGE_PIXELS = None
    sources = [snapshot(c) for c in cols]
    # Models differ intentionally; constructor and forward RNG states need not
    # match HYPIR. Only actual inputs, order, original renderer and encodings
    # must agree. Full score audit is independent from this visual helper.
    for source in sources[1:]:
        for key in ['tasks', 'input_encoding', 'output_encoding', 'warmup', 'inference_rng', 'source_sha256', 'torch']:
            need(source['protocol'][key] == sources[0]['protocol'][key], 'Unmatched full input/order/encoding: ' + key)
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
                need(im.mode == 'RGB' and im.format == 'PNG' and list(im.size) == row['native_size'], 'Actual RGB PNG/native size differs')
                need(box[2] <= im.width and box[3] <= im.height, 'Fixed ROI exceeds native image')
                tiles.append(im.crop(box))
        need(len({t.size for t in tiles}) == 1, 'Source ROI shape differs')
        w, h = tiles[0].size
        panel = Image.new('RGB', (16 + len(cols) * (w + 16), h + 86), '#f4f4f4')
        d = ImageDraw.Draw(panel)
        d.text((16, 10), f'case{case} | native 1:1 ROI', font=font, fill='black')
        for i, (c, tile) in enumerate(zip(cols, tiles)):
            x = 16 + i * (w + 16)
            d.text((x, 42), c['title'], font=font, fill='black')
            panel.paste(tile, (x, 70))
        path = out / f'case{case}_fresh_distill_ROI_1to1.png'
        panel.save(path, compress_level=1)
        with Image.open(path) as saved:
            for i, tile in enumerate(tiles):
                x = 16 + i * (w + 16)
                need(saved.crop((x, 70, x + w, 70 + h)).tobytes() == tile.tobytes(), 'Stored crop pixels changed')
        outputs.append(dict(case=case, ROI=list(box), crop_pixels_exact=True, **record(path)))
    page = '<!doctype html><meta charset="utf-8"><title>Fresh PixG5i distillation</title><style>body{font:16px system-ui;margin:24px}figure{overflow:auto;margin:24px 0}img{max-width:none}</style><h1>Fresh original PixG5i / fixed HYPIR / distilled PixG5i</h1><p>All columns use actual committed full103 native PNG outputs with the same ordered inputs. Fixed 1:1 ROIs, no resize, enhancement or filter. Original packaged JPEGs are not used. Different model RNG consumption is allowed and no output RNG equivalence or quality-score gain is inferred from selected cases.</p>'
    for row in outputs:
        name = Path(row['path']).name
        page += f'<figure><figcaption>case{row["case"]} | {html.escape(str(row["ROI"]))}</figcaption><img src="{name}"></figure>'
    page += '<details><summary>Actual input, PNG and source provenance</summary><a href="proof.json">proof.json</a></details>'
    (out / 'index.html').write_text(page)
    for path in list(SEEN):
        sha(path)
    proof = dict(status='PASS_FRESH_FULL_NATIVE_ROI4', CPU_only=True, GPU_used=False,
                 spec=record(a.spec), helper=record(__file__), sources=sources, outputs=outputs,
                 no_resize_filter_or_enhancement=True, packaged_JPEG_used=False,
                 external_HYPIR_raw_and_engine_bytes_rehashed=False,
                 Pix_model_dir_files_including_model_pth_rehashed=True,
                 source_scope='Render source/model-directory byte fingerprints and supplied actual family bindings; independent quality audit verifies model/engine bytes',
                 inference_postRNG_equality_claim=False, score_gain_from_cases_claim=False,
                 HTML=record(out / 'index.html'))
    (out / 'proof.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=proof['status'], out_dir=str(out), proof_sha256=sha(out / 'proof.json'))))


if __name__ == '__main__':
    main()
