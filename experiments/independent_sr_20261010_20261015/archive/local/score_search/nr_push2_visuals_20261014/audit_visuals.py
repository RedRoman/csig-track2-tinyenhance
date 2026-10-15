"""Final full-manifest, source SHA and native-ROI audit; no image modification."""
from pathlib import Path
from PIL import Image, ImageChops
from html.parser import HTMLParser
import datetime
import hashlib
import json
import shutil

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    state = json.loads((OUT / 'download_state.json').read_text())
    assert state['all_4_cases_downloaded'] and state['all_final_manifests_verified']
    manifest = json.loads((OUT / 'preview_manifest.json').read_text())
    assert manifest['cases_available_in_all_models'] == [7, 16, 41, 83]
    assert manifest['all_final_manifests_verified']
    sources = {row['id']: row for row in manifest['source_files']}
    assert len(sources) == 12
    parent_manifest = OUT.parent / 'nr_recheck_visual_20261014/provenance/nr_s6250_render_manifest.json'
    parent_copy = OUT / 'provenance/parent_s6250_final_render_manifest.json'
    shutil.copyfile(parent_manifest, parent_copy)
    assert sha(parent_manifest) == sha(parent_copy)
    final_manifests = {}
    original_checks = []
    for key, source in sources.items():
        assert source['source_kind'] == 'final_full_render_manifest'
        assert source['final_manifest_verified']
        provenance = Path(source['manifest_path'])
        model_manifest = json.loads(provenance.read_text())
        assert model_manifest['scope'] == 'test100+validation3'
        assert len(model_manifest['records']) == 103
        expected = next(row for row in model_manifest['records']
                        if row['split'] == 'test' and row['case'] == source['case'])
        actual_sha = sha(source['local_path'])
        with Image.open(source['local_path']) as image:
            size = list(image.size)
        assert actual_sha == expected['sha256'] == source['sha256']
        assert size == expected['size'] == source['size_wh']
        assert sorted(size) == [3072, 4096]
        if 'state_file_sha256' in source:
            assert sha(provenance) == source['state_file_sha256']
        final_manifests[source['label']] = {'path': str(provenance), 'sha256': sha(provenance),
                                           'scope': model_manifest['scope'], 'record_count': 103,
                                           'protocol_id': model_manifest['protocol_id']}
        original_checks.append({'source_id': key, 'local_path': source['local_path'],
                                'remote_path': source['remote_path'], 'actual_sha256': actual_sha,
                                'expected_final_manifest_sha256': expected['sha256'],
                                'size_wh': size, 'sha_and_dimensions_match': True})
    roi_checks = []
    for preview in manifest['previews']:
        if preview['type'] != 'native_roi':
            continue
        assert preview['scale_xy'] == [1, 1] and preview['resampling'] is None
        with Image.open(sources[preview['source']]['local_path']) as image:
            expected = image.convert('RGB').crop(preview['box_xyxy_exclusive'])
        x, y = preview['paste_xy']
        with Image.open(preview['output_path']) as image:
            actual = image.crop((x, y, x + expected.width, y + expected.height))
        assert ImageChops.difference(expected, actual).getbbox() is None
        roi_checks.append({'source_id': preview['source'], 'box_xyxy_exclusive': preview['box_xyxy_exclusive'],
                           'output_path': preview['output_path'], 'paste_xy': [x, y],
                           'exact_source_pixel_match': True, 'resampling': None})
    assert len(roi_checks) == 12
    for row in manifest['outputs']:
        assert sha(row['path']) == row['sha256']

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('href', 'src'):
                    assert (OUT / value).exists(), value
    Links().feed((OUT / 'index.html').read_text())
    for source in sources.values():
        assert sha(source['local_path']) == source['sha256']
    snapshots = sorted((OUT / 'provenance').glob('remote_full_snapshot_*.json'))
    latest = snapshots[-1]
    last = json.loads(latest.read_text())
    assert all(item['source_kind'] == 'final_full_render_manifest' for item in last['labels'].values())
    shutil.copyfile(latest, OUT / 'provenance/final_full_snapshot.json')
    audit = {'schema': 'final_full_visual_audit_v1', 'status': 'complete',
             'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'models': 3, 'cases': [7, 16, 41, 83], 'original_png_count': len(original_checks),
             'original_png_checks': original_checks, 'final_manifests': final_manifests,
             'native_roi_pane_count': len(roi_checks), 'native_roi_checks': roi_checks,
             'all_original_png_sha_unchanged_after_previews': True,
             'all_html_links_exist': True,
             'overview_scale_xy': [0.125, 0.125], 'overview_resampling': 'Lanczos thumbnails only',
             'full_render_only_no_pilot_pngs': True,
             'metrics_scope': 'Only available per-image metrics from full sequences; no full mean NR inferred',
             'preview_manifest_sha256': sha(OUT / 'preview_manifest.json'),
             'selected_per_image_metrics_sha256': sha(OUT / 'selected_per_image_metrics.json'),
             'observations_sha256': sha(OUT / 'visual_observations_zh.txt')}
    (OUT / 'final_visual_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'complete', 'original_png_sha_dimension_checks': 12,
                      'native_roi_exact_pixel_checks': 12, 'full_manifest_models': len(final_manifests),
                      'audit': str(OUT / 'final_visual_audit.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
