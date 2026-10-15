"""Append/refresh verified small metadata after the complete NR core archive."""
import json
from pathlib import Path
import shutil
from sync_nr_artifacts import ROOT, DEST, SKIP, digest, save


def main():
    done_path = DEST / 'DONE.json'
    previous = json.loads(done_path.read_text())
    manifest_path = DEST / 'manifest.json'
    assert previous['status'] == 'complete' and digest(manifest_path) == previous['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    records = {record['path']: record for record in manifest['files']}
    refreshed = []
    for name in ['code', 'calibration', 'audits', 'reports', 'results', 'metric_rechecks', 'review_previews/nr_recheck_visual_20261014']:
        folder = ROOT / name
        if not folder.is_dir():
            raise RuntimeError(f'Required metadata/preview directory missing: {folder}')
        for source in sorted(folder.rglob('*')):
            rel = source.relative_to(ROOT)
            if not source.is_file() or any(part in SKIP for part in rel.parts):
                continue
            if source.stat().st_size > 100 * 1024 ** 2:
                raise RuntimeError(f'Metadata unexpectedly large: {rel}')
            target = DEST / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = digest(source)
            with source.open('rb') as reader, target.open('wb') as writer:
                shutil.copyfileobj(reader, writer, 4 * 1024 ** 2)
            assert digest(target) == expected == digest(source), f'Metadata SHA mismatch: {rel}'
            record = dict(path=str(rel), bytes=source.stat().st_size, sha256=expected)
            records[str(rel)] = record
            refreshed.append(record)
    manifest['files'] = [records[key] for key in sorted(records)]
    manifest['count'] = len(records)
    manifest['bytes'] = sum(record['bytes'] for record in records.values())
    manifest['metadata_refresh'] = dict(previous_verified_manifest_sha256=previous['manifest_sha256'], files=refreshed,
                                       scope='Core weights/PNG/data remain the previously SHA-verified copies; every final small metadata file was SHA-verified again.')
    save(manifest_path, manifest)
    result = dict(status='complete', count=manifest['count'], bytes=manifest['bytes'], destination=str(DEST),
                  manifest_sha256=digest(manifest_path), metadata_refreshed_count=len(refreshed),
                  core_verified_count=previous['count'], sfs_free_bytes=shutil.disk_usage(DEST).free)
    save(done_path, result)
    save(ROOT / 'preflight/nr_artifact_backup_progress.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
