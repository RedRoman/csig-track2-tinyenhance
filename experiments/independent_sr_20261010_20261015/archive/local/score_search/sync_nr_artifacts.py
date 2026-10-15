"""Verified direct writes to shared SFS; no NFS chmod/rename dependency."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/cache/independent_sr_cuda_20261013')
DEST = Path('/temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014')
SKIP = {'__pycache__', '.git', '_work'}
MIN_FREE = 30 * 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    result = set()
    for name in ['code', 'nr_full_evaluations', 'nr_pilot_evaluations', 'calibration', 'audits', 'reports', 'results',
                 'repos/HYPIR_cuda', 'source_payload/assets/input', 'source_payload/scoring_assets', 'scoring_tool']:
        folder = ROOT / name
        for path in folder.rglob('*'):
            rel = path.relative_to(ROOT)
            if not path.is_file() or any(part in SKIP for part in rel.parts):
                continue
            # Original assets are preserved at their canonical source paths.
            if name == 'scoring_tool' and ('data' in rel.parts or 'weights' in rel.parts):
                continue
            result.add(rel)
    for branch in ['conservative', 'aligned']:
        folder = ROOT / f'adaptation/runs/cuda_nr_s6000_ct175_{branch}_pilot100'
        for path in folder.iterdir():
            if path.is_file() and (path.suffix not in {'.pt', '.pth'} or path.name in {'step6100.pth', 'step6250.pth', 'resume_latest.pt'}):
                result.add(path.relative_to(ROOT))
    result.add(Path('source_payload/adaptation/runs/ascend_balanced/step6000.pth'))
    for folder in (ROOT / 'submissions').glob('hypir_cuda_nr*'):
        for path in folder.rglob('*'):
            if path.is_file() and not any(part in SKIP for part in path.relative_to(ROOT).parts):
                result.add(path.relative_to(ROOT))
    for path in (ROOT / 'logs').glob('*nr*'):
        if path.is_file() and path.name != 'nr_artifact_backup.log':
            result.add(path.relative_to(ROOT))
    return sorted(result)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    done = DEST / 'DONE.json'
    if done.exists():
        done.unlink()
    files = inventory()
    pending_bytes = sum((ROOT / rel).stat().st_size for rel in files if not (DEST / rel).is_file())
    free = shutil.disk_usage(DEST).free
    if free - pending_bytes < MIN_FREE:
        raise RuntimeError(f'Need {pending_bytes} bytes; free {free}; own reserve {MIN_FREE}')
    started = time.time()
    records = []
    for index, rel in enumerate(files, 1):
        source, target = ROOT / rel, DEST / rel
        source_hash = digest(source)
        reused = target.is_file() and target.stat().st_size == source.stat().st_size and digest(target) == source_hash
        if not reused:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as reader, target.open('wb') as writer:
                shutil.copyfileobj(reader, writer, 4 * 1024 ** 2)
            if digest(target) != source_hash or digest(source) != source_hash:
                raise RuntimeError(f'Unstable or corrupted copy: {rel}')
        records.append(dict(path=str(rel), bytes=source.stat().st_size, sha256=source_hash))
        save(ROOT / 'preflight/nr_artifact_backup_progress.json', dict(status='running', completed=index,
             expected=len(files), last_path=str(rel), verified_bytes=sum(r['bytes'] for r in records), reused=reused,
             elapsed_seconds=time.time()-started, destination=str(DEST)))
    manifest = dict(status='complete', source_root=str(ROOT), destination=str(DEST), files=records,
                    count=len(records), bytes=sum(r['bytes'] for r in records), shared_base_model='/temp/zihaohe_stage/baselines/sd21b',
                    exclusions='Intermediate checkpoints6050/6150/6200, env runtime and duplicate scoring symlinks. Selected checkpoints6100/6250 and both exact optimizer/RNG resumes are included.')
    save(DEST / 'manifest.json', manifest)
    result = dict(status='complete', count=len(records), bytes=manifest['bytes'], destination=str(DEST),
                  manifest_sha256=digest(DEST / 'manifest.json'), elapsed_seconds=time.time()-started,
                  sfs_free_bytes=shutil.disk_usage(DEST).free)
    save(done, result)
    save(ROOT / 'preflight/nr_artifact_backup_progress.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
