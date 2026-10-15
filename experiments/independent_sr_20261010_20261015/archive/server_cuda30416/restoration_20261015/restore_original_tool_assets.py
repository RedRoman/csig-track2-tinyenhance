"""Safely restore original CSIG tool/assets from its byte-verified supplied tar.

CPU filesystem operations only. Refuses conflicting existing files; never writes
to persistent /temp or imports scoring/model packages.
"""
import argparse, hashlib, json, os, shutil, tarfile
from pathlib import Path, PurePosixPath

R = Path('/cache/independent_sr_cuda_20261013')
P = Path('/temp/independent_sr_cuda_20261013_artifacts')
ARCHIVE_SHA = 'c962f2d0c4bfed44e81f9d6ae251b8252f41394e5983828523fac4b983c5e90a'
ARCHIVE_BYTES = 1833751106

def need(ok, message):
    if not ok: raise RuntimeError(message)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def immutable_link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): need(dst.is_file() and sha(dst) == sha(src), 'Conflicting existing file: '+str(dst))
    else:
        need(not dst.is_symlink(), 'Dangling destination symlink: '+str(dst)); os.link(src, dst)

def exact_symlink(dst, target):
    if dst.is_symlink(): need(os.readlink(dst) == target, 'Different existing symlink: '+str(dst))
    else:
        need(not dst.exists(), 'Non-symlink already occupies alias: '+str(dst))
        dst.parent.mkdir(parents=True, exist_ok=True); os.symlink(target, dst)

def main():
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument('--archive', type=Path, default=R/'restoration_20261015/csig_scoring_tool.tar.gz')
    q.add_argument('--execute', action='store_true'); a = q.parse_args()
    need(a.archive.resolve().parent == R/'restoration_20261015', 'Archive must be in new restoration directory')
    if not a.execute:
        print(json.dumps(dict(status='CPU_RESTORE_PLAN_ONLY', archive=str(a.archive), expected_sha256=ARCHIVE_SHA,
            expected_bytes=ARCHIVE_BYTES, root=str(R), persistent_source=str(P), writes_persistent_temp=False,
            scores_or_GPU=False))); return
    need(a.archive.stat().st_size == ARCHIVE_BYTES and sha(a.archive) == ARCHIVE_SHA, 'Transferred archive identity mismatch')
    with tarfile.open(a.archive, 'r:gz') as archive:
        members = archive.getmembers(); names = set(); regular = []
        for m in members:
            rel = PurePosixPath(m.name)
            need(not rel.is_absolute() and rel.parts and rel.parts[0] == 'csig_scoring_tool' and
                 '..' not in rel.parts and '\\' not in m.name and m.name not in names,
                 'Unsafe or duplicate tar member: '+m.name)
            need(m.isfile() or m.isdir(), 'Links/devices/special tar members forbidden: '+m.name)
            need(m.size >= 0, 'Negative member size'); names.add(m.name)
            if m.isfile(): regular.append(m)
        need(len(regular) == 131 and sum(m.size for m in regular) == 2021193287, 'Original archive member inventory differs')
        stage = R/'restoration_20261015/original_archive'; stage.mkdir(exist_ok=True)
        for m in regular:
            dst = stage.joinpath(*PurePosixPath(m.name).parts); dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                need(dst.is_file() and dst.stat().st_size == m.size, 'Conflicting extraction file'); continue
            need(not dst.is_symlink(), 'Extraction symlink forbidden')
            with archive.extractfile(m) as src, dst.open('xb') as target: shutil.copyfileobj(src, target, 8*1024**2)
            os.chmod(dst, 0o644)
    original = stage/'csig_scoring_tool'; tool = R/'scoring_tool'
    tool.mkdir(exist_ok=True)
    for m in regular:
        rel = PurePosixPath(m.name).relative_to('csig_scoring_tool'); src = original.joinpath(*rel.parts)
        if rel.parts[:2] == ('data', '测试集'):
            dst = R/'source_payload/assets/input'/rel.name
        elif rel.parts[:2] == ('data', '验证集'):
            dst = R/'source_payload/scoring_assets/data/验证集'/rel.name
        elif rel.parts[0] == 'weights':
            dst = R/'source_payload/scoring_assets/weights'/Path(*rel.parts[1:])
        else: dst = tool.joinpath(*rel.parts)
        immutable_link(src, dst)
    exact_symlink(R/'source_payload/scoring_assets/data/测试集', '../../assets/input')
    exact_symlink(tool/'data', '../source_payload/scoring_assets/data')
    exact_symlink(tool/'weights', '../source_payload/scoring_assets/weights')
    for name in ['deployment_manifest.json', 'original_tool_provenance_manifest.json', 'archive_data_manifest.json']:
        source = P/'scoring_tool'/name; dest = tool/name
        if dest.exists(): need(sha(dest) == sha(source), 'Conflicting manifest: '+name)
        else: shutil.copyfile(source, dest)
    deployment = json.loads((tool/'deployment_manifest.json').read_text()); checked = []
    for row in deployment['files']:
        dst = tool/row['path']
        need(dst.is_file() and dst.stat().st_size == row['bytes'] and sha(dst) == row['sha256'] and
             str(dst.resolve()) == row['resolved_path'], 'Restored deployment file/hash/path mismatch: '+str(dst))
        checked.append(dict(path=str(dst), resolved_path=str(dst.resolve()), bytes=dst.stat().st_size, sha256=row['sha256']))
    data = json.loads((tool/'archive_data_manifest.json').read_text())
    for row in data:
        dst = tool/row['path']
        need(dst.stat().st_size == row['size'] and sha(dst) == row['sha256'], 'Original data manifest mismatch')
    need(len(data) == 106 and len(list((R/'source_payload/assets/input').glob('case*.jpg'))) == 100,
         'Expected original test100+val6 inventory differs')
    for name, digest in {
        'render_cuda_submission.py':'1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5',
        'score_cuda_stream.py':'20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074',
        'numpy_legacy_compat.py':'87cdcdd5fe12b7824e219ac6a25c8358815d870d2c1f333c161c0cb41afd010f'
    }.items(): need(sha(R/'code'/name) == digest, 'Root-restored frozen code mismatch: '+name)
    receipt = dict(status='PASS_ORIGINAL_TOOL_ASSETS_RESTORED', archive=str(a.archive), archive_bytes=ARCHIVE_BYTES,
        archive_sha256=ARCHIVE_SHA, safe_archive_members=len(members), regular_archive_files=len(regular),
        deployment_manifest_sha256=sha(tool/'deployment_manifest.json'), actual_deployment_files=checked,
        actual_test_inputs=100, actual_validation_files=6, original_metric_weight_files=8,
        restored_payload_bytes=2021193287, file_payload_hardlinked_within_cache=True,
        persistent_temp_written=False, metric_sources_modified=False, GPU_used=False, scoring_performed=False,
        environment_restoration_owned_elsewhere=True, script_sha256=sha(__file__))
    out = R/'restoration_20261015/tool_assets_restore_receipt.json'
    need(not out.exists(), 'Receipt already exists; do not overwrite frozen receipt')
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=receipt['status'], receipt=str(out), receipt_sha256=sha(out),
        actual_test_inputs=100, actual_validation_files=6, original_metric_weight_files=8,
        deployment_checked_files=len(checked), GPU_used=False, scoring_performed=False)))

if __name__ == '__main__': main()
