"""Independent read-only SFS full-byte verification; receipt is post-commit only.

Script/log/progress live outside the frozen experiment root. No tensor libraries,
model imports, GPU launches, archived-file changes, deletion or large downloads.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import io
import json
import shutil
import subprocess
import time

R = Path('/cache/independent_sr_cuda_20261013')
D = Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push3_20261014')
MANIFEST_SHA = '6c4cc6db3d963762cb1868d83d89969083326cc0e3932d64743a23b598c48eaa'
SELECTION_SHA = 'dc41fefc4077b6b157e310065d5017dfc8c0c2fa791e943543abd97a9a91d6af'
SOURCE_HELPER_SHA = '60fbeae585b4fcb6475f728477166f868880d4dd4707d8ce5695702a718156b7'
ROOT_EXIT = Path('/cache/push3_backup_root_exit.json')
ROOT_LOG = Path('/cache/nr_push3_backup_execute.log')
OUTPUT = R / 'results/nr_push3_backup_receipt_20261014.json'
PRIOR = [
    {'directory': '/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014', 'marker': 'DONE',
     'manifest_sha256': 'ef284432a338b0a4574f52be645f4a40571ef94e775e1e39a2c0d1cfc1d13e8d',
     'DONE_sha256': '0a9cb512123ca402baaf43e21db2204853b2033fbcb61a1739f802306b36703f'},
    {'directory': '/temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014', 'marker': 'DONE.json',
     'manifest_sha256': '99f461e6e95045cf0af053dc3b24bde74cd3531715225d0ccf95eef2d2dbd0d4',
     'DONE_sha256': '92cadf2e81aba1864e027e1fded512c008b15585018e1c1eda7d9658b45b8d12'},
]


def need(ok, message):
    if not ok:raise RuntimeError(message)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):h.update(block)
    return h.hexdigest()


def safe(relative):
    p = Path(relative)
    need(not p.is_absolute() and '..' not in p.parts, 'Unsafe manifest relative path')
    q = D / p
    need(q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(D), 'Missing/symlinked/escaped archive file')
    return q


def previous_heads():
    out = []
    for row in PRIOR:
        p = Path(row['directory']); m = p / 'manifest.json'; marker = p / row['marker']
        ms, ds = sha(m), sha(marker)
        need(ms == row['manifest_sha256'] and ds == row['DONE_sha256'], 'Previous immutable backup head changed')
        done = read(marker);need(done['status'] == 'complete' and done['manifest_sha256'] == ms, 'Previous marker checksum inconsistency')
        out.append(dict(row, reverified_manifest_and_DONE_byte_SHA=True))
    return out


def gpu_process_evidence():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = [r for r in csv.reader(io.StringIO(raw)) if len(r) >= 3]
    apps = [{'uuid': r[0].strip(), 'pid': r[1].strip(), 'process': ','.join(r[2:]).strip()} for r in rows]
    ps = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    names = ['train_cuda_nr_stage_v4.py', 'render_cuda_submission.py', 'score_cuda_stream.py', 'benchmark_cuda.py',
             'run_push3_graph_probe', 'probe_push3_graph']
    own = [line.strip() for line in ps.splitlines()
           if str(R) in line and any(name in line for name in names)
           and 'ps -eo' not in line]
    need(not own, 'This round still has a GPU training/render/scoring/benchmark process')
    return {'NVML_visible_compute_processes': apps, 'all_GPUs_have_no_visible_compute_processes': not apps,
            'this_round_GPU_command_lines': own, 'this_round_GPU_processes_absent': True,
            'method': 'read-only NVML process query plus ps command inspection; no CUDA context/model imported'}


def main():
    started = now(); print(json.dumps({'independent_verify_started_utc': started}), flush=True)
    deadline = time.monotonic() + 900
    while not (D / 'DONE').is_file() or not ROOT_EXIT.is_file():
        need(time.monotonic() < deadline, 'Timed out waiting for complete DONE and root exit marker')
        time.sleep(15)
    root_exit = read(ROOT_EXIT)
    need(root_exit['exit_code'] == 0 and root_exit['session_id'] == 88655, 'Root session did not confirm exit0')
    need(sha(D / 'manifest.json') == MANIFEST_SHA, 'Final manifest SHA differs from root commit')
    done = read(D / 'DONE'); done_sha = sha(D / 'DONE'); manifest = read(D / 'manifest.json')
    need(done['status'] == manifest['status'] == 'complete' and done['manifest_sha256'] == MANIFEST_SHA, 'Manifest/DONE commit mismatch')
    need(done['selection_sha256'] == manifest['selection_sha256'] == SELECTION_SHA, 'Commit selection differs')
    need(done['scope'] == 'final_selected_only' and manifest['destination'] == str(D), 'Backup scope/destination differs')
    need(manifest['backup_script_sha256'] == SOURCE_HELPER_SHA, 'Committed helper source differs')
    rows = manifest['files']; index = {r['path']: r for r in rows}
    need(len(rows) == len(index) == done['files'] == manifest['count'], 'Duplicate or inconsistent file count')
    need(sum(r['bytes'] for r in rows) == done['bytes'] == manifest['bytes'], 'Manifest total bytes mismatch')
    total = 0
    for n, record in enumerate(rows, 1):
        path = safe(record['path'])
        need(path.stat().st_size == record['bytes'], 'Actual SFS file size differs: ' + record['path'])
        need(sha(path) == record['sha256'], 'Actual SFS byte SHA differs: ' + record['path'])
        total += record['bytes']
        if n % 50 == 0 or n == len(rows):
            print(json.dumps({'verified_SFS_full_byte_files': n, 'expected_files': len(rows), 'verified_bytes': total}), flush=True)
    selection_path = safe('results/nr_push3_selection.json'); selection = read(selection_path)
    need(sha(selection_path) == sha(R / 'results/nr_push3_selection.json') == SELECTION_SHA, 'Archive/source selection bytes differ')
    need(selection['status'] == 'selected_after_independent_full_audit' and selection['global_step'] == 7250
         and selection['local_step'] == 750 and selection['optimizer_step'] == 1250, 'Selected exact-resume counters differ')
    raw = str(Path(selection['checkpoint']).relative_to(R)); resume = str(Path(selection['resume']).relative_to(R))
    archives = [r['path'] for r in rows if Path(r['path']).suffix.lower() in ('.pt', '.pth', '.safetensors', '.ckpt')]
    need(set(archives) == {raw, resume} and len(archives) == 2, 'Backup must contain exactly selected raw and immutable resume tensor archives')
    need(Path(resume).name == 'resume_step7250.pt', 'Mutable/latest resume included instead of exact immutable milestone')
    need(index[raw]['sha256'] == selection['checkpoint_sha256']
         and index[resume]['sha256'] == selection['resume_sha256'], 'Selected archive SHA differs from selection')
    q = str(Path(selection['quality_dir']).relative_to(R)); render = read(safe(q + '/render_manifest.json'))
    png_records = [r for r in rows if r['path'].endswith('.png')]
    png_paths = {str(Path(q) / rec['output']) for rec in render['records']}
    need(len(png_records) == len(png_paths) == len(render['records']) == 103
         and {r['path'] for r in png_records} == png_paths, 'Native103PNG scope differs or other full outputs included')
    need([(r['split'], r['case']) for r in render['records']] == [('val', i) for i in range(1, 4)] + [('test', i) for i in range(1, 101)], 'PNG split/count/order differs')
    for rec in render['records']:
        need(index[str(Path(q) / rec['output'])]['sha256'] == rec['sha256'], 'Archived PNG/render-manifest SHA mismatch')
    result = read(safe(q + '/scores/result.json'))
    measurements = [json.loads(line) for line in safe(q + '/scores/measurements.jsonl').read_text().splitlines() if line.strip()]
    need(result['status'] == 'complete' and result['completed'] == result['expected'] == 103
         and len(measurements) == 103 and sum(len(row['metrics']) for row in measurements) == 418, 'Complete full scores missing')
    for row, rec in zip(measurements, render['records']):need(row['sr_sha256'] == rec['sha256'], 'Archived scored PNG SHA differs')
    audit_relative = str(Path(selection['training_audit']).relative_to(R)); train_audit = read(safe(audit_relative))
    need(index[audit_relative]['sha256'] == selection['training_audit_sha256'] and train_audit['status'] == 'PASS', 'Selected independent CPU training audit missing/different')
    run_name = Path(selection['config']).parent.name
    branch = [b for b in train_audit['branches'] if b['run'] == run_name and b['audited_global_step'] == 7250]
    need(len(branch) == 1 and branch[0]['final_weight_exact_equal_resume_FP32_LoRA'] is True
         and branch[0]['LoRA']['keys'] == 514 and branch[0]['LoRA']['all_finite'] is True
         and branch[0]['Adam']['state_entries'] == 514 and branch[0]['Adam']['all_steps'] == [1250]
         and branch[0]['Adam']['all_moments_FP32_CPU_finite'] is True
         and set(branch[0]['final_rng']['state_sha256']) == {'python', 'numpy', 'cpu', 'cuda'}, 'Archived exact514/Adam1250/four-RNG evidence incomplete')
    report_relative = 'reports/NR_PUSH3_REPORT_20261014.md'; report_sha = sha(safe(report_relative))
    need(report_sha == sha(R / report_relative) == index[report_relative]['sha256'], 'Archived/source final report bytes differ')
    plan_path = R / 'nr_push3_20261014/preparation/selected_backup_plan.json'; plan = read(plan_path)
    need(plan['selection_sha256'] == SELECTION_SHA and plan['files'] == manifest['files']
         and plan['count'] == manifest['count'] and plan['bytes'] == manifest['bytes'], 'Reviewed selection plan differs from final committed payload')
    prior = previous_heads(); gpu = gpu_process_evidence()
    log_lines = ROOT_LOG.read_text().splitlines(); commits = []
    for line in log_lines:
        try:value = json.loads(line)
        except json.JSONDecodeError:continue
        if 'commit_complete' in value:commits.append(value)
    need(commits and commits[-1]['commit_complete'] == done, 'Root successful commit log does not match final DONE')
    need(sha(D / 'manifest.json') == MANIFEST_SHA and sha(D / 'DONE') == done_sha, 'Committed metadata changed during full-byte verification')
    free = shutil.disk_usage(D).free
    need(free >= done['self_chosen_reserve_bytes'], 'Actual remaining SFS free below recorded self-chosen reserve')
    receipt = {'schema': 'nr_push3_independent_final_backup_receipt_v1', 'status': 'verified_complete',
               'verified_utc': now(), 'verification_started_utc': started, 'CPU_only': True,
               'destination': str(D), 'manifest_sha256': MANIFEST_SHA, 'DONE_sha256': done_sha,
               'files': len(rows), 'bytes': total, 'scope': 'final_selected_only',
               'all_file_actual_SFS_bytes_full_SHA_and_size_reread': True,
               'all_file_reread_files': len(rows), 'all_file_reread_bytes': total,
               'selected_global_step': 7250, 'selected_local_step': 750, 'selected_optimizer_step': 1250,
               'selected_raw_path': raw, 'selected_raw_sha256': index[raw]['sha256'],
               'selected_immutable_resume_path': resume, 'selected_immutable_resume_sha256': index[resume]['sha256'],
               'tensor_archive_count': 2, 'native_full_PNG_count': 103,
               'native_full_PNG_bytes': sum(r['bytes'] for r in png_records), 'full_metric_rows': 103,
               'full_metric_scalars': 418, 'archived_exact514_FP32_Adam1250_and_four_RNG_audit_PASS': True,
               'previous_backup_heads_reverified': prior, 'SFS_free_bytes_at_independent_verify': free,
               'self_chosen_reserve_bytes': done['self_chosen_reserve_bytes'], 'reserve_is_user_requirement': False,
               'report_sha256': report_sha, 'report_source_bytes_match_archive': True,
               'selection_sha256': SELECTION_SHA, 'selection_source_bytes_match_archive': True,
               'selected_backup_plan_sha256': sha(plan_path), 'reviewed_plan_files_equal_committed_manifest': True,
               'root_backup_exit': root_exit, 'root_backup_exit_marker_sha256': sha(ROOT_EXIT),
               'root_backup_log_sha256': sha(ROOT_LOG), 'root_backup_log_commit_matches_DONE': True,
               'GPU_process_evidence': gpu, 'verification_script': str(Path(__file__).resolve()),
               'verification_script_sha256': sha(__file__),
               'verification_scope': 'Independent complete reread of all destination files after root exit0, then post-commit receipt only. Archived manifest/DONE/report/status and prior prefixes remain read-only.'}
    need(not OUTPUT.exists(), 'Independent receipt already exists; do not overwrite')
    with OUTPUT.open('x') as stream:json.dump(receipt, stream, ensure_ascii=False, indent=2);stream.write('\n')
    need(read(OUTPUT) == receipt, 'Post-commit receipt readback failed')
    print(json.dumps({'status': 'verified_complete', 'receipt': str(OUTPUT), 'receipt_sha256': sha(OUTPUT),
                      'manifest_sha256': MANIFEST_SHA, 'DONE_sha256': done_sha, 'files': len(rows),
                      'bytes': total, 'SFS_free_bytes': free, 'report_sha256': report_sha,
                      'GPU_process_evidence': gpu}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':main()
