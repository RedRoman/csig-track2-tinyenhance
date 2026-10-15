from pathlib import Path
import datetime, hashlib, json, os, shutil, tarfile

WORK = Path('/Users/hezihao/Documents/New project')
STAGING = WORK / 'csig_experiment_github_staging_20261015'
REPO = WORK / 'csig-track2-tinyenhance'
OUT = REPO / 'experiments/independent_sr_20261010_20261015'
ALLOWED = {'.py','.sh','.md','.json','.yaml','.yml','.csv','.tsv','.txt','.jsonl','.log','.toml'}
SKIP = {'env','.venv','venv','node_modules','site-packages','__pycache__','.git','original_package','source_payload','weights','data','datasets','render','rendered','inputs','input','output','output_dir','deliverables','candidates_local'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    entries=[]
    roots=[('csig_scoring_tool_review_20261013/score_search','score_search'),('independent_sr_round2_review_20261011','independent_sr_round2_review_20261011'),('independent_sr_ascend_review_20261013','independent_sr_ascend_review_20261013')]
    for source, name in roots:
        root=WORK/source
        for directory, dirs, names in os.walk(root):
            dirs[:]=[s for s in dirs if s not in SKIP and not s.startswith('backup') and not (Path(directory)/s).is_symlink()]
            for name0 in names:
                p=Path(directory)/name0
                if p.is_symlink() or not p.is_file() or p.suffix.lower() not in ALLOWED: continue
                size=p.stat().st_size
                if size > 8*1024**2: raise RuntimeError(f'Unexpected large metadata: {p}')
                q=OUT/'archive/local'/name/p.relative_to(root); q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
                assert sha(p)==sha(q)
                entries.append({'source':str(p),'archive_path':str(q.relative_to(OUT)),'bytes':size,'sha256':sha(q)})
    # Preserve every relative visual link in the original full Chinese report.
    root=WORK/'independent_sr_round2_review_20261011/visuals'
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in {'.jpg','.png'}:
            q=OUT/'archive/local/independent_sr_round2_review_20261011/visuals'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            entries.append({'source':str(p),'archive_path':str(q.relative_to(OUT)),'bytes':q.stat().st_size,'sha256':sha(q)})
    archive=STAGING/'server_metadata.tar.gz'
    assert sha(archive)=='1e10c53568b261f3ffc15f6e0567d596c137ee3be01af8b2358f138f8aa07da8'
    dest=OUT/'archive/server_cuda30416';dest.mkdir(parents=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            rel=Path(member.name)
            assert not rel.is_absolute() and '..' not in rel.parts and member.isfile()
        tf.extractall(dest,filter='data')
    mf=json.loads((dest/'SERVER_METADATA_MANIFEST.json').read_text())
    for row in mf['files']:
        p=dest/row['relative_path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    # Exact model-location and small backup receipts, without large payloads.
    D=WORK/'csig_scoring_tool_review_20261013/score_search/pixg5i_hypir_distill_20261015'
    for root in D.glob('backup*_verified'):
        for p in root.glob('*.json'):
            q=OUT/'archive/backup_receipts'/root.name/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            entries.append({'source':str(p),'archive_path':str(q.relative_to(OUT)),'bytes':q.stat().st_size,'sha256':sha(q)})
    selected=D/'visuals/coverage_v4_s200_vs_medium300_teacher_diagnostic'
    for p in selected.iterdir():
        if p.is_file() and p.suffix.lower() in {'.png','.json','.html'}:
            q=OUT/'visuals/pixg5i_latest_vs_previous_teacher'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            entries.append({'source':str(p),'archive_path':str(q.relative_to(OUT)),'bytes':q.stat().st_size,'sha256':sha(q)})
    p=D/'visuals/coverage_v4_s200_vs_medium300_teacher_full/full_diag_equivalence.json'
    q=OUT/'visuals/pixg5i_latest_vs_previous_teacher'/p.name;shutil.copyfile(p,q)
    entries.append({'source':str(p),'archive_path':str(q.relative_to(OUT)),'bytes':q.stat().st_size,'sha256':sha(q)})
    (OUT/'manifests').mkdir()
    (OUT/'manifests/LOCAL_SOURCE_MANIFEST.json').write_text(json.dumps({'snapshot_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':entries},ensure_ascii=False,indent=2))
    (OUT/'tools').mkdir();shutil.copyfile(Path(__file__),OUT/'tools/build_archive.py')
    print(json.dumps({'out':str(OUT),'local_files':len(entries),'server_files':len(mf['files']),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())},ensure_ascii=False))
if __name__=='__main__': main()
