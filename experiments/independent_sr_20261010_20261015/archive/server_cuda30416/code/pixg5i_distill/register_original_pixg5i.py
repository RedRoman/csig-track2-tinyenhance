"""Exclusive CPU registration of supplied pixG5i baseline; no inference/training."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013')
FILES={'runner.py':'6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e',
       'pix_model.py':'2f1cd3cd8311381e432138f489b01b982e08ebd5b09ecbc05ac87f7bfb577923',
       'model.pth':'f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487',
       'config.yaml':'e13bab32185ac95f07ac36133eff89133a33ed90cc5ca534b7d8496bc457250d'}
ADAPTER_SHA='70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
ARCHIVE_SHA='f8e154da5ea9cb2fe165a316f824003459661945e6bbe22efd025d8592d573d6'
def need(x,m):
    if not x:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    src=R/'source_payload/pixg5i_original/model_dir';adapter=Path(__file__).with_name('pixg5i_protocol_adapter.py')
    scope_path=R/'experiments/pixg5i_hypir_distill_20261015/USER_SCOPE.json';scope=json.loads(scope_path.read_text())
    need(sha(adapter)==ADAPTER_SHA and all(sha(src/k)==h for k,h in FILES.items()),'Original pixG5i package/adapter changed')
    need(scope['student_original_model_SHA256']==FILES['model.pth'] and scope['teacher_fixed'] is True and
         scope['teacher_raw_sha256']=='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c','Fixed user scope/teacher differs')
    md=R/'submissions/pixg5i_original_20261015/model_dir';need(not md.parent.exists(),'Never overwrite a registered baseline')
    binding=dict(schema='pixg5i_hypir_distill_binding_v1',label='pixg5i_original_20261015',stage='original_baseline',
        checkpoint_sha256=FILES['model.pth'],initial_package_checkpoint_sha256=FILES['model.pth'],archive_sha256=ARCHIVE_SHA,
        training_provenance=None,used_as_G4_reference=False,teacher=dict(label=scope['teacher'],raw_sha256=scope['teacher_raw_sha256'],
            outputs=scope['teacher_outputs'],applied_to_this_checkpoint=False),user_scope=ref(scope_path))
    plan=dict(status='CPU_PLAN_ONLY',model_dir=str(md),binding=binding,original_source_files={k:ref(src/k) for k in FILES},adapter=ref(adapter))
    if a.execute:
        md.parent.mkdir(exist_ok=False);md.mkdir()
        for old,new in [('runner.py','reference_runner.py'),('pix_model.py','pix_model.py'),('model.pth','model.pth'),('config.yaml','config.yaml')]:
            with (src/old).open('rb') as x,(md/new).open('xb') as y:shutil.copyfileobj(x,y)
            need(sha(md/new)==FILES[old],'Copied original bytes differ')
        with adapter.open('rb') as x,(md/'runner.py').open('xb') as y:shutil.copyfileobj(x,y)
        write(md/'pixg5i_binding.json',binding)
        cm=dict(schema='pixg5i_registered_candidate_v1',label=binding['label'],model_dir=str(md),stage=binding['stage'],
            model_type='supplied_pixG5i_pixel_UNet',checkpoint=str(md/'model.pth'),checkpoint_sha256=FILES['model.pth'],
            initial_package_checkpoint_sha256=FILES['model.pth'],archive_sha256=ARCHIVE_SHA,runner_sha256=sha(md/'runner.py'),
            binding=ref(md/'pixg5i_binding.json'),files={x.name:ref(x) for x in md.iterdir() if x.is_file()},
            training_provenance=None,teacher=binding['teacher'],quality_reuse_claim=False,used_as_G4_reference=False,
            registration_source=ref(__file__),CPU_only=True,GPU_used=False,original_inference_source_unchanged=True)
        write(md.parent/'candidate_manifest.json',cm)
        plan=dict(status='REGISTERED_ORIGINAL_PIXG5I_CPU_ONLY',model_dir=str(md),candidate_manifest=ref(md.parent/'candidate_manifest.json'),
            label=binding['label'],checkpoint_sha256=FILES['model.pth'],runner_sha256=ADAPTER_SHA,GPU_used=False)
    print(json.dumps(plan,ensure_ascii=False))
if __name__=='__main__':main()
