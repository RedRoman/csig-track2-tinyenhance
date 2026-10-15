"""Independent CPU exact514 FP32 torch.lerp audit, without training/GPU."""
import argparse,hashlib,json,os
from pathlib import Path
R=Path('/cache/independent_sr_cuda_20261013')
D=R/'adaptation/runs/hypir_push7_niqe768_parameter_blends'
PRODUCER=R/'code/push7/parameter_blend_teacher.py'
SOURCE='2bb83af6f7cd988b7cbf210aa241f7be48f93337c126ca0813ec4a02e802a2d3'
A='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
N='763950727fe29a009245cd9cd855743a99803fa9fbf6824a62d38994fa8e0650'
EXPECTED={.25:'daa6e782eb219222f7becc6c5d2a26038fa2a7897c90db7736c8ba38c885d993',.5:'3438c5913333729ade0560dde38fa25813932754ddabefffde7529c511cd2f58'}
SEEN={}
def need(x,m):
    if not x:raise RuntimeError(m)
def sha(p):
    p=Path(p).resolve();s=p.stat();st=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
    if p in SEEN:
        need(SEEN[p][0]==st,'Source changed: '+str(p));return SEEN[p][1]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    s=p.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed while hashing');SEEN[p]=(st,h.hexdigest());return h.hexdigest()
def read(p):sha(p);return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--out',type=Path,default=R/'audits/push7/parameter_blends/independent514_cpu.json');args=p.parse_args()
    out=args.out.resolve();need(out.is_relative_to(R/'audits/push7') and not out.exists(),'New immutable audit path required')
    if not args.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',derivation_manifest=str(D/'derivation_manifest.json'),out=str(out),formula='torch.lerp(A_FP32,N100_FP32,alpha)',alphas=[.25,.5],GPU=False)));return
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    doc=read(D/'derivation_manifest.json');need(doc['status']=='TWO_REAL_DERIVED_HYPIR_CHECKPOINTS' and len(doc['candidates'])==2 and sha(PRODUCER)==doc['script_sha256']==SOURCE,'Actual producer/manifest changed')
    first=doc['candidates'][0];ap=Path(first['initial_checkpoint']);np=Path(first['finetuned_checkpoint'])
    need(sha(ap)==A and sha(np)==N,'Actual source LoRA weights changed')
    a=torch.load(ap,map_location='cpu',weights_only=True,mmap=True);n=torch.load(np,map_location='cpu',weights_only=True,mmap=True)
    need(list(a)==list(n) and len(a)==514 and sum(x.numel() for x in a.values())==259474432,'Original514 HYPIR schema changed')
    need(all(a[k].dtype==n[k].dtype==torch.float32 and a[k].shape==n[k].shape and bool(torch.isfinite(a[k]).all()) and bool(torch.isfinite(n[k]).all()) for k in a),'Endpoints must be finite FP32')
    records=[]
    for row in doc['candidates']:
        alpha=row['alpha'];need(alpha in EXPECTED and row['checkpoint_sha256']==EXPECTED[alpha] and row['initial_checkpoint_sha256']==A and row['finetuned_checkpoint_sha256']==N,'Unexpected interpolation identity')
        need(row['recipe']=='torch.lerp(initial_A_FP32,NIQE100_FP32,alpha) on CPU per514 ordered tensors' and row['new_training'] is False and
             row['distillation'] is False and row['lightweight_model'] is False and row['no_val_GT'] is True and row['no_teacher_outputs'] is True and
             row['optimizer_state']=='not applicable: derived parameters, no new Adam/step/resume','Derived parameter lineage mislabeled')
        md=Path(row['model_dir']);need(read(md.parent/'candidate_manifest.json')==row and sha(md/'runner.py')==row['runner_sha256']=='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b','Actual native candidate registration changed')
        need(sha(md/'config.json')==row['config_sha256'] and Path(read(md/'config.json')['weight_path']).resolve()==Path(row['checkpoint']).resolve(),'Derived runtime config differs')
        need(sha(row['training_marker']['path'])==row['training_marker']['sha256'],'Upstream training marker changed')
        marker=read(row['training_marker']['path']);need(marker['raw_checkpoint_sha256']==N and marker['global_step']==7850 and marker['local_step']==marker['optimizer_step']==100 and
             marker['no_val_GT'] is True and marker['no_teacher_outputs'] is True,'Actual upstream NIQE100 lineage differs')
        for path,digest in marker['source_sha256'].items():need(sha(path)==digest,'Actual upstream training source changed')
        need(sha(row['checkpoint'])==row['checkpoint_sha256'],'Actual derived raw SHA changed')
        v=torch.load(row['checkpoint'],map_location='cpu',weights_only=True,mmap=True);need(list(v)==list(a),'Actual derived key order differs')
        same=0;changed=0
        for k in a:
            need(v[k].dtype==torch.float32 and v[k].shape==a[k].shape and bool(torch.isfinite(v[k]).all()),'Derived tensor finite/schema failure')
            expected=torch.lerp(a[k],n[k],alpha)
            need(torch.equal(v[k],expected),'Actual FP32 torch.lerp formula differs: '+k);same+=1
            changed+=not torch.equal(v[k],a[k])
        need(same==changed==row['changed_tensors_from_A']==row['FP32_tensor_count']==514,'Actual interpolation tensor count differs')
        records.append(dict(label=row['label'],alpha=alpha,checkpoint=ref(row['checkpoint']),native_candidate_manifest=ref(md.parent/'candidate_manifest.json'),
             exact_FP32_lerp_tensors=same,changed_tensors_from_A=changed,finite_FP32=True,new_training=False,derived_optimizer_or_resume=False,upstream_training_marker=ref(row['training_marker']['path'])))
        del v
    need(not list(D.glob('*resume*')) and not list(D.glob('*optimizer*')),'Derived optimizer/resume file must not be invented')
    for path,(st,digest) in SEEN.items():
        s=path.stat();need(st==(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),'Source changed before publish')
    proof=dict(status='PASS_EXACT_FP32_LERP514_BOTH',schema='push7_hypir_parameter_interpolation_cpu_v1',CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),
       derivation_manifest=ref(D/'derivation_manifest.json'),producer=ref(PRODUCER),auditor=ref(__file__),initial_checkpoint=ref(ap),finetuned_checkpoint=ref(np),candidates=records,
       formula='Actual CPU torch.lerp(A_FP32,N100_FP32,alpha), not algebraically substituted',quality_or_speed_claim=False,no_fake_derived_optimizer_or_RNG=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=proof['status'],out=str(out),sha256=sha(out),candidates=records)))
if __name__=='__main__':main()

