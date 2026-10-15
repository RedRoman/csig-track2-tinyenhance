"""Seed/provenance adapter; all inference delegates to the unchanged pixG5i package."""
from pathlib import Path
import hashlib,importlib.util,json,random,sys
sys.dont_write_bytecode=True

REFERENCE_RUNNER='6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'
PIX_MODEL='2f1cd3cd8311381e432138f489b01b982e08ebd5b09ecbc05ac87f7bfb577923'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(8*1024**2),b''):h.update(v)
    return h.hexdigest()

def need(x,m):
    if not x:raise RuntimeError(m)

class Runner:
    def __init__(self,model_dir):
        import torch
        import numpy as np
        md=Path(model_dir).resolve();b=json.loads((md/'pixg5i_binding.json').read_text())
        need(b['schema']=='pixg5i_hypir_distill_binding_v1' and b['initial_package_checkpoint_sha256']==INITIAL,
             'Wrong pixG5i original package identity')
        need(sha(md/'reference_runner.py')==REFERENCE_RUNNER and sha(md/'pix_model.py')==PIX_MODEL and
             sha(md/'model.pth')==b['checkpoint_sha256'],'Actual original inference source/current checkpoint differs')
        need(b['stage'] in ['original_baseline','distilled_checkpoint'] and b['used_as_G4_reference'] is False,
             'Explicit baseline/distilled pixG5i stage required; this is not the G4 reference')
        if b['stage']=='original_baseline':need(b['checkpoint_sha256']==INITIAL and b['training_provenance'] is None,'Baseline must be original unchanged weights')
        else:need(b['checkpoint_sha256']!=INITIAL and b['training_provenance'] is not None,'Actual new training provenance required')
        random.seed(231);np.random.seed(231);torch.manual_seed(231);torch.cuda.manual_seed_all(231)
        name='_pixg5i_unchanged_package_runner'
        s=importlib.util.spec_from_file_location(name,md/'reference_runner.py');m=importlib.util.module_from_spec(s)
        sys.modules[name]=m;s.loader.exec_module(m);self.impl=m.Runner(str(md))
        self.device=self.impl.device;self.weight_dtype=self.impl.weight_dtype;self._path=self.impl._path
        self.protocol=dict(schema='pixg5i_original_inference_metadata_adapter_v1',model_type='supplied_pixG5i_pixel_UNet',
            label=b['label'],stage=b['stage'],runner_path=self._path,runner_sha256=sha(md/'runner.py'),
            checkpoint_sha256=b['checkpoint_sha256'],weight_path=str(md/'model.pth'),repo_path=str(md),
            source_sha256={p.name:sha(p) for p in [md/'reference_runner.py',md/'pix_model.py',md/'pixg5i_binding.json']},
            initial_package_checkpoint_sha256=INITIAL,archive_sha256=b['archive_sha256'],training_provenance=b['training_provenance'],
            teacher=b['teacher'],used_as_G4_reference=False,quality_reuse_claim=False,
            input='original FP16 native [1,3,H,W] [-1,1]',output='original FP16 native [1,3,H,W] [-1,1]',
            inference='Unchanged original fold/FP16/channels_last;512 original compile/graph;native original eager with original OOM tile1024 overlap128 fallback',
            adapter='Only initialization seed and identity metadata; infer delegates unchanged original supplied Runner',
            seed='231 before initialization; continuous inference RNG',inference_RNG_matches_HYPIR_required=False)

    def infer(self,x):return self.impl.infer(x)
