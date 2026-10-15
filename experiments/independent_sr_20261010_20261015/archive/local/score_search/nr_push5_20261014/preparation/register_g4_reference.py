"""Register the supplied trained G4 as a labelled reference, without editing it."""
from pathlib import Path
import hashlib
import json
import os

R = Path('/cache/independent_sr_cuda_20261013')
SOURCE = R / 'calibration/g4/model_dir'
TARGET = R / 'submissions/push5_g4_reference/model_dir'
REFERENCE_RUNNER_SHA = '6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'

WRAPPER = '''"""Metadata-only adapter around the supplied trained G4 reference runner."""
from pathlib import Path
import hashlib
import importlib.util
import json
import random
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Runner:
    def __init__(self, modeldir):
        import numpy as np
        import torch
        root = Path(modeldir).resolve()
        random.seed(231); np.random.seed(231); torch.manual_seed(231)
        torch.cuda.manual_seed_all(231)
        name = 'push5_supplied_g4_reference_impl'
        spec = importlib.util.spec_from_file_location(name, root / 'reference_runner.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.impl = module.Runner(str(root))
        self.device = self.impl.device
        self.weight_dtype = self.impl.weight_dtype
        self._path = self.impl._path
        self.protocol = dict(
            schema='push5_supplied_g4_reference_adapter_v1',
            label='push5_g4_reference_full',
            provenance='Original user-supplied scoring-tool trained G4; reference replication, no new training',
            runner_path=self._path,
            checkpoint_sha256=sha(root / 'model.pth'),
            weight_path=str(root / 'model.pth'),
            runner_sha256=sha(root / 'runner.py'),
            source_sha256={str(p.name):sha(p) for p in [root/'reference_runner.py',root/'pix_model.py']},
            input='Native CUDA FP16 [1,3,H,W] [-1,1]',
            output='Native CUDA FP16 [1,3,H,W] [-1,1]; original fold + channels_last + optional compiled graph',
            adapter='Only fixed seed and provenance metadata added; infer delegates unchanged supplied Runner',
            seed=231, stochastic_inference=False,
        )

    def infer(self, x):
        return self.impl.infer(x)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(SOURCE / 'runner.py') == REFERENCE_RUNNER_SHA
    assert not TARGET.exists(), 'Refuse to overwrite an existing registration'
    TARGET.mkdir(parents=True)
    for old, new in [('runner.py','reference_runner.py'), ('pix_model.py','pix_model.py'),
                     ('model.pth','model.pth'), ('config.yaml','config.yaml')]:
        os.link(SOURCE / old, TARGET / new)
    (TARGET / 'runner.py').write_text(WRAPPER)
    proof = {'status':'registered_reference_only','source':str(SOURCE),'model_dir':str(TARGET),
             'new_training':False,'weight_sha256':sha(TARGET/'model.pth'),
             'source_runner_sha256':REFERENCE_RUNNER_SHA,
             'wrapper_sha256':sha(TARGET/'runner.py'),
             'infer_unchanged_delegation':True,'source_files_hardlinked_without_modification':True}
    path = R / 'nr_push5_20261014/preparation/g4_reference_registration.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(proof,ensure_ascii=False))


if __name__ == '__main__':
    main()
