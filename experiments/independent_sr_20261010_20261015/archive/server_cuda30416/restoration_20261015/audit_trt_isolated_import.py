"""CPU-only dependency/import receipt for the restored isolated TRT site.

No engine deserialization, builder, inference, CUDA visibility, or lazy CUDA init.
"""
import hashlib, importlib.metadata as md, json, os, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
R = Path('/cache/independent_sr_cuda_20261013')
SITE = R/'backends/push7_trt/site'
OUT = R/'restoration_20261015'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def need(ok, message):
    if not ok: raise RuntimeError(message)

def distributions():
    return sorted([dict(name=d.metadata.get('Name'), version=d.version, path=str(d._path))
                   for d in md.distributions()], key=lambda x: (x['name'] or '', x['path']))

before = json.loads((OUT/'trt_main_environment_before.json').read_text())
need(distributions() == before['distributions'], 'Main environment changed during target installation')
target = list(md.distributions(path=[str(SITE)]))
versions = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in target}
expected = dict(zip(['torch-tensorrt', 'tensorrt', 'tensorrt-cu12', 'tensorrt-cu12-bindings',
                     'tensorrt-cu12-libs', 'dllist'], ['2.8.0']+['10.12.0.36']*4+['2.0.0']))
need(versions == expected, 'Unexpected target distribution/version; Torch must stay outside target')
need(not (SITE/'torch').exists(), 'Target must not contain another Torch')
import torch
need(torch.__version__ == '2.8.0+cu128' and not torch.cuda.is_initialized(), 'Original Torch/no-CUDA state differs')
cuda_attempts = []
def blocked_cuda_init(*args, **kwargs):
    cuda_attempts.append('blocked_cuda_initialization')
    raise RuntimeError('CUDA initialization forbidden during CPU-only package import')
old_lazy = torch.cuda._lazy_init; old_cinit = torch._C._cuda_init
torch.cuda._lazy_init = blocked_cuda_init; torch._C._cuda_init = blocked_cuda_init
sys.path.insert(0, str(SITE))
try:
    import tensorrt
    import torch_tensorrt
    need(tensorrt.__version__ == '10.12.0.36' and torch_tensorrt.__version__ == '2.8.0', 'Imported TRT versions differ')
    need(not cuda_attempts and not torch.cuda.is_initialized(), 'Import attempted CUDA initialization')
    modules = {'torch': torch, 'tensorrt': tensorrt, 'torch_tensorrt': torch_tensorrt}
    module_sources = {name: dict(path=str(Path(m.__file__).resolve()), sha256=sha(m.__file__)) for name, m in modules.items()}
    need(all(Path(module_sources[n]['path']).is_relative_to(SITE) for n in ['tensorrt', 'torch_tensorrt']), 'TRT import escaped isolated site')
    need(not Path(torch.__file__).resolve().is_relative_to(SITE), 'Torch came from isolated target')
finally:
    torch.cuda._lazy_init = old_lazy; torch._C._cuda_init = old_cinit
    sys.path.remove(str(SITE))
need(distributions() == before['distributions'], 'Main environment changed after imports')
sources = []
for p in sorted((OUT/'trt_wheels').glob('*.whl')):
    sources.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
installed = []
for d in target:
    p = Path(d._path)
    installed.append(dict(name=d.metadata['Name'], version=d.version, metadata_path=str(p),
                          METADATA_sha256=sha(p/'METADATA'), RECORD_sha256=sha(p/'RECORD')))
libraries = []
for p in sorted(SITE.rglob('*')):
    if p.is_file() and ('.so' in p.name):
        libraries.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
receipt = dict(status='PASS_ISOLATED_TRT_CPU_IMPORT_ONLY', site=str(SITE), versions=versions,
    python=sys.version, source_wheels=sources, installed_metadata=installed, imported_modules=module_sources,
    installed_shared_libraries=libraries, main_environment_before_sha256=sha(OUT/'trt_main_environment_before.json'),
    main_environment_distribution_paths_and_versions_unchanged=True, main_Torch_unchanged=True,
    torch_version=torch.__version__, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
    torch_CUDA_initialized=False, blocked_CUDA_initialization_attempts=cuda_attempts,
    GPU_used=False, engines_deserialized=False, model_build=False, inference=False, scoring=False,
    persistent_temp_written=False, script_sha256=sha(__file__))
out = OUT/'trt_isolated_import_receipt.json'
need(not out.exists(), 'Final CPU import receipt already exists; never overwrite')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'], versions=versions, receipt=str(out), receipt_sha256=sha(out),
                     main_environment_unchanged=True, CUDA_initialized=False)))
