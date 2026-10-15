"""Authorized short GPU1 import-only validation; no default-device monkeypatch.

No engine deserialization, model construction/build, forward, or scoring.
"""
import hashlib, importlib.metadata as md, json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
R = Path('/cache/independent_sr_cuda_20261013')
SITE = R/'backends/push7_trt/site'; OUT = R/'restoration_20261015'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def need(ok, message):
    if not ok: raise RuntimeError(message)

def distributions():
    return sorted([dict(name=d.metadata.get('Name'), version=d.version, path=str(d._path))
        for d in md.distributions()], key=lambda x: (x['name'] or '', x['path']))

need(os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'Only explicitly authorized physical GPU1 import')
before = json.loads((OUT/'trt_main_environment_before.json').read_text())
need(distributions() == before['distributions'], 'Main environment changed during target installation')
target = list(md.distributions(path=[str(SITE)]))
versions = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in target}
expected = dict(zip(['torch-tensorrt', 'tensorrt', 'tensorrt-cu12', 'tensorrt-cu12-bindings',
                    'tensorrt-cu12-libs', 'dllist'], ['2.8.0']+['10.12.0.36']*4+['2.0.0']))
need(versions == expected and not (SITE/'torch').exists(), 'Unexpected target package/version or duplicate Torch')
import torch
need(torch.__version__ == '2.8.0+cu128' and not torch.cuda.is_initialized(), 'Original Torch/pre-import state differs')
sys.path.insert(0, str(SITE))
import tensorrt
import torch_tensorrt
need(tensorrt.__version__ == '10.12.0.36' and torch_tensorrt.__version__ == '2.8.0', 'Imported versions differ')
module_sources = {name: dict(path=str(Path(module.__file__).resolve()), sha256=sha(module.__file__))
    for name, module in [('torch', torch), ('tensorrt', tensorrt), ('torch_tensorrt', torch_tensorrt)]}
need(all(Path(module_sources[n]['path']).is_relative_to(SITE) for n in ['tensorrt', 'torch_tensorrt']) and
     not Path(torch.__file__).resolve().is_relative_to(SITE), 'Package import escaped intended site')
sys.path.remove(str(SITE))
need(distributions() == before['distributions'], 'Main environment changed after imports')
pip_report = json.loads((OUT/'trt_install_report.json').read_text())
installed = []
for d in target:
    p = Path(d._path)
    installed.append(dict(name=d.metadata['Name'], version=d.version, metadata_path=str(p),
        METADATA_sha256=sha(p/'METADATA'), RECORD_sha256=sha(p/'RECORD')))
sources = [dict(name=x['metadata']['name'], version=x['metadata']['version'],
    wheel_url=x['download_info']['url'], wheel_hashes=x['download_info']['archive_info']['hashes'])
    for x in pip_report['install']]
receipt = dict(status='PASS_ISOLATED_TRT_UNMODIFIED_GPU_IMPORT', site=str(SITE), versions=versions,
    python=sys.version, source_wheels_from_actual_pip_install_report=sources, installed_metadata=installed,
    imported_modules=module_sources, main_environment_before_sha256=sha(OUT/'trt_main_environment_before.json'),
    main_environment_distribution_paths_and_versions_unchanged=True, main_Torch_unchanged=True,
    torch_version=torch.__version__, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
    torch_CUDA_initialized_after_upstream_import=torch.cuda.is_initialized(), default_device_functions_monkeypatched=False,
    scope='Explicitly authorized short physicalGPU1/logicalGPU0 package import only; process release on exit',
    model_build=False, engines_deserialized=False, inference=False, scoring=False,
    independent_full_shared_library_rehash=False, persistent_temp_written=False,
    prior_strict_CPU_import_log=str(OUT/'trt_cpu_import.log'), prior_CPU_stub_diagnostic_not_used_for_acceptance=True,
    script_sha256=sha(__file__))
out = OUT/'trt_isolated_real_import_receipt.json'
need(not out.exists(), 'Final real-import receipt already exists; never overwrite')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'], versions=versions, receipt=str(out), receipt_sha256=sha(out),
    main_environment_unchanged=True, physical_visible_device='1', CUDA_initialized=receipt['torch_CUDA_initialized_after_upstream_import'],
    no_build_no_deserialize_no_forward=True)))
