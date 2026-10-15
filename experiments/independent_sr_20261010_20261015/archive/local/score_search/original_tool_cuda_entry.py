"""Run an unchanged supplied CSIG tool script with NumPy import compatibility."""
from pathlib import Path
import runpy
import sys

import numpy_legacy_compat  # noqa: F401; dtype catalog only, no RNG change


if len(sys.argv) < 2:
    raise SystemExit("Usage: original_tool_cuda_entry.py TOOL_SCRIPT [tool arguments]")

script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
import maniqa_offline_init
maniqa_offline_init.install(script.parent / "weights/torch/hub/checkpoints/ckpt_koniq10k.pt")

# Audit metric construction after the unmodified tool loads its checkpoint.
sys.path.insert(0, str(script.parent))
import csig_metrics as CM
_create_metrics = CM.create_metrics


def audited_create_metrics(*args, **kwargs):
    import json
    import os
    metrics = _create_metrics(*args, **kwargs)
    proof = maniqa_offline_init.verify_final_state(metrics["maniqa"].net)
    audit = os.environ.get("CSIG_INIT_AUDIT_JSON")
    if audit:
        Path(audit).write_text(json.dumps(proof, indent=2) + "\n")
    return metrics


CM.create_metrics = audited_create_metrics
runpy.run_path(str(script), run_name="__main__")
