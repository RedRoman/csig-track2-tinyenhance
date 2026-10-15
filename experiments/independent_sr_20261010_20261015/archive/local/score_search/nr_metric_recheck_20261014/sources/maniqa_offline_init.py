"""Offline initialization for the complete supplied MANIQA KonIQ checkpoint.

Only the redundant ViT pretrained download is suppressed. The original
pyiqa constructor and checkpoint loader, forward, preprocessing, and test
crop count remain unchanged. verify_final_state proves every final MANIQA
parameter/buffer equals the supplied checkpoint before any metric forward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import timm

_CALLS = []
_REFERENCE = None
_WEIGHT_PATH = None
_ORIGINAL = timm.create_model


def _state_manifest(state):
    records = []
    overall = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        payload = value.numpy().tobytes()
        record = {"key": key, "shape": list(value.shape), "dtype": str(value.dtype),
                  "sha256": hashlib.sha256(payload).hexdigest()}
        overall.update(json.dumps(record, sort_keys=True).encode())
        records.append(record)
    return {"keys": sorted(state), "parameters_and_buffers": records,
            "aggregate_sha256": overall.hexdigest()}


def install(weight_path):
    global _REFERENCE, _WEIGHT_PATH
    if _REFERENCE is not None:
        if Path(weight_path).resolve() != _WEIGHT_PATH:
            raise RuntimeError("Offline MANIQA adapter already installed for a different checkpoint")
        return
    _WEIGHT_PATH = Path(weight_path).resolve()
    _REFERENCE = torch.load(_WEIGHT_PATH, map_location="cpu", weights_only=True)
    if "params" in _REFERENCE:
        _REFERENCE = _REFERENCE["params"]

    def create_model(model_name, *args, **kwargs):
        if model_name == "vit_base_patch8_224" and kwargs.get("pretrained") is True:
            kwargs = {**kwargs, "pretrained": False}
            model = _ORIGINAL(model_name, *args, **kwargs)
            before = _state_manifest(model.state_dict())
            reference = {k[4:]: v for k, v in _REFERENCE.items() if k.startswith("vit.")}
            if set(model.state_dict()) != set(reference):
                raise RuntimeError("Supplied MANIQA checkpoint does not completely cover the constructed ViT")
            _CALLS.append({"model_name": model_name, "suppressed_extra_pretrained_download": True,
                           "before_original_checkpoint_load": before,
                           "checkpoint_vit_keyset": sorted(reference),
                           "pretrained_cfg_preprocessing": {
                               k: model.pretrained_cfg.get(k) for k in
                               ("input_size", "mean", "std", "interpolation", "crop_pct")}})
            return model
        return _ORIGINAL(model_name, *args, **kwargs)

    timm.create_model = create_model


def verify_final_state(net):
    if _REFERENCE is None:
        raise RuntimeError("Offline MANIQA adapter has not been installed")
    actual = net.state_dict()
    missing = sorted(set(_REFERENCE) - set(actual))
    unexpected = sorted(set(actual) - set(_REFERENCE))
    if missing or unexpected:
        raise RuntimeError(f"MANIQA full state key mismatch: missing={missing}, unexpected={unexpected}")
    differences = []
    for key, reference in _REFERENCE.items():
        value = actual[key].detach().cpu()
        if value.shape != reference.shape or value.dtype != reference.dtype or not torch.equal(value, reference):
            differences.append(key)
    if differences:
        raise RuntimeError(f"Final MANIQA state differs from supplied checkpoint: {differences}")
    after_vit = _state_manifest(net.vit.state_dict())
    reference_vit = _state_manifest({k[4:]: v for k, v in _REFERENCE.items() if k.startswith("vit.")})
    if after_vit != reference_vit:
        raise RuntimeError("Final ViT parameter/buffer hashes differ from supplied checkpoint")
    source = Path(__file__).resolve()
    weight_sha = hashlib.sha256()
    with _WEIGHT_PATH.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            weight_sha.update(chunk)
    arch_source = Path(__import__(net.__module__, fromlist=["x"]).__file__)
    if "pretrained_cfg" in arch_source.read_text():
        raise RuntimeError("MANIQA source reads pretrained_cfg; preprocessing equivalence needs re-audit")
    return {"adapter_path": str(source), "adapter_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "checkpoint_path": str(_WEIGHT_PATH), "checkpoint_sha256": weight_sha.hexdigest(),
            "before_init_records": _CALLS, "after_final_checkpoint_load": after_vit,
            "checkpoint_vit_reference": reference_vit, "missing_keys": missing,
            "unexpected_keys": unexpected, "different_values_or_dtype_or_shape": differences,
            "full_maniqa_state_keys": len(actual), "all_final_parameters_and_buffers_equal_checkpoint": True,
            "maniqa_source_sha256": hashlib.sha256(arch_source.read_bytes()).hexdigest(),
            "maniqa_source_reads_pretrained_cfg": False, "net_training": net.training,
            "all_parameters_require_grad_false": all(not p.requires_grad for p in net.parameters()),
            "test_sample": int(net.test_sample),
            "scope": "Only extra pretrained download suppressed; original CM/pyiqa forward and 100 crops unchanged."}
