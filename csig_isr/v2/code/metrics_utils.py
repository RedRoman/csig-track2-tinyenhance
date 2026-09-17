"""Unified IQA metric construction and proxy-score calculation."""
from __future__ import annotations
import os
from typing import Dict, Optional
import yaml

from settings import ROOT, DEFAULT_MANIQA


def metric_config_path() -> str:
    local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "metrics.yaml"))
    return os.environ.get("CSIG_METRICS_YAML", local)


def load_metric_names(path: Optional[str] = None) -> Dict[str, str]:
    path = path or metric_config_path()
    names = {
        "lpips": "lpips",
        "dists": "dists",
        "niqe": "niqe",
        "musiq": "musiq",
        "maniqa": DEFAULT_MANIQA,
        "clipiqa": "clipiqa",
    }
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        names.update((cfg.get("metrics") or {}))
    return names


def create_metrics(device, as_loss: bool = False, include_psnr: bool = False,
                   path: Optional[str] = None):
    import pyiqa
    names = load_metric_names(path)
    out = {}
    for key in ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa"):
        kwargs = {"device": device}
        if as_loss and key in ("lpips", "dists", "musiq", "maniqa", "clipiqa"):
            kwargs["as_loss"] = True
        out[key] = pyiqa.create_metric(names[key], **kwargs)
    if include_psnr:
        out["psnr"] = pyiqa.create_metric(
            "psnr", device=device, test_y_channel=True, color_space="ycbcr")
    return out


def proxy_score(m: Dict[str, float]) -> float:
    return float((1.0 - m["lpips"]) + (1.0 - m["dists"]) + m["clipiqa"]
                 + m["maniqa"] + m["musiq"] / 100.0
                 + max(0.0, (10.0 - m["niqe"]) / 10.0))


def nr_proxy_contrib(m: Dict[str, float]) -> float:
    return float(m["clipiqa"] + m["maniqa"] + m["musiq"] / 100.0
                 + max(0.0, (10.0 - m["niqe"]) / 10.0))
