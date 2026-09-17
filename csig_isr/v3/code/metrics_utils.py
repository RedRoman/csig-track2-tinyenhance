"""Single source of truth for CSIG track-2 proxy metrics and aggregation.

Important invariant (v3): every evaluator computes the six metric means first,
then applies ``proxy_score`` exactly once.  Do not average per-crop proxy scores;
the NIQE clamp makes those two operations mathematically different.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import yaml

from settings import DEFAULT_MANIQA

SCORE_KEYS = ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")
FR_KEYS = ("lpips", "dists")
NR_KEYS = ("niqe", "musiq", "maniqa", "clipiqa")


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
    for key in SCORE_KEYS:
        kwargs = {"device": device}
        if as_loss:
            # pyiqa needs as_loss=True for a differentiable graph. Direction is
            # handled explicitly by train_tinysr.py (higher-better metrics are
            # rewards, lower-better metrics are losses).
            kwargs["as_loss"] = True
        out[key] = pyiqa.create_metric(names[key], **kwargs)
    # MANIQA 的 forward 是 random_crop(x,224,crop_num=test_sample)，在 4K 图上只抽
    # test_sample 个 224 小块。真实图差异大，Score 抖动可达 0.011，且共用全局 RNG
    # 会让结果依赖评测顺序。评测端由环境变量提到 100；训练端保持 20，否则 OOM。
    try:
        _net = out["maniqa"].net
        if hasattr(_net, "test_sample"):
            _net.test_sample = int(os.environ.get("CSIG_MANIQA_SAMPLES", 20))
    except Exception:
        pass
    if include_psnr:
        out["psnr"] = pyiqa.create_metric(
            "psnr", device=device, test_y_channel=True, color_space="ycbcr")
    return out


def proxy_score(m: Mapping[str, float]) -> float:
    """Local proxy only; this is not claimed to be the organizer's final formula."""
    return float((1.0 - float(m["lpips"])) + (1.0 - float(m["dists"]))
                 + float(m["clipiqa"]) + float(m["maniqa"])
                 + float(m["musiq"]) / 100.0
                 + max(0.0, (10.0 - float(m["niqe"])) / 10.0))


def nr_proxy_contrib(m: Mapping[str, float]) -> float:
    return float(float(m["clipiqa"]) + float(m["maniqa"])
                 + float(m["musiq"]) / 100.0
                 + max(0.0, (10.0 - float(m["niqe"])) / 10.0))


def mean_metrics(metric_rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    rows = list(metric_rows)
    if not rows:
        raise ValueError("cannot aggregate an empty metric row set")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    return {
        k: float(np.mean([float(r[k]) for r in rows if k in r]))
        for k in keys
        if any(k in r for r in rows)
    }


def _group_summary(rows: List[Mapping], group_key: str):
    grouped = defaultdict(list)
    for row in rows:
        if row.get(group_key) is not None:
            grouped[str(row[group_key])].append(row["metrics"])
    scores, metrics = {}, {}
    for key, metric_rows in grouped.items():
        mm = mean_metrics(metric_rows)
        metrics[key] = mm
        scores[key] = proxy_score(mm)
    return scores, metrics


def aggregate_metric_rows(rows: List[Mapping], robust_beta: float = 0.15) -> Dict:
    """Canonical aggregation used by training, score_eval and sweeps.

    Each row must contain ``metrics`` and may contain ``case``, ``tier`` and
    ``alpha``. Global/case/tier proxy scores are always computed from the mean
    component metrics, never by averaging crop-level proxy scores.
    """
    if not rows:
        raise ValueError("cannot aggregate empty evaluation rows")
    mm = mean_metrics([r["metrics"] for r in rows])
    score = proxy_score(mm)
    case, case_metrics = _group_summary(rows, "case")
    tier, tier_metrics = _group_summary(rows, "tier")
    case_std = float(np.std(list(case.values()))) if case else 0.0
    robust = float(score - float(robust_beta) * case_std)
    crop_scores = [proxy_score(r["metrics"]) for r in rows]
    alphas = [float(r["alpha"]) for r in rows if r.get("alpha") is not None]
    return {
        "metrics": mm,
        "score": float(score),
        "robust_score": robust,
        "case": case,
        "tier": tier,
        "case_metrics": case_metrics,
        "tier_metrics": tier_metrics,
        "case_score_std": case_std,
        # Diagnostic only: this intentionally differs from the canonical score
        # when NIQE samples straddle the clamp at 10.
        "mean_crop_score_diagnostic": float(np.mean(crop_scores)),
        "alpha_mean": float(np.mean(alphas)) if alphas else 1.0,
        "alpha_std": float(np.std(alphas)) if alphas else 0.0,
        "num_crops": len(rows),
    }


def seed_metrics(seed: int = 1234):
    """每个候选评测前调用，消除 MANIQA 随机采样的顺序依赖。"""
    import torch, random
    import numpy as np
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
