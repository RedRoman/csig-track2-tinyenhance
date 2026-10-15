"""CSIG 赛道二 本地代理评分 —— 单一事实源(standalone,不依赖 v3/code)。

六项指标:
  有参考(需 GT):lpips, dists
  无参考:niqe, musiq(koniq), maniqa(koniq, 官方 eval.py 用的是 create_metric("maniqa")), clipiqa(CLIP RN50)
代理 Score = (1-lpips) + (1-dists) + clipiqa + maniqa + musiq/100 + max(0, (10-niqe)/10)
  NR 部分 = clipiqa + maniqa + musiq/100 + max(0,(10-niqe)/10)     (测试集 100 张无 GT,只能算这部分)
  FR 部分 = (1-lpips) + (1-dists)                                  (验证集 3 组有 GT)

不变量:先对每项指标在全部图上取均值,再算一次 Score。不要对逐图 Score 取均值——NIQE 的 clamp 使两者不同。
MANIQA 在 4K 图上随机抽 test_sample 个 224 块,默认 20 抖动可达 ±0.01 Score;评测统一用 100(CSIG_MANIQA_SAMPLES),
并在每个候选评测前 seed_all(),否则结果依赖评测顺序。即便如此,同一模型两次 test NR 仍有 ~0.005 噪声地板。

权重:pyiqa 通过 torch hub 目录取权重($TORCH_HOME/hub/checkpoints),CLIP 主干固定读 ~/.cache/clip/RN50.pt。
ensure_weights() 把本包 weights/ 接进去,离线可用。
"""
import os
import random
import shutil

import numpy as np

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("CSIG_DATA_ROOT", os.path.join(TOOL_DIR, "data"))
VAL_DIR = os.path.join(DATA_ROOT, "验证集")
TEST_DIR = os.path.join(DATA_ROOT, "测试集")

SCORE_KEYS = ("lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa")
FR_KEYS = ("lpips", "dists")
NR_KEYS = ("niqe", "musiq", "maniqa", "clipiqa")
# pyiqa 指标名。maniqa = koniq 训练版(scale 0.8),与官方 eval.py 一致;曾误用 maniqa-pipal 六周,勿改回。
METRIC_NAMES = {"lpips": "lpips", "dists": "dists", "niqe": "niqe", "musiq": "musiq",
                "maniqa": "maniqa", "clipiqa": "clipiqa"}
VAL_PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"), ("case3_lq.jpg", "case3_gt.jpg")]


def ensure_weights():
    """把包内 weights/ 接到 pyiqa/CLIP 的查找路径上;已有系统缓存时不覆盖。必须在 import pyiqa 之前调用。"""
    w = os.path.join(TOOL_DIR, "weights")
    if os.path.isdir(os.path.join(w, "torch", "hub", "checkpoints")):
        os.environ.setdefault("TORCH_HOME", os.path.join(w, "torch"))
    clip_src = os.path.join(w, "clip", "RN50.pt")
    clip_dir = os.path.expanduser("~/.cache/clip")
    clip_dst = os.path.join(clip_dir, "RN50.pt")
    if os.path.exists(clip_src) and not os.path.exists(clip_dst):
        os.makedirs(clip_dir, exist_ok=True)
        try:
            os.symlink(clip_src, clip_dst)
        except OSError:
            shutil.copy(clip_src, clip_dst)
    os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")


def seed_all(seed: int = 1234):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_metrics(device, include_psnr: bool = False, maniqa_samples: int = None):
    ensure_weights()
    import pyiqa
    out = {k: pyiqa.create_metric(METRIC_NAMES[k], device=device) for k in SCORE_KEYS}
    n = int(maniqa_samples or os.environ.get("CSIG_MANIQA_SAMPLES", 100))
    try:
        out["maniqa"].net.test_sample = n
    except Exception:
        pass
    if include_psnr:
        out["psnr"] = pyiqa.create_metric("psnr", device=device, test_y_channel=True, color_space="ycbcr")
    return out


def proxy_score(m) -> float:
    return fr_contrib(m) + nr_contrib(m)


def nr_contrib(m) -> float:
    return float(m["clipiqa"]) + float(m["maniqa"]) + float(m["musiq"]) / 100.0 + max(0.0, (10.0 - float(m["niqe"])) / 10.0)


def fr_contrib(m) -> float:
    return (1.0 - float(m["lpips"])) + (1.0 - float(m["dists"]))


def mean_metrics(rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    return {k: float(np.mean([float(r[k]) for r in rows if k in r])) for k in keys}


def load_img(path, device=None):
    """→ float32 [1,3,H,W] in [0,1]"""
    import torch
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    return t.to(device) if device is not None else t


def measure(M, sr, gt=None):
    """sr/gt: [1,3,H,W] float32 [0,1] 同 device。返回逐项 float。有 GT 时算六项,否则四项 NR。
    LPIPS/DISTS 在 4K 整图上用 fp16 autocast 省显存(与历次评测口径一致);NR 四项不 autocast。"""
    import torch
    out = {}
    with torch.no_grad():
        if gt is not None:
            with torch.autocast("cuda", dtype=torch.float16):
                for k in FR_KEYS:
                    out[k] = float(M[k](sr, gt))
                    torch.cuda.empty_cache()
            if "psnr" in M:
                out["psnr"] = float(M["psnr"](sr, gt))
        for k in NR_KEYS:
            out[k] = float(M[k](sr))
            torch.cuda.empty_cache()
    return out
