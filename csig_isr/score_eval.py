"""按 NTIRE 官方口径算 Score（与 eval.py 完全一致的指标配置）。

Score = (1-LPIPS) + (1-DISTS) + CLIPIQA + MANIQA + MUSIQ/100 + max(0,(10-NIQE)/10)
FinalScore = 2^Score * Speedup^0.2

用法：
  python score_eval.py --lq_only            # LQ 基线（什么都不做的下界）
  python score_eval.py --ckpt <path>        # 评某个训练 checkpoint
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/cache/zihaohe/csig_isr/NTIRE2026")

ROOT = "/cache/zihaohe/csig_isr"
VAL = f"{ROOT}/data/赛题二/验证集"
CKPT_S = f"{ROOT}/NTIRE2026/model_zoo/team11_SamsungAICamera/team11_mobile_samsung/weights.pth"
X1 = f"{ROOT}/x1_model/srnet_x1_init.pth"
SD21 = f"{ROOT}/sd21base"
dev = torch.device("cuda")

PAIRS = [("case1_lq.jpg", "case1_gt.png", 101),
         ("case2_lq.jpg", "case2_gt.png", 202),
         ("case3_lq.jpg", "case3_gt.jpg", 303)]


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


class ResSRNet(nn.Module):
    def __init__(self, net, refine, decoder):
        super().__init__()
        self.net, self.refine, self.decoder = net, refine, decoder
        self.zero_conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return (x + self.zero_conv(self.decoder(self.refine(self.net(x))) - 0.5)).clamp(0, 1)


def build(ckpt_path=None):
    from diffusers import UNet2DConditionModel
    from models.team11_SamsungAKCamera.model import create_srnet
    cfg = UNet2DConditionModel.load_config(SD21, subfolder="unet")
    m = create_srnet(UNet2DConditionModel.from_config(cfg), checkpoint_path=CKPT_S)
    m.net.body[1].conv_in = nn.Conv2d(192, 160, 3, padding=1)
    m.net.body[0] = nn.PixelUnshuffle(8)
    base = torch.load(X1, map_location="cpu", weights_only=False)
    m.net.load_state_dict(base["net"])
    m.refine.load_state_dict(base["refine"])
    m.decoder.load_state_dict(base["decoder"])
    model = ResSRNet(m.net, m.refine, m.decoder)
    nn.init.zeros_(model.zero_conv.weight)
    nn.init.zeros_(model.zero_conv.bias)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.net.load_state_dict(sd["net"])
        model.refine.load_state_dict(sd["refine"])
        model.decoder.load_state_dict(sd["decoder"])
        if "zero_conv" in sd:
            model.zero_conv.load_state_dict(sd["zero_conv"])
        print(f"  载入 {ckpt_path} (step {sd.get('step', '?')})")
    return model


def collect_crops(n_per=8):
    out = []
    for lqn, gtn, seed in PAIRS:
        lq_f, gt_f = load_img(f"{VAL}/{lqn}"), load_img(f"{VAL}/{gtn}")
        if lq_f.shape != gt_f.shape:
            continue
        rng = np.random.RandomState(seed)
        _, _, H, W = gt_f.shape
        got, tries = 0, 0
        while got < n_per and tries < n_per * 12:
            tries += 1
            y, x = rng.randint(0, H - 512), rng.randint(0, W - 512)
            gt = gt_f[:, :, y:y + 512, x:x + 512]
            if gt.std().item() < 0.045:
                continue
            got += 1
            out.append((lq_f[:, :, y:y + 512, x:x + 512].clone(), gt.clone()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--lq_only", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--maniqa", type=str, default="maniqa",
                    help="maniqa(官方,koniq权重) 或 maniqa-pipal(备用)")
    args = ap.parse_args()

    crops = collect_crops(args.n)
    print(f"验证 crop 数：{len(crops)}")

    import pyiqa
    M = {
        "lpips": pyiqa.create_metric("lpips", device=dev),
        "dists": pyiqa.create_metric("dists", device=dev),
        "niqe": pyiqa.create_metric("niqe", device=dev),
        "musiq": pyiqa.create_metric("musiq", device=dev),
        "maniqa": pyiqa.create_metric(args.maniqa, device=dev),
        "clipiqa": pyiqa.create_metric("clipiqa", device=dev),
        "psnr": pyiqa.create_metric("psnr", device=dev, test_y_channel=True,
                                    color_space="ycbcr"),
    }
    print(f"  指标载入完成（maniqa 用 {args.maniqa}）")

    model = None
    if not args.lq_only:
        model = build(args.ckpt).to(dev).half().eval()

    res = {k: [] for k in M}
    with torch.no_grad():
        for lq, gt in crops:
            lq, gt = lq.to(dev), gt.to(dev)
            out = lq if args.lq_only else model(lq.half()).float()
            for k, fn in M.items():
                v = fn(out, gt) if k in ("lpips", "dists", "psnr") else fn(out)
                res[k].append(v.item())
            torch.cuda.empty_cache()

    m = {k: float(np.mean(v)) for k, v in res.items()}
    score = ((1 - m["lpips"]) + (1 - m["dists"]) + m["clipiqa"]
             + m["maniqa"] + m["musiq"] / 100 + max(0, (10 - m["niqe"]) / 10))

    tag = "LQ 基线（什么都不做）" if args.lq_only else (args.ckpt or "x1 初始化")
    print()
    print("=" * 62)
    print(f"  {tag}")
    print("=" * 62)
    print(f"  {'指标':<12}{'值':>10}{'贡献 Score':>14}")
    print("  " + "-" * 38)
    for k, contrib in [("lpips", 1 - m["lpips"]), ("dists", 1 - m["dists"]),
                       ("clipiqa", m["clipiqa"]), ("maniqa", m["maniqa"]),
                       ("musiq", m["musiq"] / 100),
                       ("niqe", max(0, (10 - m["niqe"]) / 10))]:
        print(f"  {k:<12}{m[k]:>10.4f}{contrib:>14.4f}")
    print("  " + "-" * 38)
    print(f"  {'PSNR':<12}{m['psnr']:>10.4f}{'(不计分)':>14}")
    print(f"  {'Score':<12}{'':<10}{score:>14.4f}")
    print()
    print(f"  {'加速比':>8}{'FinalScore':>14}")
    for sp in [1, 5, 10, 20, 40]:
        print(f"  {sp:>8}{2 ** score * sp ** 0.2:>14.2f}")

    name = "lq" if args.lq_only else os.path.basename(args.ckpt or "init")
    out_json = f"{ROOT}/eval_out/score_{name}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump({"metrics": m, "score": score}, open(out_json, "w"), indent=1)
    print(f"\n  已存 {out_json}")


if __name__ == "__main__":
    main()
