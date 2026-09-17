"""OSEDiff 基准时延实测（加速比的分母）。

按官方 test_inference_time.py 口径，但拆出逐模块 profiling：
  RAM/DAPE tagging -> CLIP text encoder -> VAE encoder -> UNet -> VAE decoder

赛题规定 512x512 输入输出、单卡、fp16。
注意：OSEDiff 原生是 x4，输入 512 会输出 2048。这里同时报两种口径。
"""
import os, sys, time, argparse
import numpy as np
import torch

ROOT = "/cache/zihaohe/csig_isr"
sys.path.insert(0, f"{ROOT}/OSEDiff")
os.chdir(f"{ROOT}/OSEDiff")

SD21 = f"{ROOT}/sd21base"
OSE = f"{ROOT}/AdcSR/weight/pretrained/osediff.pkl"
RAM = f"{ROOT}/AdcSR/weight/pretrained/ram_swin_large_14m.pth"
DAPE = f"{ROOT}/AdcSR/weight/pretrained/DAPE.pth"
dev = torch.device("cuda")


def bench(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    from osediff import OSEDiff_test
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    from torchvision import transforms

    class A:
        pretrained_model_name_or_path = SD21
        osediff_path = OSE
        process_size = args.size
        vae_decoder_tiled_size = 224
        vae_encoder_tiled_size = 1024
        latent_tiled_size = 96
        latent_tiled_overlap = 32
        mixed_precision = "fp16"
        merge_and_unload_lora = True

    print("载入 OSEDiff ...", flush=True)
    model = OSEDiff_test(A())
    print("载入 RAM/DAPE ...", flush=True)
    DAPE_m = ram(pretrained=RAM, pretrained_condition=DAPE, image_size=384, vit="swin_l")
    DAPE_m.eval().to(dev).to(torch.float16)

    ram_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    n_ose = sum(p.numel() for p in model.unet.parameters()) if hasattr(model, "unet") else 0
    n_dape = sum(p.numel() for p in DAPE_m.parameters())
    print(f"DAPE/RAM 参数量 {n_dape/1e6:.1f} M", flush=True)

    lq = torch.rand(1, 3, args.size, args.size, device=dev, dtype=torch.float16)

    print(f"\n开始测速（{args.size}x{args.size} 输入，fp16，{args.iters} 次）...", flush=True)

    # 1) RAM/DAPE tagging
    def f_tag():
        with torch.no_grad():
            x = ram_tf(lq)
            return inference(x, DAPE_m)
    t_tag = bench(f_tag, args.iters)

    with torch.no_grad():
        prompt = f_tag()[0] if isinstance(f_tag(), (list, tuple)) else ""

    # 2) 端到端（含 tagging，官方口径）
    def f_all():
        with torch.no_grad():
            p = f_tag()
            p = p[0] if isinstance(p, (list, tuple)) else ""
            return model(lq * 2 - 1, prompt=p)
    t_all = bench(f_all, max(args.iters // 2, 10))

    # 3) 仅核心（不含 tagging）
    def f_core():
        with torch.no_grad():
            return model(lq * 2 - 1, prompt=prompt)
    t_core = bench(f_core, max(args.iters // 2, 10))

    with torch.no_grad():
        out = model(lq * 2 - 1, prompt=prompt)
    print(f"输出尺寸 {tuple(out.shape)}", flush=True)

    print()
    print("=" * 66)
    print(f"  OSEDiff 基准时延（A100, fp16, 输入 {args.size}x{args.size}）")
    print("=" * 66)
    print(f"  {'RAM/DAPE tagging':<34}{t_tag:>10.3f} ms  {t_tag/t_all*100:>6.1f}%")
    print(f"  {'核心 (VAE+UNet+CLIP+decode)':<34}{t_core:>10.3f} ms  {t_core/t_all*100:>6.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'端到端（官方口径，含tagging）':<34}{t_all:>10.3f} ms  {100.0:>6.1f}%")
    print()
    print(f"  以此为分母，各模型加速比：")
    for name, t in [("Samsung x1 (eager)", 18.80), ("Samsung x1 (jit frozen)", 14.16)]:
        sp = t_all / t
        print(f"    {name:<28}{t:>7.2f} ms  ->  {sp:>6.2f}x   Speedup^0.2 = {sp**0.2:.3f}")


if __name__ == "__main__":
    main()
