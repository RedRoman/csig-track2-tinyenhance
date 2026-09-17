"""OSEDiff 基准时延实测（加速比的分母）。

AdcSR 提供的 osediff.pkl 是【已 merge LoRA】的纯 state_dict（只有 vae/unet 两个 key），
不含官方 OSEDiff_test 期望的 rank_unet / *_lora_modules。
所以这里直接用 SD2.1 结构 + 载入合并后权重，绕过 LoRA 装配 —— 数值等价且更快。

赛题口径：512x512 输入输出、单卡、fp16、统计整网所有模块。
"""
import os, sys, argparse
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
dt = torch.float16


def bench(fn, n=30, warm=8):
    with torch.no_grad():
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
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import AutoTokenizer, CLIPTextModel

    print("载入 SD2.1 结构 ...", flush=True)
    vae = AutoencoderKL.from_pretrained(SD21, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(SD21, subfolder="unet")
    tok = AutoTokenizer.from_pretrained(SD21, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(SD21, subfolder="text_encoder")
    sched = DDPMScheduler.from_pretrained(SD21, subfolder="scheduler")
    sched.set_timesteps(1, device="cuda")

    print("载入 OSEDiff 合并后权重 ...", flush=True)
    ck = torch.load(OSE, map_location="cpu", weights_only=False)
    mv = vae.load_state_dict(ck["vae"], strict=False)
    mu = unet.load_state_dict(ck["unet"], strict=False)
    print(f"  vae  missing={len(mv.missing_keys)} unexpected={len(mv.unexpected_keys)}")
    print(f"  unet missing={len(mu.missing_keys)} unexpected={len(mu.unexpected_keys)}")

    vae = vae.to(dev, dtype=dt).eval()
    unet = unet.to(dev, dtype=dt).eval()
    te = te.to(dev, dtype=dt).eval()
    sched.alphas_cumprod = sched.alphas_cumprod.to(dev)

    print("载入 RAM/DAPE ...", flush=True)
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    from torchvision import transforms
    DAPE_m = ram(pretrained=RAM, pretrained_condition=DAPE, image_size=384, vit="swin_l")
    DAPE_m = DAPE_m.eval().to(dev).to(dt)
    ram_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    n_unet = sum(p.numel() for p in unet.parameters())
    n_vae = sum(p.numel() for p in vae.parameters())
    n_te = sum(p.numel() for p in te.parameters())
    n_ram = sum(p.numel() for p in DAPE_m.parameters())
    print(f"\n参数量  UNet {n_unet/1e6:.1f}M  VAE {n_vae/1e6:.1f}M  "
          f"TextEnc {n_te/1e6:.1f}M  RAM/DAPE {n_ram/1e6:.1f}M  "
          f"合计 {(n_unet+n_vae+n_te+n_ram)/1e6:.1f}M", flush=True)

    S = args.size
    lq = torch.rand(1, 3, S, S, device=dev, dtype=dt)
    ts = torch.tensor([999], device=dev).long()

    # ---- 各阶段函数 ----
    def f_tag():
        x = ram_tf(lq)
        res = inference(x, DAPE_m)
        return res[0] if isinstance(res, (list, tuple)) else str(res)

    with torch.no_grad():
        prompt = f_tag()

    def f_text():
        ids = tok(prompt, max_length=tok.model_max_length, padding="max_length",
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        return te(ids)[0].to(dt)

    with torch.no_grad():
        pe = f_text()
        z = vae.encode(lq * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

    def f_enc():
        return vae.encode(lq * 2 - 1).latent_dist.sample()

    def f_unet():
        return unet(z, ts, encoder_hidden_states=pe).sample

    with torch.no_grad():
        mp = f_unet()
        zd = sched.step(mp, ts, z, return_dict=True).prev_sample.to(dt)

    def f_dec():
        return vae.decode(zd / vae.config.scaling_factor).sample

    def f_all():
        p = f_tag()
        ids = tok(p, max_length=tok.model_max_length, padding="max_length",
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        e = te(ids)[0].to(dt)
        zz = vae.encode(lq * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
        m = unet(zz, ts, encoder_hidden_states=e).sample
        zo = sched.step(m, ts, zz, return_dict=True).prev_sample.to(dt)
        return vae.decode(zo / vae.config.scaling_factor).sample

    with torch.no_grad():
        out = f_all()
    print(f"输出尺寸 {tuple(out.shape)}\n", flush=True)

    print(f"测速中（{S}x{S}, fp16, {args.iters} 次）...", flush=True)
    t_tag = bench(f_tag, args.iters)
    t_text = bench(f_text, args.iters)
    t_enc = bench(f_enc, args.iters)
    t_unet = bench(f_unet, args.iters)
    t_dec = bench(f_dec, args.iters)
    t_all = bench(f_all, max(args.iters // 2, 10))

    print()
    print("=" * 70)
    print(f"  OSEDiff 基准时延（A100, fp16, {S}x{S}）")
    print("=" * 70)
    print(f"  {'模块':<30}{'ms':>10}{'占比':>10}")
    print("  " + "-" * 50)
    for nm, t in [("RAM/DAPE tagging", t_tag), ("CLIP text encoder", t_text),
                  ("VAE encoder", t_enc), ("UNet (1 step)", t_unet),
                  ("VAE decoder", t_dec)]:
        print(f"  {nm:<30}{t:>10.3f}{t/t_all*100:>9.1f}%")
    print("  " + "-" * 50)
    print(f"  {'端到端（整网，赛题口径）':<30}{t_all:>10.3f}{100.0:>9.1f}%")
    print(f"  {'(五项之和)':<30}{t_tag+t_text+t_enc+t_unet+t_dec:>10.3f}")

    print()
    print("=" * 70)
    print("  以此为分母的加速比")
    print("=" * 70)
    cands = [("Samsung x1 eager", 18.80), ("Samsung x1 jit frozen", 14.16),
             ("TinySR x1 eager", 32.11), ("TinySR x1 +colorfix", 32.60)]
    print(f"  {'模型':<28}{'ms':>9}{'加速比':>10}{'Sp^0.2':>10}")
    for nm, t in cands:
        sp = t_all / t
        print(f"  {nm:<28}{t:>9.2f}{sp:>10.2f}x{sp**0.2:>10.3f}")

    import json
    json.dump({"size": S, "t_tag": t_tag, "t_text": t_text, "t_enc": t_enc,
               "t_unet": t_unet, "t_dec": t_dec, "t_all": t_all},
              open(f"{ROOT}/eval_out/osediff_bench.json", "w"), indent=1)
    print(f"\n  已存 {ROOT}/eval_out/osediff_bench.json")


if __name__ == "__main__":
    main()
