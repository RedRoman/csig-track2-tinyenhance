"""统一的 VOSR 加载器：同时认官方 checkpoint 和我们微调产出的 .pth。

两种格式内容不同，不能互转：
  官方  <dir>/args.json + <dir>/checkpoints/ema_model.safetensors  —— 完整 DiT 权重
  我们  <path>.pth = {"lora": {...}, "dec": {...}, "args": {...}}  —— 只有 LoRA 增量
所以这里按路径形态分派，而不是做格式转换。
"""
import os, sys, torch

TRAIN_SRC = "/cache/zihaohe/train_vosr_csig.py"


def _ns():
    """把训练脚本当模块用（它末尾直接 main()，所以先剥掉）"""
    src = open(TRAIN_SRC).read().replace("\nmain()\n", "\n")
    ns = {"__name__": "tv_mod", "__file__": TRAIN_SRC}
    exec(compile(src, "train_vosr_csig.py", "exec"), ns)
    return ns


def is_official(spec: str) -> bool:
    return os.path.isdir(spec) and os.path.exists(os.path.join(spec, "args.json"))


def load_vosr(spec, device, dtype, dino_layers=None, dino_size=0, verbose=True):
    """spec 可以是官方 ckpt 目录，也可以是我们训练产出的 best.pth。返回 eval 好的 wrapper。"""
    ns = _ns()

    if is_official(spec):
        # 官方权重：build() 内部已经从 checkpoints/ema_model.safetensors 装完整权重，
        # 不叠 LoRA。dino_layers 传 None 表示不截断（官方推理就是全层）。
        model = ns["build"](spec, False,
                            dino_layers if dino_layers is not None else 0, dino_size)
        if verbose:
            print(f"  官方 ckpt 已装载：{os.path.basename(spec.rstrip('/'))}（完整权重，无 LoRA）",
                  flush=True)
    else:
        sd = torch.load(spec, map_location="cpu")
        a = sd["args"]
        model = ns["build"](a["ckpt"], bool(a.get("train_decoder")),
                            a.get("dino_layers", 9) if dino_layers is None else dino_layers,
                            a.get("dino_size", 0) if not dino_size else dino_size)
        model.tr, _ = ns["add_lora"](model.tr, a["lora_r"], a["lora_alpha"])
        model.tr = model.tr.to(device)
        if hasattr(model.tr, "base_model"):
            bm = model.tr.base_model.model
            if hasattr(bm, "forward_flexible"):
                bm.forward = bm.forward_flexible
        miss = model.tr.load_state_dict(sd["lora"], strict=False)
        if sd.get("dec") and model.dec is not None:
            model.dec.load_state_dict(sd["dec"])
        if verbose:
            print(f"  微调 ckpt 已装载：LoRA {len(sd['lora'])} 个张量"
                  f"（未匹配 {len(miss.unexpected_keys)}）"
                  f"{'，含 LightDecoder' if sd.get('dec') else ''}", flush=True)

    for p in model.parameters():
        p.data = p.data.to(dtype)
    return model.eval()
