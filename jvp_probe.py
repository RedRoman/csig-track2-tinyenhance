"""rCM 可行性的第一道闸：sCM 那一半需要对网络做前向模式自动微分（JVP）。

rCM 论文自己说这是主要基础设施障碍——他们专门写了支持并行的
FlashAttention-2 JVP 内核。我们这边 VOSR 的 LightningDiT 用了
SDPA + RoPE + SwiGLU + RMSNorm，先测能不能算出 JVP。

分层测：整网 -> 单个 block -> attention -> SDPA 本身，
定位到底是哪一层不支持，这决定了要不要改内核、还是干脆只做 score 蒸馏那一半。
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
VOSR_DIR = "/cache/zihaohe/VOSR"
sys.path.insert(0, VOSR_DIR); os.chdir(VOSR_DIR)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import jvp
from safetensors.torch import load_file

DEV = torch.device("cuda")
CK = "/cache/zihaohe/VOSR_weights/VOSR_0.5B_os"
A = json.load(open(os.path.join(CK, "args.json")))


def probe(name, fn, primals, tangents):
    try:
        out, dout = jvp(fn, primals, tangents)
        fin = bool(torch.isfinite(dout).all()) if torch.is_tensor(dout) else None
        nrm = float(dout.abs().mean()) if torch.is_tensor(dout) else float("nan")
        print(f"  {name:<34} 可以   切向量均值={nrm:.4e} 有限={fin}", flush=True)
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  {name:<34} 不行   {msg[:150]}", flush=True)
        return False


print("=== 逐层探测（fp32）===", flush=True)
x = torch.randn(1, 512, 1024, device=DEV)
v = torch.randn_like(x)

# 1. SDPA 本身
q = torch.randn(1, 8, 512, 128, device=DEV)
probe("SDPA(默认后端)", lambda a: F.scaled_dot_product_attention(a, q, q),
      (q,), (torch.randn_like(q),))
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False,
                                    enable_math=True):
    probe("SDPA(math 后端)", lambda a: F.scaled_dot_product_attention(a, q, q),
          (q,), (torch.randn_like(q),))

# 2. RMSNorm / SwiGLU 类算子
probe("F.rms_norm", lambda a: F.rms_norm(a, (1024,)), (x,), (v,))
probe("F.silu * linear (SwiGLU 形)", lambda a: F.silu(a[..., :512]) * a[..., 512:], (x,), (v,))

# 3. 整个 DiT
from models.lightningdit import LightningDiT
dit = LightningDiT(
    input_size=A["resolution"] // 8, patch_size=A["patch_size"], in_channels=8, out_channels=4,
    hidden_size=A["dim"], depth=A["depth"], num_heads=A["num_heads"], mlp_ratio=A["mlp_ratio"],
    z_dims=A["enc_dim"], encdim_ratio=A["encdim_ratio"], auxiliary_time_cond=False,
    use_qknorm=A["use_qknorm"], use_swiglu=A["use_swiglu"], use_rope=A["use_rope"],
    use_rmsnorm=A["use_rmsnorm"], wo_shift=A["wo_shift"],
    num_fused_layers=len(A["layer_dinov2b_list"]))
w = load_file(os.path.join(CK, "checkpoints", "ema_model.safetensors"))
dit.load_state_dict({k: v_.float() for k, v_ in w.items()}, strict=False)
dit = dit.to(DEV).float().eval()
if hasattr(dit, "forward_flexible"):
    dit.forward = dit.forward_flexible

lat = torch.randn(1, 8, 64, 64, device=DEV)
tt = torch.zeros(1, device=DEV)
cond = [torch.randn(1, 1024, A["enc_dim"], device=DEV)]
print(flush=True)
ok_full = probe("LightningDiT 整网 28 层", lambda a: dit(a, tt, None, cond),
                (lat,), (torch.randn_like(lat),))

# 4. 单个 block
if not ok_full:
    blk = dit.blocks[0]
    xb = torch.randn(1, 1024, A["dim"], device=DEV)
    c0 = torch.randn(1, 6 * A["dim"], device=DEV) if not A["wo_shift"] else torch.randn(1, 4 * A["dim"], device=DEV)
    z0 = dit.mlp_ca(dit.layer_norm(cond[0])) if hasattr(dit, "mlp_ca") else cond[0]
    print(flush=True)
    probe("单个 LightningDiTBlock", lambda a: blk(a, c0, z0, dit.feat_rope),
          (xb,), (torch.randn_like(xb),))
    probe("block.attn (自注意力)", lambda a: blk.attn(a, rope=dit.feat_rope),
          (xb,), (torch.randn_like(xb),))
    probe("block.cross_attn", lambda a: blk.cross_attn(a, z0),
          (xb,), (torch.randn_like(xb),))

# 5. 关掉 fused attention 再试整网
print(flush=True)
if hasattr(dit, "disable_fused_attn"):
    dit.disable_fused_attn()
    probe("整网（关掉 fused_attn）", lambda a: dit(a, tt, None, cond),
          (lat,), (torch.randn_like(lat),))

# 6. 对照：TinySR 的 DiT
print("\n=== 对照：TinySR 的 12 层 DiT ===", flush=True)
try:
    sys.path.remove(VOSR_DIR)
    for m in list(sys.modules):
        if m == "models" or m.startswith("models."):
            del sys.modules[m]
    sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
    os.chdir("/cache/zihaohe/csig_isr")
    from tinysr_runtime import build_model
    tm, _ = build_model("/cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth",
                        device=DEV, dtype=torch.float32, merge_lora=True,
                        alpha_mode="fixed", fixed_alpha=1.0)
    tm.eval()
    zt = torch.randn(1, 16, 64, 64, device=DEV)
    probe("TinySR DiT 12 层", lambda a: tm.tr(hidden_states=a, timestep=tm.ts,
                                              pooled_projections=tm.pooled,
                                              return_dict=False)[0],
          (zt,), (torch.randn_like(zt),))
except Exception as e:
    print(f"  TinySR 对照失败: {type(e).__name__}: {str(e)[:150]}", flush=True)
