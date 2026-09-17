"""补丁 v2：CSIG_VAE_PATH 生效时改用 diffusers 原版 AutoencoderTiny 类。

TinySR 的 models.vae.autoencoder_tiny.AutoencoderTiny 是可分离卷积改造版，
参数结构与官方 taesd3 不同（169 个张量里只有 67 个同名同形），官方权重装不进去。
"""
import sys, shutil

P = "/cache/zihaohe/csig_isr/v3/code/tinysr_runtime.py"
src = open(P).read()

OLD = '''    _vae_path = os.environ.get("CSIG_VAE_PATH") or os.path.join(TS, "checkpoint", "vae", "separable")
    vae = AutoencoderTiny.from_pretrained(_vae_path, torch_dtype=dtype)'''
NEW = '''    _vae_path = os.environ.get("CSIG_VAE_PATH")
    if _vae_path:
        # 官方 taesd3 是标准卷积结构，必须用 diffusers 原版类装载
        from diffusers import AutoencoderTiny as _HFAutoencoderTiny
        vae = _HFAutoencoderTiny.from_pretrained(_vae_path, torch_dtype=dtype)
    else:
        vae = AutoencoderTiny.from_pretrained(
            os.path.join(TS, "checkpoint", "vae", "separable"), torch_dtype=dtype)'''

if "_HFAutoencoderTiny" in src:
    print("v2 已打过，跳过"); sys.exit(0)
if OLD not in src:
    print("找不到 v1 补丁块，未修改"); sys.exit(1)

shutil.copy(P, P + ".bak_v2")
open(P, "w").write(src.replace(OLD, NEW))
print("v2 补丁已打")
