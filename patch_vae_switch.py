"""给 tinysr_runtime._build_raw 加一个 VAE 切换开关（环境变量 CSIG_VAE_PATH）。
幂等：已打过就跳过。"""
import re, sys, shutil, os

P = "/cache/zihaohe/csig_isr/v3/code/tinysr_runtime.py"
src = open(P).read()

OLD = '''    vae = AutoencoderTiny.from_pretrained(
        os.path.join(TS, "checkpoint", "vae", "separable"), torch_dtype=dtype)'''
NEW = '''    # TinySR 出厂的 separable TAESD 自重建只有 18.7-20.7 dB，官方 taesd3 同架构同时延
    # 却有 36.9-38.6 dB（编码器毁掉的输入细节 DiT 绕不过去）。留一个切换口。
    _vae_path = os.environ.get("CSIG_VAE_PATH") or os.path.join(TS, "checkpoint", "vae", "separable")
    vae = AutoencoderTiny.from_pretrained(_vae_path, torch_dtype=dtype)'''

if "CSIG_VAE_PATH" in src:
    print("已打过补丁，跳过")
    sys.exit(0)
if OLD not in src:
    print("找不到目标代码块，未修改")
    sys.exit(1)

shutil.copy(P, P + ".bak_vaeswitch")
open(P, "w").write(src.replace(OLD, NEW))
print("已打补丁，备份在", P + ".bak_vaeswitch")
