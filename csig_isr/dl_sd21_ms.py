"""从 modelscope 下载 SD2.1-base（hf-mirror 上该 repo 401/404）。"""
import os, sys

os.environ["MODELSCOPE_CACHE"] = "/cache/zihaohe/.mscache"

from modelscope import snapshot_download

p = snapshot_download(
    "AI-ModelScope/stable-diffusion-2-1-base",
    local_dir="/cache/zihaohe/csig_isr/sd21base",
    allow_file_pattern=["*.json", "*.txt", "unet/*.safetensors", "vae/*.safetensors",
                        "text_encoder/*.safetensors", "tokenizer/*", "scheduler/*"],
)
print("DONE", p)
