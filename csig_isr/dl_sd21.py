import os
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

p = snapshot_download(
    "stabilityai/stable-diffusion-2-1-base",
    local_dir="/cache/zihaohe/csig_isr/sd21base",
    allow_patterns=["*.json", "*.txt", "unet/*.safetensors", "vae/*.safetensors",
                    "text_encoder/*.safetensors", "tokenizer/*", "scheduler/*"],
    max_workers=4,
)
print("DONE", p)
