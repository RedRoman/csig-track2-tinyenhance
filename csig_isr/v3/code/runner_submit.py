import os
import sys

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch
import yaml
from torchvision import transforms
import time

logger = logging.getLogger(__name__)


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.config = _load_config(model_dir)

        # WARNING: All models must be using fp16 precision for fair comparisons
        self.weight_dtype = torch.float16

        models_cfg = self.config.get("models", {})

        # 本方案为单模型单步扩散：一个 jit 模型即完成 VAE 编码 -> DiT 残差预测 -> VAE 解码。
        # 不使用 RAM/DAPE 打标与 CLIP 文本编码器（提示词条件在本任务中无增益，
        # 且它们在基准 OSEDiff 中占约 28% 时延）。
        self.core_model = self._load_jit(
            models_cfg.get("core_model", "tinysr_x1.pt")
        )

        # 预热：jit 模型首次前向包含编译开销，预热后时延才稳定
        self._warmup()

    def _load_jit(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            model = torch.jit.load(path, map_location=self.device)
            model.eval()
            return model
        raise Exception(f"JIT model not found: {path}")

    @torch.no_grad()
    def _warmup(self, size: int = 512, iters: int = 10):
        dummy = torch.zeros(1, 3, size, size, device=self.device,
                            dtype=self.weight_dtype)
        for _ in range(iters):
            self.core_model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        """Run full inference pipeline.

        Args:
            image_tensor: [1, 3, H, W] tensor in [-1, 1] range.
            prompt: Optional user prompt (unused in this solution).

        Returns:
            output_tensor: [1, 3, H, W] tensor in [-1, 1] range.
        """
        # 仅在必要时转换，避免重复的 H2D 拷贝（实测 CPU->GPU 拷贝约 13ms，
        # 与模型本身 12ms 同量级，会显著影响时延测量）
        x = image_tensor
        if x.device != self.device:
            x = x.to(device=self.device)
        if x.dtype != self.weight_dtype:
            x = x.to(dtype=self.weight_dtype)
        output_tensor = self.core_model(x)
        return output_tensor
