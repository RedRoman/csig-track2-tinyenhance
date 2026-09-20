import os
import sys
import logging

import torch

logger = logging.getLogger(__name__)
_MD = os.path.dirname(os.path.abspath(__file__))


def _load_pix_model(md, device, dtype):
    """pix_model.py 与本文件同目录,按文件路径加载,不碰 sys.path。加载后做推理期折叠(固定时间步的调制常量折进权重)。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tinyenhance_pix_model", os.path.join(md, "pix_model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tinyenhance_pix_model"] = mod   # dynamo 追踪时按名解析,必须可见
    spec.loader.exec_module(mod)
    m = mod.build_pix(os.path.join(md, "model.pth"), device=device, dtype=torch.float32)
    fm = mod.fold_inference(m).to(device, dtype).eval()
    del m
    return fm


class _Core(torch.nn.Module):
    """[-1,1] 进出;单步去噪(固定时间步)。channels_last 让 cuDNN/Inductor 走 NHWC 核。"""
    def __init__(self, m):
        super().__init__()
        self.m = m.to(memory_format=torch.channels_last)

    def forward(self, x):
        return self.m(x.contiguous(memory_format=torch.channels_last))


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        # WARNING: All models must be using fp16 precision for fair comparisons
        self.weight_dtype = torch.float16
        torch.backends.cudnn.benchmark = False
        m = _load_pix_model(model_dir, self.device, self.weight_dtype)
        self.eager = _Core(m).to(self.device).eval()
        self.core_model = None
        x = torch.zeros(1, 3, 512, 512, device=self.device, dtype=self.weight_dtype)
        if self.device.type == "cuda":
            self.core_model = self._try_compile(x)
            if self.core_model is None:
                try:
                    with torch.no_grad():
                        j = torch.jit.freeze(torch.jit.trace(self.eager, x, check_trace=False).eval())
                        for _ in range(10):
                            j(x)
                    torch.cuda.synchronize()
                    self.core_model = j
                    self._path = "jit"
                except Exception as exc:
                    logger.warning("jit failed: %s", exc)
                    self.core_model = None
        if self.core_model is None:
            self.core_model = self.eager
            self._path = "eager"

    def _bench(self, fn, x, n=30):
        with torch.no_grad():
            for _ in range(5):
                fn(x)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n):
                fn(x)
            e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / n

    def _try_compile(self, x):
        """编译路径:显式可写缓存目录 + 三档模式 + 自校验(编译版必须实测快于 eager 且数值一致才采用)。"""
        import tempfile
        try:
            work = tempfile.mkdtemp(prefix="tinyenhance_")
        except Exception:
            work = os.path.join(self.model_dir, "_work"); os.makedirs(work, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(work, "inductor")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(work, "triton")
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        os.environ.pop("TORCHDYNAMO_DISABLE", None)
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.disable = False
        except Exception as exc:
            logger.warning("dynamo unavailable: %s", exc); return None
        g = torch.Generator(device=self.device); g.manual_seed(0)
        xr = torch.rand(x.shape, device=self.device, dtype=torch.float32, generator=g).to(self.weight_dtype) * 2 - 1
        with torch.no_grad():
            ref = self.eager(xr).float()
        t_eager = self._bench(self.eager, x)
        for mode in ("max-autotune", "reduce-overhead", "default"):
            try:
                _dynamo.reset()
                cm = torch.compile(self.eager, mode=mode)
                if type(cm).__name__ != "OptimizedModule":
                    continue
                with torch.no_grad():
                    for _ in range(6):
                        cm(x)
                    torch.cuda.synchronize()
                    err = float((cm(xr).float() - ref).abs().max())
                t_c = self._bench(lambda t: cm(t).clone(), x)
                logger.warning("compile %s: %.2f ms vs eager %.2f ms, max|d|=%.4f", mode, t_c, t_eager, err)
                if t_c < 0.8 * t_eager and err < 0.06:
                    self._path = "compile:" + mode
                    return cm
            except Exception as exc:
                logger.warning("compile %s failed: %s", mode, exc)
        return None

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        """Run full inference pipeline.

        Args:
            image_tensor: [1, 3, H, W] tensor in [-1, 1] range.
            prompt: Optional user prompt (unused in this solution).

        Returns:
            output_tensor: [1, 3, H, W] tensor in [-1, 1] range.
        """
        x = image_tensor
        if x.device != self.device:
            x = x.to(device=self.device)
        if x.dtype != self.weight_dtype:
            x = x.to(dtype=self.weight_dtype)
        _, _, H, W = x.shape
        if H == 512 and W == 512:
            return self.core_model(x).clone()
        # 全卷积:任意尺寸整图直推(eager 路径,形状不固定);显存不足时退化为分块
        try:
            return self.eager(x).clone()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._infer_tiled(x)

    @torch.no_grad()
    def _infer_tiled(self, x: torch.Tensor, tile: int = 1024, ov: int = 128) -> torch.Tensor:
        _, _, H, W = x.shape
        st = tile - ov
        ys = list(range(0, max(H - tile, 0) + 1, st)) or [0]
        xs = list(range(0, max(W - tile, 0) + 1, st)) or [0]
        if ys[-1] + tile < H:
            ys.append(max(H - tile, 0))
        if xs[-1] + tile < W:
            xs.append(max(W - tile, 0))
        t = torch.linspace(-1, 1, tile, device=self.device)
        g = torch.exp(-(t ** 2) / (2 * 0.45 ** 2))
        wm = (g[:, None] * g[None, :])[None, None].float()
        out = torch.zeros(1, 3, H, W, device=self.device, dtype=torch.float32)
        acc = torch.zeros(1, 1, H, W, device=self.device, dtype=torch.float32)
        for y in ys:
            for xx in xs:
                p = self.eager(x[:, :, y:y + tile, xx:xx + tile]).float()
                out[:, :, y:y + tile, xx:xx + tile] += p * wm
                acc[:, :, y:y + tile, xx:xx + tile] += wm
        return (out / acc.clamp_min(1e-8)).clamp(-1, 1).to(self.weight_dtype)
