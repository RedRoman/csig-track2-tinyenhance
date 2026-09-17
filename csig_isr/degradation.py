"""CSIG 赛题二 —— 退化合成模块（独立可用）

作用：把高质量图（GT）转成符合赛题特性的低质量图（LQ），用于构造训练配对。

===============================================================================
为什么不能直接用 Real-ESRGAN 的标准退化
===============================================================================
我们分析了赛题给的 3 组真实配对（LQ + GT），退化特性与常规超分任务完全不同：

    指标                     数值        含义
    ---------------------------------------------------------------
    低频能量比 (0-10%)       1.00        低频完全无损
    中频能量比 (10-40%)      0.61        中频衰减
    高频能量比 (40-100%)     0.33        高频只剩 1/3
    Laplacian 方差比         0.072       细节丰富度只剩 7%
    梯度能量比               0.47        边缘强度腰斩
    噪声比 (LQ/GT)           0.27        !! LQ 比 GT 还干净 !!
    PSNR(LQ, GT)             27.23 dB

最后一行是关键：**LQ 的噪声比 GT 更低**。
这说明退化不是"拍糊了"或"有噪点"，而是**手机 ISP 降噪降过头**——
算法在消除噪声时把真实纹理一起抹平了。

所以：
  - 绝对不能加噪声（方向完全相反，会让模型学错）
  - 不需要降采样（LQ 与 GT 同分辨率，是 x1 增强不是 x4 超分）
  - 核心是"低通滤波 + 边缘保持降噪 + JPEG 压缩"

===============================================================================
退化链是怎么定出来的
===============================================================================
方法：对 GT 施加候选退化算子，提取统计指纹（梯度能量比/Laplacian方差比/
      噪声比/三频段能量比/PSNR），与真实 LQ 的指纹做加权归一化距离匹配，
      网格搜索最优参数组合。

结果（距离越小越接近真实退化）：

    退化链                                    距离
    ------------------------------------------------
    双边(σ90) → 高斯(σ1.4) → JPEG(q60)       0.212   <- 最优
    双边(σ90) → 高斯(σ1.4) → JPEG(q75)       0.226
    双边(σ90) → 高斯(σ1.2) → JPEG(q85)       0.239
    ------------------------------------------------
    （对照）仅高斯(σ1.0)，无 JPEG              0.568
    （对照）仅双边(d9,σ75)                     3.256

两个发现：
  1. 加入 JPEG 后距离从 0.435 腰斩到 0.212 —— 测试集全是 .jpg，
     压缩本身就是退化链的一环，不能省
  2. 前 9 名退化链全部含双边滤波 —— 真实顺序是"边缘保持降噪 → 低通 → JPEG 存储"，
     正是手机 ISP 的物理处理顺序

===============================================================================
使用方法
===============================================================================
    from degradation import degrade, DegradationConfig

    # 方式一：直接用（推荐，参数已按标定结果随机采样）
    lq_u8 = degrade(gt_u8, np.random.RandomState(seed))

    # 方式二：自定义强度范围
    cfg = DegradationConfig(gauss_sigma=(1.2, 1.6), jpeg_quality=(50, 75))
    lq_u8 = degrade(gt_u8, rng, cfg)

    # 方式三：固定参数复现最优配置（用于验证）
    lq_u8 = degrade_fixed(gt_u8, bilateral_sigma=90, gauss_sigma=1.4, jpeg_q=60)

依赖：numpy, opencv-python, pillow
"""

import io
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2
from PIL import Image


@dataclass
class DegradationConfig:
    """退化参数的采样范围。默认值来自赛题验证集标定。

    三张真实样本的退化强度差异极大（PSNR 21.05 / 32.78 / 27.86 dB），
    所以训练时必须随机采样一个范围，不能固定单一强度。
    """
    # 双边滤波：模拟 ISP 的边缘保持降噪。d=9 为邻域直径，sigma 越大平滑越强
    bilateral_prob: float = 0.9
    bilateral_d: int = 9
    bilateral_sigma: Tuple[float, float] = (40.0, 100.0)

    # 高斯低通：模拟高频丢失
    gauss_sigma: Tuple[float, float] = (0.8, 1.6)

    # JPEG 压缩：测试集全为 jpg，压缩是退化链的必要一环
    jpeg_quality: Tuple[int, int] = (50, 95)


DEFAULT_CONFIG = DegradationConfig()


def degrade(img_u8: np.ndarray,
            rng: np.random.RandomState,
            cfg: DegradationConfig = DEFAULT_CONFIG) -> np.ndarray:
    """按标定所得的退化链合成 LQ。

    Args:
        img_u8: GT 图像，HxWx3 uint8，RGB
        rng:    随机数发生器（传入以保证 DataLoader 多进程下的可复现性）
        cfg:    退化参数范围

    Returns:
        LQ 图像，HxWx3 uint8，RGB，与输入同尺寸
    """
    x = img_u8

    # 1) 双边滤波 —— 边缘保持降噪，模拟 ISP 的去噪环节
    if rng.random() < cfg.bilateral_prob:
        s = float(rng.uniform(*cfg.bilateral_sigma))
        x = cv2.bilateralFilter(x, cfg.bilateral_d, s, s)

    # 2) 高斯低通 —— 高频衰减
    sigma = float(rng.uniform(*cfg.gauss_sigma))
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)
    x = np.clip(x, 0, 255).astype(np.uint8)

    # 3) JPEG 压缩 —— 测试集为 jpg，此步不可省
    q = int(rng.uniform(*cfg.jpeg_quality))
    buf = io.BytesIO()
    Image.fromarray(x).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def degrade_fixed(img_u8: np.ndarray,
                  bilateral_sigma: float = 90.0,
                  gauss_sigma: float = 1.4,
                  jpeg_q: int = 60,
                  bilateral_d: int = 9) -> np.ndarray:
    """固定参数版本，复现标定出的最优配置（距离 0.212）。

    用于验证退化合成是否正确，或需要确定性输出的场合。
    """
    x = cv2.bilateralFilter(img_u8, bilateral_d, bilateral_sigma, bilateral_sigma)
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), gauss_sigma)
    x = np.clip(x, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(x).save(buf, format="JPEG", quality=jpeg_q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


# =============================================================================
# 统计指纹工具 —— 用于验证自己合成的 LQ 是否匹配真实退化
# =============================================================================

def _radial_spectrum(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    nb = min(cy, cx)
    prof = np.bincount(r.ravel(), f.ravel(), minlength=nb + 1)[:nb]
    cnt = np.bincount(r.ravel(), minlength=nb + 1)[:nb]
    return prof / np.maximum(cnt, 1)


def fingerprint(lq: np.ndarray, gt: np.ndarray) -> dict:
    """提取退化的统计指纹，用于与真实 LQ 对比。

    Args:
        lq, gt: HxWx3 uint8 或 float32([0,255])，同尺寸

    Returns:
        dict，含 grad_ratio / lapvar_ratio / noise_ratio / low / mid / high / psnr
    """
    lq = lq.astype(np.float32)
    gt = gt.astype(np.float32)

    def stats(img):
        g = img.mean(axis=2)
        gx = np.diff(g, axis=1)
        gy = np.diff(g, axis=0)
        grad = np.sqrt((gx ** 2).mean() + (gy ** 2).mean())
        lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1]
               - g[1:-1, :-2] - g[1:-1, 2:])
        m = g[:-1, :-1] - g[1:, :-1] - g[:-1, 1:] + g[1:, 1:]
        return grad, lap.var(), m.std() / 2.0

    g_l, v_l, n_l = stats(lq)
    g_g, v_g, n_g = stats(gt)

    sl = _radial_spectrum(lq.mean(axis=2))
    sg = _radial_spectrum(gt.mean(axis=2))
    n = len(sl)
    bands = {}
    for name, lo, hi in [("low", 0, n // 10),
                         ("mid", n // 10, int(n * 0.4)),
                         ("high", int(n * 0.4), n)]:
        bands[name] = float(sl[lo:hi].sum() / max(sg[lo:hi].sum(), 1e-9))

    mse = np.mean((lq - gt) ** 2)
    return dict(
        grad_ratio=float(g_l / max(g_g, 1e-6)),
        lapvar_ratio=float(v_l / max(v_g, 1e-6)),
        noise_ratio=float(n_l / max(n_g, 1e-6)),
        psnr=float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else 99.0,
        **bands,
    )


# 赛题验证集实测的真实 LQ 指纹（3 组配对，48 个 512 crop 平均）
REAL_LQ_FINGERPRINT = {
    "grad_ratio": 0.4715,
    "lapvar_ratio": 0.0716,
    "noise_ratio": 0.2679,
    "low": 1.0011,
    "mid": 0.6124,
    "high": 0.3280,
    "psnr": 27.2284,
}


def match_distance(fp: dict, ref: dict = None) -> float:
    """计算合成 LQ 的指纹与真实 LQ 的加权归一化距离。越小越接近。

    重要：这个距离只在【大量 crop 的整体统计】上有意义，不适合评判单张图。

    原因：三张真实样本的退化强度差异极大（PSNR 21.05 / 32.78 / 27.86 dB），
    REAL_LQ_FINGERPRINT 是它们的平均值。任何单张图都会偏离这个均值——
    实测真实的 case2 配对与均值的距离是 1.242，比合成图还大。

    正确用法：对几十张图合成 LQ，取指纹平均后再与 REAL_LQ_FINGERPRINT 比。
    标定时的参考值（48 个 crop 整体匹配）：
        最优退化链 0.212 / 仅高斯无 JPEG 0.568 / 仅双边 3.256
    """
    ref = ref or REAL_LQ_FINGERPRINT
    w = dict(grad_ratio=1.0, lapvar_ratio=2.0, noise_ratio=1.5,
             low=2.0, mid=2.0, high=2.0, psnr=1.5)
    # 归一化尺度（取参考值的 10% 或经验标准差）
    scale = dict(grad_ratio=0.140, lapvar_ratio=0.048, noise_ratio=0.088,
                 low=0.014, mid=0.163, high=0.101, psnr=7.17)
    tot = sum(w[k] * abs(fp[k] - ref[k]) / scale[k] for k in ref)
    return tot / sum(w.values())


if __name__ == "__main__":
    """自检：对一批图合成 LQ，用【整体平均指纹】与真实退化对比。

    用法:
        python degradation.py <图片目录 或 单张图>  [采样张数]
    """
    import sys, glob, os

    if len(sys.argv) < 2:
        print(__doc__)
        print("用法: python degradation.py <图片目录|单张图> [采样张数=30]")
        sys.exit(0)

    Image.MAX_IMAGE_PIXELS = None
    src = sys.argv[1]
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.png")) +
                       glob.glob(os.path.join(src, "*.jpg")))[:n_max]
    else:
        files = [src]
    if not files:
        print(f"未在 {src} 找到图片")
        sys.exit(1)

    rng = np.random.RandomState(0)
    acc = {"rand": [], "best": []}
    for f in files:
        gt = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        if min(gt.shape[:2]) > 512:
            h, w = gt.shape[:2]
            gt = gt[h // 2 - 256:h // 2 + 256, w // 2 - 256:w // 2 + 256]
        if min(gt.shape[:2]) < 64:
            continue
        acc["rand"].append(fingerprint(degrade(gt, rng), gt))
        acc["best"].append(fingerprint(degrade_fixed(gt), gt))

    keys = ["grad_ratio", "lapvar_ratio", "noise_ratio", "low", "mid", "high", "psnr"]
    print(f"样本 {len(acc['rand'])} 张（整体平均指纹，单张对比无意义）")
    print()
    print(f"{'':<14}{'grad':>9}{'lapvar':>9}{'noise':>9}{'low':>8}"
          f"{'mid':>8}{'high':>8}{'psnr':>8}{'距离':>8}")
    print("-" * 80)
    ref = REAL_LQ_FINGERPRINT
    print(f"{'真实 LQ':<14}" + "".join(f"{ref[k]:>9.3f}" if k in
          ("grad_ratio", "lapvar_ratio", "noise_ratio") else f"{ref[k]:>8.3f}"
          for k in keys) + f"{'--':>8}")
    for tag, lab in [("rand", "随机采样"), ("best", "固定最优")]:
        m = {k: float(np.mean([f[k] for f in acc[tag]])) for k in keys}
        print(f"{lab:<14}" + "".join(f"{m[k]:>9.3f}" if k in
              ("grad_ratio", "lapvar_ratio", "noise_ratio") else f"{m[k]:>8.3f}"
              for k in keys) + f"{match_distance(m):>8.3f}")

    if len(files) == 1:
        Image.fromarray(degrade_fixed(
            np.asarray(Image.open(files[0]).convert("RGB"), dtype=np.uint8)
        )).save("_lq.png")
        print("\n已保存 _lq.png")
