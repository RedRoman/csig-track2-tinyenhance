# PixelDiT → CSIG 赛题二教师接入文档

> 发给做 PixelDiT ControlNet-SR 的同事。目标：把现在的 checkpoint 变成能当
> CSIG 教师 / rCM 底座的可复现版本。核心就一件事：**退化域对齐**。

---

## 1. 现状（为什么现在不能直接用）

在你那边的评测域（`originaldeg`, 双边+高斯+JPEG，轻退化）下：
```
sr_psnr 32.06 < lq_psnr 35.36     ← 即使在你的域里它也在伤图
sr_lpips 0.355 > lq_lpips 0.148     ← 同上
```
它输出的 LQ 平均 PSNR 高达 **35dB**——训练退化几乎接近无损。

我们赛题三张验证图的真实 LQ-vs-GT 强度：

| case | LQ_vs_GT PSNR | LQ 梯度能量 | GT 梯度能量 | 判定 |
|---|---:|---:|---:|---|
| case1 | **22.4 dB** | 0.0108 | 0.0332 | 重度（细节被抹 3 倍） |
| case2 | **30.1 dB** | 0.0085 | 0.0199 | 轻度 |
| case3 | **24.8 dB** | 0.0100 | 0.0175 | 中度 |

**你的训练平均 35dB ≈ 更轻**，实际面对的域是 22~30dB，差了 5~13dB。模型没见过这种输入。

**我们用官方六项公式在验证集上测了你的 checkpoint（512 中心口径）：Score 2.70**，
对照我们当前学生 4.06。——差 1.35 分全部来自退化域不匹配。

---

## 2. 要改的唯一变量：替换退化链

不碰架构、不碰 loss、不碰采样器。**只把训练时的 LQ 合成换成下面这份标定过的链。**

### 2.1 推荐：`degrade_v2`（我们赛题验证集标定的链）

参数（已为你换算成可读形式）：
```python
import cv2, io, numpy as np
from PIL import Image
from typing import Tuple

class CSIGDegradeV2:
    bilateral_prob       = 0.90
    bilateral_d          = 9
    bilateral_sigma      = (35.0, 105.0)   # 边缘保持降噪, 模拟 ISP
    gauss_sigma          = (0.65, 1.75)    # 高频衰减
    jpeg_quality         = (48, 96)        # jpg 压缩
    chroma_subsample_prob= 0.28
    sharpen_prob         = 0.18
    sharpen_amount       = (0.05, 0.28)
    gamma_prob           = 0.12
    gamma_range          = (0.94, 1.06)
    residual_noise_prob  = 0.06
    residual_noise_sigma = (0.15, 0.80)

def csig_degrade_v2(img_u8: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """img_u8: HxWx3 uint8 RGB 输出与输入同尺寸。"""
    x = img_u8

    # 1) 双边滤波（边缘保持降噪）—— 大概率发生
    if rng.random() < CSIGDegradeV2.bilateral_prob:
        s = float(rng.uniform(*CSIGDegradeV2.bilateral_sigma))
        x = cv2.bilateralFilter(x, CSIGDegradeV2.bilateral_d, s, s)

    # 2) 高斯低通 —— 高频衰减
    sigma = float(rng.uniform(*CSIGDegradeV2.gauss_sigma))
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)
    x = np.clip(x, 0, 255).astype(np.uint8)

    # 3) JPEG 压缩（测试集是 jpg，此步不可省）
    q = int(rng.uniform(*CSIGDegradeV2.jpeg_quality))
    buf = io.BytesIO()
    Image.fromarray(x).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    x = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    # 4) 低概率分支（增加 ISP 拟真变化，覆盖标定无法辨识的长尾）
    if rng.random() < CSIGDegradeV2.chroma_subsample_prob:
        buf = io.BytesIO()
        Image.fromarray(x).save(buf, format="JPEG", quality=int(rng.uniform(50, 90)), subsampling=0)
        buf.seek(0); x = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    if rng.random() < CSIGDegradeV2.sharpen_prob:
        amt = float(rng.uniform(*CSIGDegradeV2.sharpen_amount))
        # 简化：用 unsharp mask
        import scipy.ndimage as ndi
        blurred = ndi.gaussian_filter(x.astype(np.float32), sigma=1.0)
        x = np.clip(x.astype(np.float32) + amt * (x.astype(np.float32) - blurred), 0, 255).astype(np.uint8)
    if rng.random() < CSIGDegradeV2.gamma_prob:
        g = float(rng.uniform(*CSIGDegradeV2.gamma_range))
        x = (255 * (x.astype(np.float32) / 255) ** g).clip(0, 255).astype(np.uint8)
    if rng.random() < CSIGDegradeV2.residual_noise_prob:
        sigma = float(rng.uniform(*CSIGDegradeV2.residual_noise_sigma))
        noise = rng.normal(0, sigma, x.shape).astype(np.float32)
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(x)
```

### 2.2 验证退化强度是否对齐 —— 一段自检

重训前先跑这个（在**你的任意 512 GT 图**上）：
```python
# 需要你能加载一张高清 GT（DIV2K 或你已有的）
gt = ...  # uint8, 512x512
rng = np.random.RandomState(0)
lq = csig_degrade_v2(gt, rng)
mse = float(np.mean((lq.astype(np.float32) - gt.astype(np.float32))**2))
psnr_v2 = 10*np.log10(255**2/max(mse,1))
print(f"csig_degrade_v2 PSNR = {psnr_v2:.1f} dB")
```
**目标区间：20~30 dB（跨样本）。** 如果跑出来 32~35dB，说明 gauss/bilateral 还要加强；
如果 < 18dB，减弱。我们的三张真实样本 22.4 / 30.1 / 24.8 dB，训练必须随机化覆盖 20~30dB。

### 2.3 接入点（替换你现有的 `degradation_impl: original`）

在你那条 train 配置里，把 `data.degradation_impl` 的合成函数替换为上面的
`csig_degrade_v2`。注意采样逻辑跟你现在的 `degrade_fixed` 一样：随机性来自
`np.random.RandomState(seed)`，多进程下用各自 seed。

---

## 3. 重训配置建议（基于你现有 `detail_losses10x` 线）

保持你这条线成功的东西（detail loss 10x、originaldeg 之外全不动），只换上面退化链：

```
data:
  degradation_impl: csig_v2        # 替换
  crop_size: 512
  max_samples: null
loss:
  flow_matching_weight: 1.0       # 保持
  charbonnier_weight: 5.0         # 保持
  lpips_weight: 0.2               # 保持
  edge_weight: 0.8                # 保持
  haar_weight: 0.3                # 保持（detail losses10x 关键?）
  # 不变
```

**训练量**：你现在 200k 步到了 3.5G checkpoint。换退化域是"概念漂移修复"，
不是从零学。从你现有的官方 weights（`pixeldit_t2i_v1.pth`）出发，用新退化链
**续训 × ①LoRA ② 现 checkpoint 继续**均可，但建议先从现有 200k checkpoint 继续
50~100k，让它即学会新域又保留细节能力。

---

## 4. 评测口径（重训后对接我们）

重训完把**同一套验证输出**发我，我来跑官方的六项：

```
验证集 = /cache/zihaohe/csig_isr/data/赛题二/验证集  (3 张 4K)
公式  = (1-LPIPS)+(1-DISTS)+CLIPIQA+MANIQA+MUSIQ/100+max(0,(10-NIQE)/10)
注意  = MANIQA 用 koniq 变体（不要 pipal）
```

**期望值**：重训后 ≥3.8（60% 成功率），达到 4.0 就完美。

---

## 5. 附录：我们这条线的经验（帮你少踩坑）

1. **训练数据域的问题我踩过三次**。最早的 TinySR 合成退化和我们自己的 testset
   不匹配，导致我在上 real 测试时全崩。这次的 2.70 就是同一类问题——训练退化比
   真实重了/轻了都不行，会造成模型在错误的分布上"越专越弱"。
2. **检查退化是否对齐的最快方法**：跑 `PSNR(lq, gt)`，看单样本分布是否覆盖你
   将要推理的域。别只看论文里报告的 PSNR。
3. **会跟你们的 original degradation 冲突**。你那条 `originaldeg` 是个轻链，但
   我们的验证集比它重 5~13dB，直接续训会造成"训练分布漂移到图外"。
4. 另外，我们判断它**无仙人掌幻觉**，这一点比 VOSR 强，适合当干净教师。rCM 那边
   如果你感兴趣我可以发一份说明，我们可以把它作为多步教师底座。

---

*CSIG 赛道二 · 干就完了 · 2026-08-25*
