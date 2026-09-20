# PixelDiT 教师侧训练优化建议（R3）

> 背景：你的 CSIGv2@240k 已经**实战上岗**——我们用它换掉了自家 VOSR 教师做蒸馏，
> 跑完了 4 组对照实验。这份文档给你三样东西：
> 1) 蒸馏侧的完整实测证据（你的教师哪里强、哪里是短板）；
> 2) 教师侧训练的优化建议（按优先级，含我们踩过的反面教材）；
> 3) 全套可直接粘贴的代码：强退化链、确定性种子蒸馏对生成、幻觉自检。
>
> 结论先行：**你的教师 FR 迁移完全成立，唯一值得重训投入的是"强档退化下的保真"。
> 教师不需要任何 NR/IQA 优化——那是我们学生侧的活，你做了反而坏事（见 §3.3）。**

---

## 1. 蒸馏侧实测证据（2026-08-27 ~ 08-28）

### 1.1 教师同场对比：200 张确定性 mix-LQ（50% 强档）

| 教师 | LPIPS | DISTS | PSNR | lf_top（幻觉，越低越好） |
|---|---:|---:|---:|---:|
| LQ 基线 | 0.4414 | 0.2389 | 23.83 | 0.0000 |
| VOSR teacher_v3_mixlq（我方现役） | 0.2414 | 0.1812 | 19.66 | 0.0709 |
| **PixelDiT@240k（你）** | **0.1784** | **0.1494** | **25.04** | **0.0253** |

其中强档子集（91 张，σ_bilateral=110 / σ_gauss=1.8 / JPEG q55）：

| 教师 | LPIPS | DISTS | PSNR | lf_top |
|---|---:|---:|---:|---:|
| VOSR | 0.2561 | 0.1835 | 20.01 | 0.0705 |
| **PixelDiT@240k** | **0.2401** | **0.1832** | **24.46** | **0.0267** |

两个关键读数：
- 全集上你比我方教师 LPIPS 好 0.063、PSNR 高 5.4dB、幻觉干净 2.6 倍——**全面碾压**；
- 但强档子集上 LPIPS 从 0.178 退化到 0.240、DISTS 优势归零——
  **强档是你唯一的短板，也是 v3 重训唯一需要解决的问题**（正好印证 R2 的判断）。

### 1.2 换教师蒸馏的单变量分解（学生侧证据）

学生（TinySR 单步）配方完全不变，只把 teacher_dir 从 VOSR 换成你（20k 蒸馏对）：

| 指标 | VOSR 教师（v8=4.1741） | PixelDiT 教师（v8b=4.0567） | 变化 |
|---|---:|---:|---:|
| LPIPS | 0.3507 | **0.3260** | −0.025 |
| DISTS | 0.3004 | **0.2659** | −0.035 |
| PSNR | 22.96 | **24.15** | +1.19dB |
| MANIQA | 0.6758 | 0.6067 | −0.069 |
| MUSIQ | 62.81 | 56.74 | −6.1 |
| CLIPIQA | 0.8538 | 0.8309 | −0.023 |
| NIQE | 3.326 | 3.564 | +0.24 |

分解：FR 轴 **+0.060 分**（恰好兑现 1.1 表预测的量级），NR 轴 −0.177。
后续两组实验确认了两件事：
- v8d（换不同初始化+带 GAN/gate 的完整配方重跑）= 4.0608 ≈ v8b，
  说明 **FR 增益和 NR 塌陷都是教师本身的性质，稳定可复现，与学生初始化无关**；
- v8c（学生侧 IQA 权重 ×1.5）= 4.1046，NR 收回一半、FR 回吐三分之一，
  说明 **NR 可以由学生侧补，教师完全不需要管 NR**。

### 1.3 对你的意义

你的教师的价值 = 强档保真 + 干净（低 lf_top）。这两样我们学生侧补不了，
NR 我们自己能补。所以 v3 重训的目标函数只有一条：
**把强档子集的 LPIPS 从 0.240 拉向全集的 0.178**，其余一切保持不动。

---

## 2. 教师训练优化建议（按优先级）

### P0：混入固定强档（预期收益最大，我方已验证同款 +0.05）

50% 概率直接走固定参数强档，**不是拉宽随机范围**——拉宽会稀释重样本密度，
固定强档保证一半样本落在真实域强度上。完整代码见 §3.1/§3.2。

从你现有 240k checkpoint **续训**（不要从头），预计 30~50k 步收敛到新域。
你的 detail_losses5x 那条线的 GT 感知细节损失可以保留（它是 FR 性质的，无害）。

### P1：checkpoint 选择加两个锚点（当前按合成域 rank 选不出强档最优）

每 5k 步跑两个自检，最终 checkpoint 按二者加权选：
1. **强档合成锚**：固定 200 张强档 LQ（§3.2 代码 mix=1.0 生成），
   看 LPIPS/DISTS/PSNR。这是 v3 的主目标，必须单独监控，
   混在全集里会被轻档样本平均掉。
2. **真实域锚**：赛题验证集 case1~3 真 LQ 中心 512（R2 提过，包已发你）。
   通过线不变：SR PSNR > 23.95（LQ 基线）且 LPIPS < 0.40。

### P2：教师侧严禁 NR/IQA 优化（我方 teacher_v4 反面教材）

我们在自家 VOSR 教师上试过教师侧直推 NIQE（w_niqe=1.5），结果：
NIQE 4.78 → **6.56（反向恶化）**，LPIPS 崩到 0.509，MANIQA 冲到 0.967
（幻觉线附近，图上开始长不存在的纹理）。结论已重复验证：
**IQA 压力放教师侧必炸，放学生侧配合 FR 势垒才安全**。
你的教师现在 lf_top 0.025 比 GT 本底还干净，这是最宝贵的性质，别为任何
无参考指标牺牲它。

### P3：训练稳定性小项

- **EMA 权重**（decay 0.999）作为发布 checkpoint，few-step 推理下比裸权重稳；
- 强档样本刚混入的前 ~5k 步 loss 会抬升，属于正常域迁移，别急着调 lr；
- 如果显存允许，把强档样本在 batch 内配平（每 batch 固定一半），
  比 per-sample 抛硬币的梯度噪声更小。

### P4（可选，低优先）：逐图弱语义 prompt

维持 R2 的判断：fixed prompt 目前没观察到方向性伪影，P0 见效前不动它。

---

## 3. 强退化链代码（完整可运行）

### 3.1 固定强档（核心三步，参数是我方对齐真实域的标定值）

```python
import io
import cv2
import numpy as np
from PIL import Image

def degrade_strong(img_u8: np.ndarray) -> np.ndarray:
    """固定强档：双边过度降噪 -> 高斯糊 -> JPEG 压缩。

    参数为对齐赛题真实域的标定值，不要随机化：
    真实 ISP 的过度降噪主要伤中频结构，双边 sigma 110 模拟的就是这一档。
    输入输出均为 HxWx3 uint8 RGB。
    """
    x = cv2.bilateralFilter(img_u8, d=9, sigmaColor=110, sigmaSpace=110)
    x = cv2.GaussianBlur(x, (9, 9), 1.8)
    buf = io.BytesIO()
    Image.fromarray(x).save(buf, format="JPEG", quality=55)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8).copy()
```

### 3.2 mix 包装器（50% 强档 + 50% 你现有的 CSIGDegradeV2）

```python
def csig_degrade_v2_mix(img_u8, rng, mix=0.5):
    """rng 是 np.random.RandomState。注意判定抽样必须用同一个 rng，
    否则多 worker 下复现性会碎（我们在这上面吃过亏，见 3.3 的验证纪律）。"""
    if rng.random() < mix:
        return degrade_strong(img_u8)
    return csig_degrade_v2(img_u8, rng)   # 你现有的实现，R2 已核对无误
```

### 3.3 确定性种子蒸馏对生成（我方消费端协议，供你自测或生成蒸馏集）

我们的蒸馏管线**只存教师输出**（20k 张 PNG，`{idx:06d}.png` 命名），
LQ/GT 在训练端按同一套种子逐字重算。这要求生成端与训练端的 rng 消耗顺序
byte-exact 一致。完整 Dataset：

```python
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def crop_seed(idx: int) -> int:
    """确定性种子——生成端与训练端必须完全一致，否则教师输出对不上 lq"""
    return (idx * 104729 + 7919) % (2 ** 31)

class DetPairDataset(Dataset):
    """确定性退化对。rng 消耗顺序（一步都不能换）：
    1) 若源图大于 crop 尺寸：randint(y)、randint(x)
    2) random() 判定水平翻转
    3) randint(4) 决定旋转
    4) random() < mix 判定强档（强档分支内部无随机性）
    5) 否则 legacy 链继续消耗该 rng
    """

    def __init__(self, paths, size=512, mix=0.5):
        self.paths = paths
        self.size = int(size)
        self.mix = float(mix)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.RandomState(crop_seed(i))
        gt = np.asarray(Image.open(self.paths[i]).convert("RGB"), dtype=np.uint8)
        h, w, _ = gt.shape
        s = self.size
        if h > s or w > s:
            y = rng.randint(0, max(h - s, 0) + 1)
            x = rng.randint(0, max(w - s, 0) + 1)
            gt = gt[y:y + s, x:x + s]
        if gt.shape[0] != s or gt.shape[1] != s:
            gt = cv2.resize(gt, (s, s), interpolation=cv2.INTER_CUBIC)
        if rng.random() < 0.5:
            gt = gt[:, ::-1]
        k = int(rng.randint(4))
        if k:
            gt = np.rot90(gt, k)
        gt = np.ascontiguousarray(gt)
        if self.mix > 0 and rng.random() < self.mix:
            lq = degrade_strong(gt)
        else:
            lq = degrade_legacy(gt, rng)   # 你的 v2 链（消耗同一 rng）
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.).permute(2, 0, 1)
        return to_t(lq), to_t(gt), i
```

**验证纪律（务必照做，"实现了"不等于"生效了"）**：强档分支判定后内部无随机性，
所以可以反向自检——重放 rng 消耗序列找出强档 idx，对导出的 GT 重新施加
`degrade_strong`，与导出的 LQ 逐字节比对：

```python
def verify_pairs(gt_dir, lq_dir, src_paths, n=30, size=512):
    import glob
    ok = tested = 0
    for i in range(n):
        rng = np.random.RandomState(crop_seed(i))
        im = Image.open(src_paths[i])
        h, w = im.size[1], im.size[0]
        if h > size or w > size:
            rng.randint(0, max(h - size, 0) + 1)
            rng.randint(0, max(w - size, 0) + 1)
        rng.random(); rng.randint(4)
        if rng.random() < 0.5:                    # 强档 idx
            gt = np.asarray(Image.open(f"{gt_dir}/{i:06d}.png"))
            lq = np.asarray(Image.open(f"{lq_dir}/{i:06d}.png"))
            tested += 1
            ok += int(np.array_equal(degrade_strong(gt), lq))
    print(f"强档样本 {tested} 个，byte-exact 复现 {ok} 个")  # 必须全中
```

我们 2026-08-27 实测 14/14 全中后才敢开跑。任何一处顺序改动（比如把
翻转放到旋转后面）都会让教师输出和 LQ 静默错位——训练不会报错，只会白炼。

---

## 4. 自检工具

### 4.1 lf_top 幻觉检测（教师必须过这关）

低频偏移 top-k：SR 与 LQ 各下采 16 倍取低频，逐位置绝对差取 top 0.1% 均值。
编造出来的结构（我们这边长过仙人掌）必然改变局部低频，此指标会亮：

```python
import torch.nn.functional as F

def lf_top(sr, lq):
    """sr/lq: [1,3,H,W]，0..1。参考线：GT 本底约 0.02~0.03；
    >0.05 = 低频编造偏多（我方 VOSR 教师 0.071，有幻觉前科）；
    你的 240k 版 0.025，比 GT 本底还干净——v3 必须守住这个水平。"""
    lo = F.avg_pool2d(sr, 16)
    ll = F.avg_pool2d(lq, 16)
    dev = (lo - ll).abs().mean(1).flatten()
    k = max(1, int(dev.numel() * 0.001))
    return float(torch.topk(dev, k).values.mean())
```

### 4.2 推理协议（我们生成教师集时的固定用法，供你对齐）

```
python -m eval_checkpoint_paired_lq \
  --lq-dir <确定性LQ目录> --data <确定性GT目录> \
  --config <delivery>/train_config.yaml --checkpoint <delivery>/checkpoint-best-last10 \
  --num-images 10000 --steps 8 --cfg-scale 1.0 --seed 43 \
  --fixed-prompt "A clean, natural, high-quality image with sharp, well-defined details, ..."
```

实测 vGPU-48 上 0.9 秒/张（8 步、512px），20k 集双卡 2 小时 24 分。
v3 出来后我们重生成 20k 教师集 + 重蒸馏一轮，全流程已自动化，半天内出对照数字。

---

## 5. 验收标准与预期收益

v3 验收（我来跑，也可以你自己跑）：

```
强档合成锚（200 张固定强档）：LPIPS < 0.20（现 0.240），PSNR > 25（现 24.46）
真实域锚（case1~3）        ：SR PSNR > 23.95 且 LPIPS < 0.40
幻觉                       ：lf_top < 0.03（现 0.025，不得恶化）
轻/中档                    ：LPIPS 不得比 240k 版差 0.01 以上（防灾难遗忘）
```

预期收益测算：学生侧换你的教师已实测 FR +0.060 分；强档教师 LPIPS 每收窄
0.02，按 200 图对照的传导率估算学生侧再 +0.01~0.02 分。叠加我们学生侧
正在调的 NR 恢复配方，这是当前全队冲 100+ 唯一量化过的通路。

---
*CSIG 赛道二 · 干就完了 · 2026-08-28 第三轮反馈（R3）*
