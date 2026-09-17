"""MANIQA 变体错位探针。

发现：CSIG 评分公式来自 NTIRE 2026 Mobile RealWorld SR，其官方 eval.py 用的是
  pyiqa.create_metric('maniqa')   —— 默认 = KonIQ 权重
而我们全项目（训练损失 + 所有评测）用的是 settings.py 里的
  DEFAULT_MANIQA = 'maniqa-pipal'  —— PIPAL 权重，完全不同的网络

旁证：学弟报告对同样 100 张测试输入测得 maniqa(koniq)=0.268，
我们测得 maniqa-pipal=0.358 —— 两个变体在同一批图上差 33%。

本探针：对已存盘的输出 PNG 同时算两个变体，看排序是否一致。
排序不一致 => 六周的 MANIQA 优化瞄错了靶子，选型全部要重排。
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("CSIG_MANIQA_SAMPLES", "100")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
import numpy as np, torch, random, pyiqa
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda")

# 两个变体分别建（不走 metrics_utils，避免被 env 覆盖）
m_pipal = pyiqa.create_metric("maniqa-pipal", device=DEV)
m_koniq = pyiqa.create_metric("maniqa", device=DEV)
for m in (m_pipal, m_koniq):
    try:
        if hasattr(m.net, "test_sample"):
            m.net.test_sample = 100
    except Exception:
        pass


def seed_all(s=1234):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV)


DIRS = [
    ("LQ输入", "/cache/zihaohe/csig_isr/data/赛题二/验证集", ["case1_lq.jpg", "case2_lq.jpg", "case3_lq.jpg"]),
    ("GT真值", "/cache/zihaohe/csig_isr/data/赛题二/验证集", ["case1_gt.png", "case2_gt.png", "case3_gt.jpg"]),
    ("TinySR零样本", "/cache/zihaohe/csig_isr/eval_out/x1_zeroshot", ["case1.png", "case2.png", "case3.png"]),
    ("VOSR_0.5B_os", "/cache/zihaohe/vosr_val/out_cfg0.5", None),
    ("0.5B_ms_25步", "/cache/zihaohe/teacher_out/VOSR_0.5B_ms_s25", None),
    ("1.4B_os_1步", "/cache/zihaohe/teacher_out/VOSR_1.4B_os_s1", None),
    ("1.4B_ms_8步", "/cache/zihaohe/teacher_out/VOSR_1.4B_ms_s8", None),
]

rows = {}
for tag, d, names in DIRS:
    if names is None:
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.png")))[:3]
    ps = [os.path.join(d, n) for n in names]
    ps = [p for p in ps if os.path.exists(p)]
    if len(ps) < 3:
        print(f"  跳过 {tag}（只有 {len(ps)} 张）", flush=True); continue
    seed_all()
    vp, vk = [], []
    for p in ps:
        t = load(p)
        vp.append(float(m_pipal(t))); vk.append(float(m_koniq(t)))
        del t; torch.cuda.empty_cache()
    rows[tag] = (float(np.mean(vp)), float(np.mean(vk)))
    print(f"  {tag:<16} pipal={rows[tag][0]:.4f}  koniq={rows[tag][1]:.4f}", flush=True)

print()
tags = list(rows)
rp = sorted(tags, key=lambda t: -rows[t][0])
rk = sorted(tags, key=lambda t: -rows[t][1])
a = np.array([rows[t][0] for t in tags]); b = np.array([rows[t][1] for t in tags])
print("pipal 排序:", " > ".join(rp))
print("koniq 排序:", " > ".join(rk))
if len(tags) > 2:
    print(f"两变体相关 r = {float(np.corrcoef(a, b)[0,1]):+.3f}")
print("排序" + ("一致" if rp == rk else "不一致 <<< 六周的 MANIQA 优化可能瞄错了靶子"))
