"""离线提取 v8 教师 DiT 中间特征(block 7 输出, [1024,1536] token)→ PCA 到 256 维 → 存 fp16 [32,32,256]。
数据用与训练端逐字一致的确定性 LQ(DistillDataset 的 lq 通道)。"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code"); sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch
from torch.utils.data import DataLoader
from settings import ROOT
from tinysr_runtime import build_model
from train_distill import DistillDataset
DEV, DT = torch.device("cuda"), torch.float16
BLK = int(os.environ.get("FEAT_BLOCK", "7")); PCA_K = 256
OUT = f"/cache/zihaohe/distill_data/v8_feat_b{BLK}"; os.makedirs(OUT, exist_ok=True)
core, _ = build_model("/cache/zihaohe/csig_isr/train_out/distill_v8_best.pth", device=DEV, dtype=DT, merge_lora=True, alpha_mode="auto"); core.eval()
tr = core.tr

@torch.no_grad()
def feat(lq01):
    x = lq01 * 2 - 1
    z = core.vae.encode(x).latents * core.vae.config.scaling_factor
    h = tr.pos_embed(z)
    for i, blk in enumerate(tr.transformer_blocks):
        h = blk.forward_(h)
        if i == BLK: return h   # [B, 1024, 1536]
    return h

paths = sorted(glob.glob(os.path.join(ROOT, "train_data", "lsdir512", "*.png")))
ds = DistillDataset(paths, "/cache/zihaohe/distill_data/v8_teacher", mix=0.5)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6)
# 1) PCA 基:前 400 张
acc = []
for bi, (lq, gt, te) in enumerate(dl):
    with torch.autocast("cuda", dtype=DT):
        f = feat(lq.to(DEV, DT)).float()
    acc.append(f.reshape(-1, f.shape[-1])[::8].cpu())
    if bi >= 49: break
X = torch.cat(acc); mu = X.mean(0, keepdim=True); Xc = X - mu
U, S, V = torch.pca_lowrank(Xc, q=PCA_K, center=False)
print(f"PCA: {X.shape} -> {PCA_K}, 解释方差 {float((S**2).sum()/ (Xc**2).sum()):.3f}", flush=True)
torch.save({"mu": mu.half(), "V": V.half(), "block": BLK}, f"{OUT}/pca.pt")
mu_d = mu.to(DEV); V_d = V.to(DEV)
# 2) 全量投影
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6)
n = 0
for lq, gt, te in dl:
    idxs = [ds.items[n + j][0] for j in range(lq.shape[0])]
    with torch.autocast("cuda", dtype=DT):
        f = feat(lq.to(DEV, DT)).float()
    p = ((f - mu_d) @ V_d).reshape(f.shape[0], 32, 32, PCA_K).half().cpu()
    for j, i in enumerate(idxs): torch.save(p[j].clone(), f"{OUT}/{i:06d}.pt")
    n += lq.shape[0]
    if n % 2000 == 0: print(f"feat {n}/20000", flush=True)
print("FEAT_DONE", flush=True)
