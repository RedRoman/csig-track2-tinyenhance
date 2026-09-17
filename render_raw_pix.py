import os, sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code"); sys.path.insert(0, "/cache/zihaohe")
import numpy as np, torch
from PIL import Image
from settings import VAL, TEST
from pix_model import build_pix
Image.MAX_IMAGE_PIXELS = None
DEV, DT = torch.device("cuda"), torch.float16
CK, OUT = sys.argv[1], sys.argv[2]
m = build_pix(CK, device=DEV, dtype=DT)
def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)[None]
def save(t, p): Image.fromarray((t[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(p, compress_level=1)
os.makedirs(f"{OUT}/val", exist_ok=True); os.makedirs(f"{OUT}/test", exist_ok=True)
with torch.no_grad(), torch.autocast("cuda", dtype=DT):
    for f in ("case1_lq.jpg","case2_lq.jpg","case3_lq.jpg"):
        save(m(load(os.path.join(VAL,f)).to(DEV,DT)).float().clamp(0,1), f"{OUT}/val/{f[:-4]}.png")
    tf = sorted(glob.glob(os.path.join(TEST,"*.jpg")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    for i, f in enumerate(tf, 1):
        save(m(load(f).to(DEV,DT)).float().clamp(0,1), f"{OUT}/test/{os.path.basename(f)[:-4]}.png")
print("RAW_PIX_DONE", flush=True)
