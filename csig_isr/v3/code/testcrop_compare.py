"""在【测试图】的 crop 上横向比较候选 —— 提交前的正确第一步。

不用验证集：验证集 case1/2/3 同时被用于训练验证和 best-checkpoint 选择，
在那上面看伪影是有偏的。
"""
import os, sys, numpy as np, torch
from PIL import Image, ImageDraw
from settings import ROOT
from tinysr_runtime import build_model

Image.MAX_IMAGE_PIXELS=None
DEV, DT = torch.device("cuda"), torch.float16
TEST = os.path.join(ROOT, "data", "赛题二", "测试集")
DST  = os.path.join(ROOT, "review_testcrop"); os.makedirs(DST, exist_ok=True)
C = 512
T  = os.path.join(ROOT, "train_out")

CFGS = [
    ("run8 已提交",     f"{T}/tinysr8_fid/best.pth",        1.0),
    ("hf_w2250",        f"{T}/v3_hf_w2250/final.pth",       1.0),
    ("hf_w750",         f"{T}/v3_hf_w750/final.pth",        1.0),
    ("combo_hf1500",    f"{T}/v3_combo_hf1500/best.pth",    1.0),
]
SPOTS = [("case24.jpg",512,1792),("case25.jpg",768,512),
         ("case16.jpg",1024,2048),("case29.jpg",1792,1280)]

crops={}
for nm,x,y in SPOTS:
    a=np.asarray(Image.open(os.path.join(TEST,nm)).convert("RGB"),np.float32)/255.
    crops[nm]=torch.from_numpy(a[y:y+C,x:x+C]).permute(2,0,1)[None]

cols={nm:[("LQ 输入",(crops[nm][0].permute(1,2,0).numpy()*255).astype(np.uint8))] for nm,_,_ in SPOTS}
for tag,ck,al in CFGS:
    if not os.path.exists(ck): print("跳过(无ckpt):",tag); continue
    m,rep=build_model(ck,device=DEV,dtype=DT,merge_lora=False,alpha_mode="fixed",fixed_alpha=al)
    m.eval()
    with torch.no_grad():
        for nm,_,_ in SPOTS:
            with torch.autocast(device_type="cuda",dtype=DT):
                o=m(crops[nm].to(DEV,dtype=DT)).float().clamp(0,1)
            cols[nm].append((tag,(o[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))
    del m; torch.cuda.empty_cache(); print("done",tag,"step",rep.get("step"),flush=True)

S,Z=176,3
for nm,_,_ in SPOTS:
    items=cols[nm]; n=len(items)
    out=Image.new("RGB",(S*Z*n+8*(n-1),S*Z+26),(255,255,255)); d=ImageDraw.Draw(out)
    for i,(tag,img) in enumerate(items):
        t=Image.fromarray(img[(C-S)//2:(C-S)//2+S,(C-S)//2:(C-S)//2+S]).resize((S*Z,S*Z),Image.NEAREST)
        out.paste(t,(i*(S*Z+8),26)); d.text((i*(S*Z+8)+6,8),tag,fill=(0,0,0) if i==0 else (190,0,0))
    f=os.path.join(DST,f"tc_{nm.replace('.jpg','')}.png"); out.save(f); print(f)
