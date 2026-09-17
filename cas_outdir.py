import os, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
D = "/cache/zihaohe/csig_isr/submit_v8b/output_dir"
k = (torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16.).cuda().expand(3,1,3,3)
for f in sorted(glob.glob(os.path.join(D, "*.jpg"))):
    a = np.asarray(Image.open(f).convert("RGB"), np.float32)/255.
    x = torch.from_numpy(a).permute(2,0,1)[None].cuda()
    b = F.conv2d(F.pad(x, (1,1,1,1), mode="reflect"), k, groups=3)
    y = (x + 0.12*(x-b)).clamp(0,1)
    arr = (y[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(arr).save(f, quality=97, subsampling=0)
print("CAS_OUTDIR_DONE", len(glob.glob(os.path.join(D,"*.jpg"))))
