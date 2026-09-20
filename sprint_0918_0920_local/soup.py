"""同血统权重汤:soup.py OUT.pth A.pth B.pth alpha  → OUT = (1-alpha)*A + alpha*B(cfg 须一致)"""
import sys, torch
out, a, b, al = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
A = torch.load(a, map_location="cpu", weights_only=False); B = torch.load(b, map_location="cpu", weights_only=False)
ca={k:v for k,v in A["cfg"].items() if k!="nbs_dec"}; cb={k:v for k,v in B["cfg"].items() if k!="nbs_dec"}
assert ca == cb, (ca, cb)
st = {k: ((1 - al) * A["state"][k].float() + al * B["state"][k].float()).to(A["state"][k].dtype) for k in A["state"]}
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"cfg": A["cfg"], "state": st, "step": -1, "args": {"soup": [a, b, al]}}, out); print("soup saved", out)
