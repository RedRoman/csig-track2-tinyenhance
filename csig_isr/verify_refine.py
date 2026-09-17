"""恢复后自检：多步细化的恒等性与时延。"""
import sys, warnings, time
sys.path.insert(0, "/cache/zihaohe/csig_isr/v3/code")
warnings.filterwarnings("ignore")
import torch
from tinysr_runtime import build_model

DEV, DT = torch.device("cuda"), torch.float16
CK = "/cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth"
x = torch.rand(1, 3, 512, 512, device=DEV, dtype=DT)


def bench(m, n=40, w=15):
    with torch.no_grad():
        for _ in range(w):
            with torch.autocast(device_type="cuda", dtype=DT): m(x)
        torch.cuda.synchronize(); best = 1e9
        for _ in range(4):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n):
                with torch.autocast(device_type="cuda", dtype=DT): m(x)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter()-t0)/n*1000)
    return best


m0, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=True,
                    alpha_mode="fixed", fixed_alpha=1.0)
m0.eval()
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=DT): y0 = m0(x).float()
print(f"单步  {bench(m0):6.2f} ms", flush=True)

for N in (2, 3):
    m, _ = build_model(CK, device=DEV, dtype=DT, merge_lora=True,
                       alpha_mode="fixed", fixed_alpha=1.0)
    m.eval(); m.enable_refine(N)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=DT): y = m(x).float()
    d = (y - y0).abs().max().item()
    ok = "恒等 OK" if d < 1e-6 else "不恒等"
    print(f"{N} 步  {bench(m):6.2f} ms   与单步最大差 {d:.3e}  {ok}", flush=True)
    del m; torch.cuda.empty_cache()
