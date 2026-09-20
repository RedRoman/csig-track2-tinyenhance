"""预测榜分(以 L2c 包 138.41 为锚)。用法: predict.py --nr 2.7108 --fr 1.3406 --ratio 0.69 [--label X]
ratio = 候选(fold+graph runner)相对 L2d_g 包的同卡时延比(time_pair.py 的 ratio 列)。
模型: 榜分 ∝ 2^(NR+FR) × (1/t_eff)^0.2, t_eff = a + k×g(AutoDL 尺度 GPU ms), 四点拟合 a=0.62 k=1.54;
L2c 包 g=0.598; L2d_g 包 g=0.598×0.83=0.497(fold+CL+graph 实测 −17%)。也给出 a=0.12/k=1.0(小固定开销)情形。"""
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--nr", type=float, required=True); ap.add_argument("--fr", type=float, required=True)
ap.add_argument("--ratio", type=float, required=True); ap.add_argument("--label", default="cand")
ap.add_argument("--eye", default="", help="肉眼判读备注(可选)")
a_ = ap.parse_args()
ANCHOR, S0, G0 = 138.41, 2.7096 + 1.3406, 0.598
g = G0 * 0.83 * a_.ratio; S = a_.nr + a_.fr
for a, k, tag in ((0.62, 1.54, "四点拟合 a=0.62 k=1.54"), (0.12, 1.00, "乐观 a=0.12 k=1.0"), (1.00, 1.54, "悲观 a=1.00 k=1.54")):
    t0 = a + k * G0; t = a + k * g
    pred = ANCHOR * 2 ** (S - S0) * (t0 / t) ** 0.2
    print(f"{a_.label:10s} S={S:.4f} (ΔS {S-S0:+.4f})  g={g:.3f}ms  [{tag}] pred = {pred:6.1f}")
if a_.eye: print("eye:", a_.eye)
