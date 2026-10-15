"""榜分预测(经验模型,五个提交点拟合,残差 ±2 分)。

    榜分 ≈ anchor_score × 2^(S − anchor_S) × ((a + k·g_anchor)/(a + k·g))^0.2
      S = test NR(100 张) + val FR(3 张)        g = 512×512 fp16 单发 GPU 时间(ms)
      a=0.49(评测机固定开销), k=2.17            锚点 = G4: 152.95 @ S 4.0841, g 0.431ms

g 的口径:AutoDL vGPU-48G 上、fold+channels_last+CUDA graph runner 的独占卡 event 计时。
别的机器上量的绝对值不可直接代入(30674 A100 上同模型读数约 1.6×)。稳妥做法是同机相对标定:
用 examples/submit_pixG4_g 在你的机器上量一次 ref_ms,再 --ref_ms 传入,工具按 0.431/ref_ms 换算。

    python predict.py --nr 2.8128 --fr 1.3003 --ms 0.478
    python predict.py --nr 2.8128 --fr 1.3003 --ms 0.78 --ref_ms 0.70   # 同机以 G4 为参照换算
"""
import argparse

A, K = 0.49, 2.17
ANCHOR = dict(score=152.95, S=4.0841, g=0.431)   # G4, 2026-09-19 实得


def predict(S, g, a=A, k=K, anchor=ANCHOR):
    return anchor["score"] * 2 ** (S - anchor["S"]) * ((a + k * anchor["g"]) / (a + k * g)) ** 0.2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr", type=float, required=True, help="test NR(100 张)")
    ap.add_argument("--fr", type=float, required=True, help="val FR = (1-lpips)+(1-dists)")
    ap.add_argument("--ms", type=float, required=True, help="512 单发 GPU ms")
    ap.add_argument("--ref_ms", type=float, default=None, help="同机上 examples/submit_pixG4_g 的 512 单发 ms,用于换算到 AutoDL 口径")
    ap.add_argument("--a", type=float, default=A); ap.add_argument("--k", type=float, default=K)
    ap.add_argument("--label", default="cand")
    x = ap.parse_args()
    g = x.ms * (ANCHOR["g"] / x.ref_ms) if x.ref_ms else x.ms
    S = x.nr + x.fr
    p = predict(S, g, x.a, x.k)
    lo, hi = predict(S, g, 1.00, 1.54), predict(S, g, 0.12, 1.00)
    print(f"{x.label:12s} S={S:.4f} (ΔS vs G4 {S-ANCHOR['S']:+.4f})  g={g:.3f}ms (Δ {g/ANCHOR['g']-1:+.1%})  "
          f"pred={p:.1f}   [情景带 {min(lo,hi):.1f} ~ {max(lo,hi):.1f}]")


if __name__ == "__main__":
    main()
