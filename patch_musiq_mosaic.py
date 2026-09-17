"""给训练脚本接入 mosaic4K 的 MUSIQ 口径。

实测（musiq_proxy.py）：MUSIQ 的响应取决于图像绝对尺寸。
  crop512（现状）与真实 4K 排序 r = -0.544   —— 一直在反向优化
  mosaic4K（512 输出平铺成 4096x3072）r = +0.997
所以把 MUSIQ 项的输入换成 mosaic4K，其余口径不动（NIQE/MANIQA 实测在 512 上是对的）。

新参数 --musiq_proxy {crop,mosaic4k}，缺省 crop（不改变现有行为）。
mosaic 只对 1 张样本做，4K 前向带梯度很吃显存。
"""
import sys, shutil

P = "/cache/zihaohe/csig_isr/v3/code/train_tinysr.py"
src = open(P).read()

if "musiq_proxy" in src:
    print("已打过，跳过"); sys.exit(0)

# 1) 新增参数
OLD1 = '    ap.add_argument("--w_musiq", type=float, default=1.0)'
NEW1 = ('    ap.add_argument("--w_musiq", type=float, default=1.0)\n'
        '    ap.add_argument("--musiq_proxy", choices=["crop", "mosaic4k"], default="crop",\n'
        '                    help="MUSIQ 的度量口径。crop=直接在 512 块上算（与官方 4K 口径 r=-0.54，'
        '反向）；mosaic4k=平铺成 4096x3072 再算（r=+0.997）")\n'
        '    ap.add_argument("--musiq_mosaic_n", type=int, default=1,\n'
        '                    help="mosaic4k 时参与的样本数；4K 前向带梯度很吃显存")')

# 2) 插入 mosaic 辅助函数（放在 select_dense_params 之前）
OLD2 = "def select_dense_params(model, last_n):"
NEW2 = '''def _musiq_input(t, mode, n):
    """把 512 输出拼成 4K 尺寸，让 MUSIQ 落回官方评测的响应区间。可导。"""
    if mode != "mosaic4k":
        return t
    t = t[:max(1, int(n))]
    H, W = 3072, 4096
    ry = -(-H // t.shape[-2]); rx = -(-W // t.shape[-1])
    return t.repeat(1, 1, ry, rx)[:, :, :H, :W]


def select_dense_params(model, last_n):'''

# 3) 两处 musiq 调用改走 _musiq_input
OLD3 = '                    mu = loss_metrics["musiq"](iq).mean() / 100.0\n                    iqa_reward = args.w_clipiqa * ci + args.w_maniqa * ma + args.w_musiq * mu'
NEW3 = '                    mu = loss_metrics["musiq"](_musiq_input(iq, args.musiq_proxy, args.musiq_mosaic_n)).mean() / 100.0\n                    iqa_reward = args.w_clipiqa * ci + args.w_maniqa * ma + args.w_musiq * mu'

OLD4 = '                        mu = loss_metrics["musiq"](iq).mean() / 100.0\n                        iqa_reward = 3.0 * args.w_musiq * mu'
NEW4 = '                        mu = loss_metrics["musiq"](_musiq_input(iq, args.musiq_proxy, args.musiq_mosaic_n)).mean() / 100.0\n                        iqa_reward = 3.0 * args.w_musiq * mu'

for i, o in enumerate((OLD1, OLD2, OLD3, OLD4), 1):
    if o not in src:
        print("找不到第 {} 处目标:\\n{}".format(i, o[:120])); sys.exit(1)

shutil.copy(P, P + ".bak_mosaic")
src = src.replace(OLD1, NEW1).replace(OLD2, NEW2).replace(OLD3, NEW3).replace(OLD4, NEW4)
open(P, "w").write(src)
print("补丁已打（--musiq_proxy 缺省 crop，不影响现有行为）")
