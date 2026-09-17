"""把 FR 从「加权损失项」改成「硬约束」。

今天的数据：mosaic 口径修好 MUSIQ 之后，四项 NR 全部朝同一方向拉，
测试集 NR 冲到 2.7401（史上最高），但 LPIPS 崩到 0.5748、DISTS 0.4025，
比原始 LQ（0.4262/0.2942）还差。净值为负。

原因是 FR 只是损失里的一个加权项，训练过程一路把它换成 NR。
改成铰链势垒：低于阈值时完全不管，一旦越过阈值就以平方惩罚急剧上升，
使 FR 成为不可逾越的边界而不是可交易的筹码。

新参数：--lpips_cap --dists_cap --w_barrier（缺省 0，不改变现有行为）
"""
import sys, shutil

P = "/cache/zihaohe/csig_isr/v3/code/train_tinysr.py"
src = open(P).read()

if "lpips_cap" in src:
    print("已打过，跳过"); sys.exit(0)

OLD1 = '    ap.add_argument("--w_dists", type=float, default=1.5)'
NEW1 = ('    ap.add_argument("--w_dists", type=float, default=1.5)\n'
        '    ap.add_argument("--lpips_cap", type=float, default=0.0,\n'
        '                    help="LPIPS 上界，越界后按平方势垒重罚。0=关闭")\n'
        '    ap.add_argument("--dists_cap", type=float, default=0.0,\n'
        '                    help="DISTS 上界，同上。0=关闭")\n'
        '    ap.add_argument("--w_barrier", type=float, default=0.0,\n'
        '                    help="势垒强度。需远大于 w_iqa 才能真正挡住交易")')

OLD2 = '                loss = loss + args.w_fid * fid'
NEW2 = ('                loss = loss + args.w_fid * fid\n\n'
        '                # FR 势垒：阈值以内不施加任何力，越界后平方增长。\n'
        '                # 目的是让 NR 只能在不破坏保真的前提下上涨。\n'
        '                if args.w_barrier > 0:\n'
        '                    bar = torch.zeros((), device=DEV)\n'
        '                    if args.lpips_cap > 0:\n'
        '                        bar = bar + F.relu(lp - args.lpips_cap) ** 2\n'
        '                    if args.dists_cap > 0:\n'
        '                        bar = bar + F.relu(ds - args.dists_cap) ** 2\n'
        '                    loss = loss + args.w_barrier * bar')

for i, o in enumerate((OLD1, OLD2), 1):
    if o not in src:
        print(f"找不到第 {i} 处目标"); sys.exit(1)

shutil.copy(P, P + ".bak_frcap")
open(P, "w").write(src.replace(OLD1, NEW1).replace(OLD2, NEW2))
print("FR 势垒补丁已打（w_barrier 缺省 0，不影响现有行为）")
