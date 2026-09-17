"""把幻觉检测器接进训练损失。

标定结果（与目视排序完全一致）：
  teacher2 0.2160(整片仙人掌) > vosr_cap 0.1087 > vosr_cap2 0.0807
  > mosaic_cap 0.0781(编织纹) > distill_v1 0.0502 > run12 0.0381(干净)

判据是「局部低频偏差的极值」：真实复原只加高频、保留低频；
凭空长出物体会改变局部低频。训练里已有的 w_fid 算的是全图均值，
局部幻觉会被平均掉——所以这里取 top-k 极值，且可导。

新参数 --w_antihall --antihall_cap --antihall_topk（缺省关闭，不改变现有行为）
"""
import sys, shutil

P = "/cache/zihaohe/csig_isr/v3/code/train_tinysr.py"
src = open(P).read()

if "w_antihall" in src:
    print("已打过，跳过"); sys.exit(0)

OLD1 = '    ap.add_argument("--w_barrier", type=float, default=0.0,'
NEW1 = ('    ap.add_argument("--w_antihall", type=float, default=0.0,\n'
        '                    help="反幻觉势垒强度。0=关闭")\n'
        '    ap.add_argument("--antihall_cap", type=float, default=0.038,\n'
        '                    help="局部低频偏差上界。run12(目视干净)实测 0.0381，'
        'distill_v1 0.0502，teacher2 0.2160")\n'
        '    ap.add_argument("--antihall_topk", type=float, default=0.001,\n'
        '                    help="取偏差最大的这个比例的位置求均值")\n'
        '    ap.add_argument("--w_barrier", type=float, default=0.0,')

OLD2 = '                loss = loss + args.w_fid * fid'
NEW2 = ('                loss = loss + args.w_fid * fid\n\n'
        '                # 反幻觉：局部低频偏差的极值。w_fid 管的是均值，\n'
        '                # 凭空长出的物体是稀疏强信号，只有取极值才看得见。\n'
        '                if args.w_antihall > 0:\n'
        '                    _lo = F.avg_pool2d(out, 16); _ll = F.avg_pool2d(lq, 16)\n'
        '                    _dev = (_lo - _ll).abs().mean(1).flatten(1)\n'
        '                    _k = max(1, int(_dev.shape[1] * args.antihall_topk))\n'
        '                    _top = torch.topk(_dev, _k, dim=1).values.mean()\n'
        '                    loss = loss + args.w_antihall * F.relu(_top - args.antihall_cap) ** 2\n'
        '                else:\n'
        '                    _top = torch.tensor(float("nan"), device=DEV)')

for i, o in enumerate((OLD1, OLD2), 1):
    if o not in src:
        print(f"找不到第 {i} 处目标"); sys.exit(1)

src = src.replace(OLD1, NEW1, 1).replace(OLD2, NEW2, 1)

# 训练日志里打出来，好盯着它有没有真被压住
OLD3 = 'f"lr={opt.param_groups[0][\'lr\']:.3e}'
if OLD3 in src:
    src = src.replace(OLD3, 'f"lf_top={_top.item():.4f} ' + OLD3[2:], 1)

shutil.copy(P, P + ".bak_antihall")
open(P, "w").write(src)
print("反幻觉补丁已打（w_antihall 缺省 0，不影响现有行为）")
