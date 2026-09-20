
p = "/cache/zihaohe/train_pix.py"; s = open(p).read()
old = '    ap.add_argument("--norm", default="gn", choices=["gn", "none"], help="v2 块归一化:gn=AdaGN,none=无归一化(Blk2)")\n'
new = old + '    ap.add_argument("--nbs_dec", default="", help="v2:解码侧每层块数(level0..L-2),空=镜像 enc;如 0,1")\n'
assert old in s; s = s.replace(old, new, 1)
old = 'model = PixUNet2(chs=chs, nbs=tuple(int(v) for v in args.nbs.split(",")), norm=args.norm, tdim=args.tdim, ps=args.ps)'
new = 'model = PixUNet2(chs=chs, nbs=tuple(int(v) for v in args.nbs.split(",")), norm=args.norm, tdim=args.tdim, ps=args.ps, nbs_dec=(tuple(int(v) for v in args.nbs_dec.split(",")) if args.nbs_dec else None))'
assert old in s; s = s.replace(old, new, 1)
open(p, "w").write(s); print("train_pix patched (nbs_dec)")
