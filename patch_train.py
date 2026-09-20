import re, sys
p = "/cache/zihaohe/train_pix.py"; s = open(p).read(); orig = s
s = s.replace("from pix_model import PixUNet\n", "from pix_model import PixUNet, PixUNet2\n")
s = s.replace('    ap.add_argument("--init_ckpt", default="")\n',
 '    ap.add_argument("--init_ckpt", default="")\n'
 '    ap.add_argument("--nbs", default="", help="v2:逐层块数,如 1,2,2(给出即用 PixUNet2)")\n'
 '    ap.add_argument("--norm", default="gn", choices=["gn", "none"], help="v2 块归一化:gn=AdaGN,none=无归一化(Blk2)")\n'
 '    ap.add_argument("--init_partial", default="", help="按名字+形状匹配的部分初始化(用于改块数后的热启动)")\n')
s = s.replace('''    else:
        model = PixUNet(chs=chs, nb=args.nb, tdim=args.tdim, ps=args.ps)
    model = model.to(DEV).float().train()
''', '''    elif args.nbs:
        model = PixUNet2(chs=chs, nbs=tuple(int(v) for v in args.nbs.split(",")), norm=args.norm, tdim=args.tdim, ps=args.ps)
    else:
        model = PixUNet(chs=chs, nb=args.nb, tdim=args.tdim, ps=args.ps)
    if args.init_partial:
        psd = torch.load(args.init_partial, map_location="cpu", weights_only=False)["state"]; own = model.state_dict()
        ok = {k: v for k, v in psd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(ok, strict=False)
        print(f"partial init from {args.init_partial}: {len(ok)}/{len(own)} tensors matched; missing={sorted(set(own)-set(ok))[:6]}", flush=True)
    model = model.to(DEV).float().train()
''')
s = s.replace('print(f"PixUNet chs={chs} nb={args.nb} ps={args.ps} 参数',
              'print(f"{type(model).__name__} cfg={model.cfg} 参数')
assert s != orig and "PixUNet2(" in s and "init_partial" in s and "type(model).__name__" in s, "patch failed"
open(p, "w").write(s); print("train_pix.py patched")
