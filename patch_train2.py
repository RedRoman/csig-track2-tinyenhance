p = "/cache/zihaohe/train_pix.py"; s = open(p).read(); orig = s
old = '''        sd = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model = PixUNet(**sd["cfg"]); model.load_state_dict(sd["state"]); print("init from", args.init_ckpt, flush=True)'''
new = '''        sd = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        _cfg = dict(sd["cfg"]); _ver = _cfg.pop("arch_ver", 1)
        model = PixUNet2(**_cfg) if _ver == 2 else PixUNet(**_cfg)
        model.load_state_dict(sd["state"]); print("init from", args.init_ckpt, "step", sd.get("step"), flush=True)'''
assert old in s, "anchor missing"
s = s.replace(old, new, 1); open(p, "w").write(s); print("patched init_ckpt for arch_ver 2")
