"""验证 TinySR 自带的 ProjectedDiscriminator 能否加载运行。

它基于 DINO ViT-S/16 特征 + 多层 DiscHead（StyleGAN-T 那套 projected GAN），
这是 TinySR 论文里用的对抗训练判别器，与我们的 backbone 完全匹配。
"""
import os, sys, torch
ROOT="/cache/zihaohe/csig_isr"; TS=f"{ROOT}/TinySR"
sys.path.insert(0, TS); os.chdir(TS)
dev=torch.device("cuda")

from models.discriminator import ProjectedDiscriminator
D = ProjectedDiscriminator(c_dim=0).to(dev)
n_tot=sum(p.numel() for p in D.parameters())
n_tr =sum(p.numel() for p in D.parameters() if p.requires_grad)
n_head=sum(p.numel() for p in D.heads.parameters())
print(f"判别器总参数 {n_tot/1e6:.2f}M   可训练 {n_tr/1e6:.2f}M   heads {n_head/1e6:.2f}M")
print(f"DINO 特征 hook 数 {D.dino.n_hooks}  embed_dim {D.dino.embed_dim}  分辨率 {D.dino.img_resolution}")

D.train()
x=torch.rand(2,3,512,512,device=dev)*2-1     # [-1,1]
c=torch.zeros(2,0,device=dev)
out=D(x,c)
def as_list(o):
    if torch.is_tensor(o): return [o]
    if isinstance(o,(list,tuple)):
        r=[]
        for t in o:
            r += as_list(t)
        return [t for t in r if torch.is_tensor(t)]
    return []
outs=as_list(out)
print(f"输出类型 {type(out).__name__}，展开后 {len(outs)} 个张量")
print(f"  形状 {[tuple(o.shape) for o in outs]}")
loss=sum(torch.nn.functional.relu(1-o).mean() for o in outs)
print(f"hinge loss 试算 {loss.item():.4f}")
loss.backward()
g=sum((p.grad is not None) for p in D.heads.parameters())
print(f"反向传播 OK，heads 中有梯度的参数 {g} 个")

# 显存
torch.cuda.reset_peak_memory_stats()
x=torch.rand(4,3,512,512,device=dev)*2-1
o=D(x,torch.zeros(4,0,device=dev))
l=sum(torch.nn.functional.relu(1-t).mean() for t in as_list(o))
l.backward()
print(f"bs=4 判别器峰值显存 {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
