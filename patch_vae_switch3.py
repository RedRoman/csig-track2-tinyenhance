"""补丁 v3：隐空间标度桥接。

官方 taesd3 的 latent std 是 separable 的 1.977 倍（相关 0.96）。DiT 是在
separable 标度下训出来的，直接换 VAE 等于把它扔进一个陌生分布，实测起始
验证分 2.51、4 步里 2 步梯度 nan。

做法：在 DiT 前后各插一次标量换算，让 DiT 始终工作在它熟悉的标度上，
而编码/解码用高保真的 taesd3。这样零样本就接近原状，训练只需消化
两个隐空间之间那部分非线性残差（逐通道仿射 R2=0.739）。

  z_dit = z_taesd3 / s
  zo_dit = z_dit - alpha * pred
  z_out  = zo_dit * s   -> taesd3 解码

s 由环境变量 CSIG_LATENT_SCALE 给出，缺省 1.0（不改变现有行为）。
"""
import sys, shutil

P = "/cache/zihaohe/csig_isr/v3/code/tinysr_runtime.py"
src = open(P).read()

if "CSIG_LATENT_SCALE" in src:
    print("v3 已打过，跳过"); sys.exit(0)

OLD = """    def _impl(self, lq01):
        x = lq01 * 2 - 1
        sf = self.vae.config.scaling_factor
        z = self.vae.encode(x).latents * sf"""
NEW = """    def _impl(self, lq01):
        x = lq01 * 2 - 1
        sf = self.vae.config.scaling_factor
        ls = float(os.environ.get("CSIG_LATENT_SCALE", "1.0"))
        z = self.vae.encode(x).latents * sf / ls"""

OLD2 = """        img = self.vae.decode(zo / sf, return_dict=False)[0]
        return (img * 0.5 + 0.5).clamp(0, 1), alpha"""
NEW2 = """        img = self.vae.decode(zo * ls / sf, return_dict=False)[0]
        return (img * 0.5 + 0.5).clamp(0, 1), alpha"""

for o in (OLD, OLD2):
    if o not in src:
        print("找不到目标块：\\n" + o[:80]); sys.exit(1)

shutil.copy(P, P + ".bak_v3")
src = src.replace(OLD, NEW).replace(OLD2, NEW2)
open(P, "w").write(src)
print("v3 补丁已打（CSIG_LATENT_SCALE 缺省 1.0，不影响现有行为）")
