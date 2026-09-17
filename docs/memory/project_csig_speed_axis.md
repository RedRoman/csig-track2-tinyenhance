---
name: project-csig-speed-axis
description: 时延轴突破——现场 torch.compile 7.38ms 使 v8 预测 102 分;v8 提交件已就绪含回退链
metadata: 
  node_type: memory
  type: project
  originSessionId: 104e347e-c0e8-47a0-8633-b9d5b60702af
---

2026-08-31:教师侧枯竭(v9/v10 证据)后转时延轴,当天完成全链:

**核心数字**:榜分=3.357×2^S×(100.15/ms)^0.2。v8(S=4.1741)@11.31ms=93.8;
@7.38ms=**102.1 过百线**。compile max-autotune 实测 7.38ms(1.53×),
真图数值 p50=0.0005(bench 随机噪声输入的 |d|=0.145 是混沌放大,虚惊,勿再被骗)。

**踩坑记录**:AOTI(torch._export.aot_compile)导出后输出彩色涂抹,
弃用;compile(jit ScriptModule) 零加速(dynamo 黑盒);
TinySD3 是首次前向懒折叠结构(temb 烘焙进 per-block modulate 后删模块),
from_config 重建键不匹配 → 解法=pickle 整模块(tr_folded.pt 684MB);
pickle 需要 SeverityGate 类 → model_dir 放 tinysr_runtime.py stub;
tinysd3.py 硬编码 dataset/cache/pos_embed.pt 相对路径已 patch。

**提交件**:30674 `/cache/zihaohe/csig_isr/submit_v8/赛题二_TinyEnhance_干就完了_18020768360.zip`
(2.07GB,output_dir=v8 渲染含 wavelet colorfix + model_dir)。
runner 链:现场 compile max-autotune(冷 init 149s 实测)→失败自动回退 jit(11.3ms,93.8分)。
从 zip 解包冷缓存端到端冒烟通过(OptimizedModule 7.38ms)。
评测机若 compile 失败(torch 老/无 triton)只回退到 93.8,不会更差。
未提交,等用户决定。旧 submit/ 与 submit_probe/ 两个 zip 未动。

相关:[[project-csig-track2]] [[project-vosr2-fill2sr-recon]]
