# A7750 质量优先：训练后备 CPU 审阅（2026-10-15）

结论：先推进同权重安全加速。暂不启动新训练。相同 HYPIR 架构的125步微调不会直接减少推理计算量；旧500步完整S仅改善0.0017624，不能把短训练当作达到速度目标的主要手段。

## 已核事实

父为 `cuda_nr_stage5_s7250_teacher6500_continue500` 的封存A7750：global7750、该阶段local500、Adam1750。raw SHA `589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d`；immutable resume SHA `e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b`；config SHA `86015a361f5fa1429f1d797f175d27450cb6f798e1df4483badf01d876b2ab5b`；protocol `20b8568b1a8591a8bc0fdde05a02c07ade1608621c970bcc0d0718b56cacce83`。

2026-10-15通过SSH只读重核小文件实际SHA：v5源码 `4b5bdf40aab5d2fd35c92f8fc714c675e62bd86228bc3fb79e48e896b70278f2`，marker `860df8ada386586bb845435e6ee961b861d8b6154a1554783d1e61aa4b2ccc5f`，gradient_preflight `2edc67396b3cf28e31ee5c8217022d7ba11a6b952cdfb4b477746bf2f56dfe14`。此前冻结CPU里程碑审计已检查514份FP32 LoRA/raw=resume、514份有限FP32 Adam矩及step1750、四组RNG；本次未重复加载/重哈希3.1GB resume，也未启动CUDA。

旧训练目标：`L1(teacher6500) + .1 LPIPS(teacher6500) + .3 gradient_L1 - .15*(1.5*MUSIQ/100 + MANIQA + .5*CLIPIQA)`。100 test输入随机512裁块；同一个随机VAE latent供student和冻结teacher使用；两者同样解码、wavelet reconstruction、clamp。训练MANIQA20 crops，完整评分100；无验证GT进入训练。NIQE不在可微训练奖励中，验证FR的DISTS也不在训练loss中。保真于教师不等于FR对GT必然改善。

旧首步三指标真实输入梯度范数：MUSIQ .15570244、MANIQA .13146532、CLIPIQA 1.83034325；全部514参数梯度有限非零、teacher/metrics无参数梯度。CLIP的梯度尺度较大，但单个首crop不能代表全分布。500行loss由既有审计重算通过。旧A相对7250：NR +.00660815、FR -.00484572、S +.00176243；C提高MUSIQ同时减reward未超过A，D更保真却使S下降，因此不能把提高单一NR分量当作完整质量提升。

## 唯一后备（尚未授权运行）

仅若安全加速不足且root决定继续质量微调，创建全新入口/新run；旧v5固定7250/Adam1250、teacher6500/7250、save_every250，不能仅换CLI或monkeypatch使用A7750。

候选单分支：A7750 exact FP32 student + 同A7750冻结FP32 teacher；继承Adam1750及四RNG、加载后显式LR=1e-7；reward=.025（从.15下调），MUSIQ1.5、CLIP.5保持；保真加强为L1 2、LPIPS .2、gradient .6。沿用原crop/latent/wavelet/metric协议，100test且不使用验证GT，不增加新的频率损失、锐化、OCR或其他架构。

先仅25步并强停：local25/global7775/Adam1775；仅在原1:1质量和完整评测合格后，由同一不可变resume继续，最多local125/global7875/Adam1875。保存实际raw、匹配immutable resume、实际SHA/计数、首梯度与log。不能为方便而一口气自动跑125；不存在新的默认多分支。

关键限制：同权重教师使初始fidelity近零，但其L1/gradient约束在相等点也近零；继承Adam1750的一阶矩仍会推动旧方向。降低reward不能保证首步等比变小，Adam归一化也不等于缩小参数更新。必须保留精确迁移证明与25步停审。策略独立意见倾向reset Adam0+LR5e-8，但那会改变root要求的exact Adam继承，故不作为本方案默认，需root另行决定。

最小实现改动仅包含：固定A7750三文件/协议及父计数；teacher强制同A7750；2倍原保真项；新奖励/LR；25/125两个显式边界与对应计数/证明。保持全部514 FP32复制和矩精确校验、构造结束后恢复Python/NumPy/TorchCPU/CUDA RNG，拒绝mutable父resume和覆盖旧目录。新参数更新后必须重新完整103/418评测，不能沿用旧A的评分或质量结论。

## 四例视觉红线

已直接查看既有原1:1同坐标ROI（`nr_push5_visuals_20261014/roi_case{7,16,41,83}_native.png`），未编辑原图或重新选ROI。

- case7：A7750的中文字形、笔画交接和窗框平直；学生仍有笔画鼓包/边缘融合。不能接受文字连通关系改变或平面出现浮雕纹。
- case16：A7750栏杆的细横线、交点与把手轮廓最稳定；学生仍明显融合、起波浪。不能接受细线消失、断裂、弯折或相邻杆合并。
- case41：A7750花瓣边缘和绿色细枝更完整，学生变软并保留边缘亮晕。不能以平滑掩盖边缘位移或增加亮/暗光圈。
- case83：A7750花瓣折叠层次、叶片界限更清楚，学生细节简化且边缘有晕。不能接受花瓣折线或叶片分界融化。

四例仅是质量红线，不能代表103张全分布。候选必须同时通过固定四例1:1、原验证三图FR与完整S复核；任何可见文字/直线结构退化都拒绝，即使工具分上升。验证GT仅用于原正式评估，不用于训练loss/裁块选择。若125步内无可靠改善，保留A7750并停止该后备，不延长同目标训练。

本次仅CPU读源码/小证明及本地原图，未改任何冻结文件，未启动GPU、训练、评分、备份或新大图库。
