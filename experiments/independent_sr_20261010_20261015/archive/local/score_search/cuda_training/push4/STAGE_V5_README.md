# CUDA NR Push4 / Stage5：从最佳7250短续训

新源 `train_cuda_nr_stage_v5.py` SHA256 `4b5bdf40aab5d2fd35c92f8fc714c675e62bd86228bc3fb79e48e896b70278f2`；旧v1/v2/v3/v4及已归档Push3文件保持不变。只做CPU准备，不自动启动GPU。

父版 `cuda_nr_stage4_s6500_teacher6500_nrstrong750` 必须使用封存的 `step7250.pth` 与不可变 `resume_step7250.pt`，拒绝mutable latest。raw SHA `fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e`；resume SHA `b8b709f284ce627113e392ede00e43bcd522c8dc23b63719e9fcbaa573c7256d`；config SHA `8bcf24e6f8965593f26d31aeedcc0c58fc62af185d443d8ff46d2602336b59f7`；protocol `4d12b7aae1486eb1425e3d1dedca737f152033e91c62a290c569e721899f5f71`。父状态local750/global7250/Adam1250，三分支都从这同一个精确状态独立开始。

本次每组只新增500步、每250步保存：local250/global7500/Adam1500，local500/global7750/Adam1750；不能把三组相加成一条1500步轨迹。新阶段计数统一取封存父常量7250/1250，不沿用旧入口6500/500计数。

|分支|run|MUSIQ系数|reward系数|
|---|---|---:|---:|
|A 严格延续|cuda_nr_stage5_s7250_teacher6500_continue500|1.5|0.15|
|B 增加MUSIQ权重|cuda_nr_stage5_s7250_teacher6500_m25_500|2.5|0.15|
|C 同B并减弱奖励|cuda_nr_stage5_s7250_teacher6500_m25_r10_500|2.5|0.10|

三组共同为LR2e-7、CLIP系数0.5、空间梯度保真0.3；L1固定1、LPIPS固定0.1，按root决定不增加LPIPS parser。所有组继续使用6500 teacher，保留较早锚点的保真约束，因此初始fidelity允许非零。入口另外保留teacher7250选项，但本轮3配置均不用；只有该可选重锚才要求首次fidelity绝对值<=1e-6。配置MUSIQ系数仅影响训练损失，原评分工具/原正式汇总系数不修改。

迁移顺序保持：raw6000初始化BF16基础模型 → trainable/teacher LoRA提升FP32 → 全514张量精确复制封存resume.lora → 继承514份Adam一/二阶FP32矩及step1250 → 显式应用配置LR → 全模型/指标构造完成后恢复Python、NumPy、Torch CPU、CUDA四组RNG。父raw与immutable resume的514份LoRA另作CPU逐项torch.equal校验，绝不将resume当weight文件加载。Adam矩加载后也逐份精确相等校验。教师6500由其封存resume FP32逐项复制、冻结，不经过新的BF16量化载入。

首步验证原MUSIQ/MANIQA/CLIPIQA网络的真实输入梯度均有限非零，每步全部514份LoRA梯度须有限非零；教师和评分网络冻结。训练MANIQA20crop、正式评分100crop；NIQE仅评测；原100张输入随机512裁块，验证GT不进训练。基础SD2全部权重/元数据、输入、原评分网络、库版本和旧源SHA继续核验。

每个checkpoint保存原始step7500/7750.pth，同时原子替换resume_latest.pt，再hardlink成不可变resume_step7500/7750.pt，原子写checkpoint_step7500/7750.json记录实际raw/resume SHA、大小、源mtime/inode及步数。每个里程碑都保留可精确继续的FP32 LoRA、Adam和4RNG。所有新结果只写新run目录，拒绝覆写父/teacher目录。

实际3份CLI配置位于 `code/cuda_training/push4/config_stage5_A_continue500.json`、`config_stage5_B_m25_500.json`、`config_stage5_C_m25_r10_500.json`。例如由root调度指定GPU后执行：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v5.py --config /cache/independent_sr_cuda_20261013/code/cuda_training/push4/config_stage5_A_continue500.json
```

静态审核见stage5_preparation_static_audit.json，变更差异见v4_to_v5_review.diff。全部新GPU构造/梯度与训练质量需由root实际运行后验证；本准备步骤未启动GPU或训练。
