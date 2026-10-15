# CUDA NR Push3：封存6500继续优化

本阶段只使用新文件 train_cuda_nr_stage_v4.py，旧v1/v2/v3源均保持原字节。源SHA256：`5f2d40b41034db7f9ae5f6a5f64d90b77bff6bc849422e1fc03ba3eb1600f311`。这里只完成CPU准备与解析检查；GPU运行由主代理调度。

共同父版为 cuda_nr_stage3_s6250_anchor6250_m15_250 的 global6500/local250/Adam500。父resume SHA `1f8d6991b490ebe9438b7e7b62c8ad958bac98979516d405f97a829c3c46461f`，config SHA `c6beaad0a6427a3eaaf3e829c04a25982c7ddbcb2df5c157f0aa37df686e0c5a`，protocol `995cae4104717ed2e578cb0867efae4171a36d97b281925534fd1f74f662bff1`。v4拒绝更换父标识。

三组均从同一父版独立分叉，新增750次更新，global=6500+local，Adam=500+local；不是合计2250步的一条训练轨迹。每组LR=2e-7，MUSIQ奖励系数1.5，空间梯度保真权重0.3，L1固定1，LPIPS固定0.1。

|分支|teacher|reward权重|CLIPIQA奖励系数|run|
|---|---|---:|---:|---|
|A 严格延续|6250精确FP32|0.075|0.1|cuda_nr_stage4_s6500_teacher6250_cont750|
|B 更新锚点|6500精确FP32|0.075|0.1|cuda_nr_stage4_s6500_teacher6500_cont750|
|C 增强NR奖励|6500精确FP32|0.15|0.5|cuda_nr_stage4_s6500_teacher6500_nrstrong750|

学生和教师均先从封存raw6000初始化BF16基础模型，再把LoRA提升FP32。学生逐张量精确复制6500 resume.lora；教师逐张量复制选定封存resume.lora，绝不把resume文件当权重文件加载。514张量/259474432元素、FP32和有限值、命名顺序及Adam形状均校验。Adam继承全部514份FP32一阶/二阶矩和step500，并逐份torch.equal验证；加载后显式覆盖全部组LR（本次三组数值都未改变）。Python、NumPy、Torch CPU、CUDA RNG在全部模型/评分网络初始化、参数与优化器校验之后恢复。

训练采用原始评分网络的真实MUSIQ/MANIQA/CLIPIQA输入梯度；首步验证3项非零有限，所有步骤要求全部514份LoRA梯度非零且有限，教师和评分网络保持冻结。A初始保真损失可以非零；B/C首次期望<=1e-6。无PNG STE、无融合或编译训练。MANIQA训练20crop，正式评测100crop；NIQE仅评测。仅使用原100张输入的随机512裁块，验证GT不进入训练。评分工具公式和模型不修改。

每250次更新及最终保存step6750/7000/7250.pth，同时atomic_torch替换resume_latest.pt。随后os.link建立不可变resume_step6750/7000/7250.pt；后续atomic替换latest不影响旧inode，不额外复制同一状态。checkpoint_step*.json原子落盘后作为该里程碑完成标记，记录raw和resume SHA、字节数、mtime、inode以及local/global/Adam步数，审计应读取不可变resume_step文件。全部里程碑先留/cache；最终只按主代理计划备份到共享存储。

运行示例（由主代理决定CUDA_VISIBLE_DEVICES，不应在这里自动启动）：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v4.py --config /cache/independent_sr_cuda_20261013/code/cuda_training/push3/config_stage4_A_continue6250.json
```

配置只允许明确字段。stage_steps限定1..1000、save_every固定250；LR限定5e-8..4e-7，reward .025...3，CLIP 0..1，MUSIQ .5..2.5，gradient_anchor .1...6，拒绝非有限数。恢复本阶段只能显式指向自身resume_latest.pt，且protocol、目标步数和保存策略必须匹配。所有父源/基础SD2字节/输入/评分网络权重和运行库版本继续核对；结束时再次核对父/teacher文件状态及所有源SHA。

CPU静态审计见stage4_preparation_static_audit.json：AST解析、3个配置解析、10项非法参数拒绝、1/250/750/1000步边界、旧源SHA均通过。GPU参数/梯度与实际质量提升仍须真实运行后单独确认。
