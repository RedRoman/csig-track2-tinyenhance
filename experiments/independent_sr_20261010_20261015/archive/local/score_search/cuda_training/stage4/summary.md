# 第三轮CUDA微调训练完整性审核

三条独立分支从sealed6500/Adam500各继续750步，终点均global7250/Adam1250；不能将三条分叉合并称为一个模型继续2250步。三个250/500/750边界共9套raw权重与完整Adam/RNG快照均通过CPU审核。

| 分支 | global/Adam | FP32 LoRA ΔL2 vs6500 | 审核 |
|---|---|---:|---|
| cuda_nr_stage4_s6500_teacher6250_cont750 | 6750/750 | 0.047590645 | PASS |
| cuda_nr_stage4_s6500_teacher6500_cont750 | 6750/750 | 0.055585688 | PASS |
| cuda_nr_stage4_s6500_teacher6500_nrstrong750 | 6750/750 | 0.093863209 | PASS |
| cuda_nr_stage4_s6500_teacher6250_cont750 | 7000/1000 | 0.069900630 | PASS |
| cuda_nr_stage4_s6500_teacher6500_cont750 | 7000/1000 | 0.076366691 | PASS |
| cuda_nr_stage4_s6500_teacher6500_nrstrong750 | 7000/1000 | 0.134497816 | PASS |
| cuda_nr_stage4_s6500_teacher6250_cont750 | 7250/1250 | 0.088976849 | PASS |
| cuda_nr_stage4_s6500_teacher6500_cont750 | 7250/1250 | 0.094630548 | PASS |
| cuda_nr_stage4_s6500_teacher6500_nrstrong750 | 7250/1250 | 0.166257050 | PASS |

每套514份FP32 LoRA均实际变化，standalone raw与resume.lora逐值相等；514份Adam状态的步数、形状、有限FP32一/二阶矩及参数组均正确。所有边界的跨分支case/crop完整序列与Python/NumPy/TorchCPU/CUDA四套RNG内容完全一致。父配置/resume、训练源、checkpoint marker实际SHA和原始checkpoint inode/mtime均核验，逐步loss/reward由已记录指标复算。CPU审核未构建模型或初始化CUDA。

报告及审核源码见本目录`milestone250_cpu_audit.json`、`milestone500_cpu_audit.json`、`milestone750_cpu_audit.json`与`audit_stage4_cpu.py`。这些证明参数/续训完整性，实际质量由独立固定20+val3 pilot和原评分工具完整100+3测量决定。
