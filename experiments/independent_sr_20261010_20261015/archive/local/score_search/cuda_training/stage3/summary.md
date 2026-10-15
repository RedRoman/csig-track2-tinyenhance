# CUDA NR v3训练完成审计

2026-10-14T17:49:01.059828+08:00：两run完成各250步（global6251–6500、Adam251–500），父6250为新的冻结保真anchor，MUSIQ训练权重分别1和1.5，STE均关闭。正式PNG质量评分仍待完成。

| 分支 | step6500 SHA256 | 相对父6250真实FP32 ΔL2 | 变化元素/259474432 |
|---|---|---:|---:|
| MUSIQ1.0 | `b05e784a08b7082a4ecae63a814d71834068ceb5e9baf26eded3836b29bcc0b8` | 0.05333105739543628 | 259463830 |
| MUSIQ1.5 | `cfcc3013fa87aa721f346d6cc285e70b6d6ff2c88315ce9f1465b07ffe020e68` | 0.06014763041079606 | 259465588 |

每组514份FP32 LoRA有限，raw6500与resume LoRA逐值相等；514份Adam FP32 moments有限、状态/形状/参数ID完整并全部step500。父resume/config/source封印不变，v1/v2/v3/helper源码保持冻结。真实元素增量已逐项计算，非仅序列化SHA变化。

teacher启动时先raw6000初始化再升FP32，逐项精确复制父6250的514份LoRA并冻结。初始fidelity两组均1.1390932108960774e-14，首case13/crop[967,2015]符合父Python RNG独立重放。teacher审计依据是运行时migration exact-copy证明、冻结源码和每步梯度guards；CPU审计没有重建teacher。

两组250个case/crop逐条相同，最终四组Python/NumPy/TorchCPU/CUDA RNG内容hash完全相同，协议仅显式MUSIQ训练权重不同。三真实NR首梯度和250条514个LoRA非零梯度日志均通过，训练约248.82/246.53秒，峰值分配约29.95GB。

14份最终小记录下载SHA通过，另6份首步证明与完成后的文件字节相同。未下载大权重/optimizer，未调用GPU；Torch CUDA上下文前后False。审计经SSH stdin执行，审计文件只落本地，父任务可上传归档。

[初始化证明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage3/initial_audit.json>) · [参数/Adam/RNG审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage3/parameter_resume_audit.json>) · [日志/源码](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage3/logs_source_audit.json>) · [summary JSON](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage3/summary.json>) · [下载SHA](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage3/download_sha_manifest.json>) · [配对训练日志](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage3/paired_training_log.csv>)
