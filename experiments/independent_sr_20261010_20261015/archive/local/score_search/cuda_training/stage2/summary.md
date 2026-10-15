# CUDA NR stage2训练完成审计

2026-10-14T17:32:16.635275+08:00：两新run均完成阶段250步，对应global6251–6500、Adam251–500。这里只证明训练状态与恢复完整性，模型质量待独立PNG评测。

| 分支 | step6500 SHA256 | 真实FP32 ΔL2（相对6250） | 变化元素/259474432 |
|---|---|---:|---:|
| continuous | `dc7bbf712e3039302a52951075409a23771b8d832024d1800bbe3398c41d4afa` | 0.04899644177529621 | 259463697 |
| roundSTE | `095f93f17d998d708639f2a3b3641052c576b3036efcbc881eb2e88360e18070` | 0.04840120814972133 | 259463713 |

两run各514份有限FP32 LoRA；resume_latest中的LoRA与独立step6500逐值相等。514份Adam状态均step500、FP32 moments有限且参数ID/形状齐全。相对父6250的514个tensor均有实际优化增量，并非仅序列化SHA变化。父resume/config/v1source的SHA封印保持不变；v2与PNG helper source也保持冻结。

两分支250个case/crop序列逐条相同，最终Python/NumPy/TorchCPU/CUDA四组RNG内容hash全部相同。父Python RNG独立重放的首case13、crop[967,2015]与两run首行一致。每条训练日志记录514个非零LoRA梯度，首步三真实NR输入梯度有限非零，冻结teacher/metric guard通过。

continuous保留父objective；roundSTE明确为新量化objective，并有记录地继承父Adam/RNG。两者都保留原6000/coeff175冻结同latent anchor。正式训练约249.59/243.90秒，median step约0.8400/0.8168秒，峰值分配约29.95GB。

审计仅CPU，Torch CUDA上下文前后均未初始化；14份小记录下载SHA匹配，未下载大权重或优化器文件。CPU审计以SSH stdin执行并仅在本地保存结果，未写远端审计文件。

[参数/Adam/RNG审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage2/parameter_resume_audit.json>) · [日志/源码审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage2/logs_source_audit.json>) · [summary JSON](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage2/summary.json>) · [下载SHA](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage2/download_sha_manifest.json>) · [配对250行日志](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage2/paired_training_log.csv>)
