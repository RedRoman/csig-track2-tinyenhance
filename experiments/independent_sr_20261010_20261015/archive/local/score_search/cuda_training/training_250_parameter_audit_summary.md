CPU只读审计通过：CUDA_VISIBLE_DEVICES 为空，torch.cuda.is_initialized 前后均为False。

step6250 SHA256: 7b57349706372122ea67173fbb944e7e806d86e12379e3fd66ea2ebf1f96ecc3

相对step6100，514/514 LoRA tensors发生真实FP32变化；259461000/259474432元素变化，max_abs=1.9158236682415009e-05，L2=0.039516587739302764，mean_abs=1.9106188822795366e-06。

resume_latest.pt LoRA与step6250逐tensor exact equal；514份Adam state全部step250，moment为FP32且finite，状态字段/参数ID/形状完整。local_step250、global_step6250，最新配置target250/6250，250行日志连续且6250完成标记存在。

报告：/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/training_250_parameter_audit.json
