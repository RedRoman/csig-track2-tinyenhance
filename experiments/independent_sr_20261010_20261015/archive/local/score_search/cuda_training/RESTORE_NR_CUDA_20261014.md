# 30416 CUDA NR微调恢复说明

本轮已完成两分支100步筛选，保守分支继续至新增250步/global6250。最新完整NR=2.266385043、FR=1.463716487、S=3.730101530；单A100 512输入p10=45.780479ms。完整结果见同目录上级的 `NR_FINETUNE_REPORT_20261014.md`。

共享备份根为 `/temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014`，相对路径对应 `/cache/independent_sr_cuda_20261013`。`manifest.json`列出文件SHA；只有 `DONE.json` 的status=complete且manifest摘要一致才表示校验完成。首次核心归档670文件、18,543,300,225字节；随后更新的说明文档和审计小文件纳入最终manifest。

包含保守step6100/step6250、aligned step6100、两分支各自完整AdamW/FP32 LoRA/RNG resume、raw6000 anchor、训练代码/配置/逐步日志、原评分工具/全部指标权重/原输入与3组验证GT、两个full的206张原PNG和四个pilot的92张原PNG。未归档中间6050/6150/6200重复断点，也未复制Python环境和SD2大底座。

临时容器更换后，先将共享备份恢复至相同根路径 `/cache/independent_sr_cuda_20261013`，保留目录层级。上一轮完整基线与提速实验另保存在 `/temp/independent_sr_cuda_20261013_artifacts` 原目录；恢复时不要把新轮次子目录再复制成重复嵌套。训练协议记录了源码、底座、权重和输入身份，保持原路径/源码后才能严格恢复本轮状态。

底座已经在共享存储 `/temp/zihaohe_stage/baselines/sd21b`，完整SHA记录在 `calibration/base_model_provenance_shared.json`；恢复时先核验该目录。CUDA环境为Python3.13.5、Torch2.8.0+cu128、torchvision0.23.0+cu128、diffusers0.33.1、peft0.17.1、pyiqa0.1.10、timm1.0.28、transformers4.56.2、accelerate1.10.1。完整已安装版本记录在 `code/runtime_freeze_cuda_20261014.txt`。Python虚拟环境需要在新容器重建。

重建评分软链接：`scoring_tool/weights`指向 `source_payload/scoring_assets/weights`；`scoring_tool/data/测试集`指向 `source_payload/assets/input`；`scoring_tool/data/验证集`指向 `source_payload/scoring_assets/data/验证集`。均使用恢复后的服务器绝对路径。保留 `numpy_legacy_compat.py` 和 `maniqa_offline_init.py`，它们与原指标权重一致性证据一起归档。

当前保守resume的全部514份Adam状态均为step250，FP32 LoRA逐值等于step6250，Python/NumPy/TorchCPU/CUDA RNG完整。`step6250.pth` SHA256为 `7b57349706372122ea67173fbb944e7e806d86e12379e3fd66ea2ebf1f96ecc3`；`resume_latest.pt` SHA256为 `3dcf896028f41c7c4c6e15074cc69c4cdb2ff7679e6756f5ee57a9a657f61c1f`。目录名仍为 `cuda_nr_s6000_ct175_conservative_pilot100`，实际目标已250。

当前训练入口的单段目标上限是250，已达该目标。继续更长训练需要建立明确的新阶段/协议版本，并验证恢复身份、AdamW与RNG连续性；不要覆写本轮冻结源码或把重新加载BF16权重当作FP32优化器连续恢复。当前部署选用 `submissions/hypir_cuda_nr_conservative_s6250_compiled/model_dir`，推理融合/编译与训练raw LoRA分别保存。
