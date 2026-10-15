这是 CPU 准备包，未构建引擎、未登记候选、未测质量或计时。

已实际绑定并 CPU dry-run：
- selected2: push7_a7750_lastconv_lr1e6_nr_s7850，native manifest SHA 45cee1c8e9af789c13a151a9954bbe443edeca5677c1d89d96c48f34433f777b。
- selected6: push7_a7750_lastres_nr_s7850，native manifest SHA bf9ac78249c2c50b0855c35152228fea46bc7f39b8f02346514f81d9145fbf3a。
两者 builder default 和 registration_plan 合计四进程 exit0。未 import Torch/CUDA。原始计划与回执存 R/audits/push7/trained_backend_preparation，回执 SHA d2f97e0333a23417fc298ead794e30591e4ca6b2332c347821a1d0119a11a54b。

R=/cache/independent_sr_cuda_20261013
S=$R/code/push7/hypir_direct_nr/trained_backend_v1
示例 native MF：
M=$R/submissions/hypir_push7_a7750_lastres_nr_s7850/candidate_manifest.json
H=bf9ac78249c2c50b0855c35152228fea46bc7f39b8f02346514f81d9145fbf3a
D=$R/backends/push7_trt/push7_a7750_lastres_nr_s7850_unet_fp16

Root 在实际完整 pilot 与原生 ROI 通过后调度，才可执行：
CUDA_VISIBLE_DEVICES=<root分配物理卡> "$R/env/bin/python" "$S/build_finetuned_unet_trt.py" --native-manifest "$M" --native-manifest-sha256 "$H" --out-dir "$D" --execute
此命令会实际重哈希 current raw/base，按原 BF16 load+safe PEFT fuse→FP16→require_full_compilation 构建新 UNet，再作组件诊断；组件时间不是正式评分工具计时。

构建成功后，以真实新 component_manifest.json SHA 调用：
CUDA_VISIBLE_DEVICES='' "$R/env/bin/python" "$S/registration_plan.py" --native-manifest "$M" --native-manifest-sha256 "$H" --engine-manifest "$D/component_manifest.json" --engine-manifest-sha256 <实际SHA>
它只打印文件复制计划和可审查配置，不写候选。没有 engine 时只返回 CPU_PLAN_WAITING_REAL_NEW_UNET_ENGINE，绝不输出 runtime_config。

限制：
- 白名单仅四条真实指定训练 run 的 local50/100：低LR conv_out2、LR1e-6 conv_out2、LR1e-6 lastres6、LR5e-6 lastres6。缺真实 nativeMF/marker 即失败。
- 新 UNet 当前权重来自 native MF 真实 current raw SHA；initial A 是独立来源字段，A-only UNet engine 不可借用。
- VAE 重用已锁定原 encoder/decoder；模型架构、原 native512 stride256、t200/coeff175、空提示、原 posterior/DDPM/tiling/wavelet 保留。
- 运行时仍读取已封存 native MF、config、marker、catalog、parity proof 和源码元数据的引用路径，归档需保留这些恢复路径。没有宣称 model_dir-only 便携。
- 新后端必须 fresh103 PNG + 原418 scalar 正式评分和原尺寸视觉，不能复用 A 或 native pilot 的质量。
- 基线、旧 builder4281/wrapper270d/VAE4b/engine/证明均不改。
