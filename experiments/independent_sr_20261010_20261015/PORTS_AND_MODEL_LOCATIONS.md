# 端口、服务器与模型位置

2026-10-15 同步快照。只有 30416 在本次实际连接并检查；其余端口来自用户历史消息或既有项目记录，未重新探测，不承诺仍在线、仍挂载旧目录或仍有空闲 CUDA 卡。机器表见 [registry/ports.json](registry/ports.json)。本页不保存密码、token、SSH 私钥或其他访问凭据。

## 当前现场

```text
SSH alias: huawei-modelarts-31514
显式覆盖端口: 30416（alias 名字中的31514不是本次使用端口）
hostname: os-node-created-fg9f4
GPU: 4 × NVIDIA A100-SXM4-80GB
driver: 580.126.20
root: /cache/independent_sr_cuda_20261013
Python: /cache/independent_sr_cuda_20261013/env/bin/python
Torch: 2.8.0+cu128
```

GPU UUID、观察时间和磁盘容量见[现场 JSON](archive/server_cuda30416/LIVE_RESOURCE_SNAPSHOT.json)。快照时没有 GPU 计算进程，训练/评分已结束；不是未来资源预留。30416 曾经历实例与 `/cache` 重置；早期记录 hostname 为 `os-node-created-slr9d`，后来为 `os-node-created-fg9f4`，不能将同一端口视为相同物理卡。

使用已有授权 SSH 配置检查状态：

```bash
ssh -p 30416 huawei-modelarts-31514 'cat /cache/independent_sr_cuda_20261013/results/status_push7_20261015.json'
ssh -p 30416 huawei-modelarts-31514 'cat /cache/independent_sr_cuda_20261013/results/status_pixg5i_distill_20261015.json'
ssh -p 30416 huawei-modelarts-31514 'nvidia-smi'
```

## 历史端口

| 端口 | 来源与用途 | 本次确认状态 |
|---|---|---|
| 32316 | 最初 round2 交接，根目录 `/cache/independent_sr_round2_20261010` | 历史；未复测 |
| 32193 | 随后切换；round2 `nr_v2_strength` / `nr_v2_balanced`；后有昇腾阶段记录 | 历史；不能据此认定GPU/断点仍在 |
| 31320、31051、31731、30794、30213 | 用户曾允许使用的替代端口 | 仅历史授权；未探测硬件或当前在线性 |
| 30207 | 用户曾称 CUDA；后续真实报告为 Ascend910B3/NPU 参数搜索 | 按实际协议标 NPU，不能套 CUDA 测速 |
| 30416 | 临时 CUDA 及本次所有后续 CUDA 训练/评分 | 本次现场核实，4张A100，见上方时间快照 |
| 30138、30674 | 更早 CSIG ModelArts 训练/蒸馏历史，根目录 `/cache/zihaohe` | 仅保留原仓库历史资料 |
| 24686 | 更早 AutoDL；`/root/autodl-tmp/zihaohe`，旧 `/cache/zihaohe` 为软链 | 历史，见原迁移文档；本次未连接 |

## 当前 HYPIR 教师

```text
model_dir:
  /cache/independent_sr_cuda_20261013/submissions/hypir_push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae/model_dir
raw:
  /cache/independent_sr_cuda_20261013/adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100/step7850.pth
raw SHA256:
  23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c
resume:
  /cache/independent_sr_cuda_20261013/adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100/resume_step7850.pt
resume SHA256:
  35e339656581c89c772e6d9b00592a73cd397481f00d4d9bfd78c4bab7d05745
teacher test100 PNG:
  /cache/independent_sr_cuda_20261013/nr_push7_recovered_full_evaluations/push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all/render/test
```

部署为该权重对应的 UNet/VAE FP16 TensorRT 后端，不能加载旧 A7750 UNet engine 当作新模型。engine SHA、底模和 config 详见[候选 manifest](archive/server_cuda30416/submissions/hypir_push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae/candidate_manifest.json)及其 `model_dir/config.json`。大模型/引擎本身没有上传。

## PixG5i 原模型和实验模型

原压缩包本地位置：`/Users/hezihao/Downloads/赛题二_pixG5i_g_b4e0686c.zip`，SHA256 `f8e154da5ea9cb2fe165a316f824003459661945e6bbe22efd025d8592d573d6`；评分工具原包 `/Users/hezihao/Downloads/csig_scoring_tool.tar.gz`。两者文件内容未作为本次 Git 归档上传。

| 对象 | 当前/历史位置及状态 |
|---|---|
| 原始权重 | `/cache/independent_sr_cuda_20261013/source_payload/pixg5i_original/model_dir/model.pth`；SHA `f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487`，保留未覆盖 |
| 原始可调用模型 | `submissions/pixg5i_original_20261015/model_dir`，相对当前 CUDA root |
| 此前 medium300 | `submissions/pixg5i_hypir_medium_s300/model_dir`；raw SHA `005d13cb5b003d54e576678c52908181016cb9fa1746a9455465f5f2f619c458`；仍有视觉伪影 |
| 最新 coverage200 | `submissions/pixg5i_hypir_coverage_v4_s200/model_dir`；raw SHA `503074449893247332df973a85f9be1f348d9f130f31a28455b62790fb50c929`；**未采用** |
| 全部学生 run | `/cache/independent_sr_cuda_20261013/experiments/pixg5i_hypir_distill_20261015/runs/`，每run有config、train.jsonl、raw/resume、checkpoint JSON |
| 逐图指标 | `/cache/independent_sr_cuda_20261013/nr_pixg5i_distill_full_evaluations/<label>/scores/measurements.jsonl` |
| 完整原尺寸输出 | 同一 `<label>/render/{test,val}/`；本仓库只保存manifest与代表ROI |
| 本地精确恢复备份 | `/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/pixg5i_hypir_distill_20261015/backup*_verified/`；Git仅保存小回执 |
| 此前打包模型 | 同一Pix根目录 `deliverables/pixg5i_hypir_medium_s300_model.zip`；未上传Git |
| 最新实验包 | 同一Pix根目录 `deliverables/pixg5i_coverage_v4_s200_experimental_model.zip`；SHA `604ca5bfbe7c118add2e13a16f289f079bc21b4ebd387b0edb4c21fea4ed75ba`，未选主模型、未上传Git |

全部14版本、父权重、raw/resume SHA、局部/血缘累计步数见 [pixg5i_versions.json](pixg5i_versions.json)。先看版本的 `quality` / `formal_timing` 是否存在，不能给未全量评分的 structure/早期 headreset 候选补分，也不能把不同分支步数相加。

## 历史独立 SR 根目录

`/cache/independent_sr_round2_20261010`：`code/`、`adaptation/runs/`、`review_previews/`、`results/status_*.json`、`logs/`。用户最初给出的交接本体路径为 `reports/progress_20261010/AGENT_HANDOFF.md`。当前30416实例中该路径不存在，本地保留完整中文报告；未伪造交接正文或两组 nr_v2 训练的精确终态。

昇腾根目录 `/cache/independent_sr_ascend_20261012`，后续搜索根目录 `/cache/independent_sr_score_search_20261013`。旧模型、输出、共享归档和Windows备份位置只是历史定位，不能把“日志曾写入”当作本次再次核验完备。当前 `/cache` 可用约11TB，`/temp` 共享盘接近满；本次新增元数据快照写在 `/cache`，没有往SFS新增大资产。
