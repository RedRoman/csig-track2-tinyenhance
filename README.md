# CSIG 2026 赛道二 — TinyEnhance（团队：干就完了）

天池 raceId 532500，4K 图像超分（Diffusion 架构约束），截止 2026-09-20。
评分公式：`榜分 = 3.357 × 2^S × spd^0.2`，`spd = 100.15/时延(ms)`，
`S = (1-LPIPS)+(1-DISTS)+CLIPIQA+MANIQA+MUSIQ/100+max(0,(10-NIQE)/10)`。

## 内容范围

本仓库只保留**代码、脚本、日志、评测结果、可视化对比图、决策文档**，
按用户要求不包含数据集和大 checkpoint：

- 不包含：官方数据集（`赛题二.zip`）、训练裁剪数据（`train_data/`）、教师蒸馏输出与特征缓存
  （`distill_data/`）、任何 `*.pth/*.pt/*.ckpt/*.safetensors/*.bin` 权重文件、提交 zip 包、
  第三方 baseline 仓库克隆（AdcSR/OSEDiff/TinySR/VOSR 等）及其 venv/conda 环境。
- 包含：全部自研训练/评测/诊断脚本（`train_pix.py`、`pix_model.py`、`eval_pix.py`、
  `train_distill.py`、`eval_ck.py` 等）、训练与实验日志（`*.log`）、评测分数 json
  （`csig_isr/eval_out/`）、幻觉/AOTI 诊断对比图（`viz_*/`、`diag_*.png`、`montage_*.png`）、
  以及 `docs/` 下的提交台账和项目记忆文档。

提交包（`submit_*/model_dir`）里 vendor 进去的第三方依赖（diffusers/yaml/regex 等，
用于让评测机在 Python≤3.9 环境下离线跑通）体积和文件数都很大且非自研代码，未收录；
部署链路的关键设计（compile→jit→eager 回退、vendor 改名避免 sys.path 冲突等）见
`docs/memory/reference_csig_hosts_layout.md` 与 `docs/CSIG_SUBMISSION_LEDGER.md`。

## 关键文档

- `docs/CSIG_SUBMISSION_LEDGER.md` — 每次提交的完整台账（md5、预测分/实际分、技术动机）
- `docs/memory/project_csig_track2.md` — 项目主线技术日志
- `docs/memory/project_csig_hallucination.md` — "幻觉"伪影（仙人掌/眼睛斑）根因与对策
- `docs/memory/project_csig_speed_axis.md` — 时延轴突破（torch.compile 提速）
- `docs/memory/reference_csig_hosts_layout.md` — 双机布局与踩坑记录

## 目录结构（节选）

- `train_pix.py` / `pix_model.py` / `eval_pix.py` — 像素空间一步去噪 UNet（moonshot 学生）
- `train_distill.py` / `eval_ck.py` — TinySR 族蒸馏主线
- `csig_isr/train_out/*/log.json` — 各训练 run 的日志（不含 checkpoint 权重）
- `csig_isr/eval_out/*.json` — 评测分数记录
- `csig_isr/v3/` — 早期版本代码与配置
- `viz_*/`、`diag_*.png`、`montage_*.png` — 视觉对比证据图
- `v2b_finetune_bundle/` — 给学弟的微调交接（不含权重，仅代码+说明）
