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

## 2026-09-18 ~ 09-20 冲刺增量(第二次提交)

榜分 138.41 (L2c) → **152.95 (G4, 09-19 实得)**;终局候选 G5c / G5i 包预测约 154。

**新增自研代码**(仓库根目录):
- `pix_model.py`:PixUNet2(逐层块数 `nbs`、`norm∈{gn,none}`、非对称解码器 `nbs_dec`)+ `fold_inference()`(固定时间步的 AdaGN 调制常量折进权重,推理期零 temb 开销)
- `runner_fold.py` / `runner_graph.py`:提交用 runner,链 = 手工 CUDA graph 回放 Inductor(max-autotune-no-cudagraphs)编译体 → max-autotune → reduce-overhead → default → jit → eager,每级带速度与数值自检
- `train_pix.py`:加 `--nbs/--norm/--nbs_dec/--init_partial`(net2net 生长热启动),`--init_ckpt` 支持 arch_ver 2
- `build_fold.sh`:一键打包(渲染 output_dir → zip → 解包冒烟)
- 探针:`speed_probe.py`(kernel 归因)、`arch_probe.py`、`wall_probe.py`(墙钟 vs event)、`time_one.py`+`time_loop.sh`(独占卡按序计时,首尾校漂移)、`prune_probe.py`、`width_probe.py`、`pp_sweep.py`(后处理扫描)、`soup.py`、`viz_cands.py`(肉眼幻觉对比条)、`predict.py`(榜分预测)
- 启动/运维:`launch_N*.sh`、`watchdog_30674.sh`(定时清场让机)

**结果与日志**:`pix_runs/`(30674 训练/评测日志)、`autodl_pix_out/`(AutoDL 训练日志 + 每条 log.json)、`csig_isr/eval_out/*.json`、`viz_h1..h6/`(各站肉眼对比条)。

**提交件 model_dir**(含权重,均 <16MB):`csig_isr/submit_pixL2d_g`(保底)、`submit_pixG4_g`(已交,152.95)、`submit_pixG5c_g`、`submit_pixG5i_g`(终局两张彩票)。

**结论**(详见 `docs/memory/project_csig_track2.md` 与 `docs/CSIG_SUBMISSION_LEDGER.md`):
去 GroupNorm、瓶颈加深到 5 块、折叠 + channels_last + CUDA graph 三步把 138 推到 153;剪解码器(−0.24 NR)、IQA 加权(1:1 换 FR)、加宽(+38-59% 时延)、权重汤、更深/更久续训五条路均在同一前沿上滑动不越线;v8 教师 + 本配方下前沿 S≈4.11 @0.48ms ≈ 154。
