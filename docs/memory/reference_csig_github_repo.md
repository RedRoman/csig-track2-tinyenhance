---
name: reference-csig-github-repo
description: CSIG 赛道二代码/进度已备份至私有 GitHub 仓库 RedRoman/csig-track2-tinyenhance
metadata:
  type: reference
  originSessionId: 1a85b862-e600-464c-b9b5-d5e560c37b3b
  modified: 2026-09-17T16:05:14.446Z
---

2026-09-17 按用户要求（"不要push数据集和大ckpt"）把 30674 `/cache/zihaohe/` 的实验代码、
日志、评测结果、可视化诊断图 push 到 **https://github.com/RedRoman/csig-track2-tinyenhance**
（private）。单次提交，999 文件，164MB。

**排除内容**：官方数据集（赛题二.zip）、`train_data/`、`distill_data/`（教师输出+特征缓存，
82G+26G）、所有 `*.pth/*.pt/*.ckpt/*.safetensors/*.bin`、所有 `submit_*/` 下的提交 zip 与
vendor 第三方依赖（diffusers/yaml/regex 等，体积大且非自研）、第三方 baseline 仓库克隆
（AdcSR/OSEDiff/TinySR/VOSR/sd21base 等）及其 venv/conda 环境、fastwam_lab（不相关项目）。

**包含内容**：`train_pix.py`/`pix_model.py`/`eval_pix.py`/`train_distill.py`/`eval_ck.py` 等
全部自研脚本（top-level 99 个 .py + csig_isr 下另外 ~99 个）、训练/诊断日志、
`csig_isr/eval_out/*.json` 评测分数、`viz_*/`+`diag_*.png`+`montage_*.png` 视觉证据图
（幻觉/AOTI 损坏诊断）、`docs/CSIG_SUBMISSION_LEDGER.md`（提交台账）、
`docs/memory/` 下的项目记忆文档快照。

后续再有新实验结果要同步，直接在本地
`/private/tmp/.../scratchpad/csig_track2_repo`（session 临时目录，可能已被清理，
需要时从 30674 重新 rsync 一份新文件列表）或直接 clone 该仓库后增量提交，
排除规则同上（见仓库根目录 `.gitignore` 与 `README.md`）。

git identity 本机用的是仓库既有约定：`RedRoman <86876614+RedRoman@users.noreply.github.com>`（无全局 ~/.gitconfig，
从既有仓库 `~/Documents/generation-supervision-study` 读取到的本地 config）。

相关：[[project-csig-track2]] [[reference-csig-hosts-layout]]
