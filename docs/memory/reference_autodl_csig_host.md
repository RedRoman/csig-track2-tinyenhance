---
name: reference-autodl-csig-host
description: AutoDL 机器（VOSR/TinySR 工作机）的连接方式、路径布局与两个必须知道的约束
metadata: 
  node_type: memory
  type: reference
  originSessionId: 8e5b7e1d-72f5-4289-ad74-10d03f4df098
---

CSIG 超分工作 2026-08-19 从 ModelArts host 30674 迁到 AutoDL：
`ssh -p 24686 root@connect.westc.seetacloud.com`（30674 与本地 Mac 均已装公钥；
30674 上直接 `ssh autodl` 即可，配置在 `~/.ssh/config`）。4×vGPU-48GB，driver 580 / CUDA 12.8。

**路径**：全部在数据盘 `/root/autodl-tmp/zihaohe`，并建了 `/cache/zihaohe` 软链指向它——
因为代码里到处硬编码 `/cache/zihaohe/...`（`train_vosr_csig.py` 的 `VOSR_DIR`、
`v3/code/settings.py` 的 `ROOT`、`VOSR/preset/ckpts/` 下的绝对路径软链），
有了这个软链**代码零改动**。容器重建后 `/cache` 会没，跑 `/root/autodl-tmp/zihaohe/_relink.sh` 恢复。

环境：`source /cache/zihaohe/_env_autodl.sh` 后用 `$PY`（搬过来的 venv，torch 2.5.1+cu124）。
完整交接文档在 `/cache/zihaohe/MIGRATION_README.md`。

**约束一：系统盘 `/` 只剩约 1.2G**（miniconda3 占 21G）。任何 pip install / HF 下载 /
conda create 都必须显式指向 `/root/autodl-tmp`，否则爆盘。机上已有约定把 envs 放
`/root/autodl-tmp/envs`。数据盘 1.1T，迁完后剩约 145G。

**约束二：venv 的 `bin/python` 指向 `/usr/bin/python3`**，AutoDL 原本没有系统 python3，
现补了软链到 `/root/autodl-tmp/envs/flashvaed/bin/python3.10`。若 flashvaed 被删，
改指向任一 3.10.x 解释器即可。

反常识的一点：源端 30674 出网受限，跑完整评测会 HF `LocalEntryNotFoundError`，
AutoDL 反而能——迁过去的环境比原地更完整。

相关：[[project-csig-track2]] [[reference-remote-paths]]

**盘况更新(2026-09-18)**:数据盘 1.6T 曾被占到 100%(剩 286MB),其中队友目录占 1.3T
(songhua 494G、generative_image_enhancement_20260812 352G、pixeldit_* 等),我方 zihaohe 292G。
按用户要求清理了我方在 30674 有完整副本的冗余(md5/文件数逐一核对后删):pdit_teacher_a/b 分片 44G、
distill_full 15G、VOSR_weights 7.2G、sd21base 7.3G、9 个 submit_* 的 zip+output_dir 19G(model_dir 保留)、
raw_v8/raw_pixB2/v11cold 6G、train_out 旧 TinySR 期 run(v2_soup 等)与 smoke 14G。**清后剩 112G,zihaohe 180G。**
未删待用户定:distill_data/v8_feat_b7+b10(20G,特征蒸馏缓存,仅此处有,方向已判负)、ebgr/(55G,EBGR 项目)。
zhuyan_work(49G)与 zihaohe 外的目录全是队友的,不动。新训练 ckpt 暂写系统盘 /root/pix_out(系统盘剩 3.8G)。
