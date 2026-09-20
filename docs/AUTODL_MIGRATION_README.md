# VOSR / TinySR 迁移说明（30674 → AutoDL）

迁移时间：2026-08-19
来源：华为 ModelArts host 30674 `/cache/zihaohe`
目标：AutoDL `connect.westc.seetacloud.com:24686` → `/root/autodl-tmp/zihaohe`

## 关键设计：路径软链

代码里到处硬编码 `/cache/zihaohe/...`（`train_vosr_csig.py` 的 `VOSR_DIR`、
`v3/code/settings.py` 的 `ROOT`、`eval_*.py` 的 `sys.path.insert`），
且 `VOSR/preset/ckpts/` 下 4 个符号链接也是绝对路径。

因此本机建了：

    /cache/zihaohe -> /root/autodl-tmp/zihaohe

所有硬编码路径原样生效，**代码零改动**。

容器重建后 `/cache` 会消失，跑一次即可恢复：

    bash /root/autodl-tmp/zihaohe/_relink.sh

## Python 环境

直接搬的 venv（未重建）：`/cache/zihaohe/csig_isr/venv`
torch 2.5.1+cu124 / torchvision 0.20.1+cu124 / Python 3.10。

venv 的 `bin/python` 指向 `/usr/bin/python3`，而 AutoDL 原本没有系统 python3，
所以补了一个软链（纯增量，不影响机上其它 conda 项目）：

    /usr/bin/python3 -> /root/autodl-tmp/envs/flashvaed/bin/python3.10

注意这依赖 flashvaed 这个 env 存在。若它被删，改指向任一 3.10.x 解释器即可，
或用 conda 单独建一个：`conda create -p /root/autodl-tmp/envs/py310host python=3.10`。

## 环境变量

跑任何脚本前先 source：

    source /cache/zihaohe/_env_autodl.sh

它设置 CSIG_ROOT / TINYSR_ROOT / HF_HOME / TORCH_HOME / PIP_CACHE_DIR / TMPDIR / $PY。

**本机系统盘 `/` 只剩约 1.2G**，所有 cache 必须落数据盘。任何 pip install、
HF 下载、conda 操作都要确认目标在 `/root/autodl-tmp` 下，否则直接爆盘。

## 已验证（两端同口径同 ckpt 对数）

VOSR `train_out/vosr_iqa1.0/best.pth`，验证集全尺度：

| 指标 | 30674 | AutoDL |
|---|---|---|
| lpips | 0.3447 | 0.3447 |
| dists | 0.2380 | 0.2380 |
| niqe | 4.778 | 4.779 |
| musiq | 51.78 | 51.79 |
| maniqa | 0.6090 | 0.6090 |
| clipiqa | 0.8294 | 0.8294 |
| psnr | 24.39 | 24.39 |
| Score | 3.8957 | 3.8957 |

TinySR `train_out/combo_exact/best.pth`：Score 3.9286 → 3.9289。

测试集 100 张原生 4K，四项无参考 NR：

| ckpt | 指标 | 30674 | AutoDL |
|---|---|---|---|
| vosr_iqa1.0 | niqe / musiq / maniqa / clipiqa | 4.620 / 44.57 / 0.6504 / 0.8392 | 4.620 / 44.57 / 0.6504 / 0.8392 |
| vosr_iqa1.0 | **NR** | **2.4733** | **2.4733** |
| combo_exact | niqe / musiq / maniqa / clipiqa | 3.260 / 38.45 / 0.6664 / 0.8618 | 3.260 / 38.44 / 0.6664 / 0.8618 |
| combo_exact | **NR** | **2.5867** | **2.5866** |

VOSR 那条线测试集逐位完全相同；TinySR 仅 musiq 末位与 NR 第四位有 1e-4 级差异。

差异均在 MANIQA 采样噪声级别（`CSIG_MANIQA_SAMPLES=100`）。

复现命令：

    source /cache/zihaohe/_env_autodl.sh && cd /cache/zihaohe
    $PY eval_vosr.py 标签=/cache/zihaohe/csig_isr/train_out/<run>/best.pth --test 100
    $PY eval_ck.py   标签=/cache/zihaohe/csig_isr/train_out/<run>/best.pth --test 100

`eval_vosr.py` 用于 VOSR 系 ckpt，`eval_ck.py` 用于 TinySR 系。

## 口径陷阱

训练 `log.json` 里的 `score` 是训练中的 **crop 验证**口径（`--val_crops`），
`eval_*.py` 输出的 `[验证集全尺度]` 是原生 4K 全尺度口径。**两者不可比**。
例如 vosr_cap 训练 log 记 4.2787，全尺度实测 4.1095，不是退化。

## 已知缺口

`VOSR/preset/ckpts/Qwen-Image-vae-2d` 在**源端 30674 上就是断链**，
VOSR_weights 里根本没有这个目录。`train_vosr_csig.py` 走 sd21 VAE 不受影响，
但官方 `inference_vosr.py` 若走 Qwen VAE 分支则需要另行下载。

## 目录清单（两端已逐项校验一致）

| 目录 | 文件数 | 大小 |
|---|---|---|
| VOSR | 59 | 0.018 GB |
| VOSR_weights | 433 | 5.360 GB |
| hf | 12 | 0.350 GB |
| vosr_val | 6 | 0.035 GB |
| vosr_test | 2 | 小 |
| v3_incoming | 57 | 小 |
| csig_isr/TinySR | 80 | 1.299 GB |
| csig_isr/v3 | 57 | 小 |
| csig_isr/data | 219 | 0.539 GB |
| csig_isr/venv | 31266 | 6.059 GB |
| csig_isr/sd21base | 17 | 7.210 GB |
| csig_isr/train_data | 57300 | 25.517 GB |
| csig_isr/train_out | 136 | 19.785 GB |

另补传（不在原清单，但评测链路需要）：
`~/.cache/torch/hub/checkpoints` 1.45G（pyiqa 的 LPIPS/DISTS/MUSIQ/MANIQA 权重，
其中 `MANIQA_PIPAL-ae6d356b.pth` 543M 仅存在于此）→ `/root/autodl-tmp/.cache/torch`，
由 `TORCH_HOME` 指向。

**未迁移**（源端 `/cache/zihaohe` 剩余部分，判定为可重建或与本次目标无关）：
`conda_envs` 75G、`csig_isr/AdcSR` 14G、`submit` 4.4G、`bases` 1.9G、
`cmpmodels` 1.5G、`NTIRE2026` 1.5G、`x1_model` 1.4G、`venv_ose` 6.5G、`OSEDiff` 393M。
