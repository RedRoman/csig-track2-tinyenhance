# TinySR 蒸馏微调套件(基于 distill_v2b,榜分 85.30 那版)

## 里面有什么
- `ckpt/distill_v2b_best.pth`:v2b 权重(LoRA r=64 + alpha 配置,format_version 3)。**这是要微调的起点。**
- `ckpt/run12_iqa30_best.pth`:v2b 自己的初始化起点(更早一代),备用。
- `code/train_distill.py`:蒸馏训练脚本(教师 L1/LPIPS/DISTS + GT 保真 + FR 势垒 + 可微 IQA + 可选 GAN/EMA/hinge)。
- `code/eval_ck.py`:全尺度评测(3 张验证图六项 + 100 张测试图 NR),口径与官方对齐。
- `code/gen_teacher.py`:离线生成教师输出(确定性种子,与训练端 crop/退化逐字一致)。
- `csig_isr/v3_code/`:模型运行时(`tinysr_runtime.py` 的 `build_model` 是唯一加载入口)、指标(`metrics_utils.py`)、退化链(`degradation.py`)、评测工具。
- `csig_isr/v3_configs/metrics.yaml`:**maniqa 必须是 koniq 版**(`maniqa`,不是 `maniqa-pipal`),我们曾因此错优化六周。
- `csig_isr/验证集/`:case1~3 的 LQ/GT。
- 另需(未打包,太大):TinySR 底座 `TinySR/`(1.3G,同目录 `tinysr_base.tar`)、教师集(见下)、训练图 `train_data/lsdir512/*.png`(26G,LSDIR 512 crop,自行下载或找我拷)。

## 目录约定(代码里硬编码,照抄最省事)
```
/cache/zihaohe/csig_isr/            <- CSIG_ROOT
    TinySR/                          <- TINYSR_ROOT(底座权重+模型代码)
    v3/code/  v3/configs/            <- 本套件的 csig_isr/v3_code, v3_configs
    train_data/lsdir512/*.png        <- 训练图
    train_out/<run_name>/best.pth    <- 输出
    data/赛题二/验证集/  data/赛题二/测试集/
/cache/zihaohe/distill_data/<teacher_name>/000000.png ...   <- 教师输出(按 idx 命名)
/cache/zihaohe/train_distill.py 等
```
不想用 /cache/zihaohe 的话,`ln -s <你的目录> /cache/zihaohe` 一条软链搞定(我们在 AutoDL 就这么干)。

## 环境
python 3.10 + torch 2.5.1(cu124)+ diffusers 0.29.1 + peft + pyiqa + huggingface_hub 0.25。
```
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf
export CSIG_ROOT=/cache/zihaohe/csig_isr TINYSR_ROOT=/cache/zihaohe/csig_isr/TinySR
export PYTHONPATH= TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
pyiqa 首次运行会下载 LPIPS/DISTS/MUSIQ/MANIQA(koniq)/CLIPIQA/NIQE 权重到 ~/.cache/torch/hub。

## v2b 当年的训练命令(复现起点)
```
python train_distill.py \
  --init_ckpt /cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth \
  --teacher_dir /cache/zihaohe/distill_data/teacher_koniq \
  --steps 2500 --bs 4 --lr 2e-5 --w_iqa 0.5 \
  --out /cache/zihaohe/csig_isr/train_out/distill_v2b
```
## 从 v2b 继续微调(推荐起手,单卡 80G 约 1 小时)
```
python train_distill.py \
  --init_ckpt /cache/zihaohe/csig_isr/train_out/distill_v2b/best.pth \
  --teacher_dir /cache/zihaohe/distill_data/teacher_v3_mixlq --data_mix 0.5 \
  --steps 3000 --bs 4 --lr 8e-6 --val_every 250 --iqa_batch 1 \
  --w_iqa 2.0 --w_maniqa 3.0 --w_clipiqa 2.0 --w_musiq 0.3 --w_niqe 1.0 --w_hf_over 1500 \
  --out /cache/zihaohe/csig_isr/train_out/<你的run名>
python eval_ck.py <标签>=/cache/zihaohe/csig_isr/train_out/<你的run名>/best.pth
```
这套权重(v6 配方)把 v2b 的 3.99 推到了 4.17(全尺度)。显存约 30G;IQA 项每步只算 1 张(`--iqa_batch 1`),否则 MANIQA 会 OOM。

## 教师集
- `teacher_koniq`(v2b 用的,VOSR 0.5B 教师,20k 张)与 `teacher_v3_mixlq`(现役最强 VOSR 教师,mix 退化,20k 张)各 11G,另附 tar。
- 自己生成:`python gen_teacher.py --teacher <VOSR ckpt> --mix 0.5 --out /cache/zihaohe/distill_data/<name>`(需 VOSR repo,`vosr_load.py`)。
  **生成端与训练端的 rng 消耗顺序必须逐字一致**,否则教师输出与 LQ 静默错位、训练白跑——改 `DetPairDataset` 前先读 gen_teacher.py 顶部注释。

## 几个已知的坑
1. 训练 log 的 val 是 512 crop 分,**≠ 全尺度分**,出分以 eval_ck.py 为准(曾多次符号反转)。
2. IQA 权重推高 → MANIQA/CLIPIQA 涨但会长幻觉(鳞纹/眼睛斑),**每个 checkpoint 出分后必须看图**。
3. 提交 zip:`output_dir/`(100 张 jpg)+ `model_dir/`(runner.py + config.yaml + tinysr_x1.pt)平铺在 zip 根;Runner 类签名不能改。
4. 评测机 Python ≤3.9,runner 及其依赖不能用 `X | Y` 类型注解,不要 vendor 第三方包遮蔽评测环境。
