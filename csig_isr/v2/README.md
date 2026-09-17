# CSIG 赛题二 TinySR 冲榜 v2

这是基于原始实验包的兼容升级版。旧版关键脚本保存在 `code/legacy/`，v2 主路径统一使用 `code/tinysr_runtime.py`，避免 checkpoint / metric / export 各走各的代码。

## v2 修复的 P0 问题

1. `infer_4k.py` / `export_submit.py` 不再硬编码 `tinysr8_fid/best.pth`：`--ckpt` 现在必须显式给出。
2. `score_eval.py` 已改为真正评估 TinySR。旧脚本实际构建的是 Samsung/NTIRE ResSRNet，不能作为 TinySR checkpoint 的统一 scorer。
3. 所有 v2 评测统一从 `configs/metrics.yaml` 读取 MANIQA 等指标；默认 `maniqa-pipal`，不再训练/验证/score 各自漂移。**这里仍然只是本地 surrogate 设置**：当前赛题公开材料没有把 MANIQA 的具体权重变体写死在你们这份工程里，因此先沿用历史 sweep 一直使用的 `maniqa-pipal`；若组委会后续确认其他变体，只改这一份 YAML，所有 v2 脚本会同步切换。
4. 每次 4K 出图和 JIT 导出都会写 `candidate_manifest.json`，包含 checkpoint SHA256。`validate_candidate.py` 会阻止“100 张图和 JIT 不是同一个模型”的提交。
5. 去掉硬编码 OSEDiff 时延。`benchmark_runner.py` 分别给出 GPU-resident 与 CPU-origin 输入时延，必须和 OSEDiff 用同一 protocol 比较。

> 本地六指标 Score 只是 surrogate。不要把任何经验榜单公式当作官方最终公式。

## 新增核心思路

### 1. Fixed / Severity-Adaptive Generative Strength

TinySR 原来固定：

`z_out = z - pred`

v2 改为：

`z_out = z - alpha(x) * pred`

先用 `alpha_sweep.py` 做零训练扫描；若 light / medium / heavy 的最优 alpha 不同，再训练 `SeverityGate`。Gate 只读取 16-channel latent 的 mean / MAD / gradient-energy，MLP 约几千参数，几乎不增加推理开销。

### 2. Exact Dense Weight Soup

`weight_soup.py` **先 merge LoRA，再在实际 Transformer 权重空间插值**。不要直接平均 LoRA A/B 因子；那一般不等价于平均两个模型。

### 3. Balanced IQA

训练目标现在同时包含 CLIP-IQA、MANIQA、MUSIQ，并通过 `configs/metrics.yaml` 与验证保持一致。`--iqa_batch` 可限制 IQA 分支显存。

### 4. Zero-inference-cost adaptation

支持：

- `--train_vae_decoder`
- `--unfreeze_last_n N`
- LoRA 继续微调

这些都不增加最终网络的额外算子（dense fine-tune 只改变已有权重）。最后 N 个 DiT block 的 dense fine-tune 默认关闭，建议最后再试。

### 5. Narrow ISP Mixture

`degradation.py` 新增 `degrade_v2()`：仍以 bilateral + Gaussian + JPEG 为主，只加入低概率 chroma subsampling / mild sharpen / gamma / 极弱 residual noise。它不是宽泛 Real-ESRGAN degradation。使用 `--degradation v2` 打开。

## 环境

保持原环境：PyTorch 2.5.1+cu124、diffusers 0.29.1、numpy 1.26.4、pyiqa 0.1.10 等。

默认服务器根目录仍为：

```bash
/cache/zihaohe/csig_isr
```

可避免改源码：

```bash
export CSIG_ROOT=/your/path/csig_isr
export TINYSR_ROOT=/your/path/TinySR
unset PYTHONPATH
```

## 推荐实验顺序

### P0 — 先重新建立可信 baseline

```bash
python code/score_eval.py \
  --ckpt /cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth \
  --alpha_mode fixed --alpha 1.0 \
  --out_json /cache/zihaohe/csig_isr/eval_out/v2_run12_baseline.json
```

这个数字应作为后续所有 v2 ablation 的唯一 baseline。

### P1-A — 零训练 alpha sweep（优先级最高）

```bash
python code/alpha_sweep.py \
  --ckpt /cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth \
  --alphas 0.70,0.78,0.84,0.90,0.96,1.00,1.04,1.08,1.12
```

重点看输出 JSON 中 `tier.heavy / medium / light` 是否呈现不同最优 alpha。若 light 明显偏小而 heavy 接近 1 或更大，adaptive-alpha 假设成立。

### P1-B — run11 / run12 exact weight soup（零训练）

```bash
python code/weight_soup.py \
  --a /cache/zihaohe/csig_isr/train_out/run11_big_iqa20/best.pth \
  --b /cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth \
  --lambdas 0.0,0.2,0.4,0.6,0.8,1.0,1.1 \
  --out_dir /cache/zihaohe/csig_isr/train_out/v2_soup
```

然后逐个：

```bash
python code/score_eval.py --ckpt .../v2_soup/soup_0.60.pth
```

对最好的 2–3 个 soup 再跑 `alpha_sweep.py`。这是一个完全零推理成本的二维搜索 `(lambda, alpha)`。

> `weight_soup.py` 会保存 merged Transformer 全量权重，每个候选 checkpoint 会明显大于只含 LoRA 的旧 checkpoint；磁盘紧张时分批扫 lambda，评完及时只保留 top-k。

### P1-C — Adaptive alpha，先做 gate-only clean ablation

```bash
python code/train_tinysr.py \
  --init_ckpt /cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth \
  --adaptive_alpha --freeze_lora \
  --alpha_min 0.72 --alpha_max 1.10 \
  --w_alpha_prior 0.02 \
  --w_iqa 1.0 --w_clipiqa 1.0 --w_maniqa 1.0 --w_musiq 1.0 \
  --steps 1200 --lr 2e-4 --bs 6 \
  --out /cache/zihaohe/csig_isr/train_out/v2_alpha_gateonly
```

如果 gate-only 已经涨分，再允许 LoRA 小学习率联合适配：

```bash
python code/train_tinysr.py \
  --init_ckpt /cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth \
  --adaptive_alpha \
  --alpha_min 0.72 --alpha_max 1.10 \
  --w_alpha_prior 0.02 \
  --w_iqa 1.0 --w_clipiqa 1.0 --w_maniqa 1.0 --w_musiq 1.0 \
  --steps 1800 --lr 1e-5 --bs 6 \
  --out /cache/zihaohe/csig_isr/train_out/v2_alpha_joint
```

**不要一上来把 `w_iqa` 再推到 3/4/5。** 你们旧实验已经显示 LPIPS/DISTS 与伪纹理在恶化。

### P2 — Decoder / dense FT

只在 P1 最优策略确定后尝试：

```bash
# decoder only
python code/train_tinysr.py ... --train_vae_decoder --lr 5e-6

# 最后 1 个 DiT block + LoRA；dense LR 要远小于 LoRA LR
python code/train_tinysr.py ... --unfreeze_last_n 1 --lr 8e-6 --dense_lr 5e-7
```

若显存或数值稳定性有问题，优先放弃 dense FT，而不是为了它牺牲 batch / validation。

### P2 — ISP mixture 泛化

确认模型结构后再切：

```bash
python code/train_tinysr.py ... --degradation v2
```

必须和 `legacy` degradation 做同设置 A/B；不要仅凭 3 张验证图判断 v2 一定更好。

## 4K / tile / JIT 最终链路

先对最终 checkpoint 重跑 tile sweep：

```bash
python code/tile_sweep.py --ckpt FINAL.pth --alpha_mode auto
```

出 100 张：

```bash
python code/infer_4k.py \
  --ckpt FINAL.pth \
  --out /cache/zihaohe/csig_isr/submit/output_dir \
  --tile 512 --overlap 256 --alpha_mode auto
```

导出同一个 checkpoint：

```bash
python code/export_submit.py \
  --ckpt FINAL.pth \
  --out_dir /cache/zihaohe/csig_isr/submit/model_dir \
  --alpha_mode auto
```

一致性检查：

```bash
python code/validate_candidate.py \
  --render_dir /cache/zihaohe/csig_isr/submit/output_dir \
  --model_dir /cache/zihaohe/csig_isr/submit/model_dir
```

runner 时延：

```bash
python code/benchmark_runner.py \
  --model_dir /cache/zihaohe/csig_isr/submit/model_dir
```

## 我建议第一轮只跑的 5 个结果

1. run12 canonical baseline
2. run12 fixed-alpha sweep
3. run11/run12 weight soup sweep
4. 最佳 soup / run12 的二维 alpha sweep
5. adaptive-alpha gate-only

先把这五项结果发回来。尤其需要：每个候选的六指标、case 分数、heavy/medium/light 分数和 alpha sweep JSON。之后再决定是否值得上 decoder、dense FT 和 v2 degradation。

## 关于旧结论

`decoder ceiling LPIPS=0.5167` 不应再称为硬上限，因为当前模型 LPIPS 已能低于该值；它只代表一次有限 latent optimization 的结果。`arch_diag.py` 的临时 bypass 也不能直接证明某 DiT block 可安全剪除。v2 暂不沿这两个结论继续加大工程投入。
