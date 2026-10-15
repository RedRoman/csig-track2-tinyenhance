# CSIG 赛道二 本地评分工具(csig_scoring_tool)

2026-09-20 打包。自包含:代码 + 六项指标全部权重(1.6GB,离线可用)+ 官方验证集 3 组 / 测试集 100 张 + 一个已知榜分的参照提交件。
只需要一个带 CUDA 的 Python 环境(见 `requirements.txt`,与 30674/AutoDL 的 venv 同版本)。

## 评分口径(与我们全部提交对齐的本地代理)

```
Score = (1-lpips) + (1-dists) + clipiqa + maniqa + musiq/100 + max(0, (10-niqe)/10)
```
- 有参考两项(lpips/dists)只能在验证集 3 组算 → **val FR**
- 无参考四项在官方测试集 100 张原生 4K 上算 → **test NR**(选模型看这个,验证集 Score 会说反话)
- 代理 **S = test NR + val FR**;榜分 ≈ 152.95 × 2^(S−4.0841) × (时延项)^0.2,见 `predict.py`
- 指标实现:pyiqa 0.1.10;maniqa = **koniq 版**(官方 eval.py 用的就是 `create_metric("maniqa")`,不是 pipal);
  clipiqa = CLIP RN50 固定 prompt 版;musiq = koniq 版
- 先对每项在全部图上取均值再算一次 Score;MANIQA 抽样数固定 100、评测前固定种子;同一模型两次 test NR 仍有 ~0.005 噪声

## 目录

```
csig_metrics.py       指标/公式/权重接入的单一事实源(其余脚本都 import 它)
score_dir.py          给"已渲染好的输出图目录"打分(测试集 NR / 验证集六项)
eval_submission.py    端到端模拟官方评测:加载 model_dir/runner.py → 量 512 时延 → 现跑 103 张 → 打分 → 预测榜分
eval_ckpt.py          直接给像素 UNet ckpt 打分(训练中选 checkpoint 用,需 pix_model.py)
predict.py            榜分预测(五点拟合,残差 ±2 分)
pix_model.py          PixUNet/PixUNet2 定义 + fold_inference(eval_ckpt 依赖)
env.sh                source 后把权重/数据指到本包
configs/metrics.yaml  口径记录
weights/torch/hub/checkpoints/   alexnet, LPIPS, vgg16, DISTS, NIQE .mat, MUSIQ koniq, MANIQA koniq
weights/clip/RN50.pt             CLIPIQA 主干(pyiqa 固定读 ~/.cache/clip/RN50.pt,工具会自动软链)
data/验证集  data/测试集          官方数据(case{i}_lq/gt;case1..100.jpg)
examples/submit_pixG4_g/model_dir   参照提交件:榜分 152.95,test NR 2.7620,val FR 1.3221,AutoDL 0.431ms
examples/runner_official_template.py 官方 runner 模板(接口不能改的部分)
```

## 用法

```bash
source env.sh                       # 或直接跑,csig_metrics.ensure_weights() 会自行接入权重

# 1) 评一个提交件(最常用;约 6-8 分钟,含渲染 103 张 4K)
python eval_submission.py --model_dir path/to/model_dir --label mymodel
python eval_submission.py --zip 赛题二_xxx.zip                     # 直接给 zip,自动只解 model_dir/
python eval_submission.py --model_dir ... --skip_render --out_dir render_mymodel   # 图已渲染,只重打分

# 2) 给已有输出图目录打分
python score_dir.py --test_dir A=out_A/test --val_dir A=out_A/val --out A.json --csv

# 3) 训练中选 checkpoint(像素 UNet)
python eval_ckpt.py N3b=/path/pixN3b/last.pth G4=/path/pixG4/last.pth

# 4) 只算预测分
python predict.py --nr 2.8128 --fr 1.3003 --ms 0.478
```

## 时延口径(重要)

`predict.py` 的常数是按 **AutoDL vGPU-48G、fold+channels_last+CUDA graph runner、独占卡 CUDA event p10** 拟合的。
换机器后绝对毫秒数不可比(30674 的 A100 上同一模型读数约 1.6×)。稳妥做法:先在你的机器上跑
`eval_submission.py --model_dir examples/submit_pixG4_g/model_dir` 得到 G4 的 512 时延 `ref_ms`,
再给候选传 `--ref_ms <那个数>`,工具按 0.431/ref_ms 换算到 AutoDL 口径。
独占卡与共享卡的读数差别很大;共享卡上看 event p10/min,不看均值。

## 官方评测已确认/未确认的行为

- 已确认:无视 output_dir,拿 runner 现跑 4K(整图直接喂 infer);python ≤3.9(打包依赖不能遮蔽评测环境的包);
  torch.compile 在评测机能生效;时延对 GPU 时间敏感(G4 实得 152.95 比 k=1.5 的旧模型高 6 分)
- 未确认:时延的精确量法与固定开销;输出是否经 JPEG 再评(工具默认存 PNG 无损,`--jpg_quality 97` 可模拟)
- 本工具的预测残差 ±2 分(五个提交点)

## 自检

`selftest.log` 是打包前在 30674 上用参照 G4 跑 eval_submission 的记录(HOME 指到空目录,强制走包内权重,验证离线可用)。
期望:test NR 2.762±0.006,val FR 1.322±0.002。
