# 第三轮 CUDA NR 微调：独立策略与审核记录

本轮从第二轮实际完整评分最佳6500版出发，完整NR为2.278310360288、FR为1.461440682411、S为3.739751042699。三条分支各自从同一sealed FP32 LoRA、Adam500及四套RNG继续750步，不能相加称为一个模型继续2250步。每250步保存原始FP32权重与不可变的完整Adam/RNG恢复快照，因此6750/7000/7250都可以作为实际选型与续训边界。

| 分支 | 精确FP32保真teacher | MUSIQ训练系数 | CLIP训练系数 | NR reward总系数 |
|---|---|---:|---:|---:|
| A | 旧6250 | 1.5 | 0.1 | 0.075 |
| B | 当前6500 | 1.5 | 0.1 | 0.075 |
| C | 当前6500 | 1.5 | 0.5 | 0.15 |

A严格延续旧M1.5目标；B单独检验将保真teacher移至当前最佳6500的作用；C同时提高总reward和CLIP系数，属于联合加强NR目标的探索，不能归因成只改CLIP的单变量实验。三条分支LR均为2e-7，L1/LPIPS/空间梯度保真项系数均维持1/0.1/0.3，原评分工具的正式公式、权重与网络不变。训练MANIQA仍20 crops，正式测量100 crops，NIQE仅参与正式评测。

静态审核通过的v4源SHA256为`5f2d40b41034db7f9ae5f6a5f64d90b77bff6bc849422e1fc03ba3eb1600f311`。独立plan SHA256为`ff5a763f17a17d5658d21ef6279d7f98d94f41412b1fe96e69fd26e2891401eb`。旧v1-v3与父run未改写。student和teacher均先从原始raw6000构建、LoRA转FP32后复制selected resume的514份FP32值，逐项精确比对；Adam载入后逐项核验1028份一/二阶矩，再显式写入相同LR，最后恢复Python/NumPy/TorchCPU/CUDA四套RNG。

首步CPU检查已通过：独立重放父Python RNG给出case9、crop[3135,193]，三分支一致。更新前MUSIQ/100、MANIQA、CLIPIQA三个标量分别为0.705217897892、0.621563792229、0.703832805157，三分支精确一致。A首fidelity为0.008083939552，符合student6500与teacher6250的既有差异；B/C首fidelity均约2.25256e-12，符合重锚后数值零。三分支514份LoRA梯度均有限且非零，三个可微指标均有真实非零输入梯度；CPU审核没有初始化CUDA上下文。详见`initial_cpu_audit.json`。

`audit_stage4_cpu.py --local-step 250`、500或750会读取对应`resume_step6750.pt`等不可变快照与最后原子写入的`checkpoint_step6750.json`，核对实际raw/resume SHA、inode/mtime、514份FP32真实参数变化、所有Adam状态/步数、逐步loss/reward、case/crop顺序及四套RNG内容。中间250/500审核不把仍运行的750步stage标成完成，也不要求旧teacher A首fidelity为零。最终750审核才要求run完成标记。

新旧pilot若保持同一实际Runner/底模/源码/权重/RNG协议，可沿用旧6500的固定20图+3验证pilot作为父对照；不能直接从full100的逐图指标抽20项当pilot对照，因跳过其余图会改变MANIQA随机crop序列。pilot只报告S_subset，最后必须重新生成并评分完整100测试+3验证后选择正式最佳模型。

存储检查时/cache约9.6TB可用，共享/temp约9.8GB可用，本机约7.7GiB可用。新一轮完整备份优先保存最终选中边界的raw约1.038GB、完整resume约3.114GB、full103图约2.13GB与小型审核/配置/日志，总计约6.3GB。其余新候选保留/cache；不删旧资产，不重复复制旧父权重、底模或env。本机仅下载指标文档及少量原图/ROI，避免一次下载所有候选权重和full图。
