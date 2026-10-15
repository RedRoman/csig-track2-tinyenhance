# CUDA NR微调实验说明（2026年10月14日）

截至北京时间 2026-10-14T00:19:25.887595+08:00，保守分支已完成 CUDA 新增250步训练及完整103图评测。6250版的本地质量代理 S_proxy 为 **3.7301015304490415**，比本次 compiled6000完整基线高 **0.013001492**；增益主要来自 MANIQA、CLIPIQA，存在 MUSIQ、NIQE 和部分保真指标的取舍。6250版 fresh 512计时 event p10 为 **45.780478668毫秒**，原工具经验预测 total 为 **55.515220246**；这是本地估计，不是正式榜分。

本表三版均采用同一 CUDA cache+LoRA fuse+torch.compile 推理实现，model_t200、coeff_t175、rank256、BF16内部计算、empty caption、x1、patch512/stride256。训练保存的是 raw unfused LoRA，推理的融合/编译版本独立评测；不能把旧 NPU FP32/floor结果当作这些 CUDA round-PNG 的成绩。

| 完整版本 | NR：test100 | FR：val3 | S_proxy | 相对compiled6000 | 512 event p10(ms) | 工具预测total |
|---|---:|---:|---:|---:|---:|---:|
| compiled6000 | 2.253171081 | 1.463928958 | 3.717100039 | +0.000000000 | 46.439320374 | 54.86131655 |
| conservative6100 | 2.260917031 | 1.464764992 | 3.725682023 | +0.008581984 | 46.625793457 | 55.14474260 |
| conservative6250 | 2.266385043 | 1.463716487 | 3.730101530 | +0.013001492 | 45.780478668 | 55.51522025 |

完整评分逐项均值后再代入：`NR = CLIPIQA + MANIQA + MUSIQ/100 + max(0,(10-NIQE)/10)`；`FR = 2 - LPIPS - DISTS`；`S_proxy = NR(test100) + FR(val3)`。NR按原工具FP32，FR按原工具FP16 autocast，MANIQA正式评分100个随机crop；native输出尺寸保留。每个split只seed1234一次，逐图遵循原工具metric顺序，断点恢复四套RNG。

| 完整版本 | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ | CLIPIQA ↑ |
|---|---:|---:|---:|---:|
| compiled6000 | 4.008259636 | 43.187275162 | 0.499498821 | 0.722625471 |
| conservative6100 | 4.021619041 | 42.693643131 | 0.507724795 | 0.728417708 |
| conservative6250 | 4.024103260 | 42.354991055 | 0.512805884 | 0.732439575 |

6250相对6100的 NR 增加 0.005468013，FR减少 0.001048505，S_proxy净增加 0.004419507。相对6000，6250的NIQE略升、MUSIQ下降，MANIQA和CLIPIQA升高；并非四项同时改善。6250的LPIPS均值为0.334798177、DISTS为0.201485336，FR较6000降低0.000212471。

三版均在同一A100 GPU0的独立fresh进程，512输入、warm30后计时200次，取CUDA event p10。原工具G4同机参考为0.6502400040626526ms，换算 `g = event_ms * 0.431 / ref_ms` 后为30.781476003 /30.905076363 /30.344774518ms；使用未经修改的原predict.py得到表中预测total。其SHA为 `0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1`，CPU独立重算预测公式通过。模型来自五个pixel-model提交点的经验拟合，对HYPIR属于外推，没有可保证的±2分置信范围。三版使用同一推理算法，约0.2–0.8ms的测量差不能当作微调带来的算法加速。6250相对6000的工具估计增加0.653903697分，仍不等于官方提升。

固定pilot只含20个test病例和3个验证病例，独立输出 S_subset。test顺序为 `[1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100]`，不与test100的NR混比；pilot与完整运行有独立顺序/RNG协议。

| matched pilot版本 | NR：test20 | FR：val3 | S_subset |
|---|---:|---:|---:|
| compiled6000 control | 2.250856000 | 1.463881711 | 3.714737711 |
| conservative6100 | 2.259910396 | 1.464764992 | 3.724675388 |
| aligned6100 | 2.255283384 | 1.464345475 | 3.719628859 |
| conservative6250 | 2.266482978 | 1.463716487 | 3.730199465 |

两个100步分支均从raw6000起步，使用同latent的冻结raw6000/coeff175保真锚点。conservative采用lr2e-7、reward0.075、CLIPIQA权重0.1、空间梯度锚点0.3；aligned采用lr1e-7、reward0.05、CLIPIQA权重1.0、空间梯度锚点0.3。匹配pilot选出conservative后，只将该分支用同一AdamW/RNG从100继续到250。

训练形式为：BF16 base、仅514个FP32 rank256 LoRA tensor更新，batch1、随机512 crop、seed20261013。初始阶段采用新的AdamW（weight_decay0、foreachFalse），没有恢复旧NPU优化器/RNG。实际起点是原raw FP32权重经plain HYPIR BF16加载后再转FP32；首次CPU审计已将这部分量化误差与真实优化增量分开。100→250恢复在FP32参数上复制已保存LoRA，再恢复AdamW和Python/NumPy/TorchCPU/CUDA RNG，保留该CUDA训练run的优化连续性。

保真loss为 `L1(anchor)+0.1*LPIPS(anchor)+0.3*spatial_gradient_L1`，减去 `0.075*(MUSIQ/100+MANIQA+0.1*CLIPIQA)`。MUSIQ/MANIQA/CLIPIQA使用真实网络、FP32、eval/frozen、as_loss梯度；NIQE仅用于评测。训练MANIQA使用20crop近似，正式评分100crop。输入采用部署FP16[-1,1]往返，pred和anchor均加入原HYPIR可微wavelet_reconstruction；当前禁用uint8 round STE，训练loss针对未编码RGB，正式质量由实际rounduint8 PNG检验。anchor与student共享同一detached sampled latent，训练未fuse/compile。

首步在任何opt.step前验证三个真实NR输入梯度非零且有限，并检查LoRA梯度与teacher/metric冻结。250条日志连续覆盖local1..250/global6001..6250，每步记录514个非零有限LoRA梯度。首100训练约95.48秒；恢复150步约141.68秒，稳态median0.8206秒/步、峰值分配约29.95GB。这些时间不含首次模型初始化与独立整图渲染/评分。

CPU恢复审计确认：step6250相对step6100有514/514个tensor、259461000/259474432个FP32元素变化，最大绝对差1.915823668e-5、L2 0.039516588、平均绝对差1.910618882e-6。`resume_latest.pt`全部514个LoRA逐值等于step6250；514份Adam状态全部step250，FP32 moments有限且形状/参数ID齐全，所有必要RNG/step/protocol字段可读。最新完成标记、250行日志与目标6250一致；算法protocol ID保持 `fa5ea19179561aac206b26f7dd3596406f1ad376dccd38f49821146623b98d52`，100步配置已另存config_step6100.json。本报告未更改训练源码或身份。

| 权重/状态 | SHA256 |
|---|---|
| raw6000起点/anchor | `2c82a71a94aa754bd4a26a2a474c2bbe4989bde1d38d2e15d9e58eb0c3e408bc` |
| conservative step6100.pth | `04468a0134579c3e5f63aea93eea2aa1ca956b102f7ed32bb3dc03c918e08ff4` |
| conservative step6250.pth | `7b57349706372122ea67173fbb944e7e806d86e12379e3fd66ea2ebf1f96ecc3` |
| conservative resume_latest.pt（250） | `3dcf896028f41c7c4c6e15074cc69c4cdb2ff7679e6756f5ee57a9a657f61c1f` |
| aligned step6100.pth | `5ff99a06f0a0e723f534b159c62206250bdd91c58caa6a70a1db5944ee6095b5` |

服务器SSH端口30416，根目录 `/cache/independent_sr_cuda_20261013`。训练run在 `adaptation/runs/cuda_nr_s6000_ct175_conservative_pilot100/`（目录名保留pilot100，实际已完成250）；其中step6250.pth约1.038GB、resume_latest.pt约3.114GB。推理协议、逐图PNG/hash和418标量在 `nr_full_evaluations/nr_conservative_s6100_compiled_all/` 与 `nr_full_evaluations/nr_conservative_s6250_compiled_all/`。原compiled6000完整基线在 `evaluations/s6000_mt200_ct175_compiled_cuda_all/`。训练入口 `code/cuda_training/train_cuda_nr.py` SHA为 `0998967aa6052ec9a5090219e17cb73c05468b06d7acc1eae27108c1c412fb4b`；compiled Runner SHA为 `396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`。

两份新full的CPU最终审计均PASS103图/418标量：实际PNG SHA等于manifest与scored记录、103个LQ与3个GT SHA一致，当前checkpoint与renderer/scorer/runner/原工具源码SHA一致；逐项均值后clamp重算最大误差8.88e-16。两full共同输入/顺序/推理参数/评分协议一致，最终四套渲染RNG和四套评分RNG逐字节相等。审计使用CUDA_VISIBLE_DEVICES为空且Torch未创建CUDA上下文，没有GPU模型调用。8个指标权重此前已实际哈希并与原工具压缩包记录匹配。

31个下载的小JSON/JSONL/完成标记均与远端原始字节SHA和大小一致；另核验的pilot6100 manifest也一致。conservative6100和6250的pilot/full前三张验证PNG均字节SHA相同，六张pilot PNG已实际读取核验，full侧对应PASS103实际图像审计。

compiled6000旧full与重做control pilot的val1/2 PNG存在小漂移，val3字节全等；记录的源码、输入、配置、flags、版本一致，具体编译backend/cache原因未证实。该prefix差异造成FR约4.7e-5变化，已单独保留证据，不能声称旧control prefix严格字节复现。所检case7/16/23 ROI未见conservative6100相对control整片规则纹明确增强，部分细纹略弱；这是有限ROI观察，不证明纹理真实性或全图视觉优势。6250随后另外抽查完整评测case7文字和case16栅栏/墙面的原像素ROI：主要字形/布局稳定，细网格和蜂窝墙纹仍有残留，局部暗部略平滑。这只覆盖两个病例，不代表全图视觉排名；原4K PNG已下载并匹配完整manifest SHA。

这些分数来自用户提供工具的本地proxy，当前 S_proxy只包含质量部分，不是工具含速度拟合的predicted total或官方成绩。训练reward和评分共享NR网络，且test100图参与了无GT crop训练，因此不能据此宣称独立泛化或观感必然改善；validation GT未进入训练。速度已按上述同机基准实测并由原工具公式预测；SFS核心归档已经逐文件SHA校验完成，共670文件、18,543,300,225字节；最新文档和审计小文件再并入最终manifest。共享目录为 `/temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014`，同时包含两个分支优化器/RNG恢复状态、raw6000 anchor、6100/6250选中权重、206张full PNG、92张pilot PNG、训练日志和原评分资产。

可审计的本地记录：

- [full6100 CPU审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/audits/nr_cuda_full_s6100_audit.json>)
- [full6250 CPU审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/audits/nr_cuda_full_s6250_audit.json>)
- [三版full汇总](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/audits/nr_cuda_full_summary_20261014.json>)
- [250步参数与Adam恢复审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/training_250_parameter_audit.json>)
- [matched pilot审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/matched_pilot/matchedpilot_summary.json>)
- [ROI视觉说明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/matched_pilot_visual/inspection_roi/visual_review.md>)
- [6250原始完整评分结果](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_cuda_evaluations/nr_conservative_s6250_compiled_all/scores/result.json>)
- [6250逐图指标JSONL](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_cuda_evaluations/nr_conservative_s6250_compiled_all/scores/measurements.jsonl>)
- [250步配置与原始训练日志](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/results/cuda_nr_s6000_ct175_conservative_step6250/config.json>)

- [下载字节一致性及验证prefix证明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/audits/nr_cuda_small_download_manifest.json>)
- [原工具质量/延时/预测对照](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_full_quality_latency_comparison.json>)

恢复训练和运行环境说明见 [恢复说明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/RESTORE_NR_CUDA_20261014.md>)；最新full视觉原像素对比见 [case7文字](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_cuda_outputs/inspection_rois/case7_text_exact.png>)、[case16栅栏](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_cuda_outputs/inspection_rois/case16_grid_exact.png>)。
