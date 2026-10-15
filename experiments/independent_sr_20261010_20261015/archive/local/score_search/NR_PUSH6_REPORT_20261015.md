# Push6：A7750保画质推理加速（2026-10-15）

当前采用 `push6_a7750_ro_wavelet_all`。训练权重仍是用户认可的 HYPIR A7750，没有新增训练；本轮只改变编译执行方式。正式103张新生成PNG与原A7750逐一字节相同，最终Python、NumPy、Torch CPU/CUDA四项随机状态也相同，因此继承原已独立审计的103图/418项指标，没有伪造一次新评分运行。

|同轮、同卡比较|原A7750|保画质加速版|
|---|---:|---:|
|NR|2.31973328|2.31973328|
|FR|1.44172329|1.44172329|
|S|3.76145657|3.76145657|
|512输入 event p10|46.9421ms|41.4382ms|
|原工具估分|56.4358|57.8501|

耗时下降11.7249%，工具估分增加1.4143。70分尚未达到。估分使用原工具经验公式；本轮G4 p10为0.649215996ms，两列共用这一标定。历史58.15/58.70使用其他轮标定，不能与本轮新分直接相减。

改动仅为UNet.forward、VAE.encode、VAE.decode与完整wavelet后处理的reduce-overhead编译及每次infer的CUDA graph step marker。保留原内存布局、A7750权重、BF16内部精度、输入FP16接口、Gaussian tiling、posterior.sample和DDPM随机调用；没有图像/结果缓存或按评测图像分支。此前39ms的channels-last方案存在像素变化，留作诊断，本轮未选用。

正式质量队列由原renderer在fresh process中初始化seed231，连续val3→test1..100，无测速热身。完整103 PNG、输入、三张GT和旧418项指标逐条对上冻结审计；最终四RNG一致。速度在同一GPU0上串行G4→原A→候选，各fresh process、seed231、warm30、n200，计时包含完整infer。独立CPU脚本重算原始200样本及原predict公式，全部PASS。物理卡UUID/PCI快照取自candidate前后，不冒称有before-G4快照。渲染SSH连接最终返回255，但完整103输出/manifest与独立审计均实际通过，不记作进程exit0。

- [当前版本指针](results/current_model_preference.json) · [最终选择](results/nr_push6_selection.json)
- [同卡估分与原始记录](nr_push6_20261015/calibration/push6_quality_first/paired_prediction.json)
- [103张完全一致证明](nr_push6_20261015/results/ro_wavelet_exact103_v2.json)
- [独立核验](nr_push6_20261015/results/paired_prediction_independent_cpu.json)
- [正式四例1:1可视化](nr_push6_20261015/visuals/formal_A7750_vs_RO_wavelet_final/index.html)

服务器模型目录：`/cache/independent_sr_cuda_20261013/submissions/hypir_push6_a7750_ro_wavelet/model_dir`。新完整输出：`/cache/independent_sr_cuda_20261013/nr_push6_full_evaluations/push6_a7750_ro_wavelet_all/render`。原权重、resume_step7750、旧实验selection/报告和五份共享归档均保持不变；新小增量备份使用独立Push6目录，复用Push4的精确权重恢复方案。
