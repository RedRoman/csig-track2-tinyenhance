# 2026年10月14日 原评分工具独立复测

2026-10-14T17:05:54.871904+08:00：两组复测各完成103图/418标量。与旧完整评测逐项比较均为 **418/418完全相等，最大绝对差0**；NR、FR、S_proxy聚合差也为0。

| 原图版本 | fresh NR(test100) | fresh FR(val3) | fresh S_proxy | 完全相等标量 |
|---|---:|---:|---:|---:|
| compiled6000 | 2.253171081 | 1.463928958 | 3.717100039 | 418/418 |
| conservative6250 | 2.266385043 | 1.463716487 | 3.730101530 | 418/418 |

复测实际调用原score_dir.main及原CM.measure，每split只seed1234一次，MANIQA正式100crop；只增加CPU路径/shape/哈希和返回float日志。离线初始化仍严格核验完整MANIQA最终状态，原计量forward和数学口径不变。GT软链采用canonical路径，原图及训练权重未改变。

每版103个实际PNG的SHA与render manifest、旧记录、新记录一致，103个源LQ与3个GT SHA通过；8个指标权重、冻结源码、最终输出artifact字节验证通过。12个fresh小文件（含最终status）和5份运行源码快照共17文件下载SHA一致。原JSON的val/test/S_proxy原样保留，逐项对照见metric_deltas.csv。独立CPU顺序求均值后clamp的最大浮点差为8.88e-16。

此次没有重新生成图像、训练模型或重测延时。原工具本地proxy不代表官方榜分；这里的“完全相等”针对这两组固定PNG的实际418标量。fresh入口没有保存RNG字节快照，因此不声称比较了两轮RNG状态字节。

文件均仅保存于本地，CPU审计由ssh stdin执行，未在远端写audit。主任务可将audit.json、summary.json、download_sha_manifest.json和metric_deltas.csv连同labels/sources目录归档。

[完整审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/audit.json>) · [summary JSON](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/summary.json>) · [836项差值](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/metric_deltas.csv>) · [下载SHA证明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/download_sha_manifest.json>)
