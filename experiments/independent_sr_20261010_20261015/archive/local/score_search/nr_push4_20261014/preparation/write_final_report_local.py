"""Build the report only from completed, independently audited local records."""
from pathlib import Path
import hashlib
import json

L = Path('/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search')
R = Path('/cache/independent_sr_cuda_20261013')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    s = read(L / 'results/nr_push4_selection.json')
    p = read(L / 'calibration/push4_paired_prediction.json')
    full = read(L / 'nr_push4_20261014/evaluation_audits/fair_full_extended_table.json')
    pilot = read(L / 'nr_push4_20261014/evaluation_audits/fair_pilot_table.json')
    assert s['status'] == 'selected_after_independent_full_audit'
    assert p['status'] == 'PASS' and p['selection_sha256'] == sha(L / 'results/nr_push4_selection.json')
    assert full['status'] == pilot['status'] == 'complete'
    assert full['complete_candidates'] == 3 and pilot['complete_candidates'] == 6
    winner = next(x for x in full['rows'] if x['label'] == s['label'])
    parent = next(x for x in full['rows'] if x.get('is_reference'))
    assert winner['S_proxy'] > parent['S_proxy']
    branch = Path(s['config']).parent.name
    full_md = (L / 'nr_push4_20261014/evaluation_audits/fair_full_extended_table.md').read_text().split('| 版本 |', 1)[1]
    pilot_md = (L / 'nr_push4_20261014/evaluation_audits/fair_pilot_table.md').read_text().split('| 版本 |', 1)[1]
    c = winner['contribution_deltas_vs7250']
    comparison = '\n'.join(f'| {key} | {parent["test_means"][key]:.9f} | {winner["test_means"][key]:.9f} |'
                           for key in ('clipiqa', 'maniqa', 'musiq', 'niqe'))
    report = f'''# 第四轮 CUDA 短微调与完整复测（2026-10-14）

本轮完成三路各500步微调及额外一路250步短试、六个固定小样本评测和三组完整103张复测。最终选中 **{s['label']}**：NR **{s['NR']:.9f}**、FR **{s['FR']:.9f}**、S_proxy **{s['S_proxy']:.9f}**。相对7250步父模型，NR {p['delta_NR_vs_parent7250']:+.9f}、FR {p['delta_FR_vs_parent7250']:+.9f}、S {p['delta_S_vs_parent7250']:+.9f}。

本次新鲜同卡原工具预估 **{p['parent7250_current_paired_prediction']:.5f} → {p['selected_current_paired_prediction']:.5f} 分**；固定本次父模型速度，仅质量变化为 **{p['selected_parent_latency_fixed_quality_only_prediction']:.5f}**。这些是原五点经验工具预测，不是官方榜分。512×512 A100单卡event p10为 **{p['selected_event_p10_ms']:.5f}ms**；不得将其说成原4K全图延迟。

## 本轮训练的变化

端口30416，实验根目录 `{R}`。A/B/C三路从同一sealed C7250 FP32 LoRA、Adam1250和Python/NumPy/Torch CPU/CUDA四组随机状态起步，各训练500次，保存global7500/Adam1500和global7750/Adam1750；额外D从同一起点训练250次，保存global7500/Adam1500。四路是独立分叉，不是同一模型累计1750步。

| 分支 | MUSIQ训练系数 | 总NR奖励系数 | 其他参数 |
|---|---:|---:|---|
| A：原目标延续 | 1.5 | 0.15 | 共同参数见下 |
| B：提高MUSIQ | 2.5 | 0.15 | 共同参数见下 |
| C：降低总奖励 | 2.5 | 0.10 | 共同参数见下 |
| D：进一步降低奖励，250步 | 2.5 | 0.075 | 共同参数见下 |

共同参数：teacher6500精确FP32冻结权重、LR2e-7、CLIPIQA训练权重0.5、L1保真1、LPIPS保真0.1、空间梯度保真0.3；MANIQA训练20 crops。损失为 `L1+.1*LPIPS+.3*gradient_L1 − reward_weight*(musiq_weight*MUSIQ/100+MANIQA+.5*CLIPIQA)`。NIQE仅用于评测。A是相同损失的精确延续；B/C显式改变目标并继承Adam矩，不将其描述为原损失的连续优化。

之前最后250步的NR增益约77%被FR回退抵消，因此本轮比较奖励与保真的平衡。实际六组pilot表明，单独提高MUSIQ系数B在两个步数都降低了正式MUSIQ和综合S；C通过降低总奖励改善了B的保真损失。C7750首组完整S仅增加0.001158，仍很微弱，因此随后真实追加D250，只将总奖励改为0.075，并直接做同协议103图完整评测，未将其填入不存在的pilot结果。超参变化不保证对应正式指标单调上升。

训练源 `{R}/code/cuda_training/train_cuda_nr_stage_v5.py`，SHA256 `4b5bdf40aab5d2fd35c92f8fc714c675e62bd86228bc3fb79e48e896b70278f2`。四份实际CLI配置与说明位于 `{R}/code/cuda_training/push4/`。独立CPU审计覆盖初始迁移和实际里程碑：514份FP32权重有限且真实变化、原始权重与不可变resume逐值相等、1028份Adam矩正确、四组RNG可恢复；相同步数的case/crop和终点RNG对照一致；未初始化CUDA。首case93/crop[60,1399]与首步三个指标逐值一致。证据见[三路训练审计](<{L}/cuda_training/stage5/>)、[D短试审计](<{L}/cuda_training/stage5_D/>)和服务器同名 `audits/push4/training_stage5*` 目录。

仍使用这100张测试输入做无GT的NR优化，验证GT不参与训练。本报告描述这套评分任务上的选型，不声称独立测试集泛化。

## 原工具评测

正式NR固定为 `CLIPIQA+MANIQA+MUSIQ/100+max(0,(10−meanNIQE)/10)`；FR为 `2−meanLPIPS−meanDISTS`；S_proxy=NR(test100)+FR(val3)。正式MANIQA仍100 crops、NR FP32、FR FP16 autocast，评分代码、网络权重和公式未改。

推理仍为HYPIR rank256、model_t200/coeff_t175、空提示、BF16内部计算、缓存+LoRA融合+原tile compile。原尺寸输入FP16[-1,1]，输出FP32[-1,1]并round成uint8原尺寸PNG。原runner SHA256 `396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`。各候选只改变checkpoint和label，source、输入、GT、原指标及随机策略由独立审计实际SHA验证。

下表均为完整103张/418标量独立审核PASS；在三组完整结果齐备后按最高S选型，并与父7250比较：

| 版本 |{full_md}

以下是固定test20+val3、23图/98标量小样本，只称S_subset；用于筛选，不作为完整S或工具榜分：

| 版本 |{pilot_md}

完整100张测试分项均值：

| 指标 | 父7250 | 本轮选中 |
|---|---:|---:|
{comparison}

NR增量分解：CLIPIQA {c['clipiqa']:+.9f}、MANIQA {c['maniqa']:+.9f}、MUSIQ/100 {c['musiq_div100']:+.9f}、NIQE项 {c['niqe']:+.9f}，合计 {p['delta_NR_vs_parent7250']:+.9f}；再加FR变化 {p['delta_FR_vs_parent7250']:+.9f} 得到完整S变化。保留所有分项及负例，不将NR改善等同于每项指标或真实细节都改善。完整证据见[评测审核目录](<{L}/nr_push4_20261014/evaluation_audits/>)。

## 本次同卡测速

G4、父7250、新选中模型分别用独立新进程，在同一物理GPU0顺序warm30/n200、seed231、512×512 FP16；每次开始前无其他计算进程。编译、加载、warmup和hash不计推理时间。

| 项目 | 本次实测/原工具计算 |
|---|---:|
| G4 event p10 | {p['G4_event_p10_ms']:.9f}ms |
| 父7250 event p10 | {p['parent7250_event_p10_ms']:.9f}ms |
| 新模型event p10 | {p['selected_event_p10_ms']:.9f}ms |
| 父7250本次预估 | {p['parent7250_current_paired_prediction']:.9f} |
| 新模型本次预估 | {p['selected_current_paired_prediction']:.9f} |
| 固定父速度、仅质量变化 | {p['selected_parent_latency_fixed_quality_only_prediction']:.9f} |

物理卡 `{p['physical_card_proof']['uuid']}`、PCI `{p['physical_card_proof']['pci_bus_id']}`。质量因子 {p['quality_factor_vs_parent7250']:.12f}；本次计时因子 {p['measured_timing_factor_not_proven_algorithm_speedup']:.12f}，单次细小延迟差不证明算法稳定加速。只比较本次新鲜配对结果，不混用上轮56.41263及不同G4标定。

原predict.py SHA256 `{p['original_predict_sha256']}`，原benchmark SHA256 `{p['benchmark_script_sha256']}`。CPU join重新核验实际完整PNG、逐图指标、权重与推理配置以及三组各200份原始样本。见[配对预测核验](<{L}/calibration/push4_paired_prediction.json>)和同目录 `push4_*_latency.json`、`push4_latency_launch.json`。

## 权重、续训和原图

- 选型：[nr_push4_selection.json](<{L}/results/nr_push4_selection.json>)，SHA256 `{p['selection_sha256']}`。
- 原始LoRA：`{s['checkpoint']}`，SHA256 `{s['checkpoint_sha256']}`。
- 精确不可变resume：`{s['resume']}`，SHA256 `{s['resume_sha256']}`；local{s['local_step']}/global{s['global_step']}/Adam{s['optimizer_step']}。
- 配置：`{s['config']}`；训练run `{branch}`。resume包含同一步514份FP32权重、Adam矩/步数和四组RNG，续训使用新run，不覆盖父模型。
- 实际推理模型：`{s['model_dir']}`。
- 完整103张原PNG：`{s['quality_dir']}/render/{{test,val}}/`；逐图指标：`{s['quality_dir']}/scores/measurements.jsonl`；正式汇总同目录 `result.json`。
- 训练独立审核：`{s['training_audit']}`，SHA256 `{s['training_audit_sha256']}`。
- 完整独立审核：`{s['full_audit']}`，SHA256 `{s['full_audit_sha256']}`。

[最新4例对比图库](<{L}/nr_push4_visuals_20261014/index.html>)保留父7250及本轮完整候选在case7、16、41、83的原4K PNG、同坐标1:1 ROI和1/8概览。父图复用旧原PNG硬链接，新图必须匹配最终full manifest；裁剪逐像素核验，不锐化、不润色。视觉变化较细，不能据评分微增宣称明显结构改善。

[验证GT负例](<{L}/nr_push4_val_visuals_20261014/index.html>)使用固定旧val1 ROI[2304,2688,2944,3200]，保留GT/父7250/候选原图与感知指标；未用GT重新选区域。单个局部裁剪不能解释整图全部FR变化。图库与诊断源码/来源SHA/最终检查记录在 `{R}/audits/push4/visuals/`。

## 持久化和精确恢复

新归档prefix `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014`。旧三轮归档保持原样。由于共享存储余量有限，本轮只持久化选中的完整immutable resume、原始raw ZIP的精确重建recipe/gaps、4张原尺寸案例以及新源码/配置/全部小型训练与评测证明。

**完整103张PNG保留/cache，未全部复制到共享存储；raw没有重复占用1GB，但可由resume和recipe逐字节重建成原文件。** 全部103张仍逐张核验SHA并保留指标与manifest。recipe不是精度压缩：514份FP32 tensor字节来自同一步resume，原ZIP其余字节原样保留，已实际重建完整raw并验证原SHA。

选中recipe目录 `{s['raw_recipe_dir']}`，实际恢复证明 `{s['raw_recipe_restore_proof']}`。共享归档mirror相同相对路径。恢复时用归档 `code/push4/checkpoint_archive_recipe.py restore --directory <recipe目录> --resume <immutable-resume> --output <新的raw输出路径>`，成功后raw完整字节SHA必须为 `{s['checkpoint_sha256']}`。工具仅用Python标准库，拒绝覆盖已有输出。

复制助手只在全部源稳定、实际空间充足时流式写入新prefix，再完整SHA读回，最后写manifest和DONE。提交以 `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/DONE` 的manifest SHA为准。提交后独立回执在 `{R}/results/nr_push4_backup_receipt_20261014.json`，本地[归档回执](<{L}/results/nr_push4_backup_receipt_20261014.json>)在独立核验后写入；报告复制前冻结，回执单独保存避免自引用。

共享基础SD2仍在 `/temp/zihaohe_stage/baselines/sd21b`，原输入、GT、指标及旧代码依赖由旧三轮prefix和共享SD2共同提供，本轮不是整套环境的独立副本。实时状态 `{R}/results/status_cuda_nr_push4_20261014.json` 在提交后更新complete，归档内状态是复制前的选中快照。
'''
    target = L / 'NR_PUSH4_REPORT_20261014.md'
    target.write_text(report)
    print(json.dumps({'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target)}, ensure_ascii=False))


if __name__ == '__main__':main()
