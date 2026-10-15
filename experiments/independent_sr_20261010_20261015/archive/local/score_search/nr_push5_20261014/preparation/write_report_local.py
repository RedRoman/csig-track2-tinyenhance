"""Write the target70 report only from the completed measured records."""
from pathlib import Path
import hashlib
import json

L=Path('/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search')
R=Path('/cache/independent_sr_cuda_20261013')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection=L/'results/nr_push5_selection.json';s=read(selection)
    p=read(L/'calibration/push5_final_paired_prediction.json')
    assert p['status']=='PASS' and p['newly_trained_student_reaches70'] is True
    a=p['hypir_A7750'];b=p['g4_reference_reproduction'];n=p['newly_trained_student']
    assert n['original_tool_prediction']==s['predicted_tool_score']>=70
    package=L/'nr_push5_20261014/pix_push5_student250_model_dir.zip'
    resume=L/'nr_push5_20261014/student_resume_step250.pt'
    assert sha(package)==s['package_sha256'] and sha(resume)==s['resume_sha256']
    assert sha(L/'nr_push5_20261014/analysis/training_audit.json')==s['training_audit_sha256']
    assert sha(L/'nr_push5_20261014/analysis/student250_full_audit.json')==s['full_audit_sha256']
    table='\n'.join(f'| {name} | {q["NR"]:.8f} | {q["FR"]:.8f} | {q["S_proxy"]:.8f} | {q["event_p10_ms"]:.8f} | {q["original_tool_prediction"]:.5f} |'
                     for name,q in [('HYPIR A7750，旧模型/伪标签教师',a),('随包G4，仅参考复现',b),('本轮新学生，实际训练250步',n)])
    report=f'''# 70分目标：新学生原工具预估{n['original_tool_prediction']:.5f}分（2026-10-14）

已完成新的轻量像素模型微调、完整103张/418标量评分及一次新鲜同卡配对测速。**本轮新训练学生的原工具预估为{n['original_tool_prediction']:.5f}分，超过70分目标。** 这是用户评分工具的经验预测，不是官方榜单实测，也不保证对这类模型存在±2分误差界。

实现方式是从工具包提供的训练好G4初始化，向HYPIR A7750的现成测试输出蒸馏250步。它是新的轻量学生，不是把原HYPIR同架构微调到128分；随包参考模型的复现成绩另列，未冒充本轮训练成果。

## 完整评分与同卡速度

三组均为相同100张测试输入和3张验证图、相同原始指标/权重/种子策略、原生尺寸round-PNG。下表速度均为本次同一物理A100的512×512 FP16、warm30+n200、CUDA event p10。

| 模型与来源 | test NR ↑ | val FR ↑ | S_proxy ↑ | 512 p10/ms | 本次原工具预估 |
|---|---:|---:|---:|---:|---:|
{table}

新学生相对旧HYPIR：NR {n['NR']-a['NR']:+.8f}、FR {n['FR']-a['FR']:+.8f}、综合S {n['S_proxy']-a['S_proxy']:+.8f}；512输入约快{a['event_p10_ms']/n['event_p10_ms']:.2f}倍。这个倍率只对应512计时，不是原4K整图加速倍率。不要将本次HYPIR估分{a['original_tool_prediction']:.5f}与上一轮58.69787混用：两次G4标定不同。

参考G4的综合S高于新学生。蒸馏后相对G4的NR下降{b['NR']-n['NR']:.8f}，FR提高{n['FR']-b['FR']:.8f}；这是为保真作出的取舍，不称每项指标都提高。

原公式：`152.95 * 2**(S-4.0841) * ((.49+2.17*.431)/(.49+2.17*g))**.2`，`g=本机p10*.431/本次G4 p10`。本次G4为{b['event_p10_ms']:.9f}ms，新学生归一化g={n['normalized_g_ms']:.9f}ms。原predict.py SHA `{p['original_predict_sha256']}`，原benchmark SHA `{p['benchmark_script_sha256']}`，均未修改。

完整计时证据见[本次配对汇总](<{L}/calibration/push5_final_paired_prediction.json>)和同目录`push5_final_{{g4,hypir,student}}_latency.json`。三进程固定G4→HYPIR→学生，分别新建进程，全部使用同一GPU0 UUID/PCI，开始前物理GPU0空闲。没有为了达到目标反复选择G4样本或运行结果；600份event与600份wall原始样本的p10/median/min均从原数据重算。

## 画质取舍与对照

[四例原PNG与固定ROI对比](<{L}/nr_push5_visuals_20261014/index.html>)保留HYPIR、随包参考G4和新学生三列，case7/16/41/83，12张原图与12块ROI均核验完整manifest原SHA，ROI沿用上一轮坐标并逐像素等于原图裁剪。原图不锐化、不润色。

实际看过case16：随包G4在栏杆、把手处有明显融化和细密波纹；新学生减弱部分细波纹，但粗细结构仍明显弱于HYPIR。**高NR和高工具分不等于所有细节更真实。** 新学生的FR仍低于HYPIR，应根据评分和视觉需求选择。HYPIR原版本及其全部证据继续保留。

## 实际训练

服务器30416，实验根目录`{R}`。新run为`adaptation/runs/pix_push5_g4_nr250/`。

- 初始化：用户工具包原G4，PixUNet2，chs32/64/128、nbs1/2/4、无norm、pixel-unshuffle2，100组FP32参数共3,163,404项；参考原训练step30000只是来源，本轮新训练计数从0起。
- 新AdamW：LR1e-5、weight_decay0、foreach=False，独立新随机种子20261014；不继承HYPIR LoRA/Adam或声称HYPIR精确续训。
- 伪标签：A7750完整测试100张原PNG，仅test记录；teacher raw SHA `{s['teacher_raw_sha256']}`，manifest SHA `{s['teacher_manifest_sha256']}`。验证GT不参与训练。
- 输入FP16[-1,1]往返后FP32；1024原生上下文、中央512计算损失，8像素相位对齐，输入/teacher同case同坐标，无缩放。
- 损失：`L1 + .1*LPIPS + .3*gradient_L1 - .01*(MUSIQ/100 + MANIQA + .5*CLIPIQA)`；MANIQA训练20 crops，正式评测100 crops，NIQE只评测。
- 实际训练250步，保存0/125/250的raw与不可变resume。0步逐值等于原G4；125与250的100组FP32参数全部真实变化，raw=resume逐值一致；Adam100组矩有限且步数匹配，四组RNG齐备。冻结指标仍能向RGB传递真实非零梯度。

训练源码`{R}/code/push5/train_pix_g4_nr_stage1.py`，SHA `3a389e55e898be8a64cdda7637339126b1caf94d0b9e7fb19c9f66abe61e8a16`；CLI配置`{R}/code/push5/g4_stage1_conservative250.json`，SHA `d0b1d8560b9bdf43e628fe6fa0b8191b24a85cadc4c4f27fe9964823a7cd4b1c`。实际协议见[训练配置](<{L}/nr_push5_20261014/training/config.json>)，[独立训练审核](<{L}/nr_push5_20261014/analysis/training_audit.json>)。

这100张测试输入参与了无GT适应和选型，因此这些数字描述当前评分任务，不代表独立测试集泛化。伪标签继承HYPIR自身的生成细节，也不是GT。

## 评测核验与提速探索

正式NR=`CLIPIQA+MANIQA+MUSIQ/100+max(0,(10-meanNIQE)/10)`；FR=`2-meanLPIPS-meanDISTS`；S=NR(test100)+FR(val3)。NR FP32、FR FP16 autocast，MANIQA100 crops，每个split起始seed1234后连续采样。原评分网络、权重和代码未改。

原renderer/scorer直接处理学生与参考原尺寸输出。部署沿用随包fold固定时间步、channels_last与512 CUDA graph，原尺寸整图eager；运行时只需学生model_dir内文件和Torch，不依赖SD/HYPIR大模型。metadata中的训练绝对路径只是来源说明。

两组新full均逐张核验103原PNG/输入/3GT/源码/指标权重，从418条原始标量独立重算均值和公式：[参考full审核](<{L}/nr_push5_20261014/analysis/reference_full_audit.json>)、[新学生full审核](<{L}/nr_push5_20261014/analysis/student250_full_audit.json>)。最终CPU join还再次核验实际模型/raw/resume与原计时样本。

同时只探索了一组HYPIR channels_last+reduce-overhead+wavelet compile，512 p10实测39.14435ms，未达到固定质量70所需约19ms。该候选没有完整质量评分，未将旧A质量嫁接到新速度，未生成新完整分数，也未选中。探索记录单独保留。

## 可用文件与恢复

- [学生提交包，顶层model_dir/](<{package}>)：{s['package_bytes']:,}字节，SHA `{s['package_sha256']}`。包内模型字节与实际评测模型完全一致；包含runner、原推理实现、pix架构、权重、来源绑定和config，排除pycache/编译缓存。
- [匹配250步完整续训状态](<{resume}>)：SHA `{s['resume_sha256']}`，FP32参数、新Adam250及四组RNG。续训仍需要100张原输入和对应teacherPNG；本地resume不等于所有训练数据都已复制。
- [正式选型与全部路径](<{selection}>)：SHA `{sha(selection)}`。
- 原raw：`{s['checkpoint']}`，SHA `{s['checkpoint_sha256']}`；部署`model_dir/model.pth`与它字节完全相同，可以原样复制回raw路径。
- 实际部署：`{s['model_dir']}`。
- 全103原图：`{s['quality_dir']}/render/{{test,val}}/`；逐图指标和正式汇总：同目录`scores/measurements.jsonl`、`scores/result.json`。
- 完整源码、配置和训练日志：`{R}/code/push5/`、`{R}/adaptation/runs/pix_push5_g4_nr250/`、`{R}/logs/push5*`。

新共享归档prefix为`/temp/independent_sr_cuda_20261013_artifacts/nr_push5_20261014`，保存学生可携带model_dir、精确resume、原G4初始化、4例选中原PNG及必要小证据。全部103学生原PNG和100教师原PNG仍在/cache，未全部复制到共享存储；精确续训需保留它们，或按原协议重新生成并匹配teacher SHA。之前四轮归档保持完整。

本报告在复制前冻结。归档提交以新prefix的`manifest.json`和`DONE`为准；提交后的独立回执另存于[归档回执](<{L}/results/nr_push5_backup_receipt_20261014.json>)。实时状态`{R}/results/status_cuda_nr_push5_20261014.json`在回执验证后更新complete；归档内是复制前选中快照。
'''
    path=L/'NR_PUSH5_REPORT_20261014.md';path.write_text(report)
    print(json.dumps({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size},ensure_ascii=False))


if __name__=='__main__':main()
