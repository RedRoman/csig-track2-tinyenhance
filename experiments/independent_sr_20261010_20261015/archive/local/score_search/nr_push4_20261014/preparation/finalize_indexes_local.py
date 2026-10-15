"""Prepend the completed Push4 measured result while preserving older records."""
from pathlib import Path
import hashlib
import json

L = Path('/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search')
STATUS = Path('/Users/hezihao/Documents/New project/independent_sr_ascend_review_20261013/ASCEND_STATUS_20261012.md')
MARKER = '2026-10-14第四轮CUDA短微调'


def main():
    selection_path = L / 'results/nr_push4_selection.json'
    s = json.loads(selection_path.read_text())
    p = json.loads((L / 'calibration/push4_paired_prediction.json').read_text())
    assert s['status'] == 'selected_after_independent_full_audit' and p['status'] == 'PASS'
    assert p['selection_sha256'] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert (L / 'NR_PUSH4_REPORT_20261014.md').is_file()
    summary = (
        f'{MARKER}：完成三路各500步及额外一路250步训练、6组pilot与3组完整100测试+3验证评分。'
        f'最终选中 **A7750／原目标继续500步**，NR **{s["NR"]:.8f}**、FR **{s["FR"]:.8f}**、'
        f'S_proxy **{s["S_proxy"]:.8f}**；相对父7250，NR{p["delta_NR_vs_parent7250"]:+.8f}、'
        f'FR{p["delta_FR_vs_parent7250"]:+.8f}、S{p["delta_S_vs_parent7250"]:+.8f}。'
        f'本次同A100单卡新鲜配对的原工具预估 **{p["parent7250_current_paired_prediction"]:.5f}→'
        f'{p["selected_current_paired_prediction"]:.5f}分**；固定本次父速度仅质量变化为'
        f'**{p["selected_parent_latency_fixed_quality_only_prediction"]:.5f}分**，均为经验预测。'
        f'512×512 event p10 **{p["selected_event_p10_ms"]:.5f}ms**。'
        '本轮增益较小，约73%的NR收益被FR回退抵消；肉眼主要是细微纹理差异。'
        '全部103原PNG保留/cache；共享归档仅保存精确resume、可逐字节恢复raw的recipe、4例原图和小型证据，'
        '完成状态以新prefix `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/DONE` '
        f'及[独立归档回执](<{L}/results/nr_push4_backup_receipt_20261014.json>)为准。'
        f'[完整中文报告](<{L}/NR_PUSH4_REPORT_20261014.md>) · '
        f'[最新4例对比](<{L}/nr_push4_visuals_20261014/index.html>) · '
        f'[GT负例](<{L}/nr_push4_val_visuals_20261014/index.html>) · '
        f'[权重与续训位置](<{selection_path}>)。'
        '30416端口，实验根目录 `/cache/independent_sr_cuda_20261013`。以下保留历史记录。'
    )
    readme = L / 'README.md'
    old = readme.read_text()
    assert MARKER not in old, 'Push4 index already exists; do not duplicate or overwrite history'
    first, rest = old.split('\n', 1)
    readme.write_text(first + '\n\n' + summary + '\n' + rest)
    old_status = STATUS.read_text()
    assert MARKER not in old_status
    STATUS.write_text('## 2026-10-14 第四轮CUDA短微调已完成\n\n' + summary + '\n\n' + old_status)
    print(json.dumps({'readme': str(readme), 'status': str(STATUS)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
