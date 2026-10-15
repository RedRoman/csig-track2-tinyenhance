"""Select actual new student only after full quality and fresh original-tool fit."""
from pathlib import Path
import datetime
import hashlib
import json

R = Path('/cache/independent_sr_cuda_20261013')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    pair = R/'calibration/push5_final_paired_prediction.json'
    assert sha(pair)=='6cb05c63189c6e2c0a3804ee5534463c2fac80a30d3004bdfb42c4d7bb41fd51'
    p = read(pair)
    assert p['status']=='PASS' and p['newly_trained_student_reaches70'] is True
    measured = p['newly_trained_student']
    assert measured['original_tool_prediction']>=70
    run = R/'adaptation/runs/pix_push5_g4_nr250'
    marker_path = run/'checkpoint_step250.json'
    marker = read(marker_path)
    assert marker['local_step']==marker['optimizer_step']==250
    raw,resume = Path(marker['raw_checkpoint']),Path(marker['resume_checkpoint'])
    assert sha(raw)==marker['raw_sha256'] and sha(resume)==marker['resume_sha256']
    config = run/'config.json'
    protocol = read(config)['protocol']
    model = R/'submissions/pix_push5_g4_nr250_s250/model_dir'
    assert sha(model/'model.pth')==marker['raw_sha256']
    quality = R/'nr_push5_full_evaluations/push5_pix_g4_distill250_full'
    result = read(quality/'scores/result.json')
    assert result['status']=='complete' and result['completed']==result['expected']==103
    rows = [json.loads(line) for line in (quality/'scores/measurements.jsonl').read_text().splitlines()]
    assert len(rows)==103 and sum(len(row['metrics']) for row in rows)==418
    for a,b in [(result['test']['nr'],measured['NR']),
                (result['validation']['fr'],measured['FR']),
                (result['S_proxy'],measured['S_proxy'])]:
        assert abs(a-b)<1e-12
    training = R/'analysis/push5/training_audit.json'
    full = R/'analysis/push5/student250_full_audit.json'
    assert sha(training)=='113dae57958284e6dc6ee32d036e663ea846f7703607132109484c13aa4ee9b1'
    assert sha(full)=='da93f0d71954164226f95dce57081e15b3c7347d0fb66677f39ac0c14ca8e8e0'
    package = R/'packages/pix_push5_student250_model_dir.zip'
    assert sha(package)=='b263c875b2f1aa3e7606b5d54727f3646ccbf50accf181c0ef2321a698f81f22'
    selection = dict(schema='nr_push5_selection_v1',
        status='selected_new_student_full103_paired_prediction_ge70',
        selected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        label='push5_pix_g4_distill250_full',NR=measured['NR'],FR=measured['FR'],S_proxy=measured['S_proxy'],
        predicted_tool_score=measured['original_tool_prediction'],target70=70,reaches_target=True,
        prediction_scope='Original supplied five-point empirical tool, not official leaderboard score',
        event_p10_512_ms=measured['event_p10_ms'],local_step=250,optimizer_step=250,
        reference_training_step=30000,new_training_steps=250,
        origin='New G4-initialized pixel-student training, offline HYPIR A7750 test100 PNG distillation; new Adam and RNG',
        checkpoint=str(raw),checkpoint_sha256=marker['raw_sha256'],resume=str(resume),resume_sha256=marker['resume_sha256'],
        model_dir=str(model),quality_dir=str(quality),config=str(config),config_sha256=sha(config),
        marker=str(marker_path),marker_sha256=sha(marker_path),
        training_audit=str(training),training_audit_sha256=sha(training),
        full_audit=str(full),full_audit_sha256=sha(full),
        paired_prediction=str(pair),paired_prediction_sha256=sha(pair),
        package=str(package),package_sha256=sha(package),package_bytes=package.stat().st_size,
        teacher_raw_sha256=protocol['teacher_raw_sha256'],teacher_manifest_sha256=protocol['teacher_manifest_sha256'],
        no_validation_GT_training=True,
        selection_basis='Actual newly-trained student complete103/418 and one fixed fresh G4-HYPIR-student triplet exceeds70; supplied G4 reproduction remains a separate reference',
        visual_limit='Some reference-model fine ripple artifacts reduced; native fine structures still visibly inferior to HYPIR in fixed examples; higher proxy score is not uniformly better fidelity')
    target = R/'results/nr_push5_selection.json'
    with target.open('x') as stream:
        json.dump(selection,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'path':str(target),'sha256':sha(target),'predicted_tool_score':selection['predicted_tool_score']},ensure_ascii=False))


if __name__=='__main__':
    main()
