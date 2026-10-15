"""Compare already completed CPU proof reports for D and the sealed old C control.

Standard library only; no model, weights, CUDA or training is run. This provides
the real D/C comparison separately from a one-branch auditor's trivial pairing.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/cache/independent_sr_cuda_20261013'))
    ap.add_argument('--terminal',action='store_true')
    args=ap.parse_args();r=args.root
    old=r/'audits/push4/training_stage5';new=r/'audits/push4/training_stage5_D'
    cp=r/'code/cuda_training/push4/config_stage5_C_m25_r10_500.json'
    dp=r/'code/cuda_training/push4/config_stage5_D_m25_r075_250.json'
    c,d=read(cp),read(dp)
    diff={k:{'C':c.get(k),'D':d.get(k)} for k in set(c)|set(d) if c.get(k)!=d.get(k)}
    need(set(diff)=={'run','stage_steps','reward_weight'},'D CLI changed beyond run/target/reward')
    need(c['stage_steps']==500 and d['stage_steps']==250 and c['reward_weight']==.10 and d['reward_weight']==.075,
         'Unexpected C/D targets or reward')
    ci,di=read(old/'initial_cpu_audit.json'),read(new/'initial_cpu_audit.json')
    need(ci['status']==di['status']=='PASS' and ci['CPU_only'] and di['CPU_only'] and
         ci['CUDA_context_initialized'] is di['CUDA_context_initialized'] is False,'Initial CPU proofs not PASS')
    cr=next(b for b in ci['branches'] if b['run']=='cuda_nr_stage5_s7250_teacher6500_m25_r10_500')
    need(len(di['branches'])==1,'D initial proof must cover exactly D')
    dr=di['branches'][0]
    need(dr['run']=='cuda_nr_stage5_s7250_teacher6500_m25_r075_250','Wrong D run')
    for k in ['parent_raw_SHA256','parent_resume_SHA256','parent_config_SHA256','first_case','first_crop_xy']:
        need(ci[k]==di[k],f'Parent identity or replayed crop differs: {k}')
    cf,df=cr['first_row'],dr['first_row']
    need(cf['scores']==df['scores'] and cf['fidelity']==df['fidelity'] and cf['reward']==df['reward'],
         'Initial student/teacher/reward metrics not exactly equal')
    need(abs((df['loss']-cf['loss'])-(.10-.075)*cf['reward'])<1e-6,'Initial sole reward-factor loss delta differs')
    proofs=[cp,dp,old/'initial_cpu_audit.json',new/'initial_cpu_audit.json']
    report={'schema':'stage5_D_vs_C_independent_CPU_control_comparison_v1','status':'PASS','CPU_only':True,
            'GPU_used':False,'audited_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'CLI_only_diff':diff,'same_parent_raw_resume_config_SHA':True,
            'same_independently_replayed_first_case_crop':True,'first_case':di['first_case'],
            'first_crop_xy':di['first_crop_xy'],'pre_update_student_metric_scores_exact_equal':True,
            'pre_update_teacher_fidelity_exact_equal':True,'initial_loss_delta_matches_only_reward_change':True,
            'scope':'Initial migration/CLI/first sample only; no terminal-state or quality claim.'}
    if args.terminal:
        ca,da=read(old/'milestone250_cpu_audit.json'),read(new/'milestone250_cpu_audit.json')
        need(ca['status']==da['status']=='PASS' and ca['CPU_only'] and da['CPU_only'] and
             ca['CUDA_context_initialized'] is da['CUDA_context_initialized'] is False,'Terminal CPU proofs not PASS')
        cb=next(b for b in ca['branches'] if b['run']==cr['run']);need(len(da['branches'])==1,'Single D terminal proof required')
        db=da['branches'][0];need(db['run']==dr['run'],'Wrong D terminal run')
        for key,value in [('audited_stage_local_step',250),('audited_global_step',7500),('audited_Adam_step',1500)]:
            need(cb[key]==db[key]==value,'D/C terminal counters differ')
        # Compare the complete ordered sampling prefix and actual stored RNG
        # evidence emitted by the tensor auditor after reading sealed resumes.
        ck='paired_case_crop_sequence_SHA256'
        need(cb['training_log'][ck]==db['training_log'][ck],'D/C complete250 sample/crop sequence differs')
        need(cb['final_rng']['state_sha256']==db['final_rng']['state_sha256'],'D/C terminal four RNG differ')
        proofs += [old/'milestone250_cpu_audit.json',new/'milestone250_cpu_audit.json']
        report.update(scope='Initial plus complete250-update D/C control identity; no scoring claim.',
                      complete250_case_crop_sequence_exact_equal=True,
                      complete250_case_crop_sequence_SHA256=db['training_log'][ck],
                      terminal_four_RNG_state_SHA256_exact_equal=True,
                      terminal_four_RNG_state_SHA256=db['final_rng']['state_sha256'],
                      global_step=7500,Adam_step=1500)
    report['evidence']={str(p):sha(p) for p in proofs}
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
