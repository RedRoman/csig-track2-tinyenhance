# Final Push5 paired verification

Driver: `paired_latency_push5_final.py` SHA `60778d21970a5ec9eaa003115111c3a58c725da5290654d67a280adcb306bb2c`.
CPU join: `join_paired_prediction_push5_final.py` SHA `a00797509cb73e1e7dfc89cc63958b70e50e564555f8960abba5e788fc7a3621`.
Original benchmark/predict remain unchanged. Default driver is dryrun. Root exclusively schedules GPUs.

Wait for actual completed student full103 and immutable step250 marker. Planned paths below are CLI parameters,
not a registration or a claim that those outputs already exist.

```sh
R=/cache/independent_sr_cuda_20261013
CUDA_VISIBLE_DEVICES=0 "$R/env/bin/python" "$R/code/push5/final_pair/paired_latency_push5_final.py" \
  --student-evaluation "$R/nr_push5_full_evaluations/push5_pix_g4_distill250_full" \
  --student-marker "$R/adaptation/runs/pix_push5_g4_nr250/checkpoint_step250.json" \
  --student-step 250 --plan-out "$R/calibration/push5_final_paired_plan.json"
```

Only root adds `--execute` when GPU0 is idle. Exactly one fresh G4 -> HYPIR A7750 -> student triplet,
separate processes, seed231, warm30/n200, original p10; no repeated G4/sample selection to meet a target.
The G4 metadata adapter is the same model used in actual reference full103 and delegates original inference.
All three timings bind their own exact full-quality model/checkpoint/config/source hashes.

```sh
CUDA_VISIBLE_DEVICES='' "$R/env/bin/python" "$R/code/push5/final_pair/join_paired_prediction_push5_final.py"
```

Join reads calibration/push5_final_latency_launch.json and three original timing artifacts; recomputes all200
event/wall sample statistics and exact original predict(). It re-hashes actual full103 PNG/input/GT and code/metric
weights, checks unchanged formal100-crop policy, and recomputes418 scalars from existing records without any metric rerun.
It compares raw/resume100 FP32 weights, new Adam steps/moments, four RNG presence, actual training step>0,
reference step30000, origin, configuration protocol/source SHA, original G4 initialization and offline A7750
teacher manifest/100 PNG dependencies. Raw step125 may be used only as a CPU schema smoke, never as final250 quality.

Result: calibration/push5_final_paired_prediction.json, immutable once created.
Separate fields: g4_reference_reproduction, hypir_A7750, newly_trained_student.
Only newly_trained_student_reaches70 decides the new-training target; G4 supplied-reference reproduction is not new work.
The original five-point empirical prediction is not an official measured leaderboard score.
No old Push4 selection, original model, scoring source, or old result is modified.

CPU-only compatibility checks already completed: static compile/import (no Torch imported), and actual A7750 +
G4 full103 quality-source checks. Generic G4 recomputation: NR2.7597173432143904/FR1.3216533660888672/S4.081370709303258.
HYPIR speed exploration stopped at39.144345092773435ms (unscored candidate); its old full quality is never joined to that timing.
