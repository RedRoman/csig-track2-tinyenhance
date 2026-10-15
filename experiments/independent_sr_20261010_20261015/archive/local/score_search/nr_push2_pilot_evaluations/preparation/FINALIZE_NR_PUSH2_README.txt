Append stage3/final artifacts to nr_push2_20261014

Script: /cache/independent_sr_cuda_20261013/code/finalize_nr_push2_artifacts.py
Destination: /temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014

Read-only size/guard check for stage3 checkpoint append:
  /cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/finalize_nr_push2_artifacts.py --checkpoint-only

Explicit checkpoint append, only after the scheduler chooses to run it:
  /cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/finalize_nr_push2_artifacts.py --checkpoint-only --execute

Read-only final check, then explicit final append:
  /cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/finalize_nr_push2_artifacts.py
  /cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/finalize_nr_push2_artifacts.py --execute

Both modes reject either unfinished stage3 run. Checks include local250/global6500,
all514 actual Adam scalar step tensors=500, FP32 LoRA/RNG state presence, completion
marker, exact counters in250 log rows, gradient preflight and training source SHA.
No torch or NumPy is imported; ZIP/pickle metadata references remain inert.

Checkpoint-only adds two stage3 step6500.pth/resume_latest.pt plus their small run
records, training console logs, training code/config/helper/README and candidate
runner/config files. Other intermediate step weights are excluded.

Final mode additionally includes all eight required pilots (and any extra complete
pilot directories), all existing formal full directories, every nontransient PNG,
render protocol/progress/RNG checkpoint, scores protocol/measurements/RNG checkpoint,
and small code/audits/reports/results/submissions files. Pilots must have val3+fixed20
complete rows; each formal full directory must have val3+test100 complete rows.
Pilot scores are labeled S_subset. Formal full records are identified as S_full.

Inventory and exact write-byte/space guards are printed before any SFS write.
The append/finalize reserve is10GiB, chosen for this task, not a user requirement.
The earlier20GiB reserve applied only to the initial checkpoint phase. Insufficient
space prints free/needed/reserve bytes and stops without deleting data.

The script requires the current manifest/DONE checksum chain to reach the verified
stage2 genesis. Before changes, it preserves the exact prior manifest and DONE in
.commits/<prior-manifest-sha>/. Superseded selected data is preserved at
.commits/<prior-manifest-sha>/superseded/<original-relative-path>. The next manifest
retains all previous records, archives old versions, and links the exact prior
manifest/DONE hashes. A new root DONE is written only after the new manifest and all
new/changed data verify. Commit history remains available if an interrupted direct
metadata write needs manual recovery; a checksum mismatch stops further mutation.

Previously verified large files may retain their committed SHA records after an
existence/size check; the manifest discloses this. New/changed/archived files always
receive full SHA-256 readback. Direct final-path streams are used throughout: no
copy2, chmod, timestamp preservation, rename, or deletion. Unrelated destination
files are untouched. Parent prefix data, SD2 weights and env are not recopied.

The original sync_nr_push2_artifacts.py and checkpoint_backup_plan.json remain
frozen and are checked against their original SHA values by this separate script.
