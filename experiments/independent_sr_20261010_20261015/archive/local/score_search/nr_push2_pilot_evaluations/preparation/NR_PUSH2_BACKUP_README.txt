Stage2 checkpoint-only backup, 2026-10-14

Source root: /cache/independent_sr_cuda_20261013
Destination: /temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014
Parent reference: /temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014

The script defaults to a read-only verification and space report:
  /cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/sync_nr_push2_artifacts.py

Only the parent scheduler should explicitly start the backup after reviewing it:
  /cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/sync_nr_push2_artifacts.py --execute

Before any SFS write, both runs must be complete at local250/global6500/Adam500.
The script verifies status/config/log counters, the completion markers, frozen
training-source hashes, and all 514 Adam scalar step tensors from the real ZIP
checkpoint storage. Its restricted metadata parser imports no torch or NumPy,
constructs no tensors, and never allocates accelerator memory.

Only step6500.pth and resume_latest.pt are copied as large run files. The small
run configs/proofs/logs/completion markers, training code/configs/README, helper
sources, training console logs, candidate runner/config files and preparation
commands are pinned and copied. Other step weights, training locks, environment,
base SD2 weights and parent checkpoints are excluded. The existing parent prefix
and its verified manifest are referenced instead of duplicating those assets.

20 GiB of SFS free space is a conservative reserve chosen for this task. It is
not a user-imposed storage limit. Insufficient space produces the exact free,
copy-needed and reserve byte counts; the script never deletes other files.

Copies use direct byte streams to their final SFS paths, followed by SHA-256
readback. No copy2, metadata preservation, chmod, unlink, or rename is used.
On an interrupted incomplete backup, a rerun verifies matching files and only
recopies missing/incorrect files enumerated in the pinned plan.

manifest.json and DONE are written only after all selected copies verify.
DONE means the checkpoint backup is complete. Final best/full/pilot evaluation
artifacts are deliberately pending a separate finalize operation.
