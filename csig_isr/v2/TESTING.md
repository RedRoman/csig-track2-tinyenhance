# Package self-check

Completed before packaging:

- `python -m py_compile code/*.py code/legacy/*.py` — passed.
- `degrade_v2()` smoke test on a synthetic 512×512 RGB uint8 image — passed; output shape/dtype and severity range checked.
- `configs/metrics.yaml` load and proxy-score smoke test — passed.
- `validate_candidate.py` mock test with 100 images and matching manifests — passed.
- stale `tinysr8_fid/best.pth` hard-coding removed from the v2 render/export path.

Not executable in this container:

- Full TinySR forward/backward, because the uploaded archive intentionally contains only experiment code/results and does not contain the external TinySR repository, pretrained weights, competition data, or the team's run11/run12 `.pth` checkpoints.
- pyiqa GPU metric execution for the same reason/environment mismatch.

Therefore the first server-side action should be the P0 canonical baseline command in `README.md`. If that fails before producing the old run12-quality range, stop and fix environment/checkpoint loading before training any v2 experiment.
