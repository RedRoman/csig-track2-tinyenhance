"""CPU-only final audit of original assets and completed CUDA render/scores.

No model construction, metric forward, RNG draw, or accelerator call occurs.
Rehashes actual assets/PNGs/GTs, validates full order/checkpoints, and recomputes
aggregation from recorded scalars with the unchanged original CSIG functions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, tool = args.root.resolve(), args.root.resolve() / "scoring_tool"
    hash_cache = {}

    def sha(path):
        path = Path(path).resolve()
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns)
        if identity not in hash_cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hash_cache[identity] = digest.hexdigest()
        return hash_cache[identity]

    def read(path):
        return json.loads(Path(path).read_text())

    def check(condition, description):
        if not condition:
            raise AssertionError(description)

    def number_equal(actual, expected, description):
        check(isinstance(actual, (int, float)) and not isinstance(actual, bool)
              and math.isfinite(actual), description + " nonfinite/not numeric")
        check(math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
              f"{description}: {actual} != {expected}")

    expected_assets = read(tool / "original_tool_provenance_manifest.json")["files"]
    expected_assets += read(tool / "archive_data_manifest.json")
    check(len(expected_assets) == 122, "Expected exact122 original assets")
    asset_records = []
    for record in expected_assets:
        path = tool / record["path"]
        actual = sha(path)
        check(actual == record["sha256"], "Original asset SHA changed:" + record["path"])
        asset_records.append({"path": record["path"], "sha256": actual,
                              "resolved_path": str(path.resolve())})
    sys.path.insert(0, str(tool))
    import csig_metrics as CM
    import torch  # CPU deserialization of our own generated RNG checkpoints only
    from PIL import Image

    original_sha = {r["path"]: r["sha256"] for r in asset_records}
    clip_path = Path.home() / ".cache/clip/RN50.pt"
    check(sha(clip_path) == original_sha["weights/clip/RN50.pt"], "Effective CLIP cache differs from original weight")
    expected_order = [("validation", i) for i in range(1, 4)] + [("test", i) for i in range(1, 101)]
    results = []
    for label in args.label:
        evaluation, scores = root / "evaluations" / label, root / "evaluations" / label / "scores"
        result, protocol, render = read(scores / "result.json"), read(scores / "protocol.json"), read(evaluation / "render_manifest.json")
        check(result["status"] == "complete" and result["completed"] == result["expected"] == 103,
              label + " score not complete103")
        check(result["official_full_count"] is True and result["S_proxy_scope"] == "full100_test_NR_plus3val_FR", label + " result scope")
        check(render["scope"] == "test100+validation3" and render["protocol"]["cases"] == "all", label + " render scope")
        rows = [json.loads(line) for line in (scores / "measurements.jsonl").read_text().splitlines()]
        order = [(r["split"], r["case"]) for r in rows]
        check(order == expected_order, label + " scored order must be exact val1..3 then test1..100")
        render_records, tasks = render["records"], render["protocol"]["tasks"]
        render_order = [("validation" if r["split"] == "val" else r["split"], r["case"]) for r in render_records]
        task_order = [("validation" if r["split"] == "val" else r["split"], r["case"]) for r in tasks]
        check(render_order == task_order == expected_order, label + " renderer order mismatch")
        check(protocol["sequence"] == [list(x) for x in expected_order], label + " protocol order mismatch")
        check(protocol["seed"] == 1234 and protocol["cuda"]["physical_visible_devices"] == "3", label + " scorer seed/GPU")
        check(sha(root / "code/score_cuda_stream.py") == protocol["script_sha256"], label + " stream source changed")
        check(sha(tool / "deployment_manifest.json") == protocol["original_tool_deployment_sha256"], label + " deployment manifest changed")
        check(sha(root / "calibration/environment_protocol.json") == protocol["environment_protocol_sha256"], label + " env protocol changed")
        for name, version in protocol["environment_versions"].items():
            check(importlib.metadata.version(name) == version, label + " runtime package version changed:" + name)
        numpy_compat = protocol["numpy_legacy_compat"]
        check(sha(numpy_compat["path"]) == numpy_compat["sha256"], label + " NumPy compatibility source changed")
        proof = protocol["maniqa_offline_init_proof"]
        check(sha(proof["adapter_path"]) == proof["adapter_sha256"], label + " init adapter changed")
        check(sha(proof["checkpoint_path"]) == proof["checkpoint_sha256"], label + " MANIQA weight changed")
        check(proof["all_final_parameters_and_buffers_equal_checkpoint"] is True
              and not proof["missing_keys"] and not proof["unexpected_keys"]
              and not proof["different_values_or_dtype_or_shape"] and proof["test_sample"] == 100
              and proof["net_training"] is False, label + " MANIQA final state proof invalid")
        for relative, digest in render["protocol"]["model_dir_sha256"].items():
            check(sha(Path(render["protocol"]["model_dir"]) / relative) == digest, label + " modeldir source/config changed:" + relative)
        runner = render["protocol"]["runner"]
        check(sha(runner["weight_path"]) == runner["checkpoint_sha256"], label + " enhancement checkpoint changed")
        checkpoint = torch.load(scores / "rng_checkpoint.pt", map_location="cpu", weights_only=False)
        check(checkpoint["rows"] == rows and checkpoint["active_split"] == "test", label + " checkpoint rows/split mismatch")
        invariant = json.loads(json.dumps(protocol))
        for call in invariant["maniqa_offline_init_proof"]["before_init_records"]:
            call.pop("before_original_checkpoint_load", None)
        fingerprint = hashlib.sha256(json.dumps(invariant, sort_keys=True).encode()).hexdigest()
        check(checkpoint["fingerprint"] == fingerprint, label + " normalized checkpoint fingerprint mismatch")
        check(set(checkpoint["rng"]) == {"torch_cpu", "torch_cuda", "numpy", "python"}, label + " incomplete RNG state")
        png_records, scalar_count = [], 0
        for index, (row, record, task) in enumerate(zip(rows, render_records, tasks)):
            split, case = expected_order[index]
            check(row["label"] == label and record["task_index"] == index, label + " label/task_index mismatch")
            expected_relative = f'render/{"val" if split == "validation" else "test"}/case{case}.png'
            check(record["output"] == expected_relative, label + " PNG output path/order mismatch")
            png = evaluation / expected_relative
            check(Path(row["sr_path"]).resolve() == png.resolve(), label + " scored different PNG path")
            digest = sha(png)
            check(digest == row["sr_sha256"] == record["sha256"], label + " actual/scored/render PNG hash mismatch:" + expected_relative)
            check(sha(task["input"]) == task["input_sha256"], label + " renderer original input changed")
            with Image.open(png) as image:
                size = list(image.size)
            check(size == record["size"] == task["size"] and row["size_hw"] == size[::-1], label + " native shape mismatch")
            required = set(CM.SCORE_KEYS if split == "validation" else CM.NR_KEYS)
            check(set(row["metrics"]) == required, label + " metric keys mismatch")
            for key, value in row["metrics"].items():
                number_equal(value, value, label + f" {split}{case}/{key}")
            scalar_count += len(row["metrics"])
            number_equal(row["nr_per_image"], CM.nr_contrib(row["metrics"]), label + " diagnostic per-image NR")
            if split == "validation":
                gt = tool / "data/验证集" / CM.VAL_PAIRS[case - 1][1]
                check(sha(gt) == row["gt_sha256"] == original_sha[str(gt.relative_to(tool))], label + " actual/scored/original GT mismatch")
            else:
                check(row["gt_sha256"] is None, label + " test unexpectedly has GT")
            png_records.append({"split": split, "case": case, "path": str(png), "sha256": digest})
        check(scalar_count == 418, label + " expected418 finite metric scalars")
        actual_pngs = {str(p.relative_to(evaluation)) for p in (evaluation / "render").rglob("*.png")}
        check(actual_pngs == {r["output"] for r in render_records}, label + " extra/missing PNG files")
        val = CM.mean_metrics([r["metrics"] for r in rows if r["split"] == "validation"])
        test = CM.mean_metrics([r["metrics"] for r in rows if r["split"] == "test"])
        val.update(fr=CM.fr_contrib(val), nr_val=CM.nr_contrib(val), score=CM.proxy_score(val), n=3)
        test.update(nr=CM.nr_contrib(test), n=100)
        for split, expected in (("validation", val), ("test", test)):
            check(set(result[split]) == set(expected), label + " aggregate keys mismatch")
            for key, value in expected.items():
                number_equal(result[split][key], value, label + f" mean-then-clamp {split}/{key}")
        final_s = val["fr"] + test["nr"]
        number_equal(result["S_proxy"], final_s, label + " full S_proxy")
        results.append({"label": label, "status": "passed", "validation_count": 3, "test_count": 100,
                        "PNG_count": 103, "finite_scalar_count": scalar_count, "order_exact": True,
                        "current_PNG_scored_render_hashes_all_equal": True, "current_GT_scored_original_hashes_all_equal": True,
                        "checkpoint_rows_and_normalized_fingerprint_match": True, "all_aggregates_recomputed_from_original_CM": True,
                        "S_proxy": final_s, "files": png_records,
                        "result_sha256": sha(scores / "result.json"), "measurements_sha256": sha(scores / "measurements.jsonl"),
                        "render_manifest_sha256": sha(evaluation / "render_manifest.json"), "protocol_sha256": sha(scores / "protocol.json")})
    output = {"status": "passed", "audit_time_unix": time.time(), "audit_source_sha256": sha(__file__),
              "audit_scope": "CPU-only actual final-file hash/order/checkpoint/scalar aggregation audit; no new inference or GPU test",
              "original_assets_checked": len(asset_records), "current_PNGs_checked": sum(r["PNG_count"] for r in results),
              "finite_metric_scalars_checked": sum(r["finite_scalar_count"] for r in results),
              "assets": asset_records, "effective_CLIP_cache_sha256": sha(clip_path), "evaluations": results,
              "resume_limit_note": "Stream resume does not itself rehash all historical GT/non-MANIQA assets or validate saved row order. This final audit independently verifies every current original asset, historical PNG/GT, exact order, checkpoint and aggregation; it proves this completed run, not safety after future asset changes."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.out) + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({k: output[k] for k in ("status", "original_assets_checked", "current_PNGs_checked", "finite_metric_scalars_checked")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
