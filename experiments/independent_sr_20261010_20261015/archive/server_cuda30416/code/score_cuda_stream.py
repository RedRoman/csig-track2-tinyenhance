"""Score original CSIG metrics on arriving CUDA PNGs with exact RNG resume.

Each label has its own continuous RNG context, reset once per split to1234.
Multiple labels can be interleaved on one GPU without changing crop draws.
CM, score_dir, metric forward, dtype policy, and MANIQA100 crops are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def save_context(path, state, torch):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def capture_rng(torch, np):
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state, torch, np):
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def publish(ctx, CM):
    rows, out = ctx["state"]["rows"], ctx["out"]
    with tempfile.NamedTemporaryFile("w", dir=out, delete=False) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = stream.name
    os.replace(temporary, out / "measurements.jsonl")
    result = {"label": ctx["label"], "status": "complete" if len(rows) == len(ctx["sequence"]) else "running",
              "completed": len(rows), "expected": len(ctx["sequence"]), "official_full_count": len(ctx["sequence"]) == 103}
    for split in ("validation", "test"):
        selected = [r["metrics"] for r in rows if r["split"] == split]
        if selected:
            mean = CM.mean_metrics(selected)
            mean["n"] = len(selected)
            if split == "validation":
                mean.update(fr=CM.fr_contrib(mean), nr_val=CM.nr_contrib(mean), score=CM.proxy_score(mean))
            else:
                mean["nr"] = CM.nr_contrib(mean)
            result[split] = mean
    if len([r for r in rows if r["split"] == "validation"]) == 3 and "test" in result:
        result["S_proxy"] = result["validation"]["fr"] + result["test"]["nr"]
        result["S_proxy_scope"] = "full100_test_NR_plus3val_FR" if result["official_full_count"] and result["status"] == "complete" else "partial_or_preflight_only"
    atomic_json(out / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-dir", type=Path, required=True)
    parser.add_argument("--evaluation", action="append", required=True, help="label=render directory")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--test-cases", default=",".join(map(str, range(1, 101))))
    parser.add_argument("--poll", type=float, default=3.0)
    args = parser.parse_args()
    tool = args.tool_dir.resolve()
    os.environ["CSIG_MANIQA_SAMPLES"] = "100"
    sys.path.insert(0, str(tool))
    import numpy_legacy_compat
    import maniqa_offline_init
    import numpy as np
    import torch
    import csig_metrics as CM
    if not torch.cuda.is_available():
        raise RuntimeError("This scorer requires NVIDIA CUDA; CPU/NPU scoring is not accepted")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    maniqa_offline_init.install(tool / "weights/torch/hub/checkpoints/ckpt_koniq10k.pt")
    metrics = CM.create_metrics(device, maniqa_samples=100)
    audit = maniqa_offline_init.verify_final_state(metrics["maniqa"].net)
    if audit["test_sample"] != 100 or audit["net_training"]:
        raise RuntimeError("MANIQA must remain in eval mode with original full100-crop forward")
    deployment = json.loads((tool / "deployment_manifest.json").read_text())
    environment = json.loads((tool.parent / "calibration/environment_protocol.json").read_text())
    cases = [int(x) for x in args.test_cases.split(",") if x]
    if not cases or cases != sorted(set(cases)) or any(x < 1 or x > 100 for x in cases):
        raise ValueError("--test-cases must contain distinct ascending cases in1..100")
    sequence = [("validation", i) for i in range(1, 4)] + [("test", i) for i in cases]
    common = {"metric_source": "unmodified original CSIG csig_metrics.py",
              "original_tool_deployment_sha256": sha256(tool / "deployment_manifest.json"),
              "original_sources": [r for r in deployment["files"] if r["path"].endswith(".py")],
              "script_sha256": sha256(__file__), "environment_protocol_sha256": sha256(tool.parent / "calibration/environment_protocol.json"),
              "environment_versions": environment["versions"], "numpy_legacy_compat": environment["numpy_legacy_compat"],
              "maniqa_offline_init_proof": audit, "seed": 1234,
              "seed_scope": "once at start of each split for each label, continuous draw order within that split",
              "metric_order": {"validation": list(CM.SCORE_KEYS), "test": list(CM.NR_KEYS)},
              "cuda": {"physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_device": 0,
                       "name": torch.cuda.get_device_name(0), "torch_version": torch.__version__},
              "image_policy": "CM.load_img original FP32 RGB[0,1]; FR FP16 autocast only; all NR original FP32; no resize or edits",
              "resume_policy": "Atomic trusted local checkpoint persists rows and all CPU/CUDA/NumPy/Python RNG states. No completed metric call is replayed or skipped without restoring its post-call state.",
              "interleaving_policy": "Independent per-label RNG contexts restored immediately before each image. Metrics share fixed weights under the original no_grad calls; forward and100-crop batch unchanged."}
    contexts = []
    for spec in args.evaluation:
        label, directory = spec.split("=", 1)
        if not label or "/" in label or label in {".", ".."}:
            raise ValueError("Evaluation label must be a single directory name")
        out = args.out_root.resolve() / label / "scores"
        out.mkdir(parents=True, exist_ok=True)
        protocol = {**common, "label": label, "render_dir": str(Path(directory).resolve()), "sequence": sequence}
        # Random untrained init hashes differ by process; they are evidence only,
        # so fingerprint the invariant final checkpoint proof for exact resume.
        invariant = json.loads(json.dumps(protocol))
        for call in invariant["maniqa_offline_init_proof"]["before_init_records"]:
            call.pop("before_original_checkpoint_load", None)
        fingerprint = hashlib.sha256(json.dumps(invariant, sort_keys=True).encode()).hexdigest()
        checkpoint = out / "rng_checkpoint.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["fingerprint"] != fingerprint:
                raise RuntimeError(f"Resume protocol changed for{label}; use a new output directory")
            for row in state["rows"]:
                if sha256(row["sr_path"]) != row["sr_sha256"]:
                    raise RuntimeError(f"Previously scored PNG changed:{row['sr_path']}")
        else:
            state = {"fingerprint": fingerprint, "rows": [], "active_split": None, "rng": None}
        atomic_json(out / "protocol.json", protocol)
        ctx = {"label": label, "render": Path(directory).resolve(), "out": out,
               "sequence": sequence, "checkpoint": checkpoint, "state": state}
        contexts.append(ctx)
        publish(ctx, CM)
    print(json.dumps({"status": "initialized", "labels": [c["label"] for c in contexts], "GPU": common["cuda"]}), flush=True)
    while any(len(c["state"]["rows"]) < len(sequence) for c in contexts):
        progressed = False
        for ctx in contexts:
            state = ctx["state"]
            index = len(state["rows"])
            if index == len(sequence):
                continue
            split, case = sequence[index]
            sr_path = ctx["render"] / ("val" if split == "validation" else "test") / f"case{case}.png"
            if not sr_path.is_file():
                atomic_json(ctx["out"] / "status.json", {"status": "waiting_for_png", "completed": index,
                            "expected": len(sequence), "next_split": split, "next_case": case, "next_path": str(sr_path)})
                continue
            if state["rng"] is not None:
                restore_rng(state["rng"], torch, np)
            if state["active_split"] != split:
                CM.seed_all(1234)
                state["active_split"] = split
            started = time.perf_counter()
            sr = CM.load_img(str(sr_path), device)
            gt_path = tool / "data/验证集" / CM.VAL_PAIRS[case - 1][1] if split == "validation" else None
            gt = CM.load_img(str(gt_path), device) if gt_path else None
            if gt is not None and sr.shape != gt.shape:
                raise RuntimeError(f"SR/GT shape mismatch for {ctx['label']} validation{case}")
            measured = CM.measure(metrics, sr, gt)
            shape = list(sr.shape[-2:])
            del sr, gt
            torch.cuda.empty_cache()
            state["rows"].append({"label": ctx["label"], "split": split, "case": case,
                                  "metrics": measured, "nr_per_image": CM.nr_contrib(measured),
                                  "sr_path": str(sr_path), "sr_sha256": sha256(sr_path),
                                  "gt_sha256": sha256(gt_path) if gt_path else None,
                                  "size_hw": shape, "seconds": time.perf_counter() - started,
                                  "seed_scope": "split_seed1234_continuous_rng"})
            state["rng"] = capture_rng(torch, np)
            save_context(ctx["checkpoint"], state, torch)
            publish(ctx, CM)
            atomic_json(ctx["out"] / "status.json", {"status": "complete" if len(state["rows"]) == len(sequence) else "running",
                        "completed": len(state["rows"]), "expected": len(sequence), "last_split": split, "last_case": case})
            print(json.dumps({"label": ctx["label"], "split": split, "case": case,
                              "completed": len(state["rows"]), "metrics": measured}), flush=True)
            progressed = True
        if not progressed:
            time.sleep(args.poll)
    print("All requested evaluations completed", flush=True)


if __name__ == "__main__":
    main()
