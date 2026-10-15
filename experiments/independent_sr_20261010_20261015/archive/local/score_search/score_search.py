#!/usr/bin/env python3
"""Independent pilot/full scorer with the supplied tool's ordered split RNG.

--test-cases pilot scores val3+fixed20 and emits only S_subset. --test-cases all
scores val3+test100 and emits S_proxy. Native PNGs are not resized/re-encoded;
component means precede the NIQE clamp. Model/dataset seed1234 is set once per
split, not once per image. MANIQA uses 100 weighted crops in microbatches of4.
CUDA and Ascend backends are supported without changing existing eval scripts.
Each committed image includes all metrics and Python/NumPy/CPU/device RNG states,
so a restart restores the ordered crop stream rather than incorrectly skipping.
No calibrated 512 CUDA latency or official leaderboard score is produced.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time


REQUIRED_WEIGHTS = (
    "torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth",
    "torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
    "torch/hub/checkpoints/DISTS_weights-f5e65c96.pth",
    "torch/hub/checkpoints/vgg16-397923af.pth",
    "torch/hub/checkpoints/niqe_modelparameters.mat",
    "torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth",
    "torch/hub/checkpoints/ckpt_koniq10k.pt",
    "clip/RN50.pt",
)
NR_ORDER = ("niqe", "musiq", "maniqa", "clipiqa")
FR_ORDER = ("lpips", "dists")
PILOT_CASES = [1, 6, 7, 11, 16, 18, 21, 23, 25, 28, 31, 36, 41, 46, 51, 61, 71, 81, 91, 100]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replace_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def publish_jsonl(path, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def capture_rng(torch, np, index, backend):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "device": getattr(torch, backend).get_rng_state(index)}


def restore_rng(torch, np, index, state, backend):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    getattr(torch, backend).set_rng_state(state["device"], index)


def seed_once(torch, np, backend):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    getattr(torch, backend).manual_seed(1234)


def commit_state(torch, path, state):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def input_record(path, Image):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != ".png":
        raise ValueError(f"Expected existing PNG output: {path}")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"Output is not PNG-encoded: {path}")
        width, height = image.size
    return {"path": str(path), "sha256": sha256_file(path), "size_hw": [height, width]}


def make_tasks(args, cm, Image):
    tasks = []
    for case, (_, gt_name) in enumerate(cm.VAL_PAIRS, 1):
        sr = input_record(args.val_dir / args.val_pattern.format(case=case), Image)
        gt_path = (args.gt_dir / gt_name).resolve()
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing validation GT: {gt_path}")
        with Image.open(gt_path) as image:
            width, height = image.size
        if sr["size_hw"] != [height, width]:
            raise ValueError(f"case{case}: output {sr['size_hw']} != GT {[height, width]}")
        tasks.append({"split": "validation", "case": case, "sr": sr,
                      "gt": {"path": str(gt_path), "sha256": sha256_file(gt_path)}})
    for case in args.cases:
        sr = input_record(args.test_dir / args.test_pattern.format(case=case), Image)
        tasks.append({"split": "test", "case": case, "sr": sr, "gt": None})
    return tasks


def build_summary(rows, cm, protocol):
    validation = [row["metrics"] for row in rows if row["split"] == "validation"]
    test = [row["metrics"] for row in rows if row["split"] == "test"]
    expected = len(protocol["test_cases"])
    if len(validation) != 3 or len(test) != expected:
        raise ValueError(f"Summary requires validation3 + test{expected}")
    val = cm.mean_metrics(validation)
    val.update(fr=cm.fr_contrib(val), nr_val=cm.nr_contrib(val), score=cm.proxy_score(val), n=3)
    nr = cm.mean_metrics(test)
    nr.update(nr=cm.nr_contrib(nr), n=expected)
    full = protocol["test_cases"] == list(range(1, 101))
    result = {"label": protocol["label"], "status": "complete", "test": nr, "val": val,
              "scope": "test100+validation3" if full else "fixed20+validation3 pilot; NOT full test100",
              "test_cases": protocol["test_cases"], "pred": None,
              "pred_missing_reason": "No AutoDL-calibrated CUDA 512x512 event_p10 latency measured",
              "not_official_score": True, "protocol": protocol}
    result["S_proxy" if full else "S_subset"] = nr["nr"] + val["fr"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tool-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--metric-code-dir", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True, help="Complete offline weights directory named weights")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--test-pattern", default="case{case}.png")
    parser.add_argument("--val-pattern", default="case{case}.png")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New independent result directory; reusing it resumes safely")
    parser.add_argument("--npu-index", "--device-index", type=int, default=0, dest="npu_index")
    parser.add_argument("--backend", choices=["auto", "cuda", "npu"], default="auto")
    parser.add_argument("--test-cases", choices=["pilot", "all"], default="pilot")
    args = parser.parse_args()
    args.cases = PILOT_CASES if args.test_cases == "pilot" else list(range(1, 101))
    for name in ("tool_dir", "metric_code_dir", "weights_root", "test_dir", "val_dir", "gt_dir", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.weights_root.name != "weights":
        parser.error("--weights-root must be a directory named weights (server helper layout)")
    source_paths = {
        "csig_metrics": args.tool_dir / "csig_metrics.py",
        "run_metric_study": args.metric_code_dir / "run_metric_study.py",
        "metric_memory": args.metric_code_dir / "metric_memory.py",
        "adapter": Path(__file__).resolve(),
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    missing = [name for name in REQUIRED_WEIGHTS
               if not (args.weights_root / name).is_file()
               or (args.weights_root / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Offline metric weights missing/empty: " + ", ".join(missing))

    # No downloader or software installation is invoked here. Every known
    # PyIQA/torchvision checkpoint used by these six models is checked above.
    os.environ["TORCH_HOME"] = str(args.weights_root / "torch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CSIG_MANIQA_SAMPLES"] = "100"
    import torch
    backend = args.backend if args.backend != "auto" else ("cuda" if torch.cuda.is_available() else "npu")
    if backend == "npu":
        import torch_npu
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    accelerator = getattr(torch, backend)
    import numpy as np
    import pyiqa
    from PIL import Image
    if pyiqa.__version__ != "0.1.10":
        raise RuntimeError(f"Expected PyIQA 0.1.10, got {pyiqa.__version__}")
    Image.MAX_IMAGE_PIXELS = None
    accelerator.set_device(args.npu_index)
    device = torch.device(f"{backend}:{args.npu_index}")
    sys.path.insert(0, str(args.metric_code_dir))
    cm = load_module("csig_adapter_metrics", source_paths["csig_metrics"])
    study = load_module("run_metric_study", source_paths["run_metric_study"])
    # The unchanged helper uses DATA/weights. Redirect its runtime root only;
    # no source file or existing cache entry is edited.
    study.DATA = args.weights_root.parent
    os.environ["TORCH_HOME"] = str(args.weights_root / "torch")
    torch.hub.set_dir(str(args.weights_root / "torch" / "hub"))

    tasks = make_tasks(args, cm, Image)
    protocol = {
        "schema": "csig_score_search_tool_order_v1", "label": args.label, "test_cases": args.cases,
        "seed": 1234, "seed_scope": "once per dataset/model; validation then test",
        "metric_order_validation": list(FR_ORDER + NR_ORDER),
        "metric_order_test": list(NR_ORDER),
        "image_format": "unaltered native PNG; RGB FP32 [0,1]; no resize",
        "precision": f"NR FP32; LPIPS/DISTS {backend.upper()} FP16 autocast",
        "maniqa": "KonIQ; 100 crops; microbatch4; weighted numerator/denominator",
        "musiq": "all native tokens; CNN batches256; attention query rows256",
        "aggregation": "component means first, then NIQE clamp/formula",
        "backend": backend, "cross_device_equivalence": "not established",
        "device_index": args.npu_index,
        "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown") if backend == "npu" else None,
        "pyiqa": pyiqa.__version__, "numpy": np.__version__,
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "weights_sha256": {name: sha256_file(args.weights_root / name) for name in REQUIRED_WEIGHTS},
        "tasks": tasks,
    }
    render_manifest = args.val_dir.parent.parent / "render_manifest.json"
    if render_manifest.is_file():
        render_meta = json.loads(render_manifest.read_text())
        rendered_cases = render_meta.get("protocol", {}).get("test_cases")
        if rendered_cases is not None and rendered_cases != args.cases:
            raise RuntimeError("Render/scoring test-case sequence differs")
        protocol["render_manifest_sha256"] = sha256_file(render_manifest)
        protocol["candidate"] = {key: render_meta.get("protocol", {}).get(key)
                                 for key in ("checkpoint_sha256", "model_t", "coeff_t", "backend")}
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.out_dir / "resume_state.pt"
        jsonl_path = args.out_dir / "measurements.jsonl"
        rows = []
        saved_rng = None
        if state_path.exists():
            # This is the adapter's own local checkpoint, not a model file.
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state.get("protocol_id") != protocol_id:
                raise RuntimeError("Resume protocol/input/weight/helper hashes changed; use a new output directory")
            rows, saved_rng = state["rows"], state["rng"]
            if len(rows) != state["next_index"] or len(rows) > len(tasks):
                raise RuntimeError("Invalid resume cursor")
            for row, task in zip(rows, tasks):
                if (row["split"], row["case"], row["sr_sha256"]) != (
                    task["split"], task["case"], task["sr"]["sha256"]
                ):
                    raise RuntimeError("Resume record/task order mismatch")
        elif jsonl_path.exists() or (args.out_dir / "result.json").exists():
            raise RuntimeError("Existing result files lack authoritative RNG state; use a new output directory")
        replace_json(args.out_dir / "protocol.json", protocol)
        publish_jsonl(jsonl_path, rows)
        if len(rows) < len(tasks):
            # All metrics are initialized BEFORE the split seed, as in the
            # supplied tool. Initializer randomness cannot shift crop draws.
            metrics = {name: study.create_metric(name) for name in cm.SCORE_KEYS}
            if saved_rng is not None:
                restore_rng(torch, np, args.npu_index, saved_rng, backend)
            for index in range(len(rows), len(tasks)):
                task = tasks[index]
                if index == 0 or tasks[index - 1]["split"] != task["split"]:
                    seed_once(torch, np, backend)
                started = time.perf_counter()
                sr = cm.load_img(task["sr"]["path"], device)
                gt = cm.load_img(task["gt"]["path"], device) if task["gt"] else None
                values = {}
                with torch.inference_mode():
                    if gt is not None:
                        if sr.shape != gt.shape:
                            raise ValueError(f"Validation shape changed: {task['case']}")
                        with torch.autocast(backend, dtype=torch.float16):
                            for name in FR_ORDER:
                                values[name] = float(metrics[name](sr, gt))
                                accelerator.empty_cache()
                    for name in NR_ORDER:
                        values[name] = (
                            study.maniqa_score(metrics[name], sr, samples=100, microbatch=4)
                            if name == "maniqa" else float(metrics[name](sr))
                        )
                        accelerator.empty_cache()
                if not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"Nonfinite metric: {task['split']} case{task['case']}: {values}")
                accelerator.synchronize()
                row = {
                    "label": args.label, "split": task["split"], "case": task["case"],
                    "metrics": values, "nr_per_image": cm.nr_contrib(values),
                    "sr_sha256": task["sr"]["sha256"],
                    "gt_sha256": task["gt"]["sha256"] if task["gt"] else None,
                    "size_hw": list(sr.shape[-2:]), "seconds": time.perf_counter() - started,
                    "seed_scope": "split_seed1234_continuous_rng",
                }
                rows.append(row)
                state = {"protocol_id": protocol_id, "next_index": index + 1,
                         "rows": rows, "rng": capture_rng(torch, np, args.npu_index, backend)}
                commit_state(torch, state_path, state)
                publish_jsonl(jsonl_path, rows)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                del sr, gt
                accelerator.empty_cache()
        result = build_summary(rows, cm, protocol)
        replace_json(args.out_dir / "result.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "protocol"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
