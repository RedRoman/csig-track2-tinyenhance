"""Measure one submission Runner in a fresh CUDA process, without rendering/IQA.

Use the same GPU and this protocol for G4 and the candidate, in separate
processes. --device is a logical CUDA index; a Runner hardcoding cuda:0 should
be selected with CUDA_VISIBLE_DEVICES=<physical index> and --device 0.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time


ANCHOR_G4_MS = 0.431
HASH_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".pth", ".pt", ".ckpt", ".safetensors", ".bin"}


def file_record(path: Path, label: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": label, "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def artifact_manifest(model_dir: Path, extra_files: list[Path]) -> dict:
    files, symlink_directories = [], []
    for base, dirs, names in os.walk(model_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {"__pycache__", ".git", "_work"})
        for name in dirs:
            path = Path(base) / name
            if path.is_symlink():
                symlink_directories.append({"path": str(path.relative_to(model_dir)),
                                            "resolved_path": str(path.resolve())})
        for name in sorted(names):
            path = Path(base) / name
            if path.suffix.lower() in HASH_SUFFIXES and path.is_file():
                files.append(file_record(path, str(path.relative_to(model_dir))))
    extra = [file_record(p, str(p)) for p in extra_files]
    return {"files": sorted(files, key=lambda x: x["path"]), "extra_files": extra,
            "untraversed_symlink_directories": symlink_directories,
            "coverage": "Model directory Python/config/weight files; symlinked directories are not traversed. "
                        "Use --hash-file for weights/configs outside this directory."}


def load_runner(model_dir: Path, torch, device):
    # Match eval_submission.load_runner: cwd, append model_dir, module 'runner',
    # Runner(model_dir), then synchronization. Do not import the scoring tool.
    old = os.getcwd()
    os.chdir(model_dir)
    sys.path.append(str(model_dir))
    try:
        spec = importlib.util.spec_from_file_location("runner", model_dir / "runner.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import model_dir/runner.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["runner"] = module
        spec.loader.exec_module(module)
        started = time.perf_counter()
        runner = module.Runner(str(model_dir))
        torch.cuda.synchronize(device)
        return runner, time.perf_counter() - started
    finally:
        os.chdir(old)


def output_record(y, torch) -> dict:
    if not isinstance(y, torch.Tensor):
        raise TypeError("Runner.infer must return a torch.Tensor")
    if tuple(y.shape) != (1, 3, 512, 512):
        raise ValueError(f"Runner.infer returned shape {tuple(y.shape)}, expected (1, 3, 512, 512)")
    # Copy/range checks happen after timing and are not part of event/wall samples.
    cpu = y.detach().to(device="cpu", dtype=torch.float32)
    finite = bool(torch.isfinite(cpu).all().item())
    low, high = float(cpu.min().item()), float(cpu.max().item())
    return {"device": str(y.device), "dtype": str(y.dtype), "shape": list(y.shape),
            "all_finite": finite, "min": low if finite else None,
            "max": high if finite else None,
            "within_minus1_plus1": finite and low >= -1.0 and high <= 1.0,
            "clamping_applied_by_benchmark": False}


def latency_512(runner, torch, np, device, n: int, warm: int, seed: int) -> tuple:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(1, 3, 512, 512, device=device, dtype=torch.float16,
                   generator=generator) * 2 - 1
    declared_device = getattr(runner, "device", None)
    if declared_device is not None:
        declared_device = torch.device(declared_device)
        if declared_device.type != "cuda" or declared_device.index not in {None, device.index}:
            raise RuntimeError(f"Runner declares {declared_device}, benchmark selected {device}. "
                               "For a cuda:0 Runner use CUDA_VISIBLE_DEVICES and --device 0.")
    event, wall = [], []
    with torch.no_grad():
        for _ in range(warm):
            runner.infer(x)
        torch.cuda.synchronize(device)
        for _ in range(n):
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            began = time.perf_counter()
            start.record()
            y = runner.infer(x)
            finish.record()
            torch.cuda.synchronize(device)
            wall.append((time.perf_counter() - began) * 1e3)
            event.append(start.elapsed_time(finish))
    result = output_record(y, torch)
    if result["device"].startswith("cuda") and torch.device(result["device"]).index != device.index:
        raise RuntimeError(f"Runner output is on {result['device']}, selected {device}; timing is invalid")
    if not result["all_finite"]:
        raise RuntimeError("Runner.infer returned nonfinite values; no successful benchmark reported")
    stats = {"event_med": float(np.median(event)),
             "event_p10": float(np.percentile(event, 10)), "event_min": float(np.min(event)),
             "wall_med": float(np.median(wall)), "wall_min": float(np.min(wall)),
             "wall_p10": float(np.percentile(wall, 10)), "n": n, "warm": warm}
    return stats, {"event_ms": event, "wall_ms": wall}, result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def run(args) -> dict:
    model_dir = args.model_dir.resolve()
    if not (model_dir / "runner.py").is_file():
        raise FileNotFoundError(f"Missing {model_dir / 'runner.py'}")
    import torch
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("A working NVIDIA CUDA PyTorch environment is required; CPU/NPU timing is not accepted")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA logical device {args.device} is unavailable")
    import numpy as np
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.default_generator.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    manifest = artifact_manifest(model_dir, args.hash_file)
    properties = torch.cuda.get_device_properties(device)
    runner, initialization = load_runner(model_dir, torch, device)
    stats, samples, output = latency_512(runner, torch, np, device, args.n, args.warm, args.seed)
    result = {"status": "complete", "model_dir": str(model_dir), "init_s": initialization,
              "runner_path": str(getattr(runner, "_path", "unknown")), "seed": args.seed,
              "artifacts": manifest, "benchmark_script": file_record(Path(__file__), str(Path(__file__).resolve())),
              "cuda": {"logical_device": args.device, "name": properties.name,
                       "compute_capability": [properties.major, properties.minor],
                       "total_memory_bytes": properties.total_memory,
                       "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
                       "cudnn_version": torch.backends.cudnn.version()},
              "input": {"device": str(device), "shape": [1, 3, 512, 512],
                        "dtype": "torch.float16", "distribution": "uniform [-1,1)",
                        "allocation_excluded_from_timing": True},
              "output": output, "latency_512": stats, "raw_samples": samples,
              "timing_scope": {
                  "event": "CUDA events on the current selected-device stream surround the complete original Runner.infer(x) call.",
                  "wall": "perf_counter starts before start.record and ends after finish.record plus torch.cuda.synchronize(selected_device).",
                  "included": "Original Runner input/output copies, dtype changes, clamping, CPU work and CPU-return/GPU-return behavior remain untouched. CUDA-event coverage of host work and other streams depends on Runner synchronization; wall time includes selected-device completion.",
                  "excluded": "Runner construction, compilation during construction, input allocation, warmup, file hashing, output validation and output numeric-range checks.",
                  "isolation": "Exactly one Runner is loaded in this process. Run G4 and each candidate in separate processes on the same idle NVIDIA GPU."}}
    if args.ref_ms is not None:
        result.update(ref_ms=args.ref_ms, anchor_g4_ms=ANCHOR_G4_MS,
                      g_used_ms=stats["event_p10"] * ANCHOR_G4_MS / args.ref_ms)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref-ms", type=float)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warm", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--hash-file", type=Path, action="append", default=[],
                        help="Also hash an external checkpoint/config (repeatable); contents are not printed")
    args = parser.parse_args()
    if args.n <= 0 or args.warm < 0 or (args.ref_ms is not None and args.ref_ms <= 0):
        parser.error("--n must be positive, --warm nonnegative, and --ref-ms positive")
    try:
        result = run(args)
    except Exception as error:
        result = {"status": "failed", "model_dir": str(args.model_dir.resolve()),
                  "error_type": type(error).__name__, "error": str(error)}
        atomic_json(args.out.resolve(), result)
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 1
    atomic_json(args.out.resolve(), result)
    print(json.dumps({"status": "complete", "out": str(args.out.resolve()),
                      "cuda": result["cuda"], "runner_path": result["runner_path"],
                      "latency_512": result["latency_512"],
                      "g_used_ms": result.get("g_used_ms")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
