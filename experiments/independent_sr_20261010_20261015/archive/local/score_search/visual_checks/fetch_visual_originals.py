"""Download available committed candidate PNGs; no remote writes or polling.

Run again manually at intervals of at least 60 seconds while rendering.
Existing local originals with matching manifest hashes are kept untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


LABELS = ["s6000_mt200_ct200_pilot", "s6000_mt200_ct175_pilot", "s8000_mt200_ct200_pilot",
          "s10000_mt200_ct200_pilot", "s8000_mt200_ct175_pilot",
          "s8000_mt200_ct225_pilot", "s10000_mt200_ct175_pilot"]


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=30207)
    parser.add_argument("--host", default="huawei-modelarts-31514")
    parser.add_argument("--root", default="/cache/independent_sr_score_search_20261013")
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--cases", nargs="+", type=int, default=[7, 16, 23, 28, 41])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "original_download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    ssh = ["ssh", "-p", str(args.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.host]
    pending = {}
    for label in args.labels:
        if not label or "/" in label or label in {".", ".."}:
            parser.error("Each label must be a directory basename")
        for case in args.cases:
            key = f"{label}/case{case}.png"
            local = args.out / key
            known = manifest["files"].get(key)
            if known and local.is_file() and digest(local) == known["sha256"]:
                continue
            remote = f"{args.root}/evaluations/{label}/render/test/case{case}.png"
            pending[remote] = (key, local)
    if not pending:
        print(json.dumps({"downloaded_now": 0, "all_requested_available": True}))
        return
    command = "for p in " + " ".join(shlex.quote(p) for p in pending) + "; do if [ -f \"$p\" ]; then sha256sum -- \"$p\"; fi; done"
    available = subprocess.check_output(ssh + [command], text=True)
    downloaded = 0
    for line in available.splitlines():
        sha, remote = line.split(None, 1)
        remote = remote.lstrip(" *")
        key, local = pending[remote]
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(".download.tmp")
        with temporary.open("wb") as stream:
            subprocess.run(ssh + ["cat -- " + shlex.quote(remote)], stdout=stream, check=True)
        if digest(temporary) != sha:
            raise RuntimeError(f"Remote/download hash mismatch: {key}")
        temporary.replace(local)
        manifest["files"][key] = {"host": args.host, "port": args.port, "remote": remote,
                                    "local": str(local.resolve()), "sha256": sha,
                                    "bytes": local.stat().st_size, "transformation": "none; original PNG bytes"}
        temp_manifest = manifest_path.with_suffix(".tmp")
        temp_manifest.write_text(json.dumps(manifest, indent=2))
        temp_manifest.replace(manifest_path)
        downloaded += 1
    print(json.dumps({"downloaded_now": downloaded,
                      "remaining_unavailable": len(pending) - downloaded,
                      "manifest": str(manifest_path.resolve())}))


if __name__ == "__main__":
    main()
