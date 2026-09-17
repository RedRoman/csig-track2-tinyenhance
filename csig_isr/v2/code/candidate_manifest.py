"""Candidate provenance manifest helpers.

The goal is to make 'best checkpoint', 100-image output and JIT artifact
cryptographically traceable to the same candidate.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(path=None):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        return None


def write_manifest(path, **fields):
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload
