#!/usr/bin/env bash
# Shared defaults for the authors' current server layout. Override any variable
# from the shell if your paths differ.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CSIG_ROOT="${CSIG_ROOT:-/cache/zihaohe/csig_isr}"
export TINYSR_ROOT="${TINYSR_ROOT:-$CSIG_ROOT/TinySR}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/cache/zihaohe/hf}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-$CSIG_ROOT/venv/bin/python}"
cd "$HERE"
