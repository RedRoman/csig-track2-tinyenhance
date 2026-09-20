#!/usr/bin/env bash
# AutoDL 端环境。系统盘 / 只剩约 1.2G，所有 cache 必须指向数据盘 autodl-tmp。
export CSIG_ROOT="${CSIG_ROOT:-/cache/zihaohe/csig_isr}"
export TINYSR_ROOT="${TINYSR_ROOT:-$CSIG_ROOT/TinySR}"
export VOSR_DIR="${VOSR_DIR:-/cache/zihaohe/VOSR}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/cache/zihaohe/hf}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.pip-cache}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PY="${PY:-$CSIG_ROOT/venv/bin/python}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
