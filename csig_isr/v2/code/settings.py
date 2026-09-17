"""Shared paths/configuration for the CSIG track-2 v2 code.

Override server paths without editing source:
  export CSIG_ROOT=/path/to/csig_isr
  export TINYSR_ROOT=/path/to/TinySR
"""
import os

ROOT = os.environ.get("CSIG_ROOT", "/cache/zihaohe/csig_isr")
TS = os.environ.get("TINYSR_ROOT", os.path.join(ROOT, "TinySR"))
VAL = os.path.join(ROOT, "data", "赛题二", "验证集")
TEST = os.path.join(ROOT, "data", "赛题二", "测试集")
DEFAULT_MANIQA = os.environ.get("CSIG_MANIQA", "maniqa-pipal")
DEFAULT_TILE = 512
DEFAULT_OVERLAP = 256
