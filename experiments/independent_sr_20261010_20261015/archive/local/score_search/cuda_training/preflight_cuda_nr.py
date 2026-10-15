"""Explicit one-step real-gradient audit; no optimizer update or checkpoint save."""
import runpy
import sys
from pathlib import Path

if __name__=='__main__':
    sys.argv += ['--preflight-only']
    runpy.run_path(str(Path(__file__).with_name('train_cuda_nr.py')),run_name='__main__')
