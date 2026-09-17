"""Measure runner latency with both GPU-resident and CPU-origin inputs.

Use the same protocol for your model and OSEDiff. Do not mix these two numbers.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True)
    ap.add_argument("--runner",default=os.path.join(os.path.dirname(__file__),"runner_submit.py"))
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--warm",type=int,default=20)
    args=ap.parse_args()
    sys.path.insert(0,os.path.dirname(args.runner))
    from runner_submit import Runner
    r=Runner(args.model_dir)

    def measure(x):
        with torch.no_grad():
            for _ in range(args.warm): r.infer(x)
            torch.cuda.synchronize(); t0=time.perf_counter()
            for _ in range(args.n): r.infer(x)
            torch.cuda.synchronize()
        return (time.perf_counter()-t0)*1000/args.n

    gpu=torch.zeros(1,3,512,512,device="cuda",dtype=torch.float16)
    cpu=torch.zeros(1,3,512,512,device="cpu",dtype=torch.float16)
    print(f"GPU-resident input: {measure(gpu):.3f} ms")
    print(f"CPU-origin input:   {measure(cpu):.3f} ms")
    print("Compare OSEDiff only under the identical input/device protocol.")

if __name__=="__main__": main()
