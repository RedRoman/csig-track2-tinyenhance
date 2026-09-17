"""Fail-fast consistency check between rendered images and exported JIT candidate."""
import argparse
import json
import os
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    args=ap.parse_args()
    a=load(os.path.join(args.render_dir,"candidate_manifest.json"))
    b=load(os.path.join(args.model_dir,"candidate_manifest.json"))
    keys=["checkpoint_sha256","alpha_mode","effective_alpha_mode","alpha"]
    bad=[]
    for k in keys:
        if a.get(k)!=b.get(k): bad.append((k,a.get(k),b.get(k)))
    imgs=[x for x in os.listdir(args.render_dir) if x.lower().endswith((".jpg",".jpeg",".png"))]
    if len(imgs)!=100:
        bad.append(("image_count",len(imgs),100))
    if bad:
        print("CANDIDATE MISMATCH:")
        for x in bad: print(" ",x)
        sys.exit(2)
    print("OK: render and JIT use the same checkpoint/alpha candidate; image_count=100")

if __name__=="__main__": main()
