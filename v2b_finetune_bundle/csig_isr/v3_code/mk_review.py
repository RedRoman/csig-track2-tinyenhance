"""Visual-review contact sheets for a rendered submission directory.

Picks heavy/medium/light test images by LQ Laplacian variance, then selects a
high-detail crop from each. Candidate identity is explicit via CLI instead of a
hard-coded old soup label.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, cv2
from PIL import Image, ImageDraw
from settings import ROOT, TEST

Image.MAX_IMAGE_PIXELS = None


def lapvar(g):
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def best_crop(gray, crop, stride=256):
    h, w = gray.shape
    cand = []
    if h < crop or w < crop:
        return []
    for y in range(0, h - crop + 1, stride):
        for x in range(0, w - crop + 1, stride):
            cand.append((lapvar(gray[y:y+crop, x:x+crop]), y, x))
    cand.sort(reverse=True)
    return cand[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render_dir', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--lq_dir', default=TEST)
    ap.add_argument('--dst', default=os.path.join(ROOT, 'review_v3'))
    ap.add_argument('--crop', type=int, default=448)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    done = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.render_dir, '*.jpg')))
    print(f'rendered {len(done)} images')
    sev = []
    for nm in done:
        lq = np.asarray(Image.open(os.path.join(a.lq_dir, nm)).convert('RGB'))
        g = cv2.cvtColor(lq, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sev.append((lapvar(g), nm))
    sev.sort()
    if len(sev) < 8:
        raise RuntimeError('need at least 8 rendered images for tiered review')
    pick = [n for _, n in sev[:4]] + [n for _, n in sev[len(sev)//2:len(sev)//2+2]] + [n for _, n in sev[-2:]]
    tags = ['heavy']*4 + ['medium']*2 + ['light']*2
    for nm, tag in zip(pick, tags):
        lq = np.asarray(Image.open(os.path.join(a.lq_dir, nm)).convert('RGB'))
        out = np.asarray(Image.open(os.path.join(a.render_dir, nm)).convert('RGB'))
        if lq.shape != out.shape:
            print('skip shape mismatch', nm, lq.shape, out.shape); continue
        g = cv2.cvtColor(lq, cv2.COLOR_RGB2GRAY).astype(np.float32)
        for v, y, x in best_crop(g, a.crop):
            aa, bb = lq[y:y+a.crop, x:x+a.crop], out[y:y+a.crop, x:x+a.crop]
            canvas = np.full((a.crop+34, a.crop*2+12, 3), 255, np.uint8)
            canvas[34:, :a.crop] = aa; canvas[34:, a.crop+12:] = bb
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            d.text((6, 9), f'{nm} [{tag}] lapvar={v:.0f} @({x},{y}) LQ', fill=(0,0,0))
            d.text((a.crop+18, 9), a.label, fill=(180,0,0))
            path = os.path.join(a.dst, f'{tag}_{nm.rsplit(".",1)[0]}.png')
            im.save(path); print(path)


if __name__ == '__main__':
    main()
