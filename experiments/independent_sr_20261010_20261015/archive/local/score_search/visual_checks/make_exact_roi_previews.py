"""Create separately named QA crops, preserving every source image unchanged.

Only integer-pixel cropping and side-by-side pasting are used: no resizing,
filtering, sharpening, color correction, or edits within the source pixels.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from fetch_visual_originals import LABELS


ROOT = Path(__file__).resolve().parent
ROIS = {
    7: {"text": (.39, .45, .62, .56), "glass": (.50, .12, .80, .35)},
    16: {"wall": (.57, .00, .76, .20), "grid": (.56, .25, .72, .50)},
    23: {"text": (.31, .45, .71, .62), "glass": (.52, .04, .72, .27)},
    28: {"leaves": (.46, .40, .66, .62)},
    41: {"flower": (.30, .36, .64, .63), "leaves": (.00, .04, .25, .25)},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--baseline-label", default="baseline_s8000_mt200_ct200_all")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--cases", nargs="+", type=int, choices=list(ROIS), default=list(ROIS))
    args = parser.parse_args()
    prefix = args.prefix + "_" if args.prefix else ""
    output = ROOT / "derived_previews"
    output.mkdir(exist_ok=True)
    records = []
    counts = {}
    font = ImageFont.load_default(size=22)
    for case in args.cases:
        rois = ROIS[case]
        sources = [("input", ROOT / f"input/case{case}.jpg"),
                   (args.baseline_label, ROOT / f"{args.baseline_label}/case{case}.png")]
        sources.extend((label, ROOT / f"{label}/case{case}.png") for label in args.labels
                       if label != args.baseline_label)
        sources = [(label, path) for label, path in sources if path.is_file()]
        counts[str(case)] = len(sources)
        for name, roi in rois.items():
            crops, evidence = [], []
            expected_size = None
            for label, path in sources:
                with Image.open(path) as image:
                    if expected_size is None:
                        expected_size = image.size
                    if image.size != expected_size:
                        raise RuntimeError(f"Size mismatch for {path}")
                    box = tuple(round(v * image.size[i % 2]) for i, v in enumerate(roi))
                    crops.append((label, image.convert("RGB").crop(box)))
                evidence.append(dict(label=label, source=str(path), pixel_box=list(box),
                                     source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            width, height = crops[0][1].size
            header, margin = 58, 8
            columns, rows = 2, (len(crops) + 1) // 2
            canvas = Image.new("RGB", (columns * (width + margin), rows * (height + header + margin)), "white")
            draw = ImageDraw.Draw(canvas)
            for index, (label, crop) in enumerate(crops):
                x, y = (index % columns) * (width + margin), (index // columns) * (height + header + margin)
                draw.text((x + 4, y + 2), label, font=font, fill="black")
                draw.text((x + 4, y + 28), f"case{case} / {name} / original pixels 1:1", font=font, fill="black")
                canvas.paste(crop, (x, y + header))
            path = output / f"{prefix}case{case}_{name}_exact.png"
            canvas.save(path, format="PNG", compress_level=6)
            records.append(dict(preview=str(path), normalized_box=list(roi), sources=evidence,
                                operation="integer crop + paste; no resize/filter/color changes; source files untouched"))
    (output / f"{prefix}preview_manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"previews": len(records), "sources_available_per_case": counts}))


if __name__ == "__main__":
    main()
