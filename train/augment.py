"""Brown-spot synthesis — minority-class augmentation for PeelCheck.

Overlays soft brown speckles on a ripe-banana image to approximate the
overripe class. Used only on the training split (never on valid/test, so
evaluation stays honest). Standalone helper; wire into a custom Dataset
or run offline to expand data/raw/train/overripe/.

  python train/augment.py --src data/raw/train/ripe --dst data/raw/train/overripe_synth --n 300
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def add_spots(img: Image.Image, n_spots: int = 60, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    for _ in range(n_spots):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(2, 9)
        # brown with soft alpha edge fake: layered ellipses
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(101, 67, 33, 110))
        draw.ellipse([x - r / 2, y - r / 2, x + r / 2, y + r / 2],
                     fill=(62, 39, 20, 140))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    srcs = [p for p in Path(args.src).glob("*.jpg")]
    srcs += [p for p in Path(args.src).glob("*.png")]
    if not srcs:
        raise SystemExit(f"No images in {args.src}")
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img = Image.open(srcs[i % len(srcs)])
        spotted = add_spots(img, n_spots=40 + (i % 60), seed=i)
        spotted.save(Path(args.dst) / f"synth_{i:04d}.jpg", quality=90)
    print(f"Wrote {args.n} synthetic overripe images to {args.dst}")


if __name__ == "__main__":
    main()
