"""Download the PeelCheck training data.

Dataset: Kaggle "Banana Ripeness Classification Dataset" (Apache 2.0)
  https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset
  13,478 images · train 11,793 / valid 1,123 / test 562 · 4 classes
  (unripe, ripe, overripe, rotten), pre-resized to 416x416.

Kaggle requires a free account token. Two ways to run this:

  A) Colab (recommended): upload kaggle.json in the notebook, then run.
  B) Local: pip install kaggle, place kaggle.json at ~/.kaggle/kaggle.json,
     then `python train/download_data.py`

This script never embeds credentials. If the token is missing it prints
setup steps and exits cleanly so the rest of the repo still runs.
"""
import os
import subprocess
import sys

DATASET = "shahriar26s/banana-ripeness-classification-dataset"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("kaggle package not installed. Run: pip install kaggle")
        print("Then place your token at ~/.kaggle/kaggle.json (chmod 600).")
        print(f"Dataset page: https://www.kaggle.com/datasets/{DATASET}")
        return 1
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", OUT, "--unzip"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"Done. Data under {OUT} (expect train/ valid/ test/).")
    else:
        print("Download failed — check ~/.kaggle/kaggle.json and dataset terms.")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
