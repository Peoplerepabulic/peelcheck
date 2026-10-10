"""PeelCheck numpy trainer — no torch required (offline fallback that ships).

Trains multinomial logistic regression (softmax) on HSV color features
with balanced class weights. This exists because package-index downloads
are unreliable in the build VM; the MobileNetV3 fine-tune in
train_peelcheck.py remains the GPU path. Both export metrics in the same
shape and the demo prefers ONNX > NPZ > heuristic.

Run: python3 train/train_numpy.py --data data/raw --out models/ --epochs 60
Outputs: models/peelcheck_npz.npz, models/metrics.json, models/confusion_matrix.csv
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features  # noqa: E402


def load_split(data_dir, split, classes):
    xs, ys, paths = [], [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, split, cls)
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            try:
                xs.append(extract_features(Image.open(p)))
                ys.append(idx)
                paths.append(p)
            except Exception:
                continue
    return np.stack(xs), np.array(ys), paths


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(X, y, W, b, classes):
    probs = softmax(X @ W + b)
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    per_class = {}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[cls] = {"precision": float(prec), "recall": float(rec),
                          "f1-score": float(f1), "support": int(cm[i, :].sum())}
    return acc, per_class, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args()

    classes = sorted(os.listdir(os.path.join(args.data, "train")))
    print("classes:", classes, flush=True)
    splits = {}
    for split in ["train", "valid", "test"]:
        X, y, _ = load_split(args.data, split, classes)
        splits[split] = (X, y)
        print(f"{split}: X={X.shape}", flush=True)

    Xtr, ytr = splits["train"]
    mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    for split in splits:
        X, y = splits[split]
        splits[split] = ((X - mu) / sigma, y)
    Xtr, ytr = splits["train"]

    n, d = Xtr.shape
    k = len(classes)
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, size=(d, k)).astype(np.float32)
    b = np.zeros(k, dtype=np.float32)
    counts = np.bincount(ytr, minlength=k).astype(np.float32)
    sample_w = (len(ytr) / (k * counts))[ytr]

    for epoch in range(args.epochs):
        probs = softmax(Xtr @ W + b)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), ytr] = 1.0
        grad = (probs - onehot) * sample_w[:, None] / sample_w.sum() * len(ytr)
        # simplified weighted gradient (mean over batch)
        gW = Xtr.T @ grad / n + 1e-4 * W
        gb = grad.mean(axis=0)
        W -= args.lr * gW
        b -= args.lr * gb
        if (epoch + 1) % 10 == 0 or epoch == 0:
            vacc, _, _ = evaluate(*splits["valid"], W, b, classes)
            print(f"epoch {epoch + 1}/{args.epochs} valid_acc={vacc:.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "peelcheck_npz.npz"),
             W=W, b=b, mu=mu, sigma=sigma, classes=np.array(classes))
    acc, per_class, cm = evaluate(*splits["test"], W, b, classes)
    vacc, _, _ = evaluate(*splits["valid"], W, b, classes)
    metrics = {"engine": "numpy-softmax-hsv", "classes": classes,
               "test_accuracy": acc, "valid_accuracy": vacc,
               "per_class": per_class}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savetxt(os.path.join(args.out, "confusion_matrix.csv"), cm,
               fmt="%d", delimiter=",")
    print(json.dumps(metrics, indent=2), flush=True)
    print("confusion matrix (rows=true):\n", cm, flush=True)


if __name__ == "__main__":
    main()
