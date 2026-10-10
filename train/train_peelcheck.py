"""PeelCheck training — MobileNetV3 transfer learning (PyTorch).

Expects data laid out as ImageFolder:
  data/raw/train/{unripe,ripe,overripe,rotten}/*.jpg
  data/raw/valid/{...}/*.jpg
  data/raw/test/{...}/*.jpg

Run (Colab GPU recommended):
  python train/train_peelcheck.py --data data/raw --epochs 12 --out models/

Outputs:
  models/peelcheck.pt          (state_dict + class list)
  models/peelcheck.onnx        (for the Streamlit demo, if export succeeds)
  models/metrics.json          (accuracy, per-class P/R/F1)
  models/confusion_matrix.png

Design notes:
- Classes are imbalanced (rotten heaviest, unripe lightest) -> balanced
  class weights + color augmentation. Brown-spot synthesis for the
  overripe minority is in augment.py and applied only to train.
- Input 224x224 (dataset ships 416x416; center-crop + resize).
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_loaders(data_dir: str, batch: int = 32):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,
                               saturation=0.25, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(data_dir, "valid"), eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), eval_tf)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch, shuffle=True),
        "valid": DataLoader(valid_ds, batch_size=batch),
        "test": DataLoader(test_ds, batch_size=batch),
    }
    return loaders, train_ds.classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the classifier head (fast CPU path)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    loaders, classes = build_loaders(args.data)
    print("classes:", classes)
    os.makedirs(args.out, exist_ok=True)

    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    if args.freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        print("backbone frozen — training classifier head only")
    model = model.to(device)

    # balanced class weights for the imbalanced 4-class set
    counts = [0] * len(classes)
    for _, y in loaders["train"].dataset.samples:
        counts[y] += 1
    total = sum(counts)
    weights = torch.tensor([total / (len(classes) * c) for c in counts],
                           dtype=torch.float).to(device)
    print("class counts:", counts)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        # validation accuracy
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in loaders["valid"]:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                n += len(y)
        acc = correct / max(n, 1)
        print(f"epoch {epoch + 1}/{args.epochs}  valid_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "classes": classes},
                       os.path.join(args.out, "peelcheck.pt"))

    # final test evaluation + export
    ckpt = torch.load(os.path.join(args.out, "peelcheck.pt"),
                      map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in loaders["test"]:
                y_pred += model(x.to(device)).argmax(1).cpu().tolist()
                y_true += y.tolist()
        report = classification_report(y_true, y_pred, target_names=classes,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="YlOrBr")
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="left")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("PeelCheck — test confusion matrix")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=150)
        metrics = {"test_accuracy": report["accuracy"],
                   "per_class": {c: report[c] for c in classes},
                   "classes": classes}
    except ImportError:
        metrics = {"classes": classes,
                   "note": "pip install scikit-learn matplotlib for full report"}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2)[:800])

    dummy = torch.randn(1, 3, 224, 224).to(device)
    try:
        torch.onnx.export(model, dummy,
                          os.path.join(args.out, "peelcheck.onnx"),
                          input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"},
                                        "logits": {0: "batch"}})
        print("exported models/peelcheck.onnx")
    except Exception as exc:  # ONNX is a nice-to-have; .pt still ships
        print(f"ONNX export skipped: {exc}")


if __name__ == "__main__":
    main()
