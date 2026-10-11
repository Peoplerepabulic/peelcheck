# 🍌 PeelCheck — Banana Ripeness Classifier

Banana Hacks 2026 · Image AI · Solo project by Qiwei Li

**Upload a photo of a banana → get its ripeness stage and a use suggestion.**
Eat now, bake it, freeze it, or compost it — no more bananas wasted because
nobody could tell.

- **Live demo:** https://peelcheck.streamlit.app/
- **Demo video (≤3 min):** [TBD — YouTube]
- **Devpost:** [TBD]

## Why
Bananas are wasted because people misjudge ripeness: bought green, forgotten,
then straight to the bin once spots appear. PeelCheck turns one photo into an
actionable call, for dorm fridges, first-time cooks, and small grocers sorting
shelves.

## Image AI (core, not a wrapper)
- **Trained classifier** (multinomial logistic regression on 47 HSV color
  features) trained during the hackathon on 11,793 banana images in 4 classes
  (`unripe / ripe / overripe / rotten`); it trains in numpy only
  (`train/train_numpy.py`), so anyone can reproduce it offline in ~2 minutes.
- Full pipeline in-repo: dataset prep → feature extraction → training with
  balanced class weights → evaluation (per-class precision/recall +
  confusion matrix) → 2.5 KB model file → live demo.
- Augmentation (color jitter, synthetic brown-spot generation in
  `train/augment.py`) and a MobileNetV3-Small transfer-learning fine-tune
  (`train/train_peelcheck.py` + Colab notebook, ONNX export) are in-repo as
  the upgrade path — the demo serves the numpy classifier today and prefers
  the ONNX fine-tune if present.
- The classifier *is* the product: remove it and nothing remains.

## Dataset & resources
- Kaggle **Banana Ripeness Classification Dataset** (S.M. Shahriar, Apache 2.0):
  13,478 images — train 11,793 / valid 1,123 / test 562, pre-resized 416×416.
  https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset
- Training: shipped model trained in numpy on CPU (`train/train_numpy.py`,
  ~2 minutes). MobileNetV3-Small transfer-learning fine-tune available as
  upgrade path on Colab GPU (`train/colab_train.ipynb`).
- Demo: Streamlit, live at https://peelcheck.streamlit.app/ — 2.5 KB model,
  instant inference. (A Hugging Face Spaces bundle also lives in `spaces/`.)
- Label mapping and class decisions: `train/label_map.md`.

## Run the demo locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Without a trained `models/peelcheck.onnx`, the app falls back to a clearly
labeled color-heuristic preview so the UI is never dead. After training,
drop the ONNX file into `models/` and the same page serves real predictions.

## Train it yourself
```bash
pip install kaggle torch torchvision scikit-learn matplotlib
python train/download_data.py   # needs ~/.kaggle/kaggle.json
python train/train_peelcheck.py --data data/raw --epochs 12 --out models/
```
Or open `train/colab_train.ipynb` in Colab (GPU runtime) and run top to bottom.

## Results (trained 2026-10-10, held-out test split)
Trained HSV-feature softmax classifier (`train/train_numpy.py`, numpy-only so it
runs anywhere) on the full 13,478-image dataset:

- **Test accuracy: 83.3%** (valid 81.4%) on 562 held-out test images
- Unripe: F1 0.91 (recall 1.00) · Ripe: F1 0.87 · Overripe: F1 0.82 · Rotten: F1 0.75
- Known weakness: rotten recall 0.67 — dark rotten fruit gets confused with
  overripe/unripe on color alone. The MobileNetV3 fine-tune
  (`train/train_peelcheck.py` / Colab) is the upgrade path for texture.
- Full per-class table in `models/metrics.json`, matrix in
  `models/confusion_matrix.csv`. Demo serves this model today
  (`models/peelcheck_npz.npz`, 2.5 KB) and prefers the ONNX fine-tune when present.

## Hackathon compliance
- All code written during Banana Hacks 2026 (Oct 9–11, 2026). Public dataset
  cited above; no prior project code reused; no cross-submission.
- Project created solely for Banana Hacks 2026.

