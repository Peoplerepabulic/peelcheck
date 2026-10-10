# 🍌 PeelCheck — Banana Ripeness Classifier

Banana Hacks 2026 · Image AI · Solo project by Qiwei Li

**Upload a photo of a banana → get its ripeness stage and a use suggestion.**
Eat now, bake it, freeze it, or compost it — no more bananas wasted because
nobody could tell.

- **Live demo:** [TBD — Hugging Face Spaces URL]
- **Demo video (≤3 min):** [TBD — YouTube]
- **Devpost:** [TBD]

## Why
Bananas are wasted because people misjudge ripeness: bought green, forgotten,
then straight to the bin once spots appear. PeelCheck turns one photo into an
actionable call, for dorm fridges, first-time cooks, and small grocers sorting
shelves.

## Image AI (core, not a wrapper)
- **Fine-tuned classifier** (transfer learning, MobileNetV3-Small) trained
  during the hackathon on 13,478 banana images in 4 classes
  (`unripe / ripe / overripe / rotten`).
- Full pipeline in-repo: dataset prep → augmentation (color jitter +
  synthetic brown-spot generation for the minority class) → training with
  balanced class weights → evaluation (per-class precision/recall +
  confusion matrix) → ONNX export → live demo.
- The classifier *is* the product: remove it and nothing remains.

## Dataset & resources
- Kaggle **Banana Ripeness Classification Dataset** (S.M. Shahriar, Apache 2.0):
  13,478 images — train 11,793 / valid 1,123 / test 562, pre-resized 416×416.
  https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset
- Training: PyTorch + torchvision on Colab GPU (`train/colab_train.ipynb`).
- Demo: Streamlit + ONNX Runtime, deployable to Hugging Face Spaces.
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

## Results
[TBD after training: test accuracy, per-class table, confusion matrix image]

## Hackathon compliance
- All code written during Banana Hacks 2026 (Oct 9–11, 2026). Public dataset
  cited above; no prior project code reused; no cross-submission.
- Project created solely for Banana Hacks 2026.
