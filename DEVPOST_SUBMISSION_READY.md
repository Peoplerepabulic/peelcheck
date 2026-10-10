# PeelCheck — Devpost submission (ready to paste, fill 3 brackets)

**Project name:** PeelCheck
**Team:** Qiwei Li (solo) — B.S. Computer Science + Mathematics, Washington University in St. Louis (May 2027)
**Tagline:** One photo tells you if that banana is snack, bread, or bin.

## Inspiration
Bananas are bought green, forgotten, then tossed when spots appear — even though a spotted banana is perfect for baking. I wanted a tiny Image AI project a solo coder could train end-to-end in one weekend that proves a real fine-tuning pipeline, not an API wrapper.

## What it does
Upload a banana photo and PeelCheck returns one of four stages — unripe, ripe, overripe, or past best — with confidence and a concrete next step: wait, eat now, bake/freeze today, or compost. The classifier is the product; remove it and nothing remains.

## How I built it
- **Data:** Kaggle Banana Ripeness Classification Dataset (Apache 2.0): 13,478 images, pre-split train 11,793 / valid 1,123 / test 562 across the four classes, 416×416.
- **Model:** MobileNetV3-Small transfer learning in PyTorch, trained during Banana Hacks 2026 with balanced class weights for the imbalanced classes, color jitter, and synthetic brown-spot augmentation for the overripe minority. Exported to ONNX.
- **Evaluation:** [METRICS: test accuracy + per-class precision/recall + confusion_matrix.png from models/metrics.json]
- **App:** Streamlit demo (ONNX Runtime inference) deployed at [HF SPACES URL], so judges can click a live link and try held-out test images from `assets/demo_test/`.
- **Resources:** PyTorch/torchvision, Streamlit, ONNX Runtime, Hugging Face Spaces, the dataset above. All code written Oct 9–11, 2026; no cross-submission.

## Challenges
The ripe vs. overripe boundary is genuinely ambiguous under different lighting, and the rotten class dominates the data — hence balanced weights, color augmentation, and honest per-class metrics instead of a single headline accuracy.

## What's next
More fruits, a grocer shelf-scan mode, and a shareable "use-first" list for dorm fridges.

---
### Paste checklist (delete before submitting)
- [ ] Replace [METRICS] after training finishes; link confusion matrix in GitHub README
- [ ] Replace [HF SPACES URL]; open it in a private window to confirm no login needed
- [ ] YouTube video ≤3:00, **Public**: [YOUTUBE URL] — plan in VIDEO_PLAN.md, target 2:40
- [ ] GitHub: https://github.com/Peoplerepabulic/peelcheck (public, all commits Oct 9–11)
- [ ] Hard deadline used: Sun Oct 11, 2026 11:59 PM EDT = **10:59 PM CDT** (earlier of the two published deadlines)
