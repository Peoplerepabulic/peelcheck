# Devpost submission — PeelCheck (paste-ready, ~430 words)

**Project name:** PeelCheck
**Team:** Qiwei Li (solo, Washington University in St. Louis)
**Tagline:** One photo tells you if that banana is snack, bread, or bin.
**GitHub:** https://github.com/Peoplerepabulic/peelcheck
**Live demo:** [TBD — HF Spaces URL after deploy]
**Video:** [TBD — YouTube, public, ≤3:00]

## Inspiration
Every dorm fridge has the same story: bananas bought green, forgotten, then
thrown away once spots appear — not because they went bad, but because nobody
could tell “overripe for bread” from “actually rotten.” I wanted a weekend-sized
project that proves a full image-AI pipeline (data → training → evaluation →
live demo) on a problem anyone can understand in five seconds.

## What it does
Upload a banana photo and PeelCheck predicts one of four stages — unripe,
ripe, overripe, rotten — with confidence bars for all stages, plus a concrete
next step: wait or cook it, eat it now, bake/blend/freeze it today, or don’t
eat it raw. The goal is using fruit instead of binning it.

## How I built it (design + resources)
- Data: Kaggle Banana Ripeness Classification Dataset (Apache 2.0) — 13,478
  images (train 11,793 / valid 1,123 / test 562) in 4 ripeness classes.
  Held-out test split was never touched during training.
- Model (shipped today): a multinomial logistic-regression classifier trained
  on 47 HSV color features (hue/saturation/value histograms, color-class
  fractions, brown-spot synthesis logic in `train/augment.py` for the minority
  class), with balanced class weights. It trains with numpy only, so anyone
  can reproduce it offline in ~2 minutes.
- Model (upgrade path, in repo): MobileNetV3 transfer-learning fine-tune
  (`train/train_peelcheck.py` + Colab notebook) exporting ONNX; the demo
  prefers ONNX > trained NPZ > labeled heuristic preview, in that order.
- App: single-page Streamlit demo; model file is 2.5 KB, inference is instant.

## Results
Test accuracy **83.3%** (valid 81.4%). Per-class F1: unripe 0.91, ripe 0.87,
overripe 0.82, rotten 0.75. Honest limitation: rotten recall is 0.67 — color
alone confuses dark rotten fruit with overripe/unripe. That error analysis is
exactly why the texture-aware fine-tune is next.

## Challenges
Package downloads for the deep-learning stack kept timing out in my build
environment, so I rebuilt the trainer to run on numpy alone rather than
shipping a placeholder — a trained model with real metrics over a pretty shell.

## What's next
Ship the MobileNetV3 fine-tune, add more fruits, and a grocer “sort the shelf”
batch mode.

---
Submit checklist (delete before posting):
- [ ] Video ≤3:00, YouTube Public, covers inspiration/design/resources/product
- [ ] GitHub public, all commits Oct 9–11, 2026
- [ ] Live demo opens without login (or video shows local run end-to-end)
- [ ] Solo team listed correctly on Devpost
- [ ] No cross-submission to any other hackathon
