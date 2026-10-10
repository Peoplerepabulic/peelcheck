# Devpost description draft (≤500 words) — fill bracketed bits after building

**Project name:** PeelCheck
**Team:** [names + schools]
**Tagline:** One photo tells you if that banana is snack, bread, or bin.

## Inspiration (brainstorming)
[1–2 sentences: food waste moment — e.g. dorm fridge bananas going black; wanted a
classifier a beginner could train in a weekend that still proves a real pipeline.]

## What it does
Upload a banana photo → PeelCheck predicts underripe / ripe / overripe with
confidence → returns a concrete next step (eat now / bake / freeze) so the fruit
gets used instead of tossed.

## How we built it (design process + resources)
- Data: [n] photos across 3 ripeness stages, collected/shot during the hackathon
  + public ripeness images; augmented (rotation, brightness, crop).
- Model: MobileNetV3 transfer learning, fine-tuned in Colab; evaluated with a
  confusion matrix (per-class precision/recall in repo README).
- App: Streamlit single-page demo, deployed on Hugging Face Spaces so judges can
  click a live URL, not just watch a video.
- Resources: PyTorch, Streamlit, Hugging Face Spaces, [dataset links].

## Challenges
[1–2 honest bullets: lighting variance, class boundary at “ripe vs overripe”.]

## What's next
More fruit classes, grocer shelf-scan mode, and a shareable “use-first” list.

---
Submission checklist (do not publish this section):
- [ ] Video ≤3:00, YouTube **public**, covers inspiration / design / resources / product
- [ ] GitHub repo public, commits all within Oct 9–11, 2026
- [ ] Live demo URL opens without login
- [ ] Team members all listed on Devpost
- [ ] No other hackathon has this project (no cross-submission)
