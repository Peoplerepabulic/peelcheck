# HF Spaces deploy (Streamlit)

1. Create Space: huggingface.co > New Space > name `peelcheck` > SDK Streamlit > Public.
2. Push this repo's app files to the Space repo:
   `app.py`, `requirements.txt`, `models/peelcheck.onnx`, `models/metrics.json`
   (Space README needs front-matter: title/emoji/sdk — copy from DEPLOY_README below.)
3. First build takes a few minutes (onnxruntime + streamlit). Check Logs until "Running".
4. Put the Space URL into README.md + Devpost "Live demo" field. Judges will click it.

DEPLOY_README front-matter:
---
title: PeelCheck
emoji: 🍌
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

Fallback if Spaces is slow: record the local `streamlit run app.py` session for the
YouTube video and link the GitHub repo as the primary demo. A working link beats a
polished video, but a recorded working demo beats a broken link.
