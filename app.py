"""PeelCheck — Streamlit demo.

Upload a banana photo -> ripeness stage + what to do with it.

Inference order:
  1. models/peelcheck.onnx (fine-tuned MobileNetV3) if present.
  2. Color-heuristic preview otherwise, so the demo never shows a dead page.
     The heuristic is a placeholder, clearly labeled in the UI.

Run: streamlit run app.py
"""
import json
import os

import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = os.path.join("models", "peelcheck.onnx")
METRICS_PATH = os.path.join("models", "metrics.json")
DEFAULT_CLASSES = ["overripe", "ripe", "rotten", "unripe"]  # ImageFolder order

DISPLAY = {
    "unripe": ("Underripe", "Wait 1–2 days, or cook it — fried / grilled banana works great. Don't refrigerate yet."),
    "ripe": ("Ripe", "Eat it now. Peak sweetness, peak texture."),
    "overripe": ("Overripe", "Banana bread, smoothie, or freeze it for nice-cream today. Don't toss it!"),
    "rotten": ("Past best", "Don't eat it raw. If it's moldy or sour-smelling, compost it."),
}


def load_classes():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                return json.load(f).get("classes", DEFAULT_CLASSES)
        except Exception:
            pass
    return DEFAULT_CLASSES


CLASSES = load_classes()


def predict_onnx(image: Image.Image):
    import onnxruntime as ort

    img = image.convert("RGB").resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    x = np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def predict_heuristic(image: Image.Image):
    """Rough color fallback: green->yellow->brown progression."""
    img = image.convert("RGB").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # banana-ish mask: yellow/green dominant, ignore dark background
    lum = arr.mean(axis=2)
    mask = lum > 0.25
    if mask.sum() < 100:
        mask = np.ones_like(lum, dtype=bool)
    greenish = ((g > r) & mask).mean()
    brown = ((r < 0.45) & (g < 0.35) & (b < 0.25) & mask).mean()
    if brown > 0.18:
        label = "rotten" if brown > 0.35 else "overripe"
        conf = min(0.55 + brown, 0.8)
    elif greenish > 0.12:
        label, conf = "unripe", 0.6
    else:
        label, conf = "ripe", 0.62
    probs = {c: 0.08 for c in CLASSES}
    probs[label] = conf
    s = sum(probs.values())
    probs = {k: v / s for k, v in probs.items()}
    return label, probs[label], probs


def predict(image: Image.Image):
    if os.path.exists(MODEL_PATH):
        try:
            label, conf, probs = predict_onnx(image)
            return label, conf, probs, "Fine-tuned model (MobileNetV3)"
        except Exception as exc:
            st.warning(f"Model load failed ({exc}); showing heuristic preview.")
    label, conf, probs = predict_heuristic(image)
    return label, conf, probs, "Heuristic preview — train the model to replace this"


st.set_page_config(page_title="PeelCheck", page_icon="🍌")
st.title("🍌 PeelCheck")
st.caption("One photo → ripeness stage → what to do with it. No more wasted bananas.")

uploaded = st.file_uploader("Upload a banana photo", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your banana", use_container_width=True)
    raw_label, conf, probs, engine = predict(img)
    stage, tip = DISPLAY.get(raw_label, (raw_label, ""))
    st.metric("Ripeness", stage, f"{conf:.0%} confidence")
    st.info(tip)
    st.caption(f"Engine: {engine}")
    st.write("All stages:")
    for raw, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        st.progress(p, text=f"{DISPLAY.get(raw, (raw, ''))[0]} — {p:.0%}")
else:
    st.write("Tip: daylight, plain background, whole banana in frame works best.")
    st.write("No photo handy? Use any image from the dataset test split — "
             "the demo runs on held-out images too.")

st.divider()
st.caption("Built during Banana Hacks 2026 · Fine-tuned MobileNetV3 on the "
           "Kaggle Banana Ripeness dataset (Apache 2.0) · Solo project")
