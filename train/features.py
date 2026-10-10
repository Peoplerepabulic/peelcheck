"""Shared image features for PeelCheck (numpy + PIL only, no torch needed).

Feature vector (~47 dims):
  HSV histograms (H16 + S8 + V8), HSV mean/std, color-class fractions
  (green / yellow / brown / dark), RGB means.
Used by both the numpy trainer and the Streamlit demo so train/serve match.
"""
import numpy as np
from PIL import Image

H_BINS, S_BINS, V_BINS = 16, 8, 8
FEATURE_DIM = H_BINS + S_BINS + V_BINS + 6 + 6 + 3


def extract_features(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((64, 64))
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    def hist(x, bins):
        counts, _ = np.histogram(x, bins=bins, range=(0.0, 1.0))
        return counts / max(counts.sum(), 1)

    feats = [hist(h, H_BINS), hist(s, S_BINS), hist(v, V_BINS)]
    feats.append(np.array([h.mean(), h.std(), s.mean(), s.std(),
                           v.mean(), v.std()], dtype=np.float32))
    lum = rgb.mean(axis=2)
    mask = lum > 0.25
    denom = max(mask.sum(), 1)
    green = ((g > r + 0.03) & mask).sum() / denom
    yellow = ((r > 0.45) & (g > 0.40) & (b < 0.50) & mask).sum() / denom
    brown = ((r < 0.45) & (g < 0.35) & (b < 0.25) & mask).sum() / denom
    dark = (lum < 0.25).mean()
    bright = (lum > 0.75).mean()
    feats.append(np.array([green, yellow, brown, dark, bright,
                           (s < 0.15).mean()], dtype=np.float32))
    feats.append(np.array([r.mean(), g.mean(), b.mean()], dtype=np.float32))
    return np.concatenate(feats).astype(np.float32)
