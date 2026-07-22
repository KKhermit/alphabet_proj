---
license: mit
tags:
  - image-classification
  - onnx
  - pytorch
  - letters
  - digits
  - alphanumeric
  - emnist
datasets:
  - emnist
metrics:
  - accuracy
---

# Alpha-Numeric Classifier (0–9, A–Z)

A lightweight CNN that classifies grayscale images of handwritten digits (0–9), uppercase letters (A–Z), and blank boxes, trained on a combined EMNIST Digits + Letters dataset with synthetic blank images.

## Model Details

| Property | Value |
|---|---|
| Architecture | SmallAlphaNet (depthwise-separable CNN) |
| Parameters | 37,658 |
| Input | 64 × 64 grayscale image |
| Output | 37-class softmax (0–9, A–Z, blank) |
| Format | ONNX (opset 17) |
| Val accuracy | **TBD** |

The architecture uses depthwise-separable convolutions (like MobileNet) to stay small and fast while still achieving strong accuracy.

## Training

| Setting | Value |
|---|---|
| Dataset | EMNIST Digits + Letters (`train` split) + synthetic blanks |
| Train samples | 374,800 (240,000 digits + 124,800 letters + 10,000 blank) |
| Val samples | 62,800 (40,000 digits + 20,800 letters + 2,000 blank) |
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Batch size | 256 |
| Max epochs | 30 |
| Augmentation | RandomAffine, RandomPerspective, ColorJitter, GaussianBlur, RandomErasing |

## Classes

| Index | Character |
|---|---|
| 0–9 | Digits `0`–`9` |
| 10–35 | Uppercase letters `A`–`Z` |
| 36 | Blank (empty box) |

## Per-class Performance

*To be updated after training with the new 37-class model.*

## Usage

```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from PIL import Image

CHAR_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["blank"]

# Download model
path = hf_hub_download(repo_id="hermitkk/alphabet-classifier", filename="alphabet_model.onnx")
session = ort.InferenceSession(path)

# Preprocess a 64x64 grayscale image
img = Image.open("letter.png").convert("L").resize((64, 64))
x = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
x = x[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)

# Run inference
logits = session.run(None, {"input": x})[0]
pred = int(np.argmax(logits))
print(CHAR_CLASSES[pred])
```

## Reproduce Training

```bash
git clone https://huggingface.co/hermitkk/alphabet-classifier
cd alphabet-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py train --config config/config.yaml
```
