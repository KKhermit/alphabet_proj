---
license: mit
tags:
  - image-classification
  - onnx
  - pytorch
  - letters
  - emnist
datasets:
  - emnist
metrics:
  - accuracy
---

# Alphabet Classifier (A–Z)

A lightweight CNN that classifies grayscale images of handwritten uppercase letters (A–Z), trained on the EMNIST Letters dataset.

## Model Details

| Property | Value |
|---|---|
| Architecture | SmallAlphaNet (depthwise-separable CNN) |
| Parameters | 37,658 |
| Input | 64 × 64 grayscale image |
| Output | 26-class softmax (A–Z) |
| Format | ONNX (opset 17) |
| Val accuracy | **94.1%** |

The architecture uses depthwise-separable convolutions (like MobileNet) to stay small and fast while still achieving strong accuracy.

## Training

| Setting | Value |
|---|---|
| Dataset | EMNIST Letters (`train` split) |
| Train samples | 124,800 |
| Val samples | 20,800 |
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Batch size | 256 |
| Max epochs | 30 |
| Augmentation | RandomAffine, RandomPerspective, ColorJitter, GaussianBlur, RandomErasing |

## Per-class Performance

| Letter | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.908 | 0.968 | 0.937 |
| B | 0.981 | 0.970 | 0.975 |
| C | 0.975 | 0.984 | 0.979 |
| D | 0.952 | 0.950 | 0.951 |
| E | 0.977 | 0.974 | 0.976 |
| F | 0.976 | 0.959 | 0.967 |
| G | 0.899 | 0.815 | 0.855 |
| H | 0.957 | 0.953 | 0.955 |
| I | 0.749 | 0.745 | 0.747 |
| J | 0.960 | 0.938 | 0.949 |
| K | 0.982 | 0.966 | 0.974 |
| L | 0.750 | 0.758 | 0.754 |
| M | 0.983 | 0.993 | 0.988 |
| N | 0.955 | 0.960 | 0.958 |
| O | 0.950 | 0.984 | 0.967 |
| P | 0.986 | 0.989 | 0.988 |
| Q | 0.858 | 0.864 | 0.861 |
| R | 0.961 | 0.964 | 0.963 |
| S | 0.978 | 0.988 | 0.983 |
| T | 0.949 | 0.976 | 0.962 |
| U | 0.953 | 0.933 | 0.943 |
| V | 0.933 | 0.946 | 0.940 |
| W | 0.984 | 0.981 | 0.982 |
| X | 0.979 | 0.969 | 0.974 |
| Y | 0.948 | 0.955 | 0.951 |
| Z | 0.981 | 0.989 | 0.985 |

**Hardest letters:** I (F1=0.747), L (F1=0.754), G (F1=0.855) — these are visually ambiguous in handwriting.

## Usage

```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from PIL import Image

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
print("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[pred])
```

## Reproduce Training

```bash
git clone https://huggingface.co/hermitkk/alphabet-classifier
cd alphabet-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py train --config config/config.yaml
```
