from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from alphabet.dataset import CHAR_CLASSES, preprocess_crop_for_inference


class OnnxBackend:
    """ONNX runtime backend for the 36-class digit+letter model."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("Install onnxruntime to use the ONNX backend.") from exc
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities, shape (batch, 36)."""
        arr = batch_tensor.detach().cpu().numpy().astype(np.float32)
        logits = self.session.run([self.output_name], {self.input_name: arr})[0]
        # numerically stable softmax
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


def predict_crops(
    backend: OnnxBackend,
    crops: list[np.ndarray],
    img_size: int = 64,
    mean: float = 0.5,
    std: float = 0.5,
    min_confidence: float = 0.60,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """
    Run inference on a list of pre-cropped BGR images (digits or letters).

    Returns one dict per crop with:
      character        — predicted character ('0'–'9' or 'A'–'Z')
      prediction_index — 0–35
      confidence       — softmax probability of the top class
      probabilities    — full 36-element list
      flag             — True when confidence < min_confidence (needs human review)
    """
    tensors: list[torch.Tensor] = []
    for crop in crops:
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(preprocess_crop_for_inference(pil, img_size=img_size, mean=mean, std=std))

    all_probs: list[np.ndarray] = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size], dim=0)
        all_probs.append(backend.predict(batch))

    probs_matrix = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 36))

    results: list[dict[str, Any]] = []
    for probs in probs_matrix:
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        results.append({
            "character": CHAR_CLASSES[idx],
            "prediction_index": idx,
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "flag": confidence < min_confidence,
        })
    return results


__all__ = ["OnnxBackend", "predict_crops"]
