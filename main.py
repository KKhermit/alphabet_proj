from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphabet.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Alphabet (A-Z) classifier CLI")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Download EMNIST and train the model")
    p_train.add_argument("--config", default="config/config.yaml")
    p_train.add_argument("--device", default=None, help="cuda | cpu (auto-detects if omitted)")

    p_export = subparsers.add_parser("export", help="Export a checkpoint to ONNX and TorchScript")
    p_export.add_argument("--config", default="config/config.yaml")
    p_export.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p_export.add_argument("--device", default="cpu")

    p_infer = subparsers.add_parser("infer", help="Run inference on a single image (for testing)")
    p_infer.add_argument("--model", required=True, help="Path to .onnx model")
    p_infer.add_argument("--image", required=True, help="Path to a letter crop image")
    p_infer.add_argument("--config", default="config/config.yaml")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command == "train":
        from alphabet.train import train_from_config
        summary = train_from_config(args.config, device=args.device)
        print(f"\nBest val accuracy: {summary['best_val_accuracy']:.4f}")
        return

    if args.command == "export":
        from alphabet.export import export_from_config
        exported = export_from_config(args.config, args.checkpoint, device=args.device)
        print(exported)
        return

    if args.command == "infer":
        import cv2
        import numpy as np
        from alphabet.infer import OnnxBackend, predict_crops
        from alphabet.utils import load_yaml

        cfg = load_yaml(args.config)
        infer_cfg = cfg.get("infer", {})
        model_cfg = cfg["model"]

        crop = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if crop is None:
            print(f"Error: could not read image at {args.image}", file=sys.stderr)
            sys.exit(1)

        backend = OnnxBackend(args.model)
        results = predict_crops(
            backend,
            [crop],
            img_size=int(model_cfg.get("img_size", 64)),
            mean=float(model_cfg.get("mean", 0.5)),
            std=float(model_cfg.get("std", 0.5)),
            min_confidence=float(infer_cfg.get("min_confidence", 0.60)),
        )
        r = results[0]
        flag_str = " [FLAGGED for review]" if r["flag"] else ""
        print(f"Prediction : {r['letter']}{flag_str}")
        print(f"Confidence : {r['confidence']:.4f}")
        top3 = sorted(enumerate(r["probabilities"]), key=lambda x: -x[1])[:3]
        for idx, prob in top3:
            letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx]
            print(f"  {letter}: {prob:.4f}")
        return


if __name__ == "__main__":
    main()
