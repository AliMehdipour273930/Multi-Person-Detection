"""Export a trained Keras model to a real TFLite model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras-model",
        default="checkpoints/best_model.keras",
        help="Path to a trained Keras model.",
    )
    parser.add_argument(
        "--output",
        default="submission/model.tflite",
        help="Path for the exported TFLite model.",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the trained model to TFLite."""
    args = parse_args()

    keras_model_path = Path(args.keras_model)
    output_path = Path(args.output)

    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_model_path}")

    model = tf.keras.models.load_model(keras_model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Saved TFLite model: {output_path}")


if __name__ == "__main__":
    main()
