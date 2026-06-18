"""Export the trained heatmap model to a submission-ready TFLite package.

This script follows the TinyML exercises: it uses full integer quantization,
calibrates the model with representative training samples, and copies the
preprocessing artifacts needed by `submission/code.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf


def resolve_project_dir(cli_project_dir: str | None) -> Path:
    if cli_project_dir:
        return Path(cli_project_dir).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if cwd.name in {"scripts", "notebooks", "submission", "evaluation"}:
        return cwd.parent
    return cwd


def load_best_validation_decoder(metrics_dir: Path) -> dict:
    """Use validation F1 to select the heatmap threshold for deployment."""
    sweep_path = metrics_dir / "validation_threshold_sweep.csv"
    if sweep_path.exists():
        with open(sweep_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = max(rows, key=lambda row: float(row["f1"]))
            return {
                "heatmap_threshold": float(best["threshold"]),
                "nms_size": int(float(best["nms_size"])),
                "selected_on": "validation_f1",
                "validation_f1": float(best["f1"]),
                "validation_precision": float(best["precision"]),
                "validation_recall": float(best["recall"]),
            }

    final_metrics_path = metrics_dir / "final_metrics.json"
    if final_metrics_path.exists():
        with open(final_metrics_path, "r", encoding="utf-8") as f:
            final_metrics = json.load(f)
        return {
            "heatmap_threshold": float(final_metrics.get("threshold", 0.05)),
            "nms_size": int(final_metrics.get("nms_size", 3)),
            "selected_on": "fallback_final_metrics",
        }

    return {
        "heatmap_threshold": 0.05,
        "nms_size": 3,
        "selected_on": "default",
    }


def sanitized_preprocessing_config(config_path: Path) -> dict:
    """Drop absolute training paths before copying the config to submission."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for key in ("train_files", "validation_files", "test_files"):
        config.pop(key, None)
    return config


def representative_dataset(x_train: np.ndarray, max_samples: int):
    """Yield calibration samples for full INT8 quantization."""
    sample_count = min(max_samples, len(x_train))
    for i in range(sample_count):
        yield [np.asarray(x_train[i : i + 1], dtype=np.float32)]


def export_full_int8(
    keras_model_path: Path,
    x_train_path: Path,
    output_path: Path,
    representative_samples: int,
) -> bytes:
    model = tf.keras.models.load_model(keras_model_path, compile=False)
    x_train = np.load(x_train_path, mmap_mode="r")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        x_train,
        representative_samples,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    return tflite_model


def copy_submission_artifacts(
    processed_dir: Path,
    evaluation_dir: Path,
    submission_dir: Path,
) -> None:
    submission_dir.mkdir(parents=True, exist_ok=True)

    stats_path = processed_dir / "normalization_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")
    shutil.copy2(stats_path, submission_dir / "normalization_stats.npz")
    # Keep a short alias for compatibility with older local scripts.
    shutil.copy2(stats_path, submission_dir / "norm_stats.npz")

    config_path = processed_dir / "preprocessing_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing preprocessing config: {config_path}")
    config = sanitized_preprocessing_config(config_path)
    with open(submission_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    decoder_config = load_best_validation_decoder(evaluation_dir)
    with open(submission_dir / "decoder_config.json", "w", encoding="utf-8") as f:
        json.dump(decoder_config, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--keras-model", default="models/best_heatmap_model.keras")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--evaluation-dir", default="evaluation_results")
    parser.add_argument("--submission-dir", default="submission")
    parser.add_argument("--representative-samples", type=int, default=128)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_dir = resolve_project_dir(args.project_dir)
    keras_model_path = project_dir / args.keras_model
    processed_dir = project_dir / args.processed_dir
    evaluation_dir = project_dir / args.evaluation_dir
    submission_dir = project_dir / args.submission_dir

    if not keras_model_path.exists():
        raise FileNotFoundError(f"Missing Keras model: {keras_model_path}")
    x_train_path = processed_dir / "X_train.npy"
    if not x_train_path.exists():
        raise FileNotFoundError(f"Missing representative data: {x_train_path}")

    tflite_model = export_full_int8(
        keras_model_path=keras_model_path,
        x_train_path=x_train_path,
        output_path=submission_dir / "model.tflite",
        representative_samples=args.representative_samples,
    )
    copy_submission_artifacts(processed_dir, evaluation_dir, submission_dir)

    print("Export complete.")
    print(f"  model.tflite: {len(tflite_model) / 1024:.1f} KB")
    print(f"  submission:   {submission_dir}")
    print("Run the official-style checker next:")
    print("  python evaluation/evaluate_constraint.py --model-path submission/model.tflite")


if __name__ == "__main__":
    main()
