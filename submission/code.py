"""Submission inference for multi-person UWB radar localization.

The evaluator calls this script with one raw `.npy` file shaped
`(T, 6, 3, 120, 2)`.  The script mirrors the notebook preprocessing, runs the
TFLite heatmap model, decodes heatmap peaks, and writes one JSON object per
frame.

Only NumPy, SciPy, TensorFlow, and the Python standard library are used because
that is the allowed submission environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.ndimage import maximum_filter


DEFAULT_CONFIG = {
    "seq_len": 16,
    "selected_channels": [0, 4, 7, 11, 12, 15],
    "use_log": True,
    "declutter_mode": "ema",
    "ema_alpha": 0.98,
    "use_mag": False,
    "use_decluttered_mag": True,
    "use_motion_mag": True,
    "use_phase_diff": True,
    "use_local_variance": True,
    "variance_window": 9,
    "bin_start": 5,
    "bin_end": 60,
    "room_x": 4.8,
    "room_y": 7.2,
    "max_people": 4,
    "normalization_eps": 1e-6,
}

DEFAULT_DECODER_CONFIG = {
    "heatmap_threshold": 0.05,
    "nms_size": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TFLite heatmap localization on raw UWB radar input."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to a .npy input shaped (T, 6, 3, 120, 2).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where the .jsonl predictions will be written.",
    )
    return parser.parse_args()


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    merged = dict(default)
    merged.update(loaded)
    return merged


def load_normalization_stats(submission_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the training-set normalization used by the notebooks."""
    candidates = [
        submission_dir / "normalization_stats.npz",
        submission_dir / "norm_stats.npz",
    ]
    for path in candidates:
        if path.exists():
            stats = np.load(path)
            return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    raise FileNotFoundError(
        "Missing normalization stats. Run scripts/export_tflite.py first so the "
        "submission folder contains normalization_stats.npz."
    )


def build_complex_cir(radar_iq: np.ndarray) -> np.ndarray:
    return (radar_iq[..., 0] + 1j * radar_iq[..., 1]).astype(np.complex64)


def apply_ema_declutter(x: np.ndarray, alpha: float) -> np.ndarray:
    """Remove static reflections with the same EMA background used in training."""
    output = np.empty_like(x)
    background = x[0].copy()
    output[0] = 0
    for t in range(1, len(x)):
        background = alpha * background + (1.0 - alpha) * x[t]
        output[t] = x[t] - background
    return output


def local_temporal_variance(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise ValueError("variance_window must be a positive odd integer.")
    pad = window // 2
    x_pad = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, window, axis=0)
    return windows.var(axis=-1, dtype=np.float32)


def build_radar_features(raw: np.ndarray, config: dict) -> np.ndarray:
    """Convert raw I/Q frames into the notebook feature tensor `(T, bins, C)`."""
    if raw.ndim != 5 or raw.shape[1:] != (6, 3, 120, 2):
        raise ValueError(
            f"Expected input shape (T, 6, 3, 120, 2), got {raw.shape}."
        )

    selected_channels = np.array(config["selected_channels"], dtype=np.int64)
    bin_start = int(config["bin_start"])
    bin_end = int(config["bin_end"])

    # Merge the radar and antenna axes into the same channel ordering used by
    # the notebooks, then crop channels and bins before expensive operations.
    radar_iq = raw.astype(np.float32, copy=False).transpose(0, 3, 1, 2, 4)
    radar_iq = radar_iq.reshape(raw.shape[0], raw.shape[3], -1, 2)
    radar_iq = radar_iq[:, bin_start:bin_end, selected_channels]

    complex_cir = build_complex_cir(radar_iq)
    magnitude = np.abs(complex_cir).astype(np.float32)
    if config["use_log"]:
        magnitude = np.log1p(magnitude)

    feature_blocks = []
    if config["use_mag"]:
        feature_blocks.append(magnitude)

    needs_decluttered = config["use_decluttered_mag"] or config["use_motion_mag"]
    if needs_decluttered:
        if config["declutter_mode"] != "ema":
            raise ValueError("submission/code.py currently expects EMA decluttering.")
        decluttered_complex = apply_ema_declutter(
            complex_cir,
            alpha=float(config["ema_alpha"]),
        )
        decluttered_magnitude = np.abs(decluttered_complex).astype(np.float32)
        if config["use_log"]:
            decluttered_magnitude = np.log1p(decluttered_magnitude)

    if config["use_decluttered_mag"]:
        feature_blocks.append(decluttered_magnitude)

    if config["use_motion_mag"]:
        motion = np.abs(
            np.diff(
                decluttered_magnitude,
                axis=0,
                prepend=decluttered_magnitude[:1],
            )
        ).astype(np.float32)
        feature_blocks.append(motion)

    if config["use_local_variance"]:
        feature_blocks.append(
            local_temporal_variance(magnitude, int(config["variance_window"]))
        )

    if config["use_phase_diff"]:
        phase_difference = np.zeros_like(magnitude, dtype=np.float32)
        phase_difference[1:] = np.angle(
            complex_cir[1:] * np.conj(complex_cir[:-1])
        ).astype(np.float32)
        feature_blocks.append(phase_difference)

    if not feature_blocks:
        raise ValueError("At least one feature block must be enabled.")
    return np.concatenate(feature_blocks, axis=2).astype(np.float32)


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float,
) -> np.ndarray:
    return ((features - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), eps)).astype(
        np.float32
    )


def make_model_batch(
    features: np.ndarray,
    start_frame: int,
    batch_size: int,
    seq_len: int,
) -> tuple[np.ndarray, int]:
    """Build a batch of per-frame windows, padding early frames with frame 0."""
    end_frame = min(start_frame + batch_size, len(features))
    batch = np.empty(
        (end_frame - start_frame, seq_len, features.shape[1], features.shape[2]),
        dtype=np.float32,
    )
    first_frame = features[0:1]

    for out_i, frame in enumerate(range(start_frame, end_frame)):
        window_start = max(0, frame - seq_len + 1)
        window = features[window_start : frame + 1]
        if len(window) < seq_len:
            pad = np.repeat(first_frame, seq_len - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        batch[out_i] = window
    return batch, end_frame


def quantize_input(batch: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return batch.astype(np.float32)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        return batch.astype(dtype)

    q = np.round(batch / scale + zero_point)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize_output(output: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return output.astype(np.float32)

    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if not scale:
        return output.astype(np.float32)
    return (output.astype(np.float32) - zero_point) * scale


def predict_heatmaps(
    interpreter: tf.lite.Interpreter,
    features: np.ndarray,
    seq_len: int,
    batch_size: int = 32,
) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    heatmaps = []

    frame = 0
    while frame < len(features):
        batch, next_frame = make_model_batch(features, frame, batch_size, seq_len)
        model_input = quantize_input(batch, input_detail)
        interpreter.resize_tensor_input(input_detail["index"], model_input.shape)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])
        heatmaps.append(dequantize_output(raw_output, output_detail))
        frame = next_frame

    return np.concatenate(heatmaps, axis=0)


def heatmap_to_localizations(
    heatmap: np.ndarray,
    threshold: float,
    nms_size: int,
    max_people: int,
    room_x: float,
    room_y: float,
) -> list[list[float]]:
    """Decode local maxima from a heatmap into room coordinates."""
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]

    h, w = heatmap.shape
    local_max = maximum_filter(heatmap, size=nms_size) == heatmap
    rows, cols = np.where(local_max & (heatmap >= threshold))
    if len(rows) == 0:
        return []

    scores = heatmap[rows, cols]
    order = np.argsort(scores)[::-1][:max_people]
    localizations = []
    for row, col in zip(rows[order], cols[order]):
        x = (col / (w - 1)) * room_x
        y = (row / (h - 1)) * room_y
        localizations.append([
            float(np.clip(x, 0.0, room_x)),
            float(np.clip(y, 0.0, room_y)),
        ])
    return localizations


def main() -> None:
    args = parse_args()
    submission_dir = Path(__file__).resolve().parent
    model_path = submission_dir / "model.tflite"
    if not model_path.exists():
        print(
            f"[ERROR] Missing {model_path}. Export it with scripts/export_tflite.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_json(submission_dir / "preprocessing_config.json", DEFAULT_CONFIG)
    decoder_config = load_json(submission_dir / "decoder_config.json", DEFAULT_DECODER_CONFIG)
    mean, std = load_normalization_stats(submission_dir)

    raw = np.load(args.input_path)
    features = build_radar_features(raw, config)
    features = normalize_features(
        features,
        mean=mean,
        std=std,
        eps=float(config["normalization_eps"]),
    )

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    heatmaps = predict_heatmaps(interpreter, features, seq_len=int(config["seq_len"]))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for frame, heatmap in enumerate(heatmaps):
            record = {
                "frame": frame,
                "localizations": heatmap_to_localizations(
                    heatmap,
                    threshold=float(decoder_config["heatmap_threshold"]),
                    nms_size=int(decoder_config["nms_size"]),
                    max_people=int(config["max_people"]),
                    room_x=float(config["room_x"]),
                    room_y=float(config["room_y"]),
                ),
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
