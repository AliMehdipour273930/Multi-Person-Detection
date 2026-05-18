"""Self-contained TFLite inference script for the heatmap pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

np = None
tf = None
maximum_filter = None

SEQ_LEN = 16
USE_LOG = True
DECLUTTER_MODE = "ema"
EMA_ALPHA = 0.98
USE_MAG = False
USE_DECLUTTERED_MAG = False
USE_MOTION_MAG = True
USE_LOCAL_VARIANCE = True
USE_PHASE_DIFF = True
LOCAL_VARIANCE_WINDOW = 9
BIN_START = 0
BIN_END = 120
SELECTED_CHANNELS = [0, 4, 7, 11, 12, 15]
ROOM_X = 4.8
ROOM_Y = 7.2
HEATMAP_THRESHOLD = 0.25
NMS_SIZE = 3
MAX_PERSONS = 4


def load_runtime_dependencies() -> None:
    """Load allowed third-party runtime dependencies after argparse help."""
    global np
    global tf
    global maximum_filter

    import numpy as numpy_module
    import tensorflow as tensorflow_module
    from scipy.ndimage import maximum_filter as scipy_maximum_filter

    np = numpy_module
    tf = tensorflow_module
    maximum_filter = scipy_maximum_filter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", nargs="?", help="Input .npz file or directory.")
    parser.add_argument("output_path", nargs="?", help="Output JSONL file. Defaults to stdout.")
    parser.add_argument("--input", dest="input_option", help="Input .npz file or directory.")
    parser.add_argument("--output", dest="output_option", help="Output JSONL file. Defaults to stdout.")
    parser.add_argument("--threshold", type=float, default=HEATMAP_THRESHOLD)
    parser.add_argument("--nms-size", type=int, default=NMS_SIZE)
    return parser.parse_args()


def build_complex_cir(radar_iq: np.ndarray) -> np.ndarray:
    """Convert radar I/Q samples to complex CIR."""
    i_values = radar_iq[..., 0]
    q_values = radar_iq[..., 1]
    complex_cir = i_values + 1j * q_values
    return complex_cir.astype(np.complex64)


def declutter_mean(x: np.ndarray) -> np.ndarray:
    """Subtract the temporal mean from each bin and channel."""
    return x - x.mean(axis=0, keepdims=True)


def declutter_ema(x: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """Apply exponential moving average background subtraction."""
    out = np.zeros_like(x)
    background = x[0].copy()

    for frame_idx in range(x.shape[0]):
        background = alpha * background + (1.0 - alpha) * x[frame_idx]
        out[frame_idx] = x[frame_idx] - background

    return out


def apply_declutter(x: np.ndarray, mode: str = "mean", alpha: float = 0.98) -> np.ndarray:
    """Apply the notebook decluttering mode."""
    if mode == "mean":
        return declutter_mean(x)
    if mode == "ema":
        return declutter_ema(x, alpha=alpha)
    raise ValueError(f"Unknown declutter mode: {mode}")


def local_temporal_variance(x: np.ndarray, window: int = 9) -> np.ndarray:
    """Compute local temporal variance over a centered sliding window."""
    if window % 2 == 0:
        raise ValueError("window must be odd")

    pad = window // 2
    x_pad = np.pad(
        x,
        pad_width=((pad, pad), (0, 0), (0, 0)),
        mode="edge",
    )

    variance = np.zeros_like(x, dtype=np.float32)

    for frame_idx in range(x.shape[0]):
        segment = x_pad[frame_idx : frame_idx + window]
        variance[frame_idx] = np.var(segment, axis=0)

    return variance.astype(np.float32)


def build_radar_features(radar_iq: np.ndarray) -> np.ndarray:
    """Build the same radar features used by the notebook heatmap model."""
    complex_cir = build_complex_cir(radar_iq)

    magnitude = np.abs(complex_cir).astype(np.float32)

    if USE_LOG:
        magnitude = np.log1p(magnitude)

    magnitude = magnitude.transpose(0, 3, 1, 2).reshape(
        magnitude.shape[0],
        120,
        18,
    )
    magnitude = magnitude[:, :, SELECTED_CHANNELS]

    feature_blocks = []

    if USE_MAG:
        feature_blocks.append(magnitude.astype(np.float32))

    if USE_DECLUTTERED_MAG or USE_MOTION_MAG:
        decluttered_magnitude = apply_declutter(
            magnitude,
            mode=DECLUTTER_MODE,
            alpha=EMA_ALPHA,
        ).astype(np.float32)
    else:
        decluttered_magnitude = None

    if USE_DECLUTTERED_MAG:
        feature_blocks.append(decluttered_magnitude)

    if USE_MOTION_MAG:
        motion_source = decluttered_magnitude if decluttered_magnitude is not None else magnitude
        motion_magnitude = np.diff(
            motion_source,
            axis=0,
            prepend=motion_source[:1],
        )
        motion_magnitude = np.abs(motion_magnitude)
        feature_blocks.append(motion_magnitude.astype(np.float32))

    if USE_LOCAL_VARIANCE:
        variance_magnitude = local_temporal_variance(
            magnitude,
            window=LOCAL_VARIANCE_WINDOW,
        )
        feature_blocks.append(variance_magnitude.astype(np.float32))

    if USE_PHASE_DIFF:
        phase_diff = np.angle(complex_cir[1:] * np.conj(complex_cir[:-1]))
        phase_diff = np.concatenate(
            [np.zeros_like(phase_diff[:1]), phase_diff],
            axis=0,
        )
        phase_diff = phase_diff.astype(np.float32)
        phase_diff = phase_diff.transpose(0, 3, 1, 2).reshape(
            phase_diff.shape[0],
            120,
            18,
        )
        phase_diff = phase_diff[:, :, SELECTED_CHANNELS]
        feature_blocks.append(phase_diff)

    if len(feature_blocks) == 0:
        raise ValueError("No feature block is enabled.")

    features = np.concatenate(feature_blocks, axis=2)
    features = features[:, BIN_START:BIN_END, :]

    return features.astype(np.float32)


def build_window_ending_at(features: np.ndarray, frame_idx: int) -> np.ndarray:
    """Build one SEQ_LEN window ending at frame_idx with frame-0 padding."""
    start = frame_idx - SEQ_LEN + 1

    if start >= 0:
        return features[start : frame_idx + 1]

    pad_count = -start
    padding = np.repeat(features[:1], pad_count, axis=0)
    return np.concatenate([padding, features[: frame_idx + 1]], axis=0)


def heatmap_to_localizations(
    heatmap: np.ndarray,
    threshold: float = HEATMAP_THRESHOLD,
    nms_size: int = NMS_SIZE,
    max_persons: int = MAX_PERSONS,
) -> tuple[list[list[float]], list[float]]:
    """Decode heatmap peaks into clipped room coordinates."""
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]

    heatmap_h, heatmap_w = heatmap.shape

    local_max = maximum_filter(heatmap, size=nms_size) == heatmap
    detected = local_max & (heatmap >= threshold)
    rows, cols = np.where(detected)

    if len(rows) == 0:
        return [], []

    scores = heatmap[rows, cols]
    order = np.argsort(scores)[::-1]
    rows = rows[order][:max_persons]
    cols = cols[order][:max_persons]
    scores = scores[order][:max_persons]

    localizations = []

    for row, col in zip(rows, cols):
        x_value = (col / (heatmap_w - 1)) * ROOM_X
        y_value = (row / (heatmap_h - 1)) * ROOM_Y
        x_value = float(np.clip(x_value, 0.0, ROOM_X))
        y_value = float(np.clip(y_value, 0.0, ROOM_Y))
        localizations.append([x_value, y_value])

    return localizations, scores.astype(float).tolist()


def find_input_files(input_path: Path) -> list[Path]:
    """Find input .npz files from a file or directory path."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.npz") if path.is_file())
    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_norm_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalization stats from the submission folder."""
    with np.load(stats_path) as stats:
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)

    if mean.ndim == 1:
        mean = mean.reshape(1, 1, 1, -1)
    if std.ndim == 1:
        std = std.reshape(1, 1, 1, -1)

    return mean, std


def prepare_input_tensor(
    window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalize and batch one model input window."""
    return ((window[None].astype(np.float32) - mean) / std).astype(np.float32)


def set_tflite_input(interpreter: tf.lite.Interpreter, input_array: np.ndarray) -> None:
    """Set a TFLite input tensor, including quantized input if needed."""
    input_details = interpreter.get_input_details()[0]

    if tuple(input_details["shape"]) != tuple(input_array.shape):
        interpreter.resize_tensor_input(input_details["index"], input_array.shape, strict=False)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]

    input_tensor = input_array
    input_dtype = input_details["dtype"]

    if input_dtype != np.float32:
        scale, zero_point = input_details["quantization"]
        if scale > 0:
            input_tensor = np.round(input_array / scale + zero_point)
        limits = np.iinfo(input_dtype)
        input_tensor = np.clip(input_tensor, limits.min, limits.max).astype(input_dtype)
    else:
        input_tensor = input_tensor.astype(np.float32)

    interpreter.set_tensor(input_details["index"], input_tensor)


def get_tflite_output(interpreter: tf.lite.Interpreter) -> np.ndarray:
    """Read a TFLite output tensor, including quantized output if needed."""
    output_details = interpreter.get_output_details()[0]
    output_tensor = interpreter.get_tensor(output_details["index"])

    if output_details["dtype"] != np.float32:
        scale, zero_point = output_details["quantization"]
        if scale > 0:
            output_tensor = (output_tensor.astype(np.float32) - zero_point) * scale

    return output_tensor.astype(np.float32)


def predict_heatmap(interpreter: tf.lite.Interpreter, input_array: np.ndarray) -> np.ndarray:
    """Run one TFLite prediction and return the heatmap."""
    set_tflite_input(interpreter, input_array)
    interpreter.invoke()
    output_tensor = get_tflite_output(interpreter)
    return output_tensor[0]


def iter_prediction_rows(
    input_files: list[Path],
    interpreter: tf.lite.Interpreter,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    nms_size: int,
):
    """Yield one JSON-serializable prediction row per input frame."""
    include_file = len(input_files) > 1

    for input_file in input_files:
        with np.load(input_file) as data:
            radar = data["radar_cir_iq"].astype(np.float32)
            timestamps = data["timestamps"] if "timestamps" in data else None

        features = build_radar_features(radar)
        total_frames = features.shape[0]

        for frame_idx in range(total_frames):
            window = build_window_ending_at(features, frame_idx)
            input_array = prepare_input_tensor(window, mean, std)
            heatmap = predict_heatmap(interpreter, input_array)
            localizations, scores = heatmap_to_localizations(
                heatmap,
                threshold=threshold,
                nms_size=nms_size,
                max_persons=MAX_PERSONS,
            )

            row = {
                "frame_id": int(frame_idx),
                "localizations": localizations,
                "scores": [float(score) for score in scores],
            }

            if include_file:
                row["file"] = input_file.name

            if timestamps is not None:
                timestamp = timestamps[frame_idx]
                if np.issubdtype(type(timestamp), np.integer):
                    row["timestamp"] = int(timestamp)
                else:
                    row["timestamp"] = float(timestamp)

            yield row


def main() -> None:
    """Run self-contained inference."""
    args = parse_args()

    input_arg = args.input_option or args.input_path
    output_arg = args.output_option or args.output_path

    if input_arg is None:
        raise SystemExit("An input .npz file or directory is required.")

    load_runtime_dependencies()

    submission_dir = Path(__file__).resolve().parent
    model_path = submission_dir / "model.tflite"
    stats_path = submission_dir / "norm_stats.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing TFLite model: {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")

    input_files = find_input_files(Path(input_arg))
    if len(input_files) == 0:
        raise FileNotFoundError(f"No .npz files found under {input_arg}")

    mean, std = load_norm_stats(stats_path)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    rows = iter_prediction_rows(
        input_files,
        interpreter,
        mean,
        std,
        threshold=args.threshold,
        nms_size=args.nms_size,
    )

    if output_arg:
        output_path = Path(output_arg)
        with output_path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row) + "\n")
    else:
        for row in rows:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
