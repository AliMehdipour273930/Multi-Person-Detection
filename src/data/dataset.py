"""Dataset loading, file-level splitting, and temporal window creation."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import config
from src.features.preprocessing import FINAL_SELECTED_CHANNELS, build_radar_features


def find_npz_files(data_dir: str | Path) -> list[Path]:
    """Find all .npz files below a dataset directory."""
    data_dir = Path(data_dir)
    return sorted(path for path in data_dir.rglob("*.npz") if path.is_file())


def load_npz(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load one notebook-format .npz file."""
    with np.load(npz_path) as data:
        return {
            "radar_cir_iq": data["radar_cir_iq"].astype(np.float32),
            "people_xy": data["people_xy"].astype(np.float32),
            "people_mask": data["people_mask"].astype(np.float32),
            "timestamps": data["timestamps"] if "timestamps" in data else None,
        }


def sort_people_by_x(xy: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort valid people by x coordinate and pad to MAX_PEOPLE."""
    valid = mask.astype(bool)
    xy_valid = xy[valid]

    if len(xy_valid) > 0:
        order = np.argsort(xy_valid[:, 0])
        xy_valid = xy_valid[order]

    xy_out = np.zeros((config.MAX_PEOPLE, 2), dtype=np.float32)
    mask_out = np.zeros((config.MAX_PEOPLE,), dtype=np.float32)

    count = min(len(xy_valid), config.MAX_PEOPLE)

    if count > 0:
        xy_out[:count] = xy_valid[:count]
        mask_out[:count] = 1.0

    return xy_out, mask_out


def get_window_people_count(npz_path: str | Path) -> int:
    """Return the dominant number of people in one acquisition file."""
    with np.load(npz_path) as data:
        people_mask = data["people_mask"].astype(np.float32)

    counts_per_frame = (people_mask > 0.5).sum(axis=1).astype(int)
    values, counts = np.unique(counts_per_frame, return_counts=True)
    return int(values[np.argmax(counts)])


def split_files_by_people_count(
    npz_files: list[str | Path],
    random_state: int = config.RANDOM_STATE,
) -> tuple[list[Path], list[Path]]:
    """
    Reproduce the notebook's balanced train/test split by acquisition file.

    The notebook uses a fixed quota for the observed dataset distribution.
    """
    rng = random.Random(random_state)
    windows_by_count: dict[int, list[Path]] = defaultdict(list)

    for npz_path in npz_files:
        path = Path(npz_path)
        people_count = get_window_people_count(path)
        windows_by_count[people_count].append(path)

    for people_count in windows_by_count:
        rng.shuffle(windows_by_count[people_count])

    test_quota = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
    }

    train_files: list[Path] = []
    test_files: list[Path] = []

    for people_count in sorted(windows_by_count.keys()):
        group = windows_by_count[people_count]
        quota = test_quota.get(people_count, 0)
        test_files.extend(group[:quota])
        train_files.extend(group[quota:])

    return sorted(train_files), sorted(test_files)


def summarize_split(npz_files: list[str | Path]) -> dict[int, int]:
    """Count acquisition files by dominant people count."""
    hist = {people_count: 0 for people_count in range(config.MAX_PEOPLE + 1)}
    for npz_path in npz_files:
        people_count = get_window_people_count(npz_path)
        hist[people_count] = hist.get(people_count, 0) + 1
    return hist


def build_samples_from_window(
    npz_path: str | Path,
    seq_len: int = config.SEQ_LEN,
    frame_stride: int = config.FRAME_STRIDE,
    selected_channels: list[int] | tuple[int, ...] = tuple(FINAL_SELECTED_CHANNELS),
    only_one_person: bool = config.ONLY_ONE_PERSON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build temporal samples from one acquisition file.

    The label is taken from the last frame of each input sequence.
    """
    data = load_npz(npz_path)

    radar = data["radar_cir_iq"]
    people_xy = data["people_xy"]
    people_mask = data["people_mask"]

    features = build_radar_features(
        radar,
        selected_channels=list(selected_channels),
        use_log=config.USE_LOG,
        declutter_mode=config.DECLUTTER_MODE,
        ema_alpha=config.EMA_ALPHA,
        use_mag=config.USE_MAG,
        use_decluttered_mag=config.USE_DECLUTTERED_MAG,
        use_motion_mag=config.USE_MOTION_MAG,
        use_local_variance=config.USE_LOCAL_VARIANCE,
        variance_window=config.LOCAL_VARIANCE_WINDOW,
        use_phase_diff=config.USE_PHASE_DIFF,
        bin_start=config.BIN_START,
        bin_end=config.BIN_END,
    )

    x_list = []
    y_xy_list = []
    y_mask_list = []
    frame_idx_list = []

    total_frames = features.shape[0]

    for end_frame in range(seq_len - 1, total_frames, frame_stride):
        if only_one_person and int(people_mask[end_frame].sum()) != 1:
            continue

        sequence = features[end_frame - seq_len + 1 : end_frame + 1]
        xy, mask = sort_people_by_x(people_xy[end_frame], people_mask[end_frame])

        x_list.append(sequence)
        y_xy_list.append(xy)
        y_mask_list.append(mask)
        frame_idx_list.append(end_frame)

    feature_count = features.shape[-1]
    bin_count = features.shape[1]

    if len(x_list) == 0:
        x_seq = np.empty((0, seq_len, bin_count, feature_count), dtype=np.float32)
        y_xy = np.empty((0, config.MAX_PEOPLE, 2), dtype=np.float32)
        y_mask = np.empty((0, config.MAX_PEOPLE), dtype=np.float32)
        meta_frame = np.empty((0,), dtype=np.int32)
        return x_seq, y_xy, y_mask, meta_frame

    x_seq = np.stack(x_list).astype(np.float32)
    y_xy = np.stack(y_xy_list).astype(np.float32)
    y_mask = np.stack(y_mask_list).astype(np.float32)
    meta_frame = np.array(frame_idx_list, dtype=np.int32)

    return x_seq, y_xy, y_mask, meta_frame


def build_dataset_from_files(
    npz_files: list[str | Path],
    seq_len: int = config.SEQ_LEN,
    frame_stride: int = config.FRAME_STRIDE,
) -> dict[str, np.ndarray]:
    """Build model arrays from a list of acquisition files."""
    x_list = []
    y_xy_list = []
    y_mask_list = []
    window_name_list = []
    frame_idx_list = []

    for npz_path in npz_files:
        x_seq, y_xy, y_mask, meta_frame = build_samples_from_window(
            npz_path,
            seq_len=seq_len,
            frame_stride=frame_stride,
        )

        if len(x_seq) == 0:
            continue

        x_list.append(x_seq)
        y_xy_list.append(y_xy)
        y_mask_list.append(y_mask)
        window_name_list.extend([Path(npz_path).name] * len(x_seq))
        frame_idx_list.append(meta_frame)

    if len(x_list) == 0:
        raise ValueError("No samples were created from the provided files.")

    return {
        "X": np.concatenate(x_list, axis=0).astype(np.float32),
        "y_xy": np.concatenate(y_xy_list, axis=0).astype(np.float32),
        "y_mask": np.concatenate(y_mask_list, axis=0).astype(np.float32),
        "frame_idx": np.concatenate(frame_idx_list, axis=0).astype(np.int32),
        "window_name": np.array(window_name_list),
    }


def compute_normalization_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute notebook normalization statistics from the training split only."""
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(
    x_values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply notebook feature normalization."""
    return ((x_values - mean) / std).astype(np.float32)
