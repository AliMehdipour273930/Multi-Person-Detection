"""Heatmap target creation and heatmap decoding."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from src import config


def make_heatmap_from_points(
    xy: np.ndarray,
    mask: np.ndarray,
    H: int = config.HEATMAP_H,
    W: int = config.HEATMAP_W,
    room_x: float = config.ROOM_X,
    room_y: float = config.ROOM_Y,
    sigma: float = config.HEATMAP_SIGMA,
) -> np.ndarray:
    """
    Build one Gaussian heatmap from up to four room coordinates.

    Args:
        xy: Array with shape (4, 2).
        mask: Array with shape (4,).

    Returns:
        Array with shape (H, W, 1).
    """
    heatmap = np.zeros((H, W), dtype=np.float32)

    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    for person_idx in range(len(mask)):
        if mask[person_idx] < 0.5:
            continue

        px, py = xy[person_idx]
        px = np.clip(px, 0.0, room_x)
        py = np.clip(py, 0.0, room_y)

        col = (px / room_x) * (W - 1)
        row = (py / room_y) * (H - 1)

        gaussian = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap[..., None].astype(np.float32)


def build_heatmaps(
    y_xy: np.ndarray,
    y_mask: np.ndarray,
    H: int = config.HEATMAP_H,
    W: int = config.HEATMAP_W,
    sigma: float = config.HEATMAP_SIGMA,
) -> np.ndarray:
    """Build heatmap targets for all samples."""
    heatmaps = [
        make_heatmap_from_points(
            y_xy[sample_idx],
            y_mask[sample_idx],
            H=H,
            W=W,
            room_x=config.ROOM_X,
            room_y=config.ROOM_Y,
            sigma=sigma,
        )
        for sample_idx in range(len(y_xy))
    ]
    return np.stack(heatmaps).astype(np.float32)


def heatmap_to_localizations(
    heatmap: np.ndarray,
    threshold: float = config.HEATMAP_THRESHOLD,
    nms_size: int = config.NMS_SIZE,
    max_persons: int = config.MAX_PERSONS,
    room_x: float = config.ROOM_X,
    room_y: float = config.ROOM_Y,
) -> tuple[list[list[float]], list[float]]:
    """
    Decode heatmap peaks into room coordinates.

    Args:
        heatmap: Array with shape (H, W) or (H, W, 1).

    Returns:
        A list of [x, y] coordinates and a list of confidence scores.
    """
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
    rows = rows[order]
    cols = cols[order]
    scores = scores[order]

    rows = rows[:max_persons]
    cols = cols[:max_persons]
    scores = scores[:max_persons]

    localizations = []

    for row, col in zip(rows, cols):
        x_value = (col / (heatmap_w - 1)) * room_x
        y_value = (row / (heatmap_h - 1)) * room_y
        x_value = float(np.clip(x_value, 0.0, room_x))
        y_value = float(np.clip(y_value, 0.0, room_y))
        localizations.append([x_value, y_value])

    return localizations, scores.astype(float).tolist()


def heatmap_argmax_to_xy(
    heatmap: np.ndarray,
    room_x: float = config.ROOM_X,
    room_y: float = config.ROOM_Y,
) -> tuple[float, float, float]:
    """Convert the maximum heatmap pixel to one room coordinate."""
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]

    heatmap_h, heatmap_w = heatmap.shape
    flat_idx = np.argmax(heatmap)
    row, col = np.unravel_index(flat_idx, heatmap.shape)
    confidence = heatmap[row, col]

    x_value = (col / (heatmap_w - 1)) * room_x
    y_value = (row / (heatmap_h - 1)) * room_y

    return float(x_value), float(y_value), float(confidence)


def gt_to_localizations(
    xy: np.ndarray,
    mask: np.ndarray,
    room_x: float = config.ROOM_X,
    room_y: float = config.ROOM_Y,
) -> list[list[float]]:
    """Convert masked ground-truth coordinates to a localization list."""
    localizations = []

    for person_idx in range(len(mask)):
        if mask[person_idx] < 0.5:
            continue

        x_value, y_value = xy[person_idx]
        x_value = float(np.clip(x_value, 0.0, room_x))
        y_value = float(np.clip(y_value, 0.0, room_y))
        localizations.append([x_value, y_value])

    return localizations
