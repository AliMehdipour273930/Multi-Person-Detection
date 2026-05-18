"""Radar preprocessing and feature extraction from the notebook."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src import config


def iq_to_magnitude(radar_iq: np.ndarray) -> np.ndarray:
    """
    Convert radar I/Q samples to magnitude.

    Args:
        radar_iq: Array with shape (T, 6, 3, 120, 2).

    Returns:
        Array with shape (T, 6, 3, 120).
    """
    i_values = radar_iq[..., 0]
    q_values = radar_iq[..., 1]
    magnitude = np.sqrt(i_values**2 + q_values**2)
    return magnitude.astype(np.float32)


def build_complex_cir(radar_iq: np.ndarray) -> np.ndarray:
    """
    Convert radar I/Q samples to complex CIR.

    Args:
        radar_iq: Array with shape (T, 6, 3, 120, 2).

    Returns:
        Complex array with shape (T, 6, 3, 120).
    """
    i_values = radar_iq[..., 0]
    q_values = radar_iq[..., 1]
    complex_cir = i_values + 1j * q_values
    return complex_cir.astype(np.complex64)


def sensor_to_channels(sensor_id: int) -> list[int]:
    """Map a sensor id in [0, 5] to its three flattened channel ids."""
    base = sensor_id * 3
    return [base, base + 1, base + 2]


def resolve_selected_channels(
    selected_sensors: Iterable[int],
    selected_channels: Iterable[int],
) -> list[int]:
    """Resolve sensor and manual channel selections into sorted channel ids."""
    final_channels: list[int] = []

    for sensor_id in selected_sensors:
        if sensor_id < 0 or sensor_id > 5:
            raise ValueError(f"Invalid sensor id {sensor_id}. Must be in [0, 5].")
        final_channels.extend(sensor_to_channels(sensor_id))

    for channel_id in selected_channels:
        if channel_id < 0 or channel_id > 17:
            raise ValueError(f"Invalid channel id {channel_id}. Must be in [0, 17].")
        final_channels.append(channel_id)

    final_channels = sorted(set(final_channels))

    if len(final_channels) == 0:
        raise ValueError("No channels selected.")

    return final_channels


FINAL_SELECTED_CHANNELS = resolve_selected_channels(
    config.SELECTED_SENSORS,
    config.SELECTED_CHANNELS,
)


def declutter_mean(x: np.ndarray) -> np.ndarray:
    """
    Subtract the temporal mean from each bin and channel.

    Args:
        x: Array with shape (T, B, C).
    """
    return x - x.mean(axis=0, keepdims=True)


def declutter_ema(x: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """
    Apply exponential moving average background subtraction.

    Args:
        x: Array with shape (T, B, C).
        alpha: EMA coefficient.
    """
    out = np.zeros_like(x)
    background = x[0].copy()

    for frame_idx in range(x.shape[0]):
        background = alpha * background + (1.0 - alpha) * x[frame_idx]
        out[frame_idx] = x[frame_idx] - background

    return out


def apply_declutter(
    x: np.ndarray,
    mode: str = "mean",
    alpha: float = 0.98,
) -> np.ndarray:
    """Apply the notebook decluttering mode."""
    if mode == "mean":
        return declutter_mean(x)
    if mode == "ema":
        return declutter_ema(x, alpha=alpha)
    raise ValueError(f"Unknown declutter mode: {mode}")


def local_temporal_variance(x: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Compute local temporal variance over a centered sliding window.

    Args:
        x: Array with shape (T, B, C).
        window: Odd temporal window size.

    Returns:
        Array with shape (T, B, C).
    """
    if window % 2 == 0:
        raise ValueError("window must be odd, for example 5, 7, or 9")

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


def build_radar_features(
    radar_iq: np.ndarray,
    selected_channels: list[int] | tuple[int, ...],
    use_log: bool = True,
    declutter_mode: str = "mean",
    ema_alpha: float = 0.98,
    use_mag: bool = False,
    use_decluttered_mag: bool = False,
    use_motion_mag: bool = True,
    use_local_variance: bool = True,
    variance_window: int = 9,
    use_phase_diff: bool = False,
    bin_start: int = 0,
    bin_end: int = 120,
) -> np.ndarray:
    """
    Build radar features from I/Q input.

    Args:
        radar_iq: Array with shape (T, 6, 3, 120, 2).
        selected_channels: Flattened channel ids in [0, 17].

    Returns:
        Feature array with shape (T, B_used, C_feat).
    """
    complex_cir = build_complex_cir(radar_iq)

    magnitude = np.abs(complex_cir).astype(np.float32)

    if use_log:
        magnitude = np.log1p(magnitude)

    magnitude = magnitude.transpose(0, 3, 1, 2).reshape(
        magnitude.shape[0],
        120,
        18,
    )
    magnitude = magnitude[:, :, selected_channels]

    feature_blocks = []

    if use_mag:
        feature_blocks.append(magnitude.astype(np.float32))

    if use_decluttered_mag or use_motion_mag:
        decluttered_magnitude = apply_declutter(
            magnitude,
            mode=declutter_mode,
            alpha=ema_alpha,
        ).astype(np.float32)
    else:
        decluttered_magnitude = None

    if use_decluttered_mag:
        feature_blocks.append(decluttered_magnitude)

    if use_motion_mag:
        motion_source = decluttered_magnitude if decluttered_magnitude is not None else magnitude
        motion_magnitude = np.diff(
            motion_source,
            axis=0,
            prepend=motion_source[:1],
        )
        motion_magnitude = np.abs(motion_magnitude)
        feature_blocks.append(motion_magnitude.astype(np.float32))

    if use_local_variance:
        variance_magnitude = local_temporal_variance(
            magnitude,
            window=variance_window,
        )
        feature_blocks.append(variance_magnitude.astype(np.float32))

    if use_phase_diff:
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
        phase_diff = phase_diff[:, :, selected_channels]
        feature_blocks.append(phase_diff)

    if len(feature_blocks) == 0:
        raise ValueError("No feature block is enabled.")

    features = np.concatenate(feature_blocks, axis=2)
    features = features[:, bin_start:bin_end, :]

    return features.astype(np.float32)


def build_default_radar_features(radar_iq: np.ndarray) -> np.ndarray:
    """Build features using the notebook default heatmap configuration."""
    return build_radar_features(
        radar_iq,
        selected_channels=FINAL_SELECTED_CHANNELS,
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
