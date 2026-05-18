"""Loss functions extracted from the notebook."""

from __future__ import annotations

import tensorflow as tf


def masked_xy_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Masked MSE for the notebook direct XY regression experiment.

    y_true shape is (batch, 4, 3), with normalized xy in the first two
    channels and mask in the third channel.
    """
    true_xy = y_true[:, :, 0:2]
    mask = y_true[:, :, 2]
    mask = tf.expand_dims(mask, axis=-1)

    squared_error = tf.square(true_xy - y_pred)
    masked_error = squared_error * mask
    loss = tf.reduce_sum(masked_error) / (tf.reduce_sum(mask) * 2.0 + 1e-6)

    return loss


def masked_xy_huber(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Masked Huber loss for the notebook direct XY regression experiment."""
    true_xy = y_true[:, :, 0:2]
    mask = y_true[:, :, 2]
    huber = tf.keras.losses.huber(true_xy, y_pred, delta=0.05)
    masked = huber * mask
    return tf.reduce_sum(masked) / (tf.reduce_sum(mask) + 1e-6)


def weighted_mse_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Notebook weighted MSE heatmap loss."""
    pos_weight = 5.0
    weights = 1.0 + pos_weight * y_true
    loss = weights * tf.square(y_true - y_pred)
    return tf.reduce_mean(loss)


def improved_weighted_mse_heatmap_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    pos_weight: float = 30.0,
    bg_weight: float = 5.0,
    center_weight: float = 40.0,
    mass_weight: float = 0.05,
    empty_weight: float = 5.0,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Notebook weighted MSE with mass and empty-room terms."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    soft_pos_weights = 1.0 + pos_weight * y_true
    center_mask = tf.cast(y_true >= 0.99, tf.float32)
    center_weights = 1.0 + center_weight * center_mask
    bg_mask = tf.cast(y_true < 0.05, tf.float32)
    bg_weights = 1.0 + bg_weight * bg_mask

    weights = soft_pos_weights * center_weights * bg_weights
    mse = weights * tf.square(y_true - y_pred)
    mse_loss = tf.reduce_sum(mse, axis=[1, 2, 3]) / (
        tf.reduce_sum(weights, axis=[1, 2, 3]) + eps
    )

    true_mass = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_mass = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    mass_loss = tf.square(pred_mass - true_mass) / (tf.square(true_mass) + 1.0)

    is_empty = tf.cast(tf.reduce_max(y_true, axis=[1, 2, 3]) < 0.5, tf.float32)
    max_pred_peak = tf.reduce_max(y_pred, axis=[1, 2, 3])
    empty_loss = is_empty * tf.square(max_pred_peak)

    total_loss = mse_loss + mass_weight * mass_loss + empty_weight * empty_loss

    return tf.reduce_mean(total_loss)


def stable_focal_dice_mass_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.90,
    gamma: float = 2.0,
    dice_weight: float = 0.3,
    mse_weight: float = 1.0,
    mass_weight: float = 0.20,
    center_weight: float = 10.0,
    min_mass: float = 5.0,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Notebook focal, dice, center-weighted MSE, and mass heatmap loss."""
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    binary_crossentropy = -(
        y_true * tf.math.log(y_pred)
        + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )

    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

    focal = alpha_t * tf.pow(1.0 - p_t, gamma) * binary_crossentropy
    focal = tf.reduce_mean(focal)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])

    dice = 1.0 - tf.reduce_mean((2.0 * intersection + eps) / (union + eps))

    weights = 1.0 + center_weight * y_true
    center_mse = tf.reduce_mean(weights * tf.square(y_true - y_pred))

    true_mass = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_mass = tf.reduce_sum(y_pred, axis=[1, 2, 3])

    denominator = tf.maximum(true_mass, min_mass)
    mass_error = (pred_mass - true_mass) / denominator
    mass_error = tf.clip_by_value(mass_error, -5.0, 5.0)
    mass_loss = tf.reduce_mean(tf.square(mass_error))

    total_loss = focal + dice_weight * dice + mse_weight * center_mse + mass_weight * mass_loss

    return total_loss


def make_stable_focal_dice_mass_loss(
    alpha: float = 0.85,
    gamma: float = 2.0,
    dice_weight: float = 0.5,
    mse_weight: float = 1.0,
    mass_weight: float = 0.1,
    center_weight: float = 10.0,
    min_mass: float = 5.0,
    eps: float = 1e-6,
):
    """Create the notebook grid-search variant of the heatmap loss."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return stable_focal_dice_mass_loss(
            y_true,
            y_pred,
            alpha=alpha,
            gamma=gamma,
            dice_weight=dice_weight,
            mse_weight=mse_weight,
            mass_weight=mass_weight,
            center_weight=center_weight,
            min_mass=min_mass,
            eps=eps,
        )

    return loss_fn
