"""Model architectures extracted from the notebook."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

from src import config


def build_direct_xy_cnn(input_shape: tuple[int, ...]) -> keras.Model:
    """Build the direct XY regression CNN from the notebook."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        16,
        kernel_size=(3, 5),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)

    x = layers.Conv2D(
        32,
        kernel_size=(3, 5),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(config.MAX_PEOPLE * 2, activation="sigmoid")(x)
    outputs = layers.Reshape((config.MAX_PEOPLE, 2))(x)

    return models.Model(inputs, outputs, name="direct_xy_cnn")


def build_super_simple_heatmap_cnn(
    input_shape: tuple[int, ...],
    heatmap_h: int = config.HEATMAP_H,
    heatmap_w: int = config.HEATMAP_W,
) -> keras.Model:
    """Build the notebook's dense heatmap CNN."""
    inputs = keras.Input(shape=input_shape, name="radar_input")

    x = layers.Conv2D(32, kernel_size=(3, 7), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(pool_size=(1, 2), name="pool1")(x)

    x = layers.Conv2D(64, kernel_size=(3, 5), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.20, name="dropout")(x)

    prior_prob = 0.01
    bias_init = tf.keras.initializers.Constant(
        -np.log((1.0 - prior_prob) / prior_prob)
    )

    x = layers.Dense(
        heatmap_h * heatmap_w,
        activation="sigmoid",
        bias_initializer=bias_init,
        name="dense_heatmap",
    )(x)

    outputs = layers.Reshape(
        (heatmap_h, heatmap_w, 1),
        name="heatmap",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="super_simple_radar_heatmap_cnn",
    )


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    name: str | None = None,
) -> tf.Tensor:
    """Build one notebook U-Net convolution block."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_conv1",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn1")(x)
    x = layers.ReLU(name=None if name is None else name + "_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_conv2",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn2")(x)
    x = layers.ReLU(name=None if name is None else name + "_relu2")(x)
    return x


def build_radar_heatmap_unet(
    input_shape: tuple[int, ...],
    heatmap_h: int = config.HEATMAP_H,
    heatmap_w: int = config.HEATMAP_W,
    prior_prob: float = 0.01,
) -> keras.Model:
    """Build the notebook's radar heatmap U-Net."""
    inputs = keras.Input(shape=input_shape, name="radar_input")

    c1 = conv_block(inputs, 32, kernel_size=(3, 7), name="enc1")
    p1 = layers.MaxPooling2D(pool_size=(1, 2), name="pool1")(c1)

    c2 = conv_block(p1, 64, kernel_size=(3, 5), name="enc2")
    p2 = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(c2)

    c3 = conv_block(p2, 128, kernel_size=(3, 3), name="enc3")
    p3 = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(c3)

    bottleneck = conv_block(p3, 256, kernel_size=(3, 3), name="bottleneck")

    context = layers.GlobalAveragePooling2D(name="gap_context")(bottleneck)
    context = layers.Dense(256, activation="relu", name="context_dense1")(context)
    context = layers.Dense(256, activation="sigmoid", name="context_gate")(context)
    context = layers.Reshape((1, 1, 256), name="context_reshape")(context)
    bottleneck = layers.Multiply(name="context_modulation")([bottleneck, context])

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up3")(bottleneck)
    x = layers.Concatenate(name="skip3")([x, c3])
    x = conv_block(x, 128, name="dec3")

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up2")(x)
    x = layers.Concatenate(name="skip2")([x, c2])
    x = conv_block(x, 64, name="dec2")

    x = layers.UpSampling2D(size=(1, 2), interpolation="bilinear", name="up1")(x)
    x = layers.Concatenate(name="skip1")([x, c1])
    x = conv_block(x, 32, name="dec1")

    x = layers.Resizing(
        heatmap_h,
        heatmap_w,
        interpolation="bilinear",
        name="resize_to_heatmap",
    )(x)

    bias_init = tf.keras.initializers.Constant(
        -np.log((1.0 - prior_prob) / prior_prob)
    )

    outputs = layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        bias_initializer=bias_init,
        name="heatmap",
    )(x)

    return keras.Model(inputs, outputs, name="radar_heatmap_unet")
