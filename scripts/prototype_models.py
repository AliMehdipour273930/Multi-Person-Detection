"""
Build small separable-conv U-Net variants, convert to TFLite with dynamic-range and (if possible) full-int8 quantization using random representative data, and save outputs for constraint checks.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / 'submission'
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SHAPE = (16, 55, 24)  # seq_len, bins, channels


def build_light_unet(input_shape=INPUT_SHAPE, base_filters=16, depth=3):
    inp = keras.Input(shape=input_shape, name='input')
    x = inp
    skips = []
    # encoder
    for d in range(depth):
        f = base_filters * (2 ** d)
        x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)
    # bottleneck
    f = base_filters * (2 ** depth)
    x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # decoder
    def _crop_to_target(skip, target):
        import tensorflow as _tf
        t_shape = _tf.shape(target)
        return skip[:, :t_shape[1], :t_shape[2], :]

    for d in reversed(range(depth)):
        f = base_filters * (2 ** d)
        x = layers.UpSampling2D(2)(x)
        skip = skips[d]
        # crop skip to match x spatial dims when odd sizes occur
        skip_cropped = layers.Lambda(lambda args: _crop_to_target(args[0], args[1]))([skip, x])
        x = layers.Concatenate()([x, skip_cropped])
        x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    # heatmap head
    x = layers.SeparableConv2D(base_filters, 3, padding='same', activation='relu')(x)
    out = layers.Conv2D(1, 1, activation='sigmoid', name='heatmap')(x)
    model = keras.Model(inp, out, name='light_unet')
    return model


def representative_gen(num_samples=100):
    # yield batches of shape (1, seq_len, bins, channels) with realistic dtype float32
    for _ in range(num_samples):
        data = np.random.normal(0.0, 1.0, size=(1,) + INPUT_SHAPE).astype(np.float32)
        yield [data]


if __name__ == '__main__':
    model = build_light_unet(base_filters=16, depth=1)
    model_path = OUT_DIR / 'prototype_light_unet.keras'
    model.save(model_path, include_optimizer=False)
    print('Saved keras model to', model_path)

    # Convert to TFLite - dynamic range quantization (weights quantized)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dr = OUT_DIR / 'prototype_light_dr.tflite'
    tflite_bytes = converter.convert()
    tflite_dr.write_bytes(tflite_bytes)
    print('Wrote', tflite_dr, tflite_dr.stat().st_size)

    # Try full integer quantization (may fail if unsupported ops)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_gen(100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_int8 = OUT_DIR / 'prototype_light_int8.tflite'
        tflite_bytes = converter.convert()
        tflite_int8.write_bytes(tflite_bytes)
        print('Wrote', tflite_int8, tflite_int8.stat().st_size)
    except Exception as e:
        print('Full int8 conversion failed:', e)

    print('Done')
