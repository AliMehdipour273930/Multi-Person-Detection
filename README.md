# Multi-Person Detection

This repository contains a Python refactor of the `notebooks/EEAI2_VERSION3.ipynb`
heatmap pipeline. The code keeps the notebook approach: radar I/Q input,
complex CIR conversion, selected channels `[0, 4, 7, 11, 12, 15]`, EMA
decluttering with `alpha=0.98`, motion magnitude, local temporal variance,
phase difference, `SEQ_LEN=16` temporal windows, Gaussian heatmap targets,
the notebook heatmap CNN/U-Net models, heatmap decoding, and Hungarian/F1
evaluation.

The hardcoded Hugging Face token found in the notebook was removed. Provide
the dataset directory yourself when training.

## Project Layout

- `notebooks/`: original exploratory notebooks.
- `src/config.py`: notebook constants.
- `src/data/dataset.py`: `.npz` loading, file-level split, and temporal windows.
- `src/features/preprocessing.py`: radar I/Q preprocessing and feature extraction.
- `src/features/heatmaps.py`: heatmap targets and heatmap decoding.
- `src/modeling/model.py`: notebook model architectures.
- `src/modeling/losses.py`: notebook losses.
- `src/evaluation/metrics.py`: Hungarian matching, F1 evaluation, and threshold sweep.
- `scripts/train.py`: runnable heatmap training pipeline.
- `scripts/export_tflite.py`: Keras to TFLite export.
- `submission/code.py`: self-contained TFLite inference script.

## Training

Run training from the repository root:

```powershell
python scripts/train.py --data-dir TRAIN_DATASET_ROOT --output-dir checkpoints
```

Use a Python environment with NumPy, SciPy, and TensorFlow installed.

The script writes:

- `checkpoints/best_model.keras`
- `checkpoints/final_heatmap_model_best_weights.keras`
- `checkpoints/norm_stats.npz`
- `submission/norm_stats.npz`
- split metadata, training history, and evaluation JSON files

Use `--skip-threshold-sweep` to skip the post-training threshold sweep, and
`--save-arrays` to save the prepared NumPy arrays.

For a quick end-to-end smoke test, cap the number of samples:

```powershell
python scripts/train.py --data-dir TRAIN_DATASET_ROOT --output-dir checkpoints/smoke --epochs 1 --skip-threshold-sweep --max-train-samples 128 --max-test-samples 64
```

## TFLite Export

Export a trained Keras model:

```powershell
python scripts/export_tflite.py --keras-model checkpoints/best_model.keras --output submission/model.tflite
```

This creates a real `submission/model.tflite` from the trained model. No fake
model or fake normalization file is included in this repository.

## Submission Inference

`submission/code.py` does not import `src` modules. It uses only NumPy, SciPy,
TensorFlow, and the standard library. It loads `model.tflite` and
`norm_stats.npz` from the `submission` folder.

Run inference on one `.npz` file or a directory of `.npz` files:

```powershell
python submission/code.py --input INPUT_NPZ_OR_DIR --output predictions.jsonl
```

For every input frame, the script builds a `SEQ_LEN=16` window ending at that
frame. Early frames are padded by repeating frame 0. Each JSONL row contains
the frame id, decoded localizations, and scores. Decoding keeps at most four
coordinates and clips them to `[0, 4.8] x [0, 7.2]`.

## Remaining Manual Checks

- Train on the real dataset and inspect validation F1.
- Export the trained Keras model to TFLite.
- Run `submission/code.py` with the exported model and generated
  `norm_stats.npz` on held-out `.npz` files.
