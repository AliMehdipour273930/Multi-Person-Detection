"""Train the extracted notebook heatmap pipeline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow import keras

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.data.dataset import (
    apply_normalization,
    build_dataset_from_files,
    compute_normalization_stats,
    find_npz_files,
    split_files_by_people_count,
    summarize_split,
)
from src.evaluation.metrics import evaluate_official_style_from_heatmaps, sweep_thresholds_official_style
from src.features.heatmaps import build_heatmaps
from src.features.preprocessing import FINAL_SELECTED_CHANNELS
from src.modeling.losses import stable_focal_dice_mass_loss
from src.modeling.model import build_radar_heatmap_unet


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing notebook-format .npz files.")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for trained models and metadata.")
    parser.add_argument(
        "--submission-dir",
        default="submission",
        help="Directory where norm_stats.npz is also written for submission/code.py.",
    )
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--skip-threshold-sweep", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    return parser.parse_args()


def save_history(history: keras.callbacks.History, output_path: Path) -> None:
    """Save Keras history values as arrays."""
    arrays = {key: np.asarray(value) for key, value in history.history.items()}
    np.savez(output_path, **arrays)


class ArrayBatchSequence(keras.utils.Sequence):
    """Serve NumPy arrays to Keras in batches without copying the full dataset."""

    def __init__(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray | None = None,
        batch_size: int = config.TRAIN_BATCH_SIZE,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.x_values = x_values
        self.y_values = y_values
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(x_values))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Return the number of batches."""
        return int(np.ceil(len(self.x_values) / self.batch_size))

    def __getitem__(self, batch_idx: int):
        """Return one input batch, optionally paired with targets."""
        batch_indices = self.indices[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        x_batch = self.x_values[batch_indices]

        if self.y_values is None:
            return x_batch

        return x_batch, self.y_values[batch_indices]

    def on_epoch_end(self) -> None:
        """Shuffle sample indices between epochs when requested."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def save_split_metadata(
    train_files: list[Path],
    test_files: list[Path],
    output_path: Path,
) -> None:
    """Save split file names and people-count histograms."""
    metadata = {
        "train_files": [str(path) for path in train_files],
        "test_files": [str(path) for path in test_files],
        "train_hist": summarize_split(train_files),
        "test_hist": summarize_split(test_files),
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Run the notebook training pipeline."""
    args = parse_args()

    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    tf.random.set_seed(config.RANDOM_STATE)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_dir = Path(args.submission_dir) if args.submission_dir else None
    if submission_dir is not None:
        submission_dir.mkdir(parents=True, exist_ok=True)

    npz_files = find_npz_files(args.data_dir)
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found under {args.data_dir}")

    print("Selected channels:", FINAL_SELECTED_CHANNELS)
    print("Total acquisition files:", len(npz_files))

    train_files, test_files = split_files_by_people_count(npz_files)
    print("Train files:", len(train_files), summarize_split(train_files))
    print("Test files :", len(test_files), summarize_split(test_files))
    save_split_metadata(train_files, test_files, output_dir / "split_metadata.json")

    train_data = build_dataset_from_files(
        train_files,
        seq_len=config.SEQ_LEN,
        frame_stride=config.FRAME_STRIDE,
    )
    test_data = build_dataset_from_files(
        test_files,
        seq_len=config.SEQ_LEN,
        frame_stride=config.FRAME_STRIDE,
    )

    x_train = train_data["X"]
    y_xy_train = train_data["y_xy"]
    y_mask_train = train_data["y_mask"]

    x_test = test_data["X"]
    y_xy_test = test_data["y_xy"]
    y_mask_test = test_data["y_mask"]

    if args.max_train_samples is not None:
        max_train_samples = max(0, args.max_train_samples)
        x_train = x_train[:max_train_samples]
        y_xy_train = y_xy_train[:max_train_samples]
        y_mask_train = y_mask_train[:max_train_samples]

    if args.max_test_samples is not None:
        max_test_samples = max(0, args.max_test_samples)
        x_test = x_test[:max_test_samples]
        y_xy_test = y_xy_test[:max_test_samples]
        y_mask_test = y_mask_test[:max_test_samples]

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Train and test splits must both contain at least one sample.")

    mean, std = compute_normalization_stats(x_train)
    x_train = apply_normalization(x_train, mean, std)
    x_test = apply_normalization(x_test, mean, std)

    norm_stats_path = output_dir / "norm_stats.npz"
    np.savez(norm_stats_path, mean=mean, std=std)
    print("Saved normalization stats:", norm_stats_path)

    if submission_dir is not None:
        submission_stats_path = submission_dir / "norm_stats.npz"
        np.savez(submission_stats_path, mean=mean, std=std)
        print("Saved submission normalization stats:", submission_stats_path)

    if args.save_arrays:
        np.save(output_dir / "X_train.npy", x_train)
        np.save(output_dir / "y_xy_train.npy", y_xy_train)
        np.save(output_dir / "y_mask_train.npy", y_mask_train)
        np.save(output_dir / "X_test.npy", x_test)
        np.save(output_dir / "y_xy_test.npy", y_xy_test)
        np.save(output_dir / "y_mask_test.npy", y_mask_test)
        np.save(output_dir / "train_frame_idx.npy", train_data["frame_idx"])
        np.save(output_dir / "test_frame_idx.npy", test_data["frame_idx"])
        np.save(output_dir / "train_window_name.npy", train_data["window_name"])
        np.save(output_dir / "test_window_name.npy", test_data["window_name"])

    y_heat_train = build_heatmaps(
        y_xy_train,
        y_mask_train,
        H=config.HEATMAP_H,
        W=config.HEATMAP_W,
        sigma=config.HEATMAP_SIGMA,
    )
    y_heat_test = build_heatmaps(
        y_xy_test,
        y_mask_test,
        H=config.HEATMAP_H,
        W=config.HEATMAP_W,
        sigma=config.HEATMAP_SIGMA,
    )

    print("X_train:", x_train.shape)
    print("y_heat_train:", y_heat_train.shape)
    print("X_test:", x_test.shape)
    print("y_heat_test:", y_heat_test.shape)

    model = build_radar_heatmap_unet(
        input_shape=x_train.shape[1:],
        heatmap_h=config.HEATMAP_H,
        heatmap_w=config.HEATMAP_W,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=stable_focal_dice_mass_loss,
        metrics=["mae"],
    )

    best_model_path = output_dir / "best_model.keras"
    final_model_path = output_dir / "final_heatmap_model_best_weights.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]

    train_sequence = ArrayBatchSequence(
        x_train,
        y_heat_train,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_sequence = ArrayBatchSequence(
        x_test,
        y_heat_test,
        batch_size=args.batch_size,
        shuffle=False,
    )

    history = model.fit(
        train_sequence,
        validation_data=test_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    save_history(history, output_dir / "training_history.npz")
    model.save(final_model_path)
    print("Saved final model:", final_model_path)

    predict_sequence = ArrayBatchSequence(
        x_test,
        batch_size=args.batch_size,
        shuffle=False,
    )
    y_pred_test = model.predict(predict_sequence)

    evaluate_official_style_from_heatmaps(
        y_pred_test,
        y_xy_test,
        y_mask_test,
        threshold=config.HEATMAP_THRESHOLD,
        nms_size=config.NMS_SIZE,
        match_threshold=config.MATCH_THRESHOLD,
        max_persons=config.MAX_PERSONS,
        verbose=True,
    )

    if not args.skip_threshold_sweep:
        sweep_results, best_result = sweep_thresholds_official_style(
            y_pred_test,
            y_xy_test,
            y_mask_test,
            thresholds=np.arange(0.05, 0.99, 0.05),
            nms_size=4,
            match_threshold=config.MATCH_THRESHOLD,
            max_persons=config.MAX_PERSONS,
        )
        compact_sweep = [
            {
                "threshold": result["threshold"],
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "count_mae": result["count_mae"],
            }
            for result in sweep_results
        ]
        (output_dir / "threshold_sweep.json").write_text(
            json.dumps({"results": compact_sweep, "best": best_result["threshold"]}, indent=2),
            encoding="utf-8",
        )

        final_results = evaluate_official_style_from_heatmaps(
            y_pred_test,
            y_xy_test,
            y_mask_test,
            threshold=best_result["threshold"],
            nms_size=config.NMS_SIZE,
            match_threshold=config.MATCH_THRESHOLD,
            max_persons=config.MAX_PERSONS,
            verbose=True,
        )
        (output_dir / "final_metrics.json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in final_results.items()
                    if key not in {"gt_records", "pred_records"}
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
