"""Hungarian matching, F1 evaluation, and threshold sweeping."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from src import config
from src.features.heatmaps import gt_to_localizations, heatmap_to_localizations


def evaluate_with_hungarian(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_mask: np.ndarray,
) -> np.ndarray:
    """Evaluate direct coordinate predictions with Hungarian matching."""
    all_errors = []

    for sample_idx in range(len(y_pred)):
        true_valid = y_true[sample_idx][y_mask[sample_idx] == 1]
        pred_valid = y_pred[sample_idx][: len(true_valid)]

        if len(true_valid) == 0:
            continue

        cost = np.zeros((len(pred_valid), len(true_valid)), dtype=np.float32)

        for pred_idx in range(len(pred_valid)):
            for true_idx in range(len(true_valid)):
                cost[pred_idx, true_idx] = np.linalg.norm(
                    pred_valid[pred_idx] - true_valid[true_idx]
                )

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_errors = cost[row_ind, col_ind]
        all_errors.extend(matched_errors)

    return np.array(all_errors)


def cost_matrix(gt: list[list[float]], pred: list[list[float]]) -> np.ndarray:
    """Build a Euclidean distance matrix from GT and predicted points."""
    gt_arr = np.array(gt, dtype=np.float64)
    pred_arr = np.array(pred, dtype=np.float64)

    diff = gt_arr[:, None, :] - pred_arr[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def match_hungarian(
    gt: list[list[float]],
    pred: list[list[float]],
    threshold: float = config.MATCH_THRESHOLD,
) -> tuple[list[float], int, int]:
    """
    Match predictions to GT with the Hungarian algorithm.

    Returns true-positive distances, false negatives, and false positives.
    """
    n_gt = len(gt)
    n_pred = len(pred)

    if n_gt == 0 and n_pred == 0:
        return [], 0, 0

    if n_gt == 0:
        return [], 0, n_pred

    if n_pred == 0:
        return [], n_gt, 0

    cost = cost_matrix(gt, pred)
    row_i, col_i = linear_sum_assignment(cost)

    matched_gt = set()
    matched_pred = set()
    distances = []

    for row, col in zip(row_i, col_i):
        distance = float(cost[row, col])

        if distance <= threshold:
            distances.append(distance)
            matched_gt.add(row)
            matched_pred.add(col)

    n_fn = n_gt - len(matched_gt)
    n_fp = n_pred - len(matched_pred)

    return distances, n_fn, n_fp


def evaluate_official_style_from_heatmaps(
    y_pred_heat: np.ndarray,
    y_xy: np.ndarray,
    y_mask: np.ndarray,
    threshold: float = config.HEATMAP_THRESHOLD,
    nms_size: int = config.NMS_SIZE,
    match_threshold: float = config.MATCH_THRESHOLD,
    max_persons: int = config.MAX_PERSONS,
    verbose: bool = True,
) -> dict:
    """Compute the notebook official-style global F1 metrics."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_tp_distances = []
    count_abs_errors = []
    count_exact = 0

    gt_records = {}
    pred_records = {}

    for frame_id in range(len(y_pred_heat)):
        gt_locs = gt_to_localizations(y_xy[frame_id], y_mask[frame_id])
        pred_locs, _scores = heatmap_to_localizations(
            y_pred_heat[frame_id],
            threshold=threshold,
            nms_size=nms_size,
            max_persons=max_persons,
            room_x=config.ROOM_X,
            room_y=config.ROOM_Y,
        )

        gt_records[frame_id] = gt_locs
        pred_records[frame_id] = pred_locs

        distances, n_fn, n_fp = match_hungarian(
            gt_locs,
            pred_locs,
            threshold=match_threshold,
        )

        total_tp += len(distances)
        total_fp += n_fp
        total_fn += n_fn

        all_tp_distances.extend(distances)
        count_abs_errors.append(abs(len(pred_locs) - len(gt_locs)))

        if len(pred_locs) == len(gt_locs):
            count_exact += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if len(all_tp_distances) > 0:
        distances_arr = np.array(all_tp_distances, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(distances_arr**2)))
        mae = float(np.mean(distances_arr))
        median = float(np.median(distances_arr))
        p90 = float(np.percentile(distances_arr, 90))
        n_matches = len(distances_arr)
    else:
        rmse = None
        mae = None
        median = None
        p90 = None
        n_matches = 0

    count_mae = float(np.mean(count_abs_errors)) if len(count_abs_errors) > 0 else 0.0
    count_accuracy = count_exact / len(y_pred_heat) if len(y_pred_heat) > 0 else 0.0

    results = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "rmse": rmse,
        "mae": mae,
        "median": median,
        "p90": p90,
        "matched_pairs": n_matches,
        "count_mae": count_mae,
        "count_accuracy": count_accuracy,
        "threshold": threshold,
        "nms_size": nms_size,
        "match_threshold": match_threshold,
        "gt_records": gt_records,
        "pred_records": pred_records,
    }

    if verbose:
        print("=" * 62)
        print("  DETECTION SCORE - OFFICIAL-STYLE F1")
        print("=" * 62)
        print(f"  Heatmap threshold  : {threshold}")
        print(f"  NMS size           : {nms_size}")
        print(f"  Matching threshold : {match_threshold} m")
        print(f"  Frames evaluated   : {len(y_pred_heat)}")
        print(f"  TP / FP / FN       : {total_tp} / {total_fp} / {total_fn}")
        print(f"  Precision          : {precision:.4f}")
        print(f"  Recall             : {recall:.4f}")
        print(f"  F1 Score           : {f1:.4f}")
        print()

        print("-" * 62)
        print("LOCALIZATION ERROR  (matched pairs only)")
        print("-" * 62)

        if n_matches > 0:
            print(f"  Matched pairs      : {n_matches}")
            print(f"  RMSE               : {rmse:.4f} m")
            print(f"  MAE                : {mae:.4f} m")
            print(f"  Median error       : {median:.4f} m")
            print(f"  P90 error          : {p90:.4f} m")
        else:
            print("  No matches found; cannot compute localization error.")

        print()
        print("-" * 62)
        print("PERSON COUNT")
        print("-" * 62)
        print(f"  Count MAE          : {count_mae:.4f} persons / frame")
        print(f"  Count accuracy     : {count_accuracy:.2%}  ({count_exact} / {len(y_pred_heat)} frames)")
        print("=" * 62)

        print(
            "JSON_RESULT:",
            json.dumps(
                {
                    "f1": round(f1, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "rmse": round(rmse, 4) if rmse is not None else None,
                }
            ),
        )

    return results


def sweep_thresholds_official_style(
    y_pred_heat: np.ndarray,
    y_xy: np.ndarray,
    y_mask: np.ndarray,
    thresholds: np.ndarray = np.arange(0.05, 0.81, 0.05),
    nms_size: int = 5,
    match_threshold: float = config.MATCH_THRESHOLD,
    max_persons: int = config.MAX_PERSONS,
) -> tuple[list[dict], dict]:
    """Run the notebook threshold sweep and return all results plus the best row."""
    sweep = []

    for threshold in thresholds:
        result = evaluate_official_style_from_heatmaps(
            y_pred_heat,
            y_xy,
            y_mask,
            threshold=float(threshold),
            nms_size=nms_size,
            match_threshold=match_threshold,
            max_persons=max_persons,
            verbose=False,
        )

        sweep.append(result)

        print(
            f"threshold={threshold:.2f} | "
            f"F1={result['f1']:.4f} | "
            f"P={result['precision']:.4f} | "
            f"R={result['recall']:.4f} | "
            f"TP={result['tp']} FP={result['fp']} FN={result['fn']} | "
            f"Count MAE={result['count_mae']:.3f}"
        )

    best = max(sweep, key=lambda row: row["f1"])

    print("\n===== BEST THRESHOLD =====")
    print(f"Best threshold : {best['threshold']}")
    print(f"Best F1        : {best['f1']:.4f}")
    print(f"Precision      : {best['precision']:.4f}")
    print(f"Recall         : {best['recall']:.4f}")
    print(f"TP / FP / FN   : {best['tp']} / {best['fp']} / {best['fn']}")

    return sweep, best
