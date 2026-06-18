#!/usr/bin/env python3
"""
Performance evaluator for multi-person UWB radar localization.

Compares predicted localizations against the ground-truth and reports
an F1-based detection score plus additional diagnostic metrics.

Usage:
    python evaluation/evaluate_performance.py \
        --gt-path   evaluation/example/output_test.jsonl \
        --pred-path evaluation/example/my_output.jsonl

--- Scoring — F1 Score ---

For each frame, predicted positions are matched to GT persons via the
Hungarian algorithm (minimum-cost assignment on Euclidean distances).

  TP : predicted position matched to a GT person within 1.0 m
  FP : predicted position with no GT match within 1.0 m
  FN : GT person not matched to any prediction within 1.0 m

  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 × Precision × Recall / (Precision + Recall)

--- Additional metrics (informational) ---
  RMSE, MAE, Median error, P90  on TP matched pairs only
  Count MAE, Count accuracy     on person count per frame
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

if hasattr(sys.stdout, "reconfigure"):
    # Keep the local Windows console from failing on symbols used in reports.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ─────────────────────────────────────────────────────────────────

MATCH_THRESHOLD = 1.0   # metres — a prediction is a TP only if distance ≤ this
ROOM_X_MAX      = 4.8   # metres
ROOM_Y_MAX      = 7.2   # metres
MAX_PERSONS     = 4


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multi-person localization predictions against ground truth"
    )
    p.add_argument("--gt-path",   required=True,
                   help="Ground-truth .jsonl file")
    p.add_argument("--pred-path", required=True,
                   help="Predictions .jsonl file produced by code.py")
    return p.parse_args()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_predictions(pred_path: Path) -> list[str]:
    """Return a list of format errors; empty list means the file is valid."""
    errors: list[str] = []
    seen_frames: dict[int, int] = {}

    with open(pred_path) as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: invalid JSON — {exc}")
                continue

            if "frame" not in obj:
                errors.append(f"Line {lineno}: missing key 'frame'")
                continue
            try:
                frame = int(obj["frame"])
            except (TypeError, ValueError):
                errors.append(f"Line {lineno}: 'frame' must be an integer, got {obj['frame']!r}")
                continue

            if frame < 0:
                errors.append(f"Line {lineno}: 'frame' must be non-negative, got {frame}")
            if frame in seen_frames:
                errors.append(
                    f"Line {lineno}: duplicate frame {frame} "
                    f"(first at line {seen_frames[frame]})"
                )
            else:
                seen_frames[frame] = lineno

            if "localizations" not in obj:
                errors.append(f"Line {lineno} (frame {frame}): missing key 'localizations'")
                continue
            locs = obj["localizations"]
            if not isinstance(locs, list):
                errors.append(f"Line {lineno} (frame {frame}): 'localizations' must be a list")
                continue
            if len(locs) > MAX_PERSONS:
                errors.append(
                    f"Line {lineno} (frame {frame}): "
                    f"{len(locs)} localizations exceed max {MAX_PERSONS}"
                )

            for i, loc in enumerate(locs):
                if not isinstance(loc, (list, tuple)) or len(loc) != 2:
                    errors.append(
                        f"Line {lineno} (frame {frame}), loc {i}: "
                        f"expected [x, y], got {loc!r}"
                    )
                    continue
                try:
                    x, y = float(loc[0]), float(loc[1])
                except (TypeError, ValueError):
                    errors.append(
                        f"Line {lineno} (frame {frame}), loc {i}: coordinates must be numeric"
                    )
                    continue
                if not (0.0 <= x <= ROOM_X_MAX):
                    errors.append(
                        f"Line {lineno} (frame {frame}), loc {i}: "
                        f"x={x} outside room bounds [0, {ROOM_X_MAX}]"
                    )
                if not (0.0 <= y <= ROOM_Y_MAX):
                    errors.append(
                        f"Line {lineno} (frame {frame}), loc {i}: "
                        f"y={y} outside room bounds [0, {ROOM_Y_MAX}]"
                    )

    return errors


def load_jsonl(path: Path) -> dict[int, list]:
    """Parse a .jsonl file into {frame_id: [[x, y], ...]}."""
    records: dict[int, list] = {}
    with open(path) as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {lineno} of {path}: {exc}") from exc
            frame = int(obj["frame"])
            if frame not in records:
                locs = obj.get("localizations", [])
                records[frame] = [[float(xy[0]), float(xy[1])] for xy in locs]
    return records


# ── Matching ──────────────────────────────────────────────────────────────────

def _cost_matrix(gt: list, pred: list) -> np.ndarray:
    gt_arr   = np.array(gt,   dtype=np.float64)
    pred_arr = np.array(pred, dtype=np.float64)
    diff = gt_arr[:, np.newaxis, :] - pred_arr[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def match_hungarian(gt: list, pred: list, threshold: float) -> tuple[list, int, int]:
    """Hungarian matching → (tp_distances, n_fn, n_fp)."""
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 and n_pred == 0:
        return [], 0, 0
    if n_gt == 0:
        return [], 0, n_pred
    if n_pred == 0:
        return [], n_gt, 0

    cost = _cost_matrix(gt, pred)
    row_i, col_i = linear_sum_assignment(cost)

    matched_gt, matched_pred, distances = set(), set(), []
    for r, c in zip(row_i, col_i):
        d = float(cost[r, c])
        if d <= threshold:
            distances.append(d)
            matched_gt.add(r)
            matched_pred.add(c)

    return distances, n_gt - len(matched_gt), n_pred - len(matched_pred)


# ── Report helpers ────────────────────────────────────────────────────────────

def _header(text: str) -> None:
    print("=" * 62)
    print(f"  {text}")
    print("=" * 62)


def _rule() -> None:
    print("─" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    gt_path   = Path(args.gt_path)
    pred_path = Path(args.pred_path)

    for path, label in [(gt_path, "ground-truth"), (pred_path, "predictions")]:
        if not path.exists():
            print(f"[ERROR] {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    errors = validate_predictions(pred_path)
    if errors:
        print(f"Output validation FAILED — {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print("Output validation PASSED.")

    print("Loading files...")
    gt_all   = load_jsonl(gt_path)
    pred_all = load_jsonl(pred_path)
    print(f"  Ground truth : {len(gt_all)} frames   ({gt_path.name})")
    print(f"  Predictions  : {len(pred_all)} frames   ({pred_path.name})")

    common_frames    = sorted(set(gt_all) & set(pred_all))
    gt_only_frames   = sorted(set(gt_all) - set(pred_all))
    pred_only_frames = sorted(set(pred_all) - set(gt_all))

    if not common_frames and not gt_only_frames:
        print("[WARNING] No evaluable frames found — reporting zero scores.")
    if not common_frames:
        print("[WARNING] No frames in common between GT and predictions.")

    print(f"  Common frames                  : {len(common_frames)}")
    if gt_only_frames:
        print(f"  GT frames missing predictions  : {len(gt_only_frames)} (treated as all-FN)")
    if pred_only_frames:
        print(f"  Extra predicted frames (no GT) : {len(pred_only_frames)} (ignored)")
    print()

    total_tp, total_fp, total_fn = 0, 0, 0
    all_tp_distances: list[float] = []
    count_abs_errors: list[int]   = []
    count_exact = 0

    for frame in common_frames:
        gt   = gt_all[frame]
        pred = pred_all[frame]
        n_gt, n_pred = len(gt), len(pred)

        dists, n_fn, n_fp = match_hungarian(gt, pred, MATCH_THRESHOLD)
        total_tp += len(dists)
        total_fp += n_fp
        total_fn += n_fn
        all_tp_distances.extend(dists)
        count_abs_errors.append(abs(n_pred - n_gt))
        if n_pred == n_gt:
            count_exact += 1

    for frame in gt_only_frames:
        n_gt = len(gt_all[frame])
        total_fn += n_gt
        count_abs_errors.append(n_gt)

    n_eval_frames = len(common_frames) + len(gt_only_frames)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if all_tp_distances:
        d         = np.array(all_tp_distances, dtype=np.float64)
        rmse      = float(np.sqrt(np.mean(d ** 2)))
        mae       = float(np.mean(d))
        median    = float(np.median(d))
        p90       = float(np.percentile(d, 90))
        n_matches = len(d)
    else:
        rmse = mae = median = p90 = None
        n_matches = 0

    count_mae = float(np.mean(count_abs_errors)) if count_abs_errors else 0.0
    count_acc = count_exact / len(common_frames) if common_frames else 0.0

    _header("DETECTION SCORE — F1")
    print(f"  Matching threshold  : {MATCH_THRESHOLD} m")
    print(f"  Frames evaluated    : {n_eval_frames}")
    print(f"  TP / FP / FN        : {total_tp} / {total_fp} / {total_fn}")
    print(f"  Precision           : {precision:.4f}")
    print(f"  Recall              : {recall:.4f}")
    print(f"  F1 Score            : {f1:.4f}")
    print()

    _rule()
    print("LOCALISATION ERROR  (matched pairs only)")
    _rule()
    if n_matches > 0:
        print(f"  Matched pairs       : {n_matches}")
        print(f"  RMSE                : {rmse:.4f} m")
        print(f"  MAE                 : {mae:.4f} m")
        print(f"  Median error        : {median:.4f} m")
        print(f"  P90 error           : {p90:.4f} m")
    else:
        print("  No matches found — cannot compute localisation error.")
    print()

    _rule()
    print("PERSON COUNT")
    _rule()
    print(f"  Count MAE           : {count_mae:.4f} persons / frame")
    print(f"  Count accuracy      : {count_acc:.2%}  ({count_exact} / {len(common_frames)} frames)")
    print("=" * 62)

    print("JSON_RESULT:", json.dumps({
        "f1"       : round(f1,        4),
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "rmse"     : round(rmse,      4) if rmse is not None else None,
    }))


if __name__ == "__main__":
    main()
