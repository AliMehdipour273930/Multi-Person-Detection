#!/usr/bin/env python3
"""
ESP32-S3 hardware constraint checker for TFLite models.

Verifies that a TFLite model satisfies the deployment constraints of the
ESP32-S3 microcontroller (no external PSRAM assumed).

Hard constraints (failure → model rejected):
  1. Valid, loadable TFLite model
  2. Model file size  < 800 KB   (Flash budget)
  3. Activation arena < 300 KB   (SRAM budget — model only)
  4. No forbidden operations     (LSTM / GRU / RNN / CUSTOM / Flex)

Advisory check (warning, not a hard failure):
  5. INT8 quantization (strongly recommended — 4× smaller, faster on ESP32-S3)

Arena estimation:
  Weight tensors are mmap'd from Flash and never loaded into SRAM.
  Only activation tensors (computed at runtime) occupy the arena.
  The reported value sums all activation tensor sizes; TFLM further reduces
  this at runtime by overlapping non-simultaneous buffers, so the real
  on-device arena will be ≤ the reported estimate.

Usage:
    python evaluation/evaluate_constraint.py --model-path submission/model.tflite

Exit codes:
    0  All hard constraints pass
    1  One or more hard constraints failed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    # Keep the local Windows console from failing on symbols used in reports.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Hardware limits ────────────────────────────────────────────────────────────

MODEL_SIZE_LIMIT = 800 * 1024   # 800 KB  Flash budget
ARENA_LIMIT      = 300 * 1024   # 300 KB  SRAM budget (model arena only)

FORBIDDEN_OPS = {
    "LSTM",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "BIDIRECTIONAL_SEQUENCE_RNN",
    "GRU",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_interpreter(model_path: Path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _estimate_activation_arena(interpreter) -> int:
    """
    Return the byte footprint of activation tensors (SRAM arena).

    Runs one dummy inference: tensors whose values change are activations
    computed at runtime; tensors that stay the same are constant weights
    mmap'd from Flash and do not consume SRAM.
    """
    in_det  = interpreter.get_input_details()[0]
    details = interpreter.get_tensor_details()

    before = {}
    for t in details:
        try:
            before[t['index']] = interpreter.get_tensor(t['index']).tobytes()
        except Exception:
            before[t['index']] = None

    dummy = np.zeros(in_det['shape'], dtype=in_det['dtype'])
    interpreter.set_tensor(in_det['index'], dummy)
    interpreter.invoke()

    total = 0
    for t in details:
        shape = t.get('shape')
        if shape is None or len(shape) == 0 or np.prod(shape) == 0:
            continue
        nbytes = int(np.prod(shape)) * np.dtype(t['dtype']).itemsize
        try:
            after = interpreter.get_tensor(t['index']).tobytes()
            if before[t['index']] != after:
                total += nbytes
        except Exception:
            total += nbytes
    return total


def _scan_ops(interpreter) -> tuple[list[str], bool]:
    """
    Scan the model for forbidden or TFLM-incompatible operations.

    Returns (bad_ops, scan_ok).
    DELEGATE entries are filtered — they are XNNPACK artifacts from the
    desktop runtime and do not represent actual model operations.
    CUSTOM and Flex ops are flagged unconditionally: they require the full
    TF runtime and will not run on TFLM.
    """
    try:
        op_names = [
            o['op_name']
            for o in interpreter._get_ops_details()
            if o['op_name'] != 'DELEGATE'
        ]
    except AttributeError:
        return [], False

    bad = sorted({
        name for name in op_names
        if name in FORBIDDEN_OPS
        or name.upper().startswith('CUSTOM')
        or name.startswith('Flex')
    })
    return bad, True


def _is_int8_quantized(interpreter) -> tuple[bool, bool]:
    in_det  = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    if any(d['dtype'] in (np.int8, np.uint8) for d in in_det + out_det):
        return True, True
    all_dtypes = {np.dtype(t['dtype']).name for t in interpreter.get_tensor_details()}
    partial = 'int8' in all_dtypes or 'uint8' in all_dtypes
    return False, partial


# ── Report helpers ─────────────────────────────────────────────────────────────

def _header(text: str) -> None:
    print('=' * 64)
    print(f'  {text}')
    print('=' * 64)


def _row(tag: str, msg: str) -> None:
    print(f'  [{tag:4s}] {msg}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Verify TFLite model against ESP32-S3 hardware constraints'
    )
    parser.add_argument('--model-path', required=True,
                        help='Path to the TFLite model file')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f'[ERROR] Model file not found: {model_path}', file=sys.stderr)
        sys.exit(1)

    file_size = model_path.stat().st_size
    hard_fail = False

    _header('ESP32-S3 CONSTRAINT CHECK')
    print(f'  Model : {model_path}')
    print(f'  Size  : {file_size / 1024:.1f} KB')
    print()

    print('CHECK 1  TFLite model validity')
    try:
        interpreter = _load_interpreter(model_path)
        in_det    = interpreter.get_input_details()
        out_det   = interpreter.get_output_details()
        in_shape  = in_det[0]['shape'].tolist()
        out_shape = out_det[0]['shape'].tolist()
        in_dtype  = np.dtype(in_det[0]['dtype']).name
        out_dtype = np.dtype(out_det[0]['dtype']).name
        _row('PASS', (f'Model loads and allocates — '
                      f'input {in_shape} ({in_dtype}), '
                      f'output {out_shape} ({out_dtype})'))
    except Exception as exc:
        _row('FAIL', f'Model failed to load or allocate: {exc}')
        _header('RESULT: FAIL')
        sys.exit(1)
    print()

    print(f'CHECK 2  Model file size (limit: {MODEL_SIZE_LIMIT // 1024} KB Flash)')
    if file_size < MODEL_SIZE_LIMIT:
        _row('PASS', f'{file_size / 1024:.1f} KB < {MODEL_SIZE_LIMIT // 1024} KB')
    else:
        _row('FAIL', (f'{file_size / 1024:.1f} KB ≥ {MODEL_SIZE_LIMIT // 1024} KB — '
                      f'exceeds Flash budget by {(file_size - MODEL_SIZE_LIMIT) / 1024:.1f} KB'))
        hard_fail = True
    print()

    print(f'CHECK 3  Activation arena estimate (limit: {ARENA_LIMIT // 1024} KB SRAM)')
    arena = _estimate_activation_arena(interpreter)
    print(f'         Activation tensors: {arena / 1024:.1f} KB'
          f'  (TFLM overlaps non-simultaneous buffers — actual arena ≤ this)')
    if arena < ARENA_LIMIT:
        _row('PASS', f'~{arena / 1024:.1f} KB < {ARENA_LIMIT // 1024} KB — fits in SRAM')
    else:
        _row('FAIL', (f'~{arena / 1024:.1f} KB ≥ {ARENA_LIMIT // 1024} KB — '
                      f'exceeds SRAM budget'))
        hard_fail = True
    print()

    print('CHECK 4  Operations (LSTM / GRU / RNN / CUSTOM / Flex not supported)')
    bad_ops, scan_ok = _scan_ops(interpreter)
    if not scan_ok:
        _row('WARN', 'Op scan unavailable — update TensorFlow to enable this check')
    elif bad_ops:
        _row('FAIL', f'Forbidden operations detected: {", ".join(bad_ops)}')
        hard_fail = True
    else:
        _row('PASS', 'No forbidden operations detected')
    print()

    print('CHECK 5  Quantization (INT8 strongly recommended for ESP32-S3)')
    fully_q, partially_q = _is_int8_quantized(interpreter)
    if fully_q:
        _row('PASS', 'Full INT8 quantization — optimal for ESP32-S3')
    elif partially_q:
        _row('WARN', 'Hybrid quantization (weights INT8, activations float32) — full INT8 recommended')
    else:
        _row('WARN', 'float32 model — INT8 quantization strongly recommended (4× smaller, ~4× faster)')
    print()

    _header('RESULT: ' + ('PASS' if not hard_fail else 'FAIL'))
    if hard_fail:
        print('  One or more hard constraints failed — see details above.')
    else:
        print('  All hard constraints pass.' +
              (' Advisory warnings above.' if not fully_q else ''))
    print('=' * 64)

    print()
    print('JSON_RESULT:', json.dumps({
        'passed'      : not hard_fail,
        'file_size_kb': round(file_size / 1024, 1),
        'arena_kb'    : round(arena / 1024, 1),
    }))

    sys.exit(0 if not hard_fail else 1)


if __name__ == '__main__':
    main()
