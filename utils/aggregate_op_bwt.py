#!/usr/bin/env python3
"""
Compute OP (Overall Performance) and BWT (Backward Transfer) from TRACE inference results.

Reads results-{round}-{i_task}-{task}.json produced by inference/infer_single.py,
reconstructs the R matrix (round x task), and computes the paper's metrics (Section 3):

  OP  = (1/T) * sum_{i=0}^{T-1} R[T-1][i]                 # final-round average across tasks
  BWT = (1/(T-1)) * sum_{i=0}^{T-2} (R[T-1][i] - R[i][i]) # average backward transfer

Verified against paper Appendix Table 7 (Baichuan SeqFT): OP=0.434, BWT=-0.154.

STRICT MODE (default): the full lower-triangular R matrix (36 files for T=8) must be
present, parseable, contain each task's primary metric, and the normalized metric must
be finite and in [0,1]. Any failure exits non-zero and does NOT write op_bwt.json.
This prevents a partially-inferred run from being mistaken for a complete one (which
would otherwise trigger checkpoint cleanup downstream). Use --no-strict to override.

Usage:
  python utils/aggregate_op_bwt.py --results_dir <dir> [--tasks a,b,c] [--out op_bwt.json] [--no-strict]
"""
import os
import sys
import json
import math
import argparse
from typing import List, Optional

# primary metric per task (paper Table 4 "Metric" column)
PRIMARY_METRIC = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge-L",
    "Py150": "similarity",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
    "20Minuten": "sari",
}

# tasks whose primary metric is on a 0-100 scale (similarity / SARI); normalize to 0-1 for OP/BWT
SCALE_100 = {"Py150", "20Minuten"}

DEFAULT_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150",
                 "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def extract_primary(eval_dict, task):
    key = PRIMARY_METRIC.get(task, "accuracy")
    if not isinstance(eval_dict, dict) or key not in eval_dict:
        return None
    val = eval_dict[key]
    # robustness: sari may be nested ({"sari": x} or [{"sari": x}]) from older code
    if isinstance(val, dict):
        val = val.get("sari", val.get(key))
    elif isinstance(val, (list, tuple)) and len(val) > 0:
        v = val[0]
        val = v.get("sari", v.get(key)) if isinstance(v, dict) else v
    # normalize 0-100 metrics (similarity / SARI) to 0-1 so OP/BWT are comparable
    if task in SCALE_100 and isinstance(val, (int, float)):
        val = val / 100.0
    return val


def load_results(results_dir: str, tasks: List[str]):
    """Load the full lower-triangular R matrix, recording every missing/parseable/
    out-of-range entry. Returns (R, problems) where problems is a list of error strings."""
    T = len(tasks)
    R: List[List[Optional[float]]] = [[None] * T for _ in range(T)]
    problems: List[str] = []
    for round_i in range(T):
        for task_i in range(round_i + 1):
            task = tasks[task_i]
            fn = os.path.join(results_dir, f"results-{round_i}-{task_i}-{task}.json")
            if not os.path.exists(fn):
                problems.append(f"missing file: {fn}")
                continue
            try:
                with open(fn) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                problems.append(f"unparseable: {fn} ({e})")
                continue
            val = extract_primary(data.get("eval", {}), task)
            if val is None:
                problems.append(f"missing metric '{PRIMARY_METRIC[task]}': {fn}")
                continue
            if not isinstance(val, (int, float)) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                problems.append(f"metric out of range ({val}): {fn}")
                continue
            R[round_i][task_i] = float(val)
    return R, problems


def compute_op_bwt(R: List[List[Optional[float]]]):
    """Compute OP/BWT. In strict mode the final row and diagonal are complete, so the
    results are always finite floats; in --no-strict mode None entries are skipped and
    OP/BWT may be None if there is nothing to average over."""
    T = len(R)
    final: List[Optional[float]] = [R[T - 1][i] for i in range(T)]
    valid_final = [v for v in final if v is not None]
    OP = sum(valid_final) / len(valid_final) if valid_final else None
    bwt_terms: List[float] = []
    for i in range(T - 1):
        if R[T - 1][i] is not None and R[i][i] is not None:
            bwt_terms.append(R[T - 1][i] - R[i][i])  # type: ignore[operator]
    BWT = sum(bwt_terms) / len(bwt_terms) if bwt_terms else None
    return OP, BWT, final, bwt_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-strict", action="store_true",
                    help="disable strict completeness validation (NOT recommended)")
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t]
    T = len(tasks)

    R, problems = load_results(args.results_dir, tasks)

    if problems:
        print(f"[ERROR] strict validation failed: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        if not args.no_strict:
            print("[ERROR] refusing to write op_bwt.json (incomplete/corrupt results)")
            sys.exit(1)
        print("[WARN] --no-strict set: proceeding with incomplete matrix (results may be wrong)")

    # BWT / OP require the final row and the main diagonal to be fully present
    final_missing = [i for i in range(T) if R[T - 1][i] is None]
    diag_missing = [i for i in range(T - 1) if R[i][i] is None]
    if (final_missing or diag_missing) and not args.no_strict:
        print(f"[ERROR] final row missing tasks {final_missing}; diagonal missing rounds {diag_missing}")
        print("[ERROR] refusing to write op_bwt.json")
        sys.exit(1)

    OP, BWT, final, bwt_terms = compute_op_bwt(R)

    print("=" * 64)
    print("R matrix (round x task):")
    for i, row in enumerate(R):
        pretty = ["  --- " if v is None else f"{v:6.3f}" for v in row]
        print(f"  round {i}: [{''.join(pretty)} ]")
    print("-" * 64)
    print(f"final-round per-task (8 values used for OP): {[round(v, 4) if v is not None else None for v in final]}")
    print(f"BWT terms (7 values used for BWT)          : {[round(v, 4) for v in bwt_terms]}")
    print("-" * 64)
    print(f"OP  = {OP:.4f}" if OP is not None else "OP  = None (incomplete)")
    print(f"BWT = {BWT:.4f}" if BWT is not None else "BWT = None (incomplete)")

    if args.out:
        tmp = f"{args.out}.tmp"
        with open(tmp, "w") as f:
            json.dump({"op": OP, "bwt": BWT, "R": R,
                       "final": final, "bwt_terms": bwt_terms}, f, indent=2)
        os.replace(tmp, args.out)  # atomic write: never leave a half-written op_bwt.json
        # 回读校验：确认落盘文件可解析且 op/bwt 与本次计算结果一致，防止静默写坏
        try:
            with open(args.out) as f:
                chk = json.load(f)
            if chk.get("op") != OP or chk.get("bwt") != BWT:
                raise ValueError(f"回读不一致: {chk.get('op')} / {chk.get('bwt')}")
        except Exception as e:
            print(f"[ERROR] op_bwt.json 回读校验失败: {e}")
            sys.exit(1)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
