#!/usr/bin/env python3
"""
Compute OP (Overall Performance) and BWT (Backward Transfer) from TRACE inference results.

Reads results-{round}-{i_task}-{task}.json produced by inference/infer_single.py,
reconstructs the R matrix (round x task), and computes the paper's metrics (Section 3):

  OP  = (1/T) * sum_{i=0}^{T-1} R[T-1][i]                 # final-round average across tasks
  BWT = (1/(T-1)) * sum_{i=0}^{T-2} (R[T-1][i] - R[i][i]) # average backward transfer

Verified against paper Appendix Table 7 (Baichuan SeqFT): OP=0.434, BWT=-0.154.

Usage:
  python utils/aggregate_op_bwt.py --results_dir <dir> [--tasks a,b,c] [--out op_bwt.json]
"""
import os
import json
import argparse

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

DEFAULT_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150",
                 "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def extract_primary(eval_dict, task):
    key = PRIMARY_METRIC.get(task, "accuracy")
    if not isinstance(eval_dict, dict) or key not in eval_dict:
        return None
    val = eval_dict[key]
    # robustness: sari may be nested ({"sari": x} or [{"sari": x}]) from older code
    if isinstance(val, dict):
        return val.get("sari", val.get(key))
    if isinstance(val, (list, tuple)) and len(val) > 0:
        v = val[0]
        if isinstance(v, dict):
            return v.get("sari", v.get(key))
        return v
    return val


def load_results(results_dir, tasks):
    T = len(tasks)
    R = [[None] * T for _ in range(T)]
    missing = []
    for round_i in range(T):
        for task_i in range(round_i + 1):
            task = tasks[task_i]
            fn = os.path.join(results_dir, f"results-{round_i}-{task_i}-{task}.json")
            if not os.path.exists(fn):
                missing.append(fn)
                continue
            with open(fn) as f:
                data = json.load(f)
            R[round_i][task_i] = extract_primary(data.get("eval", {}), task)
    if missing:
        print(f"[WARN] {len(missing)} result files missing, e.g. {missing[0]}")
    return R


def compute_op_bwt(R):
    T = len(R)
    final = [R[T - 1][i] for i in range(T)]
    valid_final = [v for v in final if v is not None]
    OP = sum(valid_final) / len(valid_final) if valid_final else None
    bwt_terms = []
    for i in range(T - 1):
        if R[T - 1][i] is not None and R[i][i] is not None:
            bwt_terms.append(R[T - 1][i] - R[i][i])
    BWT = sum(bwt_terms) / len(bwt_terms) if bwt_terms else None
    return OP, BWT, final, bwt_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t]
    R = load_results(args.results_dir, tasks)
    OP, BWT, final, bwt_terms = compute_op_bwt(R)

    print("=" * 64)
    print("R matrix (round x task):")
    for i, row in enumerate(R):
        pretty = ["  --- " if v is None else f"{v:6.3f}" for v in row]
        print(f"  round {i}: [{''.join(pretty)} ]")
    print("-" * 64)
    print(f"final-round per-task: {[round(v,4) if v is not None else None for v in final]}")
    print(f"BWT terms            : {[round(v,4) for v in bwt_terms]}")
    print("-" * 64)
    print(f"OP  = {OP:.4f}" if OP is not None else "OP = None")
    print(f"BWT = {BWT:.4f}" if BWT is not None else "BWT = None")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"op": OP, "bwt": BWT, "R": R}, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
