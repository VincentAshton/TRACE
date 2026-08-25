#!/usr/bin/env python3
"""
Collect OP/BWT across all (model, ratio) runs and compare against the 10% baseline
to locate the "drop threshold" (the replay ratio below which OP/BWT falls below 10%).

Baseline policy (option B): the 10% ratio is run in the SAME environment (ratio_0.10),
so it is the PRIMARY reference. The paper's published 10% numbers are shown ONLY as a
secondary reference to gauge environment offset — they are NEVER used as the baseline
for threshold judgement. If a model's own 10% run is missing, NO threshold is computed
for that model (avoids silently mixing paper numbers into the comparison).

Paper Table 1, Replay column (raw fraction):
  llama2-7b-chat   op=0.555 bwt=0.026
  vicuna-7b        op=0.553 bwt=0.002
  baichuan2-7b     op=0.517 bwt=0.011

Usage: python scripts/summary.py --out_root <dir> [--eps 0.005]
"""
import os
import json
import argparse

PAPER = {
    "llama2-7b-chat": {"op": 0.555, "bwt": 0.026},
    "vicuna-7b": {"op": 0.553, "bwt": 0.002},
    "baichuan2-7b": {"op": 0.517, "bwt": 0.011},
}

BASELINE_RATIO = "0.10"
# descending (closest to 10% first) => first "below" ratio is the drop threshold
RATIO_ORDER = ["0.08", "0.05", "0.02", "0.01"]
EPS_DEFAULT = 0.005  # tolerance for "below baseline" (0.5 percentage point)


def load_run(out_root, model, ratio):
    """Load a run's op_bwt.json. (Note: once the run-manifest / .complete flow lands,
    this should read from the verified .complete marker instead of a bare op_bwt.json.)"""
    fn = os.path.join(out_root, model, f"ratio_{ratio}", "op_bwt.json")
    if not os.path.exists(fn):
        return None
    try:
        with open(fn) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def pct(x):
    return f"{x*100:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT,
                    help="tolerance for 'below baseline' (default 0.005 = 0.5pp)")
    args = ap.parse_args()
    eps = args.eps

    for model in PAPER:
        own = load_run(args.out_root, model, BASELINE_RATIO)
        paper = PAPER[model]
        has_own = own is not None and own.get("op") is not None and own.get("bwt") is not None

        print("=" * 78)
        print(f"MODEL: {model}")
        if has_own:
            assert own is not None
            print(f"  baseline (own 10% run): OP = {own['op']*100:.1f}   BWT = {pct(own['bwt'])}")
        else:
            print(f"  baseline (own 10% run): MISSING -> 无法判定下降阈值")
        print(f"  paper 10% (reference only): OP = {paper['op']*100:.1f}   BWT = {pct(paper['bwt'])}")
        print(f"  tolerance EPS = {eps:.3f} ({eps*100:.1f} 个百分点)")
        print("-" * 78)
        print(f"  {'ratio':>6} | {'OP':>6} | {'dOP':>7} | {'BWT':>8} | {'dBWT':>8} | below?")
        for ratio in RATIO_ORDER:
            r = load_run(args.out_root, model, ratio)
            if r is None:
                print(f"  {ratio:>6} | {'(missing)':>22}")
                continue
            op, bwt = r.get("op"), r.get("bwt")
            if op is None or bwt is None:
                print(f"  {ratio:>6} | {'(incomplete)':>22}")
                continue
            if has_own:
                assert own is not None
                dop = op - own["op"]
                dbwt = bwt - own["bwt"]
                flags = " ".join(
                    (["OP↓"] if dop < -eps else []) +
                    (["BWT↓"] if dbwt < -eps else [])
                ) or "-"
            else:
                # 无本地 baseline：只展示相对论文的参考差，不判定
                dop = op - paper["op"]
                dbwt = bwt - paper["bwt"]
                flags = "(no own baseline)"
            print(f"  {ratio:>6} | {op*100:6.1f} | {dop*100:+7.1f} | {bwt*100:+8.1f} | {dbwt*100:+8.1f} | {flags}")

        # 下降阈值判定
        if has_own:
            assert own is not None
            threshold = None
            for ratio in RATIO_ORDER:
                r = load_run(args.out_root, model, ratio)
                if r is None:
                    continue
                op, bwt = r.get("op"), r.get("bwt")
                if op is None or bwt is None:
                    continue
                if (op - own["op"]) < -eps or (bwt - own["bwt"]) < -eps:
                    threshold = ratio
                    break
            if threshold is None:
                print(f"  [threshold] 未发现低于 10% baseline 的比例（范围内）")
            else:
                print(f"  [threshold] 下降阈值 = {threshold}（离 10% 最近且 OP 或 BWT 跌破 baseline 超过 {eps:.3f}）")
        else:
            print(f"  [threshold] 跳过：本地 10% baseline 缺失，不判定")
        print()

    print("=" * 78)
    print("Drop threshold = largest ratio (closest to 10%) where OP or BWT falls below the OWN 10% baseline by more than EPS.")


if __name__ == "__main__":
    main()
