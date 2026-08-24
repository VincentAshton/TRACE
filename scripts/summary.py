#!/usr/bin/env python3
"""
Collect OP/BWT across all (model, ratio) runs and compare against the 10% baseline
to locate the "drop threshold" (the replay ratio below which OP/BWT falls below 10%).

Baseline policy (option B): the 10% ratio is run in the SAME environment (ratio_0.10),
so it is the PRIMARY reference. The paper's published 10% numbers are shown alongside
as a secondary reference to gauge environment offset.

Paper Table 1, Replay column (raw fraction):
  llama2-7b-chat   op=0.555 bwt=0.026
  llama2-13b-chat  op=0.566 bwt=0.004
  vicuna-7b        op=0.553 bwt=0.002
  vicuna-13b       op=0.569 bwt=0.006
  baichuan2-7b     op=0.517 bwt=0.011

Usage: python scripts/summary.py --out_root <dir>
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
RATIO_ORDER = ["0.08", "0.05", "0.02", "0.01"]  # descending (closest to 10% first)
EPS = 0.005  # tolerance for "below baseline"


def load_run(out_root, model, ratio):
    fn = os.path.join(out_root, model, f"ratio_{ratio}", "op_bwt.json")
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        return json.load(f)


def pct(x):
    return f"{x*100:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    for model in PAPER:
        own = load_run(args.out_root, model, BASELINE_RATIO)
        paper = PAPER[model]
        if own is not None and own.get("op") is not None and own.get("bwt") is not None:
            base, base_src = own, "own 10% run"
        else:
            base, base_src = paper, "paper 10% (own 10% missing)"

        print("=" * 78)
        print(f"MODEL: {model}")
        print(f"  baseline ({base_src}): OP = {base['op']*100:.1f}   BWT = {pct(base['bwt'])}")
        print(f"  paper 10% (reference): OP = {paper['op']*100:.1f}   BWT = {pct(paper['bwt'])}")
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
            dop = op - base["op"]
            dbwt = bwt - base["bwt"]
            flags = " ".join(
                (["OP↓"] if dop < -EPS else []) +
                (["BWT↓"] if dbwt < -EPS else [])
            ) or "-"
            print(f"  {ratio:>6} | {op*100:6.1f} | {dop*100:+7.1f} | {bwt*100:+8.1f} | {dbwt*100:+8.1f} | {flags}")
        print()

    print("=" * 78)
    print("Drop threshold = largest ratio (closest to 10%) where OP or BWT falls below baseline.")


if __name__ == "__main__":
    main()
