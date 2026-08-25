#!/usr/bin/env python3
"""创建冒烟测试用的极小数据子集（每个任务取前 N 条），避免跑完整 5000 样本。"""
import json, os, sys

SRC = "/root/TRACE/data/extracted/TRACE-Benchmark/LLM-CL-Benchmark_5000"
DST = "/dev/shm/smoke_data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "Lima"]

for task in TASKS:
    os.makedirs(os.path.join(DST, task), exist_ok=True)
    for split in ["train", "eval", "test"]:
        src = os.path.join(SRC, task, f"{split}.json")
        dst = os.path.join(DST, task, f"{split}.json")
        if not os.path.exists(src):
            print(f"  skip {task}/{split} (不存在)")
            continue
        data = json.load(open(src))
        json.dump(data[:N], open(dst, "w"), ensure_ascii=False)
        print(f"  {task}/{split}: {len(data[:N])} 条")
print(f"冒烟数据已创建到 {DST}（每任务前 {N} 条）")
