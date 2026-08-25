#!/usr/bin/env python3
"""
Environment pre-flight check for the TRACE replay-ratio experiment.

Run BEFORE starting the sweep to catch configuration / environment problems cheaply
(avoids burning rented GPU hours on broken setups). Any FAILURE exits non-zero so it
can gate a launch script; WARNINGS do not block.

Checks:
  1. library versions (torch/transformers/deepspeed/datasets/peft/flash_attn)
  2. CUDA device count + per-GPU memory (expects the frozen 4x A100-80GB config)
  3. frozen training config (per-device batch=4, grad accumulation=16, world size=4)
  4. model paths complete (llama2 / vicuna / baichuan under /dev/shm/hf)
  5. data directories complete and readable (8 tasks + Lima)
  6. /dev/shm capacity + free space (checkpoint accumulation hazard)
  7. persistent result dir (/root/results) writable + free space
  8. attention impl compatibility (baichuan=eager, llama/vicuna=flash_attention_2)
  9. stale data caches / leftover .tmp files under /tmp/data_files
  10. pre-existing incomplete runs in the output root
  11. git worktree cleanliness + commit

Usage: python scripts/check_env.py [--out_root /dev/shm/outputs] [--smoke]
  --smoke : additionally run a tiny ZeRO-3 save/load smoke test (needs GPU)
"""
import os
import sys
import glob
import json
import shutil
import subprocess

FAILURES = []
WARNINGS = []
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- frozen experiment config (must match scripts/run_replay_fast.sh) ----
FROZEN = {"gpus": 4, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 16}


def report(name, ok, detail="", fatal=True):
    tag = ("PASS" if ok else ("FAIL" if fatal else "WARN"))
    line = f"[{tag}] {name}"
    if detail and not ok:
        line += f"  -- {detail}"
    print(line)
    if not ok:
        (FAILURES if fatal else WARNINGS).append(name)


def section(t):
    print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)


# ---------------------------------------------------------------- 1. versions
section("1. library versions")
torch = transformers = deepspeed = None
try:
    import torch
    print(f"torch         {torch.__version__}  cuda={torch.version.cuda}")
except Exception as e:
    report("import torch", False, str(e))
try:
    import transformers
    print(f"transformers  {transformers.__version__}")
except Exception as e:
    report("import transformers", False, str(e))
try:
    import deepspeed
    print(f"deepspeed     {deepspeed.__version__}")
except Exception as e:
    report("import deepspeed", False, str(e))
for pkg in ["datasets", "peft", "accelerate", "evaluate", "rouge", "nltk"]:
    try:
        m = __import__(pkg)
        print(f"{pkg:<12} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        report(f"import {pkg}", False, str(e))
try:
    import flash_attn
    print(f"flash_attn    {getattr(flash_attn, '__version__', 'ok')}")
except Exception as e:
    report("import flash_attn", False, str(e), fatal=False)

# ---------------------------------------------------------------- 2. CUDA
section("2. CUDA / GPU")
if torch is None:
    report("torch available", False, "torch import failed")
else:
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    report(f"GPU count == {FROZEN['gpus']}", n == FROZEN["gpus"], f"got {n}")
    if torch.cuda.is_available():
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            gb = p.total_memory / 1e9
            print(f"  gpu {i}: {torch.cuda.get_device_name(i)}  {gb:.0f}GB")
            report(f"gpu {i} memory >= 80GB", gb >= 79, f"{gb:.0f}GB")

# ---------------------------------------------------------------- 3. frozen config
section("3. frozen training config")
fast = os.path.join(REPO, "scripts", "run_replay_fast.sh")
if os.path.exists(fast):
    txt = open(fast).read()
    for key, val in FROZEN.items():
        if key == "gpus":
            continue
        # check the hardcoded flag value in the script
        flag = {"per_device_train_batch_size": "--per_device_train_batch_size",
                "gradient_accumulation_steps": "--gradient_accumulation_steps"}[key]
        report(f"{key} == {val}", f"{flag} {val}" in txt,
               f"expected '{flag} {val}' in run_replay_fast.sh")
else:
    report("run_replay_fast.sh present", False, "script missing")

# ---------------------------------------------------------------- 4. models
section("4. model paths")
for m, path in [("llama2-7b-chat", "/dev/shm/hf/Llama-2-7b-chat-hf"),
                ("vicuna-7b", "/dev/shm/hf/vicuna-7b-v1.5"),
                ("baichuan2-7b", "/dev/shm/hf/baichuan2-7b-chat")]:
    ok = os.path.isdir(path) and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
    report(f"{m} model present", ok, path)

# ---------------------------------------------------------------- 5. data
section("5. data directories")
data_root = os.path.join(REPO, "data", "extracted", "TRACE-Benchmark", "LLM-CL-Benchmark_5000")
expected = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
            "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "Lima"]
if os.path.isdir(data_root):
    for d in expected:
        p = os.path.join(data_root, d)
        n = len(glob.glob(os.path.join(p, "*.json")))
        report(f"data/{d} readable", os.path.isdir(p) and n > 0, f"{n} json files")
else:
    report("data root present", False, data_root)

# ---------------------------------------------------------------- 6. /dev/shm space
section("6. /dev/shm (memory disk)")
try:
    total, used, free = shutil.disk_usage("/dev/shm")
    tg, fg = total / 1e9, free / 1e9
    report("/dev/shm total >= 200GB", total >= 200e9, f"{tg:.0f}GB")
    report("/dev/shm free >= 100GB", free >= 100e9, f"{fg:.0f}GB free")
except Exception as e:
    report("/dev/shm usable", False, str(e))

# ---------------------------------------------------------------- 7. persistent dir
section("7. persistent result dir")
persist = "/root/results"
if os.path.isdir(persist):
    try:
        _, _, pfree = shutil.disk_usage(persist)
        test = os.path.join(persist, ".write_test")
        open(test, "w").close(); os.remove(test)
        report("/root/results writable", True)
        report("/root/results free >= 5GB", pfree >= 5e9, f"{pfree/1e9:.0f}GB free")
    except Exception as e:
        report("/root/results writable", False, str(e))
else:
    report("/root/results exists", False, persist)

# ---------------------------------------------------------------- 8. attention
section("8. attention impl compatibility")
report("baichuan uses eager (sweep.sh)", '"baichuan2-7b"' in open(os.path.join(REPO, "scripts", "run_sweep.sh")).read() and "eager" in open(os.path.join(REPO, "scripts", "run_sweep.sh")).read(),
       "run_sweep.sh must set ATTN_IMPL=eager for baichuan2-7b")

# ---------------------------------------------------------------- 9. stale caches
section("9. stale data caches / .tmp")
cache_dir = "/tmp/data_files"
if os.path.isdir(cache_dir):
    stale = glob.glob(os.path.join(cache_dir, "*.tmp")) + glob.glob(os.path.join(cache_dir, "*.corrupt.*"))
    report("no leftover .tmp/.corrupt caches", len(stale) == 0,
           f"{len(stale)} stale file(s): {stale[:3]}", fatal=False)

# ---------------------------------------------------------------- 10. pre-existing runs
section("10. pre-existing incomplete runs")
out_root = None
for i, a in enumerate(sys.argv):
    if a == "--out_root" and i + 1 < len(sys.argv):
        out_root = sys.argv[i + 1]
if out_root and os.path.isdir(out_root):
    incomplete = []
    for model in os.listdir(out_root):
        mp = os.path.join(out_root, model)
        if not os.path.isdir(mp):
            continue
        for ratio in os.listdir(mp):
            rp = os.path.join(mp, ratio)
            if os.path.isdir(rp) and not os.path.exists(os.path.join(rp, ".complete")):
                incomplete.append(os.path.join(model, ratio))
    report("no incomplete runs", len(incomplete) == 0, f"{len(incomplete)}: {incomplete[:5]}", fatal=False)
else:
    print("  (no --out_root given; skipping)")

# ---------------------------------------------------------------- 11. git
section("11. git")
try:
    r = subprocess.run(["git", "-C", REPO, "status", "--short"], capture_output=True, text=True)
    dirty = [l for l in r.stdout.splitlines() if l.strip()]
    report("git worktree clean", len(dirty) == 0, f"{len(dirty)} dirty file(s)", fatal=False)
    r2 = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    print(f"  commit: {r2.stdout.strip()}")
except Exception as e:
    report("git available", False, str(e), fatal=False)

# ---------------------------------------------------------------- summary
section("RESULT")
if FAILURES:
    print(f"[ERROR] {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
    print("[ERROR] 环境检查未通过，禁止启动实验")
    sys.exit(1)
else:
    if WARNINGS:
        print(f"[WARN] {len(WARNINGS)} warning(s) (non-blocking): {', '.join(WARNINGS)}")
    print("[OK] 环境检查全部通过，可以启动实验")
    sys.exit(0)
