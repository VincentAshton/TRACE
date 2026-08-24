#!/usr/bin/env python3
"""
Environment sanity check for the cloud GPU machine. Run BEFORE the sweep to
catch import/API/GPU issues cheaply (avoids burning rented hours on broken setups).

Checks:
  1. import torch / transformers / deepspeed / datasets / peft and print versions
  2. CUDA availability + device count
  3. import the repo's key modules (metrics, params, base_model, data utils)
  4. a tiny forward/backward pass on a 1-layer model to smoke-test the stack

Usage: python scripts/check_env.py
"""
import sys

def section(t):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)

section("1. library versions")
try:
    import torch
    print(f"torch          {torch.__version__}  cuda={torch.version.cuda}")
except Exception as e:
    print(f"torch          FAILED: {e}")
try:
    import transformers
    print(f"transformers   {transformers.__version__}")
except Exception as e:
    print(f"transformers   FAILED: {e}")
try:
    import deepspeed
    print(f"deepspeed      {deepspeed.__version__}")
except Exception as e:
    print(f"deepspeed      FAILED: {e}")
for pkg in ["datasets", "peft", "accelerate", "evaluate", "rapidfuzz", "nltk", "rouge"]:
    try:
        m = __import__(pkg)
        print(f"{pkg:<14} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"{pkg:<14} FAILED: {e}")

section("2. CUDA")
print(f"cuda available : {torch.cuda.is_available()}")
print(f"device count   : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}: {torch.cuda.get_device_name(i)}  {torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB")

section("3. repo module imports (from repo root)")
import os
sys.path.insert(0, os.getcwd())
for mod in ["metrics", "utils.utils", "utils.data.data_collator",
            "utils.model.model_utils", "utils.ds_utils", "training.params"]:
    try:
        __import__(mod)
        print(f"import {mod:<32} OK")
    except Exception as e:
        print(f"import {mod:<32} FAILED: {type(e).__name__}: {e}")

section("4. tiny forward/backward smoke test")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-llama2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-llama2")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    ids = tok("hello world, this is a test", return_tensors="pt").input_ids.to(model.device)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    print(f"tiny forward/backward OK (loss={out.loss.item():.4f})")
except Exception as e:
    print(f"tiny smoke test FAILED: {type(e).__name__}: {e}")

section("DONE")
