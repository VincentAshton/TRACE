#!/bin/bash
# Run ONE replay experiment: train (grad ckpt ON, required for long-seq on 4 GPUs) -> parallel inference -> aggregate OP/BWT.
# Usage:  bash scripts/run_replay_fast.sh <MODEL_SHORT> <MODEL_PATH> <RATIO>
# Env:    GPUS (default "0,1,2,3"), DATA_PATH, OUT_ROOT, MAX_PROMPT_LEN, NUM_EPOCHS, LR, INFER_BATCH
set -euo pipefail

MODEL_SHORT="$1"
MODEL_PATH="$2"
RATIO="$3"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0,1,2,3}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/extracted/TRACE-Benchmark/LLM-CL-Benchmark_5000}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_ANS_LEN="${MAX_ANS_LEN:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-5,3,7,5,3,5,5,7}"
LR="${LR:-1e-5}"
ZERO_STAGE="${ZERO_STAGE:-3}"
OFFLOAD="${OFFLOAD:-}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

DATASETS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
OUT_DIR="$OUT_ROOT/$MODEL_SHORT/ratio_$RATIO"
mkdir -p "$OUT_DIR"

# 失败状态标记：任何未捕获错误写 .failed（记录时间+模型+比例），便于快速识别失败组
trap 'echo "[FAILED] $(date -Iseconds) model=$MODEL_SHORT ratio=$RATIO" > "$OUT_DIR/.failed" 2>/dev/null || true' ERR

# ---- 重跑前配置一致性校验：若已有 manifest，比对冻结配置，不匹配则拒绝（防配置漂移混用结果）----
if [ -f "$OUT_DIR/run_manifest.json" ]; then
  python - "$OUT_DIR/run_manifest.json" "$MODEL_SHORT" "$RATIO" "$ATTN_IMPL" "$NUM_EPOCHS" "$LR" "$MAX_PROMPT_LEN" "$MAX_ANS_LEN" "$ZERO_STAGE" <<'PY'
import json, sys
mf, model, ratio, attn, epochs, lr, mpl, mal, zero = sys.argv[1:10]
try:
    d = json.load(open(mf))
except Exception:
    print("[WARN] 旧 manifest 不可解析，跳过一致性校验（将重新生成）")
    sys.exit(0)
checks = {
    "model": (d.get("model"), model),
    "ratio": (d.get("ratio"), ratio),
    "attention": (d.get("attention"), attn),
    "num_train_epochs": (d.get("num_train_epochs"), epochs),
    "max_prompt_len": (d.get("max_prompt_len"), int(mpl)),
    "max_ans_len": (d.get("max_ans_len"), int(mal)),
    "zero_stage": (d.get("zero_stage"), int(zero)),
}
mismatch = [k for k, (o, n) in checks.items() if o != n]
try:
    if abs(d.get("learning_rate") - float(lr)) > 1e-9:
        mismatch.append("learning_rate")
except (TypeError, ValueError):
    mismatch.append("learning_rate")
if mismatch:
    print(f"[FATAL] 本次配置与旧 manifest 不一致: {mismatch}")
    print("[FATAL] 拒绝运行，防止配置漂移导致结果不可比。若确要改配置重跑，请先删除旧 manifest 或换输出目录。")
    sys.exit(1)
print("[manifest-check] 配置与旧 manifest 一致，继续运行")
PY
fi

# clean any stale checkpoints from a previous (failed) run
rm -rf "$OUT_DIR"/{0,1,2,3,4,5,6,7}
port=$(shuf -i25000-30000 -n1)

# ---- run manifest：记录 run ID + 冻结配置 + 环境版本，供 resume/校验使用 ----
GIT_COMMIT="$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python - "$OUT_DIR/run_manifest.json" "$MODEL_SHORT" "$MODEL_PATH" "$RATIO" \
    "$GIT_COMMIT" "$GPUS" "$ATTN_IMPL" "$NUM_EPOCHS" "$LR" "$MAX_PROMPT_LEN" "$MAX_ANS_LEN" "$ZERO_STAGE" <<'PY'
import json, sys, time, os
(_, out, model, mpath, ratio, commit, gpus, attn, epochs, lr, mpl, mal, zero) = sys.argv
import torch, transformers, deepspeed
manifest = {
    "run_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}",
    "started_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
    "git_commit": commit,
    "model": model, "model_path": mpath, "ratio": ratio,
    "seed": 1234, "gpus": gpus,
    "per_device_train_batch_size": 4, "gradient_accumulation_steps": 16,
    "num_train_epochs": epochs, "learning_rate": float(lr),
    "max_prompt_len": int(mpl), "max_ans_len": int(mal), "zero_stage": int(zero),
    "attention": attn,
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "transformers": transformers.__version__, "deepspeed": deepspeed.__version__,
}
json.dump(manifest, open(out, "w"), indent=2)
print(f"[manifest] run_id={manifest['run_id']}  commit={commit}")
PY

echo "========== TRAIN: model=$MODEL_SHORT ratio=$RATIO -> $OUT_DIR =========="
deepspeed --include="localhost:$GPUS" --master_port $port training/replay.py \
    --data_path "$DATA_PATH" \
    --dataset_name "$DATASETS" \
    --replay_dataset_name Lima \
    --model_name_or_path "$MODEL_PATH" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --max_prompt_len "$MAX_PROMPT_LEN" \
    --max_ans_len "$MAX_ANS_LEN" \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --num_train_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage "$ZERO_STAGE" \
    --gradient_checkpointing \
    $OFFLOAD \
    --deepspeed \
    --print_loss \
    --past_task_ratio "$RATIO" \
    --output_dir "$OUT_DIR" > "$OUT_DIR/train.log" 2>&1

echo "========== INFER (parallel) + AGGREGATE =========="
bash "$REPO_DIR/scripts/run_infer_parallel.sh" "$MODEL_SHORT" "$MODEL_PATH" "$RATIO"

# 成功完成：清除可能的旧失败标记（.complete 由 run_infer_parallel.sh 最后生成）
rm -f "$OUT_DIR/.failed"

echo "DONE -> $OUT_DIR"
