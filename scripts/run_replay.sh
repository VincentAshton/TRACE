#!/bin/bash
# Run ONE replay experiment: (model, ratio) -> train -> infer -> aggregate OP/BWT.
# Usage:  bash scripts/run_replay.sh <MODEL_SHORT> <MODEL_PATH> <RATIO>
# Env:    GPUS (default "0,1"), DATA_PATH, OUT_ROOT, MAX_PROMPT_LEN, NUM_EPOCHS, LR,
#         SYMLINK_LAST_CKPT (optional: symlink round-7 checkpoint to this dir, for 13B)
set -euo pipefail

MODEL_SHORT="$1"
MODEL_PATH="$2"
RATIO="$3"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0,1}"
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
# clean any stale checkpoints from a previous (failed) run
rm -rf "$OUT_DIR"/{0,1,2,3,4,5,6,7}
# 13B: symlink round-7 checkpoint to an external dir so /dev/shm (200G) is not exceeded
SYMLINK_LAST_CKPT="${SYMLINK_LAST_CKPT:-}"
if [ -n "$SYMLINK_LAST_CKPT" ]; then
  mkdir -p "$SYMLINK_LAST_CKPT"
  ln -sfn "$SYMLINK_LAST_CKPT" "$OUT_DIR/7"
fi
port=$(shuf -i25000-30000 -n1)

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
    $OFFLOAD \
    --deepspeed \
    --print_loss \
    --gradient_checkpointing \
    --past_task_ratio "$RATIO" \
    --output_dir "$OUT_DIR" > "$OUT_DIR/train.log" 2>&1

echo "========== INFER: model=$MODEL_SHORT ratio=$RATIO =========="
deepspeed --include="localhost:0" --master_port $port inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --inference_tasks "$DATASETS" \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$OUT_DIR" \
    --inference_batch 4 \
    --max_prompt_len "$MAX_PROMPT_LEN" \
    --max_ans_len "$MAX_ANS_LEN" \
    --seed 1234 \
    --deepspeed \
    --CL_method base \
    --inference_output_path "$OUT_DIR/predictions" > "$OUT_DIR/infer.log" 2>&1

echo "========== AGGREGATE OP/BWT =========="
python "$REPO_DIR/utils/aggregate_op_bwt.py" \
    --results_dir "$OUT_DIR/predictions" \
    --tasks "$DATASETS" \
    --out "$OUT_DIR/op_bwt.json" | tee "$OUT_DIR/op_bwt.txt"

echo "========== PERSIST RESULTS + CLEANUP =========="
# persist results to /root (survives /dev/shm tmpfs loss)
PERSIST_DIR="/root/results/$MODEL_SHORT/ratio_$RATIO"
mkdir -p "$PERSIST_DIR"
cp "$OUT_DIR/op_bwt.json" "$OUT_DIR/op_bwt.txt" "$PERSIST_DIR/" 2>/dev/null || true
cp -r "$OUT_DIR/predictions" "$PERSIST_DIR/" 2>/dev/null || true
# free checkpoint space (keep predictions + op_bwt + logs)
rm -rf "$OUT_DIR"/{0,1,2,3,4,5,6,7}
if [ -n "$SYMLINK_LAST_CKPT" ]; then
  rm -rf "$SYMLINK_LAST_CKPT"
fi

echo "DONE -> $OUT_DIR (results persisted to $PERSIST_DIR)"
