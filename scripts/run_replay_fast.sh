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
# clean any stale checkpoints from a previous (failed) run
rm -rf "$OUT_DIR"/{0,1,2,3,4,5,6,7}
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
    --gradient_checkpointing \
    $OFFLOAD \
    --deepspeed \
    --print_loss \
    --past_task_ratio "$RATIO" \
    --output_dir "$OUT_DIR" > "$OUT_DIR/train.log" 2>&1

echo "========== INFER (parallel) + AGGREGATE =========="
bash "$REPO_DIR/scripts/run_infer_parallel.sh" "$MODEL_SHORT" "$MODEL_PATH" "$RATIO"

echo "DONE -> $OUT_DIR"
