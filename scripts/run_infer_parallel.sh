#!/bin/bash
# Parallel (4-GPU) inference + OP/BWT aggregation for an ALREADY-TRAINED run.
# Splits the 8 rounds across 4 GPUs (rounds are independent -> each loads its own checkpoint).
# Skip-safe: skips result files that already exist.
# CRITICAL: uses explicit PID waits so a failed inference process ABORTS (and keeps checkpoints)
#           instead of `wait` (no args) which always returns 0 and would wrongly trigger cleanup.
# Usage: bash scripts/run_infer_parallel.sh <MODEL_SHORT> <MODEL_PATH> <RATIO>
set -euo pipefail

MODEL_SHORT="$1"
MODEL_PATH="$2"
RATIO="$3"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/extracted/TRACE-Benchmark/LLM-CL-Benchmark_5000}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_ANS_LEN="${MAX_ANS_LEN:-512}"
INFER_BATCH="${INFER_BATCH:-16}"

DATASETS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
OUT_DIR="$OUT_ROOT/$MODEL_SHORT/ratio_$RATIO"
port=$(shuf -i25000-30000 -n1)

echo "========== INFER (parallel 4-GPU, batch=$INFER_BATCH): $MODEL_SHORT ratio=$RATIO =========="
# (gpu_id, round_start, round_end)
SPLITS=(
  "0 0 4"
  "1 4 6"
  "2 6 7"
  "3 7 8"
)
pids=()
for s in "${SPLITS[@]}"; do
  set -- $s
  gpu=$1; rs=$2; re=$3
  deepspeed --include="localhost:$gpu" --master_port $((port + gpu)) inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --inference_tasks "$DATASETS" \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$OUT_DIR" \
    --inference_batch "$INFER_BATCH" \
    --max_prompt_len "$MAX_PROMPT_LEN" \
    --max_ans_len "$MAX_ANS_LEN" \
    --seed 1234 \
    --deepspeed \
    --CL_method base \
    --round_start "$rs" \
    --round_end "$re" \
    --inference_output_path "$OUT_DIR/predictions" > "$OUT_DIR/infer_$gpu.log" 2>&1 &
  pids+=($!)
done

# wait for EACH pid explicitly and track failures (bare `wait` always returns 0!)
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if [ "$fail" -eq 1 ]; then
  echo "[ERROR] 推理进程失败，保留 checkpoint，退出（不清理）"
  exit 1
fi

echo "========== AGGREGATE OP/BWT =========="
python "$REPO_DIR/utils/aggregate_op_bwt.py" \
    --results_dir "$OUT_DIR/predictions" \
    --tasks "$DATASETS" \
    --out "$OUT_DIR/op_bwt.json" | tee "$OUT_DIR/op_bwt.txt"

echo "========== PERSIST RESULTS + CLEANUP =========="
PERSIST_DIR="/root/results/$MODEL_SHORT/ratio_$RATIO"
mkdir -p "$PERSIST_DIR"
cp "$OUT_DIR/op_bwt.json" "$OUT_DIR/op_bwt.txt" "$PERSIST_DIR/" 2>/dev/null || true
cp -r "$OUT_DIR/predictions" "$PERSIST_DIR/" 2>/dev/null || true
# free checkpoint space (keep predictions + op_bwt + logs)
rm -rf "$OUT_DIR"/{0,1,2,3,4,5,6,7}

echo "DONE -> $OUT_DIR (results persisted to $PERSIST_DIR)"
