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

DATASETS="${DATASETS:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}"
NTASKS=$(echo "$DATASETS" | tr ',' '\n' | grep -c .)
# 结果目录名：8 任务保持原名（兼容已有 ratio_0.10/0.08），非 8 任务加 _Ntask 后缀区分「同一实验的不同版本」
if [ "$NTASKS" -eq 8 ]; then
  DIR_SUFFIX="ratio_$RATIO"
else
  DIR_SUFFIX="ratio_${RATIO}_${NTASKS}task"
fi
OUT_DIR="$OUT_ROOT/$MODEL_SHORT/$DIR_SUFFIX"
port=$(shuf -i25000-30000 -n1)

echo "========== INFER (parallel 4-GPU, batch=$INFER_BATCH): $MODEL_SHORT ratio=$RATIO =========="
# (gpu_id, round_start, round_end)；按任务数动态划分到 4 卡（8 任务保持原特调划分）
if [ "$NTASKS" -eq 8 ]; then
  SPLITS=(
    "0 0 4"
    "1 4 6"
    "2 6 7"
    "3 7 8"
  )
else
  SPLITS=()
  for gpu in 0 1 2 3; do
    rs=$(( gpu * NTASKS / 4 )); re=$(( (gpu + 1) * NTASKS / 4 ))
    if [ "$rs" -lt "$re" ]; then
      SPLITS+=("$gpu $rs $re")
    fi
  done
fi
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

echo "========== PERSIST RESULTS (verified) =========="
PERSIST_ROOT="${PERSIST_ROOT:-/root/results}"
PERSIST_DIR="$PERSIST_ROOT/$MODEL_SHORT/$DIR_SUFFIX"
PERSIST_TMP="$PERSIST_DIR.tmp.$$"

# 0) 空间预检：/root/results 需有足够空间（保守要求 >= 1GB，实际结果仅几 MB）
FREE_KB=$(df -k --output=avail "$PERSIST_ROOT" 2>/dev/null | tail -1)
if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt 1048576 ]; then
  echo "[ERROR] /root/results 可用空间不足（${FREE_KB}KB < 1GB），持久化中止，保留 checkpoint"
  exit 1
fi

# 1) 校验 op_bwt.json 可解析且 op/bwt 非空（严格聚合已保证，这里再兜底）
python - "$OUT_DIR/op_bwt.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("op") is not None and d.get("bwt") is not None, "op_bwt.json 缺 op/bwt"
print(f"[persist] op_bwt.json 校验通过: OP={d['op']:.4f} BWT={d['bwt']:.4f}")
PY

# 2) 复制到临时目录（含 manifest），再原子重命名（任一失败即中止，不清 checkpoint）
rm -rf "$PERSIST_TMP"
mkdir -p "$PERSIST_TMP"
cp "$OUT_DIR/op_bwt.json" "$OUT_DIR/op_bwt.txt" "$OUT_DIR/run_manifest.json" "$PERSIST_TMP/"
cp -r "$OUT_DIR/predictions" "$PERSIST_TMP/"

# 3) 校验预测文件数量 == N*(N+1)/2（下三角矩阵：8 任务=36，4 任务=10）
NEXPECT=$(( NTASKS * (NTASKS + 1) / 2 ))
NPRED=$(find "$PERSIST_TMP/predictions" -name "results-*.json" 2>/dev/null | wc -l)
if [ "$NPRED" -ne "$NEXPECT" ]; then
  echo "[ERROR] 预测文件数 $NPRED != $NEXPECT（$NTASKS 任务下三角矩阵），持久化中止，保留 checkpoint"
  exit 1
fi

# 4) 原子切换到正式目录
rm -rf "$PERSIST_DIR"
mv "$PERSIST_TMP" "$PERSIST_DIR"
echo "[persist] 结果已持久化到 $PERSIST_DIR"

# 5) 回读校验：确认持久化目录完整（op_bwt 可解析 + 预测数==NEXPECT），防止 cp/mv 静默丢文件
python - "$PERSIST_DIR/op_bwt.json" "$PERSIST_DIR/predictions" "$NEXPECT" <<'PY'
import json, sys, os, glob
op, preds, nexp = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(op))
assert d.get("op") is not None and d.get("bwt") is not None, "持久化 op_bwt.json 缺 op/bwt"
n = len(glob.glob(os.path.join(preds, "results-*.json")))
assert n == nexp, f"持久化预测文件数 {n} != {nexp}"
print(f"[persist] 回读校验通过: OP={d['op']:.4f} BWT={d['bwt']:.4f}, {n} predictions")
PY

echo "========== CLEANUP CHECKPOINTS =========="
# 只有持久化验证通过后才清理 checkpoint
rm -rf "$OUT_DIR"/{0,1,2,3,4,5,6,7}

echo "========== MARK COMPLETE =========="
# .complete 是整个流程最后生成的文件，是「该组已完成且已持久化」的权威标志
touch "$OUT_DIR/.complete"
# 同步完成标记到持久盘，保证 /dev/shm 清空后持久盘仍有完成标记（便于下次恢复识别已完成组）
cp "$OUT_DIR/.complete" "$PERSIST_DIR/.complete"

echo "DONE -> $OUT_DIR (results persisted to $PERSIST_DIR)"
