#!/bin/bash
# Sweep all (model x ratio) for the replay-ratio drop-threshold experiment (cloud edition).
# 7B models only; skips runs that already have op_bwt.json (resume-safe).
# Uses run_replay_fast.sh (no gradient checkpointing; bf16 + flash-attn keeps memory safe).
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# short_name|model_path
MODELS=(
  "llama2-7b-chat|/dev/shm/hf/Llama-2-7b-chat-hf"
  "vicuna-7b|/dev/shm/hf/vicuna-7b-v1.5"
  "baichuan2-7b|/dev/shm/hf/baichuan2-7b-chat"
)

RATIOS="0.10 0.08 0.05 0.02 0.01"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs}"

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  path="${entry#*|}"

  # baichuan2 不支持 flash_attention_2（会报 "does not support Flash Attention 2.0"），
  # 必须用 eager；llama2 / vicuna（llama 系架构）用 flash_attention_2 省显存。
  if [ "$model" = "baichuan2-7b" ]; then
    export ATTN_IMPL="eager"
  else
    export ATTN_IMPL="flash_attention_2"
  fi

  for ratio in $RATIOS; do
    out_dir="$OUT_ROOT/$model/ratio_$ratio"
    if [ -f "$out_dir/op_bwt.json" ]; then
      echo "[skip] $model ratio=$ratio (done)"
      continue
    fi
    echo ""
    echo "############################################################"
    echo "# MODEL=$model  RATIO=$ratio"
    echo "############################################################"
    bash "$REPO_DIR/scripts/run_replay_fast.sh" "$model" "$path" "$ratio" \
      || echo "[FAILED] $model ratio=$ratio (continuing)"
  done

  # free /dev/shm model files once a model's runs are done
  if [ "$model" = "vicuna-7b" ]; then
    rm -rf /dev/shm/hf/vicuna-7b-v1.5 && echo "[cleanup] deleted vicuna-7b model"
  elif [ "$model" = "baichuan2-7b" ]; then
    rm -rf /dev/shm/hf/baichuan2-7b-chat && echo "[cleanup] deleted baichuan model"
  fi
done

echo ""
echo "========== SUMMARY =========="
python "$REPO_DIR/scripts/summary.py" --out_root "$OUT_ROOT" 2>&1 | tee /root/results/summary.txt
