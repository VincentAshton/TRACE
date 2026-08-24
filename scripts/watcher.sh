#!/bin/bash
# Watcher: wait for the first run (llama2-7b ratio 0.10) to finish, then launch the full sweep.
set -uo pipefail

WAIT_FOR="/dev/shm/outputs/llama2-7b-chat/ratio_0.10/op_bwt.json"
MAX_WAIT=$((10*3600))   # 10h safety timeout (train ~6h w/ grad ckpt + parallel infer ~2h)
elapsed=0
echo "[watcher] waiting for $WAIT_FOR (max 10h) ..."
while [ ! -f "$WAIT_FOR" ]; do
  sleep 60
  elapsed=$((elapsed+60))
  if [ $elapsed -ge $MAX_WAIT ]; then
    echo "[watcher] TIMEOUT waiting for first run, aborting"
    exit 1
  fi
done
echo "[watcher] first run done -> cleaning its checkpoints + launching full sweep"

# first run used the older script (no self-cleanup); delete its checkpoints
rm -rf /dev/shm/outputs/llama2-7b-chat/ratio_0.10/{0,1,2,3,4,5,6,7}

cd /root/TRACE
export PATH=/root/miniconda3/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export GPUS="0,1,2,3"
export OUT_ROOT=/dev/shm/outputs
export MAX_PROMPT_LEN=1024
export ATTN_IMPL=flash_attention_2

bash scripts/run_sweep.sh > /dev/shm/sweep.log 2>&1
echo "[watcher] sweep finished (exit $?)"
