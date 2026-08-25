#!/bin/bash
# 阶段 2 冒烟测试：用极小数据跑通「训练→ZeRO-3 保存/加载→推理→严格聚合→持久化→.complete→清理 checkpoint」全链路。
# 使用独立测试目录（/dev/shm/smoke_*），不污染正式 ratio 目录和 /root/results。
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export GPUS="0,1,2,3"
export OUT_ROOT=/dev/shm/smoke_outputs
export PERSIST_ROOT=/dev/shm/smoke_persist
export MAX_PROMPT_LEN=1024
export ATTN_IMPL=flash_attention_2
export NUM_EPOCHS="1,1,1,1,1,1,1,1"
export DATA_PATH=/dev/shm/smoke_data

cd /root/TRACE

echo "========== [1/8] 创建极小数据子集 =========="
python scripts/create_smoke_data.py 300

echo "========== [2/8] 4-GPU 训练 + ZeRO-3 checkpoint 保存 =========="
# 用 vicuna（llama 系，flash-attn 兼容）跑完整 8 任务流程，每任务 1 epoch
bash scripts/run_replay_fast.sh vicuna-7b /dev/shm/hf/vicuna-7b-v1.5 0.10

echo "========== [3/8] 验证 .complete 已生成 =========="
ls -la /dev/shm/smoke_outputs/vicuna-7b/ratio_0.10/.complete

echo "========== [4/8] 验证 op_bwt.json + manifest =========="
cat /dev/shm/smoke_outputs/vicuna-7b/ratio_0.10/op_bwt.json
echo "--- manifest ---"
cat /dev/shm/smoke_outputs/vicuna-7b/ratio_0.10/run_manifest.json

echo "========== [5/8] 验证持久化结果 =========="
ls /dev/shm/smoke_persist/vicuna-7b/ratio_0.10/
echo "持久化预测数: $(find /dev/shm/smoke_persist/vicuna-7b/ratio_0.10/predictions -name 'results-*.json' 2>/dev/null | wc -l)"

echo "========== [6/8] 验证 checkpoint 已清理（持久化成功后）=========="
if ls -d /dev/shm/smoke_outputs/vicuna-7b/ratio_0.10/[0-9]* 2>/dev/null; then
  echo "[FAIL] checkpoint 目录仍残留"; exit 1
else
  echo "[PASS] checkpoint 已清理"
fi

echo "========== [7/8] 验证严格聚合可复现 =========="
python utils/aggregate_op_bwt.py --results_dir /dev/shm/smoke_outputs/vicuna-7b/ratio_0.10/predictions \
  --out /tmp/smoke_reagg.json | tail -4

echo "========== [8/8] SMOKE TEST PASSED =========="
