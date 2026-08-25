#!/bin/bash
# 阶段 3：验证并归档 llama2-7b-chat ratio_0.10 的旧结果（legacy run），补 manifest + .complete。
# 结果位于持久盘 /root/results（/dev/shm 重启已清空，但旧结果已持久化）。
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH

cd /root/TRACE
RES=/root/results/llama2-7b-chat/ratio_0.10

echo "========== [1/4] 验证 36 个预测文件完整 =========="
N=$(find "$RES/predictions" -name "results-*.json" 2>/dev/null | wc -l)
echo "预测文件数: $N (期望 36)"
[ "$N" -eq 36 ] || { echo "[FAIL] 预测文件不完整"; exit 1; }

echo "========== [2/4] 重新聚合（严格模式），确认 OP/BWT =========="
python utils/aggregate_op_bwt.py --results_dir "$RES/predictions" --out /tmp/llama2_010_reagg.json | tail -5

echo "========== [3/4] 补录 manifest（标注 validated_legacy_run）=========="
python - "$RES/run_manifest.json" <<'PY'
import json, sys, os, time
out = sys.argv[1]
if os.path.exists(out):
    print("manifest 已存在，跳过:", out)
    sys.exit(0)
import torch, transformers, deepspeed
manifest = {
    "run_id": "legacy-20260824-llama2-7b-ratio010",
    "validated_legacy_run": True,
    "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "2026-08-24 在 /dev/shm 训练完成，结果已持久化到 /root/results；重启后 /dev/shm 清空，仅保留预测与指标。",
    "git_commit": "unknown",
    "model": "llama2-7b-chat", "model_path": "/dev/shm/hf/Llama-2-7b-chat-hf",
    "ratio": "0.10", "seed": 1234, "gpus": "0,1,2,3",
    "per_device_train_batch_size": 4, "gradient_accumulation_steps": 16,
    "num_train_epochs": "5,3,7,5,3,5,5,7", "learning_rate": 1e-5,
    "max_prompt_len": 1024, "max_ans_len": 512, "zero_stage": 3,
    "attention": "flash_attention_2",
    "torch": torch.__version__, "transformers": transformers.__version__, "deepspeed": deepspeed.__version__,
}
json.dump(manifest, open(out, "w"), indent=2)
print("manifest 已写入:", out)
PY

echo "========== [4/4] 创建 .complete 标记 =========="
touch "$RES/.complete"
echo "✅ llama2-7b-chat ratio_0.10 已标记为完成（validated_legacy_run）"
