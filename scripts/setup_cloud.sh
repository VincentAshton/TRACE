#!/bin/bash
# One-time cloud machine setup. Run AFTER the ebcloud image is provisioned.
# Recommended image: Python 3.10 + PyTorch 2.3.1 + CUDA 12.1 (A100).
# This script is idempotent: it checks what's present and only installs what's missing.
set -uo pipefail

echo "========== 0. system info =========="
python --version 2>&1 || python3 --version 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -5
cat /usr/local/cuda/version.txt 2>/dev/null || echo "(no /usr/local/cuda/version.txt)"

echo "========== 1. torch check =========="
if python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" 2>/dev/null; then
    echo "torch already present; skipping torch install."
else
    echo "installing torch 2.3.1 + cu121 ..."
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
fi

echo "========== 2. install python deps =========="
pip install -r requirements.txt

echo "========== 3. HuggingFace mirror (China network) =========="
export HF_ENDPOINT=https://hf-mirror.com
grep -q HF_ENDPOINT ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo "HF_ENDPOINT=$HF_ENDPOINT"

echo "========== 4. HF login (paste token; required for LLaMA-2 gated) =========="
huggingface-cli login

echo "========== 5. self check =========="
python scripts/check_env.py

echo "========== SETUP DONE =========="
echo "Next: export GPUS=\"0,1,2,3\" && bash scripts/run_sweep.sh"
