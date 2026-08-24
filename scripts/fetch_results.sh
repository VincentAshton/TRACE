#!/bin/bash
# 从云端 /root/results 拉取实验结果到本地（供查看 + 数据备份）
# 用法: bash scripts/fetch_results.sh
set -uo pipefail

REMOTE="root@ssh-cn-huabei1.ebcloud.com"
PORT="32433"
LOCAL_DIR="/home/vincent/TRACE/results"
SSH_OPTS="-o StrictHostKeyChecking=no -p $PORT"

export DISPLAY=:0
export SSH_ASKPASS=/tmp/askpass.sh
export SSH_ASKPASS_REQUIRE=force

mkdir -p "$LOCAL_DIR"

# 云端结果目录还没生成（第一组未跑完）时静默跳过
if ! ssh $SSH_OPTS "$REMOTE" 'test -d /root/results' 2>/dev/null; then
  echo "[fetch] 云端 /root/results 尚未生成（第一组还没跑完），本次跳过"
  exit 0
fi

# 增量拉取
rsync -az -e "ssh $SSH_OPTS" "$REMOTE:/root/results/" "$LOCAL_DIR/" 2>&1 | tail -6

echo ""
echo "========== 本地结果一览 =========="
find "$LOCAL_DIR" -name "op_bwt.json" 2>/dev/null | sort
echo ""
echo "========== 已完成的 OP/BWT =========="
for f in $(find "$LOCAL_DIR" -name "op_bwt.json" 2>/dev/null | sort); do
  rel="${f#$LOCAL_DIR/}"
  op=$(python3 -c "import json,sys; d=json.load(open('$f')); print(round(d.get('op',0),4))" 2>/dev/null)
  bwt=$(python3 -c "import json,sys; d=json.load(open('$f')); print(round(d.get('bwt',0),4))" 2>/dev/null)
  echo "  $rel  ->  OP=$op  BWT=$bwt"
done
