# TRACE Replay 复现 + 回放比例下降阈值实验 — 运行手册

> ⚠️ **2026-08-24 更新**：实际运行环境与下文第 2/4 节的「初始方案」不同——
> 实际用 **3 个 7B 模型 × 5 比例 = 15 组**（13B 已取消），4×A100-80GB 租用实例，
> 脚本用 `run_replay_fast.sh`（梯度检查点**开启**、bf16 + flash-attn-2、4 卡并行推理），
> 输出在 `/dev/shm/outputs/`（内存盘），结果持久化 `/root/results/`。
> 权威最新状态见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 1. 实验设计

- 3 个 7B 模型：LLaMA-2-7B-Chat、Vicuna-7B-v1.5、Baichuan2-7B-Chat（13B 已取消）
- 扫描 past_task_ratio ∈ {0.10（基线）, 0.08, 0.05, 0.02, 0.01}
- 每个 (模型, 比例) 跑一遍完整 8 任务顺序 Replay 训练 + 推理，算 OP/BWT
- 以「自己环境跑的 10%」为基线（option B），找「下降阈值」（OP/BWT 首次跌破 10% 基线的比例）
- 共 15 组训练（3 模型 × 5 比例）

## 2. 租卡要求（重要）

- 全参数微调 7B/13B，推荐 **2× A100 80G**（ZeRO-3 即可，无需 offload 也够）
- 或 1× A100 80G + ZeRO-3 + CPU offload（慢 2~4 倍，更便宜）
- 不推荐 40G 卡跑 13B（会 OOM）
- 磁盘 ≥ 500GB（5 模型 × 4 比例 × 8 round 的 checkpoint，13B 每 round 约 26GB）

## 3. 云端环境搭建

```bash
# 1) 先装 torch（推荐 Python3.10 + torch2.3.1 + CUDA12.1；按镜像 CUDA 版本选 index）
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 2) 装其余依赖
git clone https://github.com/BeyonderXX/TRACE.git && cd TRACE
pip install -r requirements.txt

# 3) 上传已处理好的数据（本仓库已含数据 + 代码改动）
#    数据目录：data/extracted/TRACE-Benchmark/LLM-CL-Benchmark_5000

# 4) LLaMA-2 需要 HF 授权（gated），先登录
huggingface-cli login   # 粘贴你的 HF token（需已接受 meta-llama/Llama-2-*-chat-hf 授权）
# Vicuna / Baichuan 是公开的，无需 token

# 5) 环境自检（先跑这个，确认无误再烧卡）
python scripts/check_env.py
```

## 4. 跑实验

```bash
# 默认 2 卡 + offload；按需改环境变量
export GPUS="0,1"          # 卡号
export OUT_ROOT="$PWD/outputs"

# 单组试跑（先跑一个，确认能跑通再全量）
bash scripts/run_replay.sh llama2-7b-chat meta-llama/Llama-2-7b-chat-hf 0.08

# 全量扫描（自动跳过已完成的，可断点续跑）
bash scripts/run_sweep.sh
```

结果落盘结构：
```
outputs/<model>/ratio_<ratio>/
  ├── train.log
  ├── infer.log
  ├── op_bwt.json        # OP / BWT（原始分数）
  ├── op_bwt.txt
  └── predictions/results-{round}-{task}-{task}.json
```

## 5. 看结果

```bash
python scripts/summary.py --out_root outputs
```
输出每个模型的 10% 基线 vs 各比例的 OP/BWT，并标注 OP↓/BWT↓（低于基线的比例）。

## 6. 关键参数与已知偏差（务必读）

1. **epochs 不一致**：论文附录 .1 写非 LoRA 用 "1,1,5,5,1,5,5,5"，但仓库脚本
   train_replay.sh 实际用 "5,3,7,5,3,5,5,7"。本实验用脚本的 "5,3,7,5,3,5,5,7"
   （这是产出论文 Table 1 数值的真实配置）。
2. **max_prompt_len**：默认 1024（= train_replay.sh）。MeetingBank 有极长尾
   （最长 37 万字符），1024 会大幅左截断。这是论文原配置；如担心可设
   `MAX_PROMPT_LEN=2048`，但绝对分数会变，请保持全实验一致。
3. **对比口径建议**：论文 10% 基线是作者机器跑出的绝对值。最严谨的做法是
   自己也跑一组 ratio=0.10（每个模型 1 次，共 5 次），用「自己环境的 10%」
   作基准找下降阈值；直接对比论文绝对值会引入环境偏移（GPU 数→有效 batch 等）。
   要跑 10% 只需：`bash scripts/run_replay.sh <model> <path> 0.10`。
4. **随机性**：推理用 temperature=0.1 采样，非贪心，结果有轻微随机波动；
   比较下降阈值时建议看 ≥1% 量级的差异。
5. **原仓库 bug 已修**：zero_stage=3 保存目录错误、SARI 指标返回 tuple、
   params.py 导入 quadprog/qpth 强依赖（改为可选导入）。
6. **OP/BWT 量纲 bug 已修（2026-08-24）**：`utils/aggregate_op_bwt.py` 原先把
   Py150 similarity（0-100）和 20Minuten SARI（0-100）与 0-1 量纲指标直接平均，
   得到 OP≈12 的错误值。已在 `extract_primary` 加 `SCALE_100` 集合对这两个任务
   /100 归一化。修复后 llama2-7b 10% 基线 OP=0.544/BWT=+0.077（论文 0.555/0.026，
   差 ~2% 属 4 卡 vs 8 卡环境偏移）。云端已同步修复，后续组自动生效。

## 7. 成本估算（粗略，仅供参考）

- 单次 7B 全参数 Replay（8 任务）：2×A100 约 3~6 小时
- 单次 13B：约 6~10 小时
- 20 组（3×7B + 2×13B，各 4 比例）：约 150~250 GPU 小时量级
- 按 A100 80G $1~2/小时/卡：约 $150~500（看卡价与是否 offload）
- 建议：先 1 个模型跑通 4 比例验证流程，再全量
