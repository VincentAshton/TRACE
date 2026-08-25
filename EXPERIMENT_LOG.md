# TRACE 回放比例下降阈值实验 — 实验记录

> 本文件是本次实验的完整记录，用于向导师汇报 / 论文撰写。
> 结果表由 `scripts/summary.py` 自动生成到 `/root/results/summary.txt`，并同步到本地 `results/` 目录。

## 1. 实验目的

在 TRACE 基准（面向大语言模型的持续学习 benchmark，复旦大学 2023）上复现 **Replay（回放）** 方法，
并研究一个具体问题：**当回放比例（past_task_ratio）逐步下降时，模型性能何时开始明显退化 —— 即找到「下降阈值」。**

论文默认的回放比例是 **10%**。本实验把比例下调到 0.08 / 0.05 / 0.02 / 0.01，观察
OP（Overall Performance，整体性能）和 BWT（Backward Transfer，向后迁移，负值代表遗忘）
分别在哪个比例开始跌破 10% 基线。

## 2. 论文与代码来源

- 论文：*TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models*（arXiv:2310.06762）
- 代码：https://github.com/BeyonderXX/TRACE
- 本地工作目录：`/home/vincent/TRACE`（WSL）
- 云端工作目录：`/root/TRACE`（租用的 A100 实例）

## 3. 实验设计

- **方法**：Replay（训练入口 `training/replay.py`，独立于 `training/main.py`）
- **模型**（3 个 7B）：
  | 短名 | 模型 |
  |---|---|
  | llama2-7b-chat | LLaMA-2-7B-Chat |
  | vicuna-7b | Vicuna-7B-v1.5 |
  | baichuan2-7b | Baichuan2-7B-Chat |
- **回放比例**：0.10（基线）、0.08、0.05、0.02、0.01
- **组合**：3 模型 × 5 比例 = **15 组**

## 4. 指标定义（论文 Section 3）

记 R[i][j] 为「第 i 轮训练结束后，模型在第 j 个任务上的得分」，T=8 为任务数：

- **OP** = (1/T) · Σᵢ R[T-1][i]　　（最后一轮对所有任务的平均得分 = 整体性能）
- **BWT** = (1/(T-1)) · Σᵢ (R[T-1][i] − R[i][i])　　（平均向后迁移；负值 = 学新任务后遗忘了旧任务）

每个任务的主指标（论文 Table 4）：C-STANCE/FOMC/ScienceQA/NumGLUE 用 accuracy，MeetingBank 用 Rouge-L，Py150 用 similarity，20Minuten 用 SARI。

## 5. 超参数配置

| 项 | 值 |
|---|---|
| 学习率 | 1e-5，Adam (β=0.9, 0.95)，weight decay 0 |
| per_device batch | 4，梯度累积 16（全局 batch = 4×4×16 = 256，对齐论文）|
| 每任务 epoch | 5,3,7,5,3,5,5,7（对应 8 个任务）|
| 精度 / 注意力 | bf16 + flash-attention 2 |
| ZeRO | stage 3 |
| 梯度检查点 | 开启（长句任务 FOMC/MeetingBank/Py150 + Lima 回放必须开，否则 4 卡 OOM 爆 78G）|
| 序列长度 | max_prompt_len=1024, max_ans_len=512 |

## 6. 环境

- 硬件：4× A100-80GB（ebcloud 租用，论文用 8 卡，本实验用 4 卡）
- 软件：torch 2.3.0+cu121、transformers 4.44.2、deepspeed 0.14.4、flash-attn 2.5.8、Python 3.10
- 对比基线策略（option B）：**用自己环境跑的 10% 比例作为基线**，而非论文发表的 10% 数值，避免跨环境系统误差。

## 7. 结果表（每组跑完后自动填充）

| 模型 | 0.10（基线）| 0.08 | 0.05 | 0.02 | 0.01 | 下降阈值 |
|---|---|---|---|---|---|---|
| llama2-7b-chat | 0.544 / +0.077 | — | — | — | — | — |
| vicuna-7b | — | — | — | — | — | — |
| baichuan2-7b | — | — | — | — | — | — |

（每个格子为 OP / BWT；完整数值见 `/root/results/summary.txt` 或本地 `results/summary.txt`）

## 8. 下降阈值结论

（待全部跑完后填写：每个模型 OP/BWT 首次跌破 10% 基线的最大比例，即最接近 10% 的那个下降比例）

## 9. 运行状态（快照 2026-08-25）

- 实验**已暂停**：只有 llama2-7b-chat ratio=0.10 完成（**OP=0.544 / BWT=+0.077**，论文 0.555 / 0.026）。
- ratio=0.08 训练 8 轮完成但推理 round 7 失败（数据缓存并发写坏）；其余 13 组因 /dev/shm 磁盘满 + baichuan flash-attn bug 全线失败。
- 3 个 bug 已定位并修复，代码已 push（详见 [HANDOFF.md](HANDOFF.md)）。
- 云端实例当前无法连接（Connection refused），恢复后按 HANDOFF.md 第 9 节步骤重启实验。
- 云端关键路径：模型 `/dev/shm/hf/`，输出 `/dev/shm/outputs/`，结果持久化 `/root/results/`。
- 运行细节与命令见 [RUNBOOK.md](RUNBOOK.md)。
