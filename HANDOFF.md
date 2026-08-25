# TRACE 实验交接文档（HANDOFF）

> 本文件供「其他 AI agent / 协作者」快速接管本实验。它是一份自包含的交接说明：
> 读完本文件即可理解实验全貌、当前进度、已知 bug、代码改动，并判断是否还有隐藏问题。
> 权威运行手册见 `RUNBOOK.md`，实验记录见 `EXPERIMENT_LOG.md`。

---

## 1. 一句话概述

在 **TRACE 持续学习基准**（复旦 2023，arXiv:2310.06762）上复现 **Replay（回放）** 方法，
并研究一个问题：**当回放比例 `past_task_ratio` 从 10% 逐步下调时，模型性能何时开始明显退化（「下降阈值」）**。

## 2. 实验设计

- **方法**：Replay（训练入口 `training/replay.py`，独立于 `training/main.py`）
- **模型**：3 个 7B —— llama2-7b-chat、vicuna-7b-v1.5、baichuan2-7b-chat（13B 已砍掉）
- **回放比例**：0.10（基线）、0.08、0.05、0.02、0.01
- **组合**：3 模型 × 5 比例 = **15 组**
- **指标**（论文 Section 3，8 个任务 T=8）：
  - `OP` = 最后一轮对所有任务的平均得分（整体性能）
  - `BWT` = 平均向后迁移（负值 = 学新任务忘旧任务）
- **基线策略**：用「自己环境跑的 10%」做基线（不直接比论文绝对值，避免环境偏移）

## 3. 运行环境

- **云端硬件**：4× A100-80GB，**租用自 ebcloud**（按小时计费）
- **实例连接**：`ssh -p 32433 root@ssh-cn-huabei1.ebcloud.com`（密码认证）
- **系统内存**：1TB（1007GB）；**`/dev/shm` 是 tmpfs 内存盘，被人为限成 200GB**（这是磁盘满 bug 的根源）
- **软件**：torch 2.3.0+cu121、transformers 4.44.2、deepspeed 0.14.4、flash-attn 2.5.8、Python 3.10（`/root/miniconda3`）
- **关键路径**：
  - 代码 `/root/TRACE`（注意：**云端不是 git 仓库**，靠 scp 同步）
  - 模型 `/dev/shm/hf/`（llama2 51G、vicuna 13G、baichuan 14G）
  - 输出 `/dev/shm/outputs/`（内存盘）
  - 结果持久化 `/root/results/`（只有 28G 的 /root 盘）
  - 数据集 `/root/TRACE/data/extracted/TRACE-Benchmark/LLM-CL-Benchmark_5000`（8 任务 + Lima，原始 .json）
  - 数据缓存 `/tmp/data_files/*.pt`（tokenize 后的缓存）

## 4. 执行流程（脚本链路）

```
watcher.sh
  └─ 等 ratio_0.10 的 op_bwt.json 出现 → 清理 0.10 的 checkpoint → 启动 run_sweep.sh
       └─ run_sweep.sh（resume-safe：跳过已有 op_bwt.json 的组）
            └─ 每个 (model, ratio) 调 run_replay_fast.sh
                 ├─ 训练：deepspeed training/replay.py（grad ckpt + bf16 + flash-attn-2 + ZeRO-3）
                 ├─ 推理：run_infer_parallel.sh（4 卡并行，每个卡负责部分 round）
                 └─ 聚合：aggregate_op_bwt.py → op_bwt.json → 持久化到 /root/results/ → 清理 checkpoint
```

结果落盘：`outputs/<model>/ratio_<ratio>/{train.log, infer_*.log, op_bwt.json, op_bwt.txt, predictions/}`

## 5. 当前进度（2026-08-25 截至）

- ✅ **llama2-7b-chat ratio=0.10**：完整完成。**OP=0.544 / BWT=+0.077**（论文 0.555/0.026，差 ~2% 属 4 卡 vs 8 卡环境偏移）
- ⚠️ **llama2-7b-chat ratio=0.08**：训练 8 轮**全部完成**；推理 round 0-6 完成，**round 7 缺 6 个任务**（MeetingBank 起），没聚合出 op_bwt.json
- ❌ 其余 13 组全部失败（见下）
- **当前实例已无法连接**（Connection refused），需先恢复实例再继续

## 6. 已定位并修复的 3 个 bug

### Bug 1：数据缓存并发写坏（导致 ratio_0.08 推理失败的直接原因）
- **报错**：`KeyError: "filename 'storages' not found"`（torch.load 时）
- **根因**：`utils/data/data_utils.py` 的 `create_prompt_dataset` 里，`cache_found` 变量算出来但**根本没被使用**，
  每个进程的 local_rank 0 都无条件重算 + `torch.save` 覆盖缓存。推理时 `run_infer_parallel.sh` 启动 4 个**独立**的
  deepspeed 进程并发写同一个 `.pt` 缓存文件 → 文件损坏。
- **修复**：缓存命中即复用 + 跨进程文件锁（`fcntl.flock`）+ 先写 `.tmp` 再 `os.replace` 原子重命名。

### Bug 2：缓存 key 未包含 sample_ratio（隐藏 bug）
- **根因**：缓存文件名 hash 只含 `data_path + seed`，不含 `sample_ratio`。不同回放比例的 Lima 缓存会互相覆盖。
  之前因为 Bug 1 每次重写所以没暴露；修 Bug 1 后必须同时修这个，否则会引入「ratio_0.05 读到 ratio_0.08 的采样数据」。
- **修复**：hash 纳入 `sample_ratio`、`add_sys_prefix`、`for_backbone`。

### Bug 3：baichuan 不支持 flash_attention_2
- **报错**：`ValueError: BaichuanForCausalLM does not support Flash Attention 2.0 yet`
- **根因**：`run_sweep.sh` 全局 `export ATTN_IMPL=flash_attention_2`，baichuan 的自定义 modeling 代码不支持。
- **修复**：`run_sweep.sh` 里对 baichuan2-7b 用 `ATTN_IMPL=eager`，llama2/vicuna 保持 flash_attention_2。
  （`run_replay_fast.sh` 用 `${ATTN_IMPL:-flash_attention_2}` 读取，会正确继承 eager；推理脚本不设 ATTN_IMPL，继承环境变量。）

## 7. 代码改动清单（本次 commit f0b51f8）

| 文件 | 改动 |
|---|---|
| `utils/data/data_utils.py` | 缓存并发修复（锁+原子写）+ hash 纳入 sample_ratio |
| `scripts/run_sweep.sh` | baichuan 用 eager 注意力 |
| `.gitignore` | `data/` 等裸规则改为 `/data/`（原规则误伤了 `utils/data/` 源代码目录） |

## 8. 尚未解决 / 潜在风险（供其他 agent 判断）

1. **磁盘满**：`/dev/shm` 只有 200GB。8 个 checkpoint(≈107GB) + 3 模型(≈78GB) ≈ 185GB，贴着上限。
   **根治方案**：`mount -o remount,size=400G /dev/shm`（机器有 1TB 内存，完全够），这是运行时操作，不是代码。
2. **baichuan 换 eager 后显存是否够**：未验证。7B 全参数 + eager + 长序列 1024 在 4×A100-80GB 上大概率可行，但没实测过。
3. **vicuna 从没成功训练过一轮**：第一次跑就因磁盘满的 NCCL 报错挂了。需重新下载模型（被 sweep 清理逻辑删了）并验证能正常训练。
4. **数据缓存 `.tmp` 残留**：进程崩溃可能留下 `.tmp` 文件（不影响正确性，但占空间，可定期清理）。
5. **`summary.py` 的 paper 基线引用**：vicuna/baichuan 显示 "paper 10% (own 10% missing)"——目前只有 llama2 有自跑基线，其他模型暂无自跑 10% 结果，对比时要注意口径。

## 9. 恢复实验的操作步骤（实例恢复后执行）

```bash
# 1. 同步本次代码修复到云端（云端非 git 仓库，用 scp）
scp -P 32433 utils/data/data_utils.py root@<host>:/root/TRACE/utils/data/data_utils.py
scp -P 32433 scripts/run_sweep.sh root@<host>:/root/TRACE/scripts/run_sweep.sh

# 2. 扩大内存盘（根治磁盘满）
ssh root@<host> 'mount -o remount,size=400G /dev/shm && df -h /dev/shm'

# 3. 释放残留 checkpoint（ratio_0.05 的 2 个 + 失败组空日志；保留 0.10 和 0.08）
#    （需用户确认后执行 rm -rf，属于不可逆操作）

# 4. 补跑 ratio_0.08 推理（skip-safe，只补 round 7 缺失的 6 个任务，不重训）
#    bash scripts/run_infer_parallel.sh llama2-7b-chat /dev/shm/hf/Llama-2-7b-chat-hf 0.08

# 5. 重新下载 vicuna + baichuan 模型（约 27GB，被 sweep 清理逻辑删了）

# 6. 重启 sweep（自动跳过 0.10/0.08，跑剩余 13 组）
#    bash scripts/run_sweep.sh
```

## 10. 关键约束（务必遵守）

- 长句任务（FOMC/MeetingBank/Py150）+ Lima 回放必须开 `--gradient_checkpointing`，否则 4 卡 OOM 爆 78G
- 启动 run_replay_fast.sh 必须 `export OUT_ROOT=/dev/shm/outputs`（否则写 /root 的 28G 盘会爆）
- LLaMA-2 需 HF 授权 token（gated），Vicuna/Baichuan 公开
- 云端 `PATH=/root/miniconda3/bin`；远程 pkill/grep 匹配进程名须用 `[x]` 括号防杀 SSH 自身
- 云端是密码认证，密码存本地 `/tmp/askpass.sh`（WSL 重启会丢失，需重设）
