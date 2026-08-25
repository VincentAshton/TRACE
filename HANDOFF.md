# TRACE 实验交接文档（HANDOFF）

> 目标读者：接手代码修复、实验恢复和结果验收的 AI agent / 工程协作者。
> 仓库：<https://github.com/VincentAshton/TRACE>
> 本文是当前修复完成后的最终交接口径；权威运行手册见 `RUNBOOK.md`，实验记录见 `EXPERIMENT_LOG.md`。

---

## 1. 一句话概述

在 **TRACE 持续学习基准**（复旦 2023，arXiv:2310.06762）上复现 **Replay（回放）** 方法，
研究：**当回放比例 `past_task_ratio` 从 10% 下降到 8%/5%/2%/1% 时，三个 7B 模型的 OP/BWT
在什么比例开始明显低于本地 10% baseline（「下降阈值」）。**

## 2. 实验设计

- **方法**：Replay（训练入口 `training/replay.py`）
- **模型**：llama2-7b-chat、vicuna-7b-v1.5、baichuan2-7b-chat（13B 已砍掉）
- **比例**：0.10（基线）、0.08、0.05、0.02、0.01
- **组合**：3 模型 × 5 比例 = **15 组**
- **指标**（论文 Section 3，T=8 任务）：
  - `OP` = 最后一轮对所有任务平均得分
  - `BWT` = 平均向后迁移（负值 = 遗忘）
- **基线策略**：用「自己环境跑的 10%」做基线（option B）。论文数值只作参考，**绝不回退为基线**。

### 2.1 与官方代码行为的关系（重要）

本实验**遵循 TRACE 官方公开 Replay 代码的实际行为**，不是按论文文字重新设计：

- replay 阶段保留 `RandomSampler`（**不得改为 DistributedSampler**，这是有意保留的兼容行为）；
- 每个当前任务训练结束后，额外 replay 1 个 epoch（历史任务 `past_task_ratio` 子集 + 完整 LIMA）；
- `past_task_ratio` 按原代码取数据前缀，不重新随机抽样；
- 保留官方任务顺序和训练流程。

### 2.2 冻结的并行配置（不可随意改）

| 项 | 固定值 |
|---|---|
| GPU | 4 × A100-80GB |
| per_device_train_batch_size | 4 |
| gradient_accumulation_steps | 16 |
| seed | 1234 |
| learning rate | 1e-5 |
| weight decay | 0 |
| 每任务 epoch | 5,3,7,5,3,5,5,7 |
| scheduler | constant（**代码实际用 constant，脚本的 `--lr_scheduler_type cosine` 参数无效**） |
| ZeRO | stage 3 |
| precision | bf16 |
| Llama/Vicuna attention | FlashAttention 2 |
| Baichuan attention | eager |

> ⚠️ 改变 GPU 数 / batch / 梯度累积 / 任务顺序 / epoch / seed / 数据版本 / 模型版本，
> 必须重跑对应模型的 10% baseline，且不得与旧配置结果混合。

## 3. 运行环境

- **云端**：4× A100-80GB（ebcloud 租用，按小时计费），`ssh -p 32433 root@ssh-cn-huabei1.ebcloud.com`（密码认证）
- **内存**：1TB；**`/dev/shm` 是 tmpfs 内存盘，人为限成 200GB**（磁盘满 bug 的根源）
- **软件**：torch 2.3.0+cu121、transformers 4.44.2、deepspeed 0.14.4、flash-attn 2.5.8、Python 3.10（`/root/miniconda3`）
- **关键路径**：
  - 代码 `/root/TRACE`（**云端不是 git 仓库，靠 scp 同步**）
  - 模型 `/dev/shm/hf/`（llama2 51G / vicuna 13G / baichuan 14G）
  - 输出 `/dev/shm/outputs/`（内存盘）；持久化 `/root/results/`（28G 小盘）
  - 数据 `/root/TRACE/data/extracted/TRACE-Benchmark/LLM-CL-Benchmark_5000`（8 任务 + Lima，原始 .json）
  - 数据缓存 `/tmp/data_files/*.pt`（tokenize 后缓存）
- **Git 仓库（本地 fork）不包含数据与结果**（`.gitignore` 忽略 `/data/ /results/ /outputs/`）。

## 4. 执行流程（脚本链路）

```
watcher.sh（等 ratio_0.10 的 .complete）
  └─ run_sweep.sh（fail-fast；skip 依据 .complete；每模型先检查模型路径存在）
       └─ run_replay_fast.sh（每组）
            ├─ 生成 run_manifest.json（run ID + 冻结配置 + 环境版本）
            ├─ 训练 deepspeed training/replay.py
            └─ run_infer_parallel.sh
                 ├─ 4 卡并行推理（逐 PID 等待，任一失败保留 checkpoint 退出）
                 ├─ aggregate_op_bwt.py（严格模式：36 项缺失/损坏即非零退出）
                 ├─ 持久化到 /root/results（先校验 op_bwt + 预测数==36，再原子 rename）
                 ├─ 持久化验证通过后才清理 checkpoint
                 └─ 最后 touch .complete（完成的权威标志）
```

**完成判定 = `.complete` 存在**（不是 `op_bwt.json` 存在）。`.complete` 是整个流程最后生成的文件。

## 5. 当前进度（2026-08-25）

- ✅ **阶段 1 代码修复（任务 A–I）**：完成，commit `6ea5afc`
- ✅ **阶段 2 冒烟测试**：通过（vicuna 极小数据全链路，冒烟值 OP=0.3523/BWT=0.0504，不代表正式结果）
- ✅ **阶段 3 恢复 llama2 0.10 baseline**：验证通过 OP=0.5443/BWT=0.0770，补 manifest（`validated_legacy_run`）+ `.complete`
- ✅ **三个模型已下载完成**：llama2（26G）/ vicuna（13G）/ baichuan（14G）
- ⏳ **阶段 4 正式实验**：待开始。llama2 0.08 因 checkpoint 随 `/dev/shm` 重启丢失，需整组重训（不混合新旧预测）

## 6. 已修复的问题（本次全部完成）

| 编号 | 问题 | 修复 |
|---|---|---|
| P0-1 | 缺结果仍生成 op_bwt.json | `aggregate_op_bwt.py` 严格模式：36 项缺失/损坏/越界即非零退出且不写文件（任务 A） |
| P0-2 | 失败重跑混用旧预测和新 checkpoint | `run_replay_fast.sh` 生成 run_manifest.json（run ID + 配置）；`infer_single.py` skip 前校验内容，损坏隔离（任务 B/C） |
| P0-3 | 持久化失败被忽略仍删 checkpoint | `run_infer_parallel.sh` 去掉 `\|\| true`，校验后原子持久化，验证通过才清 checkpoint（任务 D） |
| P0-4 | 旧损坏缓存不能自愈 | `data_utils.py` 文件锁 + 原子写 + load 失败自动隔离重建（任务 F） |
| P0-5 | sweep 失败继续 + 删模型 | `run_sweep.sh` fail-fast + 不自动删模型 + 模型路径检查（任务 E） |
| P0-6 | 输出非原子写，存在即视为有效 | 预测 JSON + op_bwt.json 全部改为 tmp+fsync+os.replace 原子写（任务 A/C） |
| P1-1 | 完成状态只依赖易失目录单一文件 | `.complete` + manifest 判定；持久化到 `/root/results`（任务 B/D/E） |
| P1-2 | 环境检查不阻断 | `check_env.py` 全面检查 + 失败非零退出（任务 G） |
| P1-3 | summary 回退论文 baseline | `summary.py` 缺本地 10% 时不判定阈值，论文仅展示（任务 H） |
| P1-4 | 文档误导 | 本文件 + RUNBOOK 已修正（任务 I） |
| 额外 | baichuan flash-attn 不支持 | `run_sweep.sh` 对 baichuan 用 eager（任务 E 的一部分） |

## 7. 代码改动清单（本次修复）

| 文件 | 改动 |
|---|---|
| `utils/aggregate_op_bwt.py` | 严格校验 + 原子写（任务 A） |
| `utils/data/data_utils.py` | 缓存并发锁 + 原子写 + 损坏自愈 + hash 纳入 sample_ratio（任务 F + 之前的并发修复） |
| `inference/infer_single.py` | 预测原子写 + skip 前内容校验 + 损坏隔离（任务 C） |
| `scripts/run_replay_fast.sh` | 生成 run_manifest.json（任务 B） |
| `scripts/run_infer_parallel.sh` | 安全持久化 + .complete（任务 D） |
| `scripts/run_sweep.sh` | fail-fast + .complete skip + 不删模型 + baichuan eager + 模型路径检查（任务 E） |
| `scripts/watcher.sh` | 等待 .complete，去掉手动清 checkpoint（任务 E） |
| `scripts/summary.py` | 缺本地 baseline 不判定 + EPS 显式（任务 H） |
| `scripts/check_env.py` | 全面环境检查 + 失败非零退出（任务 G） |
| `.gitignore` | `/data/ /results/ /outputs/` 只匹配根目录 |

## 8. 尚未解决 / 潜在风险

1. **磁盘满（无法扩容）**：`/dev/shm` 仅 200GB，且 k8s 容器无 mount 特权（`mount -o remount` 已尝试并失败）。**缓解**：代码已加「每组成功后及时清理 checkpoint + fail-fast」，200GB 已够用（正常峰值 185G，ratio_0.10 当初即 200G 下跑通）。
2. **baichuan 换 eager 后显存未验证**：7B 全参 + eager + 长序列 1024 在 4×A100-80GB 大概率可行，未实测。
3. **vicuna 正式数据未验证**：已通过极小数据冒烟测试（全链路 OK），但正式 5000 样本未跑过，需第一组正式 canary 实测。
4. **run_manifest 的 resume 校验是基础版**：记录了配置和 run ID，但尚未实现「推理前自动比对 manifest 与当前配置拒绝不匹配」的完整逻辑（当前靠 .complete 判定跳过，未做逐字段比对）。
5. **云端非 git 仓库**：代码靠 scp 同步，`git_commit` 在云端记录为 `unknown`（除非云端也 git init）。
6. **推理偏慢（待实测）**：冒烟测试短任务 6-13s/step、长任务（MeetingBank/Py150）30-56s/step，vs 训练 2.1s/step。疑 flash-attn 推理未完全生效或 vicuna 特性；待阶段 4 llama2 正式推理实测，必要时 profiling 优化（batch 16、max_ans_len 512、temperature 采样）。

## 9. 恢复实验的操作步骤（实例恢复后，按此顺序）

```bash
# 1. scp 同步本次所有修改到云端
scp -P 32433 utils/aggregate_op_bwt.py utils/data/data_utils.py \
    inference/infer_single.py root@<host>:/root/TRACE/...
scp -P 32433 scripts/{run_sweep,run_replay_fast,run_infer_parallel,watcher,summary,check_env}.sh \
    root@<host>:/root/TRACE/scripts/

# 2. 扩大内存盘（根治磁盘满）
ssh root@<host> 'mount -o remount,size=400G /dev/shm && df -h /dev/shm'

# 3. 备份现有产物 → 清理旧损坏缓存
#    （先备份 /tmp/data_files，再删历史 .pt/.tmp/.corrupt 缓存，让新代码重建）

# 4. 验证 llama2 0.10 baseline：36 个预测齐全、重新聚合仍得 OP=0.544/BWT=0.077，
#    补录 manifest 并标注 validated_legacy_run，创建 .complete

# 5. 判断 llama2 0.08 的 round 7 checkpoint 是否可加载：
#    - 可加载 → 清理坏缓存后只补跑 round 7 缺失任务，严格聚合并持久化
#    - 不可加载 → 归档旧产物，新 run ID 整组重训 0.08

# 6. 重新下载 vicuna + baichuan 模型（~27GB）

# 7. 低成本冒烟测试（4-GPU 启动 + ZeRO-3 保存/加载 + 单任务推理 + 严格聚合 + 持久化 + .complete）

# 8. 每个模型先跑 10% canary，再跑低比例；每组完成后立即校验 + 持久化
```

## 10. 关键约束（务必遵守）

- 长句任务（FOMC/MeetingBank/Py150）+ Lima 回放必须开 `--gradient_checkpointing`（否则 4 卡 OOM）
- 启动 run_replay_fast.sh 必须 `export OUT_ROOT=/dev/shm/outputs`（否则写 /root 28G 盘会爆）
- LLaMA-2 需 HF 授权 token；Vicuna/Baichuan 公开
- 云端 `PATH=/root/miniconda3/bin`；远程 pkill/grep 匹配进程名用 `[x]` 括号防杀 SSH 自身
- **禁止**：改 RandomSampler、改任务顺序、正式组间改 batch/accumulation、用论文数值替代缺失 baseline、按 op_bwt.json 存在判定完成、持久化校验前清 checkpoint、失败后自动删模型、混用新旧训练预测
- 完成判定只看 `.complete`；失败即 fail-fast，不继续烧卡
