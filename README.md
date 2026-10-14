# Qwen two-stage safety alignment: fixed-setting seed replication

本仓库是给协作者 **yzhong7** 的可运行交接包。复现 Qwen3-1.7B 的四个第二阶段 method，以及第一阶段 harmful-only / mixed DPO 对照。每个设置使用 **2003、2004、2005 三个新训练 seed**，共 **18 个正式实验**；每个正式实验另有一个 40 步开发预检查。

给 agent 的完整任务说明：[AGENT_PROMPT.md](AGENT_PROMPT.md)。机器可读设置：[protocol.json](protocol.json)。本次只整理代码、数据和共享资产，**没有启动这 18 个实验**。

## 固定实验表

| Setting | harmful objective | benign objective | 初始化 / DPO reference | LR | 数据 | 一轮步数 |
|---|---|---|---|---:|---|---:|
| `method1_dpo` | DPO | DPO | π₁ / π₁ | 7e-6 | 1000 H + 972 B | 247 |
| `method2_chosen` | chosen-only KL → π₁ | chosen-only KL → π₀ | π₁；无 DPO | 7e-6 | 同上 | 247 |
| `method3` | chosen/rejected KL → π₁ | DPO | π₁ / π₁ | 3e-6 | 同上 | 247 |
| `method4_best` | DPO | chosen/rejected KL → π₀，λ=1 | π₁ / π₁ | 1e-6 | 同上 | 247 |
| `dpo_harmful_only` | DPO | — | π₀ / π₀ | 1e-5 | 原 828 H | 104 |
| `dpo_mixed` | DPO | DPO | π₀ / π₀ | 3e-6 | 同 828 H + 原 1000 EVO B | 229 |

所有 DPO 都是 **sequence-sum log ratio，包含 EOS，β=0.1，两侧都有梯度**；不做长度归一化。KL 是 **完整词表 forward KL（teacher‖student）**，先对回答 token 取平均，再对路径取平均；双路径不作正负相减。共同 batch=8、1 epoch、AdamW weight decay=1e-4、grad clip=1、5% warmup + cosine，FP32 参数/梯度/优化器状态，BF16 forward，dropout=0。

`method4_best` 指目前历史正式测试中数值最好的配置，不是 chosen-only λ=.3 或 chosen-only λ=1；本次选择已看过历史测试结果，不能称为新的独立开发选参。`dpo_mixed` 复现后来 LR=3e-6 的稳定对照，不是早期不稳定的 LR=1e-5 版本。

## 数据和 seed 的定义

GitHub 包含三套**原始冻结数据、原始 token encodings、原始固定 reference scores**，以及 64 条开发题和 4401 条评测题。数据压缩为带哈希的分片；还原后与原文件逐字节一致，不需要重新采样、筛选或下载上游训练集。

四个 stage2 method 固定同一原始 π₁ 和同一份 1972 对数据。新 seed 改变训练 RNG 和数据顺序；同一 seed 的四个 method 使用完全相同的顺序。π₁ 不随 seed 重新训练，三个新 harmful-only checkpoint **不替换** stage2 的初始化或 teacher。这里测量的是固定数据/π₁ 条件下的训练随机性，不涵盖数据采样或第一阶段 checkpoint 的不确定性。

harmful-only、mixed、stage2 的回答来源和总曝光不同。这是历史设置的重复实验，不能作为只改变训练时序的因果消融。原 seed=2002 的结果放在 `provenance/historical_results.json`，不计入三 seed 均值。

## 在 Talos 上运行

先从 GitHub clone 到 **yzhong7 自己可写的位置**。优先复用原环境，禁止升级或改写 Yumeng 的环境、旧数据和旧实验目录。

```bash
git clone https://github.com/YumengggZhang/DPO-Two-Stage.git
cd DPO-Two-Stage
TWOSTAGE_PYTHON=/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python
"$TWOSTAGE_PYTHON" tools/unpack_data.py
"$TWOSTAGE_PYTHON" -m unittest discover -s tests -v
"$TWOSTAGE_PYTHON" reproduce.py verify-data
"$TWOSTAGE_PYTHON" tools/check_assets.py --assets assets.talos.json
"$TWOSTAGE_PYTHON" reproduce.py plan
```

`assets.talos.json` 已指向给 yzhong7 准备的只读共享模型目录。π₀、π₁、Guard 和 Selene 都有文件身份核验；π₁ 是原始精确 FP32 checkpoint，不是从临时 method4 导出恢复的模型。大模型权重留在 Talos，不放进 Git 对象。环境精确版本见 `provenance/environment_versions.json`；`requirements.txt` 仅供新环境重建，**不要在原环境执行安装或升级**。

选择至少约 **250 GiB 可用的持久输出空间**，足以保留 18 个 FP32 最终模型、开发模型和临时恢复状态。Talos 当前磁盘总剩余空间不足以直接保留所有输出；先检查自己的 quota 和存储位置，缺空间则落实持久归档方案后再开始。不要通过删除其他实验解决空间问题，也不要把 `/dev/shm` 当持久模型保存位置。

```bash
# 替换为 yzhong7 自己已确认容量的持久目录。
TWOSTAGE_OUTPUT=/path/to/your/persistent/twostage_seeds_20261014
"$TWOSTAGE_PYTHON" reproduce.py run \
  --assets assets.talos.json --output "$TWOSTAGE_OUTPUT" \
  --gpus 0 1 --offline
"$TWOSTAGE_PYTHON" reproduce.py summarize --output "$TWOSTAGE_OUTPUT"
```

运行器串行调度，等待共享 GPU 锁且显存 <256 MiB 才占卡；每个 GPU 阶段使用独立子进程释放模型。不得绕过已有锁、抢占或终止其他用户进程。可以用 `--arms method4_best --seeds 2003` 选择已授权的子集；重新执行相同命令会复用匹配完成标记或同一次训练的恢复状态，不重复训练已完成的 seed。改变代码、设置或输入需要新的输出目录。

## 预检查和评测

每个 seed 的 40 步 pilot 使用完整一轮的 scheduler horizon。检查优化有限性、梯度、导出哈希和原 32H+32B 开发生成，再**从原初始化重新开始**完整一轮。开发质量 gate 按原标准记录，失败仍保留并按固定设置继续，避免根据 seed 质量挑选结果；执行/哈希/非有限数值错误必须修复后继续。同一规则用于所有设置，不因正式测试改参数。

完整训练后，再做相同 64 条开发诊断和全部 4401 条原评测：863 harmful 由原 Llama-Guard-3-8B 判分；3538 benign 由固定 Selene 版本独立判 target/counter。正式 benign 统计排除 200 条 `benign_dev`，分母为3338。生成 temperature=0、max new tokens=8000、repetition penalty=1.2、Qwen thinking=False，评测无 system message；训练仍保留原 helpful system prefix。

`aggregate.json` 和 `RESULTS.md` 汇总全部新 seed 的 ASR、过拒、明确 helpful 的 mean / sample SD，并保留计数、unknown 和逐 seed 结果。unknown 上下界不是置信区间。缺少任一 seed 时汇总明确标记 incomplete。原始生成、判分、完整模型和日志必须保存；不要只交一个均值。

## 文件

- `vendor/`：原成功实验的数值内核、训练、开发生成及判分逻辑；移植改动记录在 `provenance/vendor_sources.json`。
- `data/manifest.json`、`data/archive.json`：原数据指纹、原始来源、压缩分片和逐文件 SHA256。
- `provenance/checkpoint_identities.json`：π₀ / π₁ 精确文件身份。
- `tools/export_talos_data.py`：只读重新取回原文件，通常无需再运行。
- `tools/prepare_shared_assets.py`：原资产 owner 为指定协作者准备只读共享资产，通常无需再运行。
- `AGENTS.md`：本仓库 agent 执行边界。

上游数据与评测来源见 [DATA.md](DATA.md)。GPU 运行前仍需执行预检查；代码整理阶段的 CPU 验证不等于完整训练已在新 seed 上通过。
