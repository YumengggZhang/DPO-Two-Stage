# Two-stage safety alignment：多 seed 复现

本仓库交给 Talos 用户 **yzhong7**，包含两组独立比较：**Qwen3-1.7B 的四个第二阶段 method × 3 seeds =12项**，以及**历史五个模型 × harmful-only / mixed × 3 seeds =30项**，合计 **42个正式实验**。训练 seed 固定为 **2003、2004、2005**；每项另有40步开发预检查。数据、编码和 reference 分数已经提供，不需要自采样。

完整执行说明：[AGENT_PROMPT.md](AGENT_PROMPT.md)。机器可读设置：[protocol.json](protocol.json)。本次整理交接包，没有启动这42项GPU实验。

## A：四个 Qwen 第二阶段 method

| Setting | harmful objective | benign objective | LR | 数据 / 步数 |
|---|---|---|---:|---|
| `method1_dpo` | DPO | DPO | 7e-6 | 1000H+972B /247 |
| `method2_chosen` | chosen-only KL →π₁ | chosen-only KL →π₀ | 7e-6 | 同上 |
| `method3` | 双路径 KL →π₁ | DPO | 3e-6 | 同上 |
| `method4_best` | DPO | 双路径 KL →π₀，λ=1 | 1e-6 | 同上 |

初始化为同一个原始π₁；所有 DPO reference 固定为该π₁。method4采用历史数值最好的**chosen+rejected双路径**设置，不是chosen-only版本。该选择已参考历史正式测试，不能称为新的独立开发选参。同seed四个method的训练顺序一致；新基线checkpoint不替换原π₁。

## B：五个模型的公开回答 DPO 基线

| 模型 | Setting 名称中的 model key | 原历史优化器 | 训练所需GPU |
|---|---|---|---:|
| Qwen3-0.6B | `qwen3_06b` | Streaming CPU AdamW | 1 |
| Qwen3-1.7B | `qwen3_17b` | GPU AdamW | 1 |
| Qwen3-8B | `qwen3_8b` | Streaming CPU AdamW | 1 |
| DeepSeek-R1-Distill-Qwen-1.5B | `r1_15b` | CPU AdamW | 1 |
| DeepSeek-R1-Distill-Qwen-7B | `r1_7b` | GPU0模型/梯度、GPU1 moments/更新 | 2 |

每个模型分别跑 `baseline_<model key>_clean1000` 和 `baseline_<model key>_mixed2000`：

- **clean1000 / harmful-only**：1000 harmful，batch8，一轮125步。
- **mixed2000**：同1000 harmful＋1000 EVO benign；每batch固定4H+4B，一轮250步。
- 两项均从各自原始π₀初始化，reference固定为该π₀；**LR全部1e-5**。这次复现的是历史clean/mixed五模型比较，不是828H的Wild对照。

模型revision与逐文件哈希见 `provenance/baseline_model_identities.json`。Qwen3-32B曾取消，不在这五个模型中。历史Qwen0.6B harmful-only出现严重重复/截断，本次固定设置复现仍保留全部seed及退化诊断，不根据正式测试换LR或删seed。原结果见 `provenance/historical_five_model_results.md`。

## 共同协议与数据

DPO使用sequence-sum log ratio（含EOS），β=.1，正负两侧梯度；无length normalization。KL为完整词表forward teacher‖student，回答token mean后路径mean，系数1。共同batch8、1epoch、AdamW WD1e-4、grad clip1、5%warmup+cosine、dropout0，FP32参数/梯度/优化器状态、BF16 forward。保留各模型历史优化器位置，不用TRL默认设置替换。

GitHub含冻结训练bank、token encodings、固定reference sums，以及原64条开发题和4401条评测题。执行解包后逐文件核验SHA。原Wild828/1828数据保留为历史来源证据，**不进入本次42项默认清单**。828的来源是两种rejected来源对照中共同通过筛选的prompt子集，不是公开数据总量上限。

Qwen训练保留原helpful system prefix且关闭thinking；R1基线沿用历史public final_text的编码（显式结束空thinking段并含EOS），这是既有第一阶段公开回答基线，**不是已退役的R1 final-only第二阶段自采bank**。R1推理保留原生thinking前缀及完整native CoT，不能删推理或强制Qwen的non-thinking设置。

两组比较的数据分布、回答来源、参考模型和训练曝光不同，不是纯时序因果消融。新seed衡量给定冻结数据/checkpoint条件下的训练随机性；不涵盖采样或原π₁不确定性。旧seed2002不并入新三seed统计。

## Talos运行

clone到yzhong7自己的可写位置，复用既有环境，不安装/升级Yumeng的环境：

```bash
git clone https://github.com/YumengggZhang/DPO-Two-Stage.git
cd DPO-Two-Stage
TWOSTAGE_PYTHON=/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python
"$TWOSTAGE_PYTHON" tools/unpack_data.py
"$TWOSTAGE_PYTHON" reproduce.py verify-data
"$TWOSTAGE_PYTHON" -m unittest discover -s tests -v
"$TWOSTAGE_PYTHON" tools/check_assets.py --assets assets.talos.json
"$TWOSTAGE_PYTHON" reproduce.py plan
```

共享资产只读；权重留在Talos，不放Git。`assets.talos.json`同时列出原π₁、五个π₀、Guard和Selene；精确环境版本见 `provenance/environment_versions.json`。

先落实自己的持久存储与配额。保留42份FP32最终模型、BF16开发模型和单次原子恢复状态时，建议准备 **至少1.2TiB可用空间**，或事先落实逐项核验的持久归档方案。当前Talos原home和/tmp的余量不足。不得删除旧实验、改用低精度训练或仅保存内存盘模型解决空间问题。

```bash
TWOSTAGE_OUTPUT=/path/to/your/persistent/twostage_42seeds_20261014
"$TWOSTAGE_PYTHON" reproduce.py run \
  --assets assets.talos.json --output "$TWOSTAGE_OUTPUT" --gpus 0 1 --offline
"$TWOSTAGE_PYTHON" reproduce.py summarize --output "$TWOSTAGE_OUTPUT"
```

运行器串行处理全部42项，按setting/seed独立保存；可用 `--arms method4_best --seeds 2003` 选择既定子集。R1-7B训练须同时取得两张空闲卡的共享锁；评测及其他训练使用一张。显存<256MiB才占卡，任一卡忙则释放尝试中的锁并等待，不抢占其他任务。中断后检查日志/状态，以相同命令继续，不重复已完成项；配置/代码不匹配时须新目录。

每项先跑40步固定设置开发pilot，scheduler仍以完整一轮为horizon；做原32H+32B生成与判分，**从原初始化重新开始**完整训练，再做同64条开发诊断。质量gate失败保留诊断并继续固定设置，避免挑seed；非有限、完整性和执行错误必须修复后继续。新seed不调参、不按测试筛选。

每项完成原4401条评测：863H用Llama-Guard-3-8B，3538B用固定Selene（revision `023227c22156b0273b0693c82c65bd7084a50e92`）。正式benign排除200条benign_dev，分母3338。temperature0、max_new_tokens8000、repetition_penalty1.2、评测seed2002、无system message；Qwen thinking=False，R1原生thinking。保留完整回答和所有unknown。

`aggregate.json` / `RESULTS.md`报告全部42项、每个setting三seed的mean±sample SD、逐seed计数、unknown上下界及生成健康度；缺项明确标incomplete。上下界不是置信区间。完整模型、原始回答/判分和日志须一并交付。
