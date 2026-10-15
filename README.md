# Two-stage safety alignment：多 seed 复现

本仓库交给 Talos 用户 **yzhong7**，包含两组独立比较：**Qwen3-1.7B 的四个第二阶段 method × 3 seeds =12项**，以及**五个模型 × Wild harmful-only / 全Wild mixed × 3 seeds =30项**，合计 **42个正式实验**。训练 seed 固定为 **2003、2004、2005**；每项另有40步开发预检查。数据和五模型编码已提供；五模型都由运行器计算新数据上的自身π₀分数，不需要自采样。

完整执行说明：[AGENT_PROMPT.md](AGENT_PROMPT.md)。机器可读设置：[protocol.json](protocol.json)。本次整理交接包，没有启动这42项GPU实验。

## A：四个 Qwen 第二阶段 method

| Setting | harmful objective | benign objective | LR | 数据 / 步数 |
|---|---|---|---:|---|
| `method1_dpo` | DPO | DPO | 7e-6 | 1000H+972B /247 |
| `method2_chosen` | chosen-only KL →π₁ | chosen-only KL →π₀ | 7e-6 | 同上 |
| `method3` | 双路径 KL →π₁ | DPO | 3e-6 | 同上 |
| `method4_best` | DPO | 双路径 KL →π₀，λ=1 | 1e-6 | 同上 |

初始化为同一个原始π₁；所有 DPO reference 固定为该π₁。method4采用历史数值最好的**chosen+rejected双路径**设置，不是chosen-only版本。该选择已参考历史正式测试，不能称为新的独立开发选参。同seed四个method的训练顺序一致；新基线checkpoint不替换原π₁。

## B：五个模型的 Wild 公开回答 DPO 基线

| 模型 | model key | 优化器 | 训练GPU |
|---|---|---|---:|
| Qwen3-0.6B | `qwen3_06b` | Streaming CPU AdamW | 1 |
| Qwen3-1.7B | `qwen3_17b` | GPU AdamW | 1 |
| Qwen3-8B | `qwen3_8b` | Streaming CPU AdamW | 1 |
| DeepSeek-R1-Distill-Qwen-1.5B | `r1_15b` | CPU AdamW | 1 |
| DeepSeek-R1-Distill-Qwen-7B | `r1_7b` | GPU0模型/梯度、GPU1 moments/更新 | 2 |

每个模型跑以下两项、各三个新seed：

- `baseline_<model key>_wild_harmful`：同1000对Wild harmful，LR1e-5，batch8，一轮125步。
- `baseline_<model key>_wild_mixed`：同1000H＋1000对**Wild benign**，LR3e-6，batch8，一轮250步。每seed固定洗牌一次，按8条分batch；不补样、不按4H+4B重复数据。

两项均从各自原π₀初始化，reference固定为该π₀。本版已经纠正上一版clean1000/mixed2000的误解：**当前基线不使用clean1000或EVO benign**。五模型集合与三个新seed不变，Qwen四个stage2 method也不变。Qwen1.7已有828/1828的历史Wild实验；本版1000/2000是新数据版本，五模型都需重新运行，不把旧结果称为当前配置结果。固定采用既有Qwen Wild LR，所有seed先做40步开发诊断，无自动调参或测试挑seed。

harmful与benign的prompt/chosen都来自WildJailbreak；rejected是原prompt精确匹配的WildGuardTrain公开回答，保留行为和完整配对质量筛选。1000 benign为498 direct＋502 adversarial，沿用已验证冻结bank。无需生成新的训练回答。原828仅是历史on/off来源对照的共同合格prompt子集。本版按用户决定取消此限制：保留原828条，从2057条已通过公开off-policy行为及完整质量检查的记录中，按固定seed2002补172条；补充项不要求学生侧成功。原文、筛选标准与heldout排除保留。

原模型revision/逐文件身份见 `provenance/baseline_model_identities.json`。训练FP32优化器放置沿用原模型方案。原clean五模型结果仅是历史参考，不能当成本次Wild结果。

### 固定π₀ reference准备

五模型共享同1000H/2000mixed文本和相同seed顺序，各自π₀权重不同，**不能跨模型共享reference分数**。新数据改变了bank和scheduler，旧828/1828缓存不完整；运行器为五模型各自对新2000对mixed计算一次teacher-forced sequence sums，再按IDs取1000条harmful，跨三个seed复用。0优化步、0新训练回答。

`reproduce.py run`会在需要时自动执行并校验；也可先单独执行下方prepare-references。使用一张空闲GPU和共享锁，保存模型/数据/编码/代码fingerprint并支持断点续算。必须使用新输出目录，不能复用旧828/1828运行状态。制包阶段不执行这些GPU forward。

## 共同协议与数据

DPO使用sequence-sum log ratio（含EOS），β=.1，正负两侧梯度；无length normalization。KL为完整词表forward teacher‖student，回答token mean后路径mean，系数1。共同batch8、1epoch、AdamW WD1e-4、grad clip1、5%warmup+cosine、dropout0，FP32参数/梯度/优化器状态、BF16 forward。保留各模型历史优化器位置，不用TRL默认设置替换。

GitHub含冻结训练bank、token encodings、固定reference sums，以及原64条开发题和4401条评测题。执行解包后逐文件核验SHA。原828/EVO mixed文件随最初不可变归档仅作历史与π₁来源证据；本版默认基线引用新增wild1000_harmful/wild2000_mixed和对应两系列编码。

Qwen训练保留原helpful system prefix且关闭thinking；R1基线将冻结公开回答按既有public final_text协议编码（显式结束空thinking段并含EOS），这是既有第一阶段公开回答基线，**不是已退役的R1 final-only第二阶段自采bank**。R1推理保留原生thinking前缀及完整native CoT，不能删推理或强制Qwen的non-thinking设置。

两组比较的数据分布、回答来源、参考模型和训练曝光不同，不是纯时序因果消融。新seed衡量给定冻结数据/checkpoint条件下的训练随机性；不涵盖采样或原π₁不确定性。旧seed2002不并入新三seed统计。

## Talos运行

clone到yzhong7自己的可写位置，复用既有环境，不安装/升级Yumeng的环境：

```bash
git clone https://github.com/YumengggZhang/DPO-Two-Stage.git
cd DPO-Two-Stage
TWOSTAGE_PYTHON=/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python
"$TWOSTAGE_PYTHON" tools/unpack_data.py
"$TWOSTAGE_PYTHON" reproduce.py verify-data
CUDA_VISIBLE_DEVICES="" "$TWOSTAGE_PYTHON" -m unittest discover -s tests -v
"$TWOSTAGE_PYTHON" tools/check_assets.py --assets assets.talos.json
"$TWOSTAGE_PYTHON" reproduce.py plan
```

共享资产只读；权重留在Talos，不放Git。`assets.talos.json`同时列出原π₁、五个π₀、Guard和Selene；精确环境版本见 `provenance/environment_versions.json`。

先落实自己的持久存储与配额。保留42份FP32最终模型、BF16开发模型和单次原子恢复状态时，建议准备 **至少1.2TiB可用空间**，或事先落实逐项核验的持久归档方案。当前Talos原home和/tmp的余量不足。不得删除旧实验、改用低精度训练或仅保存内存盘模型解决空间问题。

```bash
TWOSTAGE_OUTPUT=/path/to/your/persistent/twostage_allwild1000_42seeds_20261015
"$TWOSTAGE_PYTHON" reproduce.py prepare-references \
  --assets assets.talos.json --output "$TWOSTAGE_OUTPUT" --gpus 0 1 --offline
"$TWOSTAGE_PYTHON" reproduce.py run \
  --assets assets.talos.json --output "$TWOSTAGE_OUTPUT" --gpus 0 1 --offline
"$TWOSTAGE_PYTHON" reproduce.py summarize --output "$TWOSTAGE_OUTPUT"
```

运行器串行处理全部42项，按setting/seed独立保存；可用 `--arms method4_best --seeds 2003` 选择既定子集。R1-7B训练须同时取得两张空闲卡的共享锁；评测及其他训练使用一张。显存<256MiB才占卡，任一卡忙则释放尝试中的锁并等待，不抢占其他任务。中断后检查日志/状态，以相同命令继续，不重复已完成项；配置/代码不匹配时须新目录。

每项先跑40步固定设置开发pilot，scheduler仍以完整一轮为horizon；做原32H+32B生成与判分，**从原初始化重新开始**完整训练，再做同64条开发诊断。质量gate失败保留诊断并继续固定设置，避免挑seed；非有限、完整性和执行错误必须修复后继续。新seed不调参、不按测试筛选。

每项完成原4401条评测：863H用Llama-Guard-3-8B，3538B用固定Selene（revision `023227c22156b0273b0693c82c65bd7084a50e92`）。正式benign排除200条benign_dev，分母3338。temperature0、max_new_tokens8000、repetition_penalty1.2、评测seed2002、无system message；Qwen thinking=False，R1原生thinking。保留完整回答和所有unknown。

`aggregate.json` / `RESULTS.md`报告全部42项、每个setting三seed的mean±sample SD、逐seed计数、unknown上下界及生成健康度；缺项明确标incomplete。上下界不是置信区间。完整模型、原始回答/判分和日志须一并交付。

此前下载过clean或828版的执行者应更新仓库并使用新的输出目录；不要把旧clean配置/缓存或旧EVO mixed结果合并进本版统计。
