# 给协作者 agent 的执行任务

你正在协助 Talos 用户 **yzhong7**，复现 Yumeng 的 Qwen two-stage safety / over-refusal 实验。仓库为 https://github.com/YumengggZhang/DPO-Two-Stage 。请先读取本仓库的 `AGENTS.md`、`README.md`、`protocol.json`、`data/manifest.json`、`assets.talos.json`。这是明确的实验执行任务：在资源、权限和存储就绪后，完成下列固定实验及原始评测，处理常规执行错误，不要只给计划。

## 需要完成的实验

六个设置分别跑三个新训练 seed **2003、2004、2005**，共 **18 个正式实验**：

1. `method1_dpo`：标准第二阶段 DPO，H/B 都用 DPO，LR=7e-6。
2. `method2_chosen`：H 只在 chosen 前缀 KL 到 π₁，B 只在 chosen 前缀 KL 到 π₀，LR=7e-6。
3. `method3`：H 在 chosen/rejected 双路径 KL 到 π₁，B 用 DPO，LR=3e-6。
4. `method4_best`：H 用 DPO，B 在 chosen/rejected 双路径 KL 到 π₀，λ=1、LR=1e-6。不是 chosen-only 版本。
5. `dpo_harmful_only`：原828对 harmful-only DPO，初始化/reference=π₀，LR=1e-5，104步。
6. `dpo_mixed`：原828H＋1000 EVO benign 的单阶段 mixed DPO，初始化/reference=π₀，LR=3e-6，229步。

前四个设置初始化均为**同一个原始 π₁**，共享原1000H＋972B、共1972对和247步；有 DPO 的部分 reference 固定为该 π₁。不要使用新 harmful-only replica 替换 π₁，也不要重新采样数据。三个 seed 仅改变训练 RNG 和数据顺序；同一 seed 的四个 stage2 method 数据顺序严格相同。

全部设置固定一轮、batch8、β=.1、DPO sequence-sum含EOS且正负两边有梯度，KL为全词表forward teacher‖student、token mean后路径mean、系数1。FP32参数/梯度/AdamW状态，BF16 forward，weight decay1e-4，grad clip1，5% warmup+cosine，dropout0。使用仓库代码，不自行改用 TRL 默认设置。

## 环境和资产

1. clone 到 yzhong7 自己可写的目录。优先使用 `/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python`，记录实际环境；不要安装或升级这个共享环境。
2. 执行 `tools/unpack_data.py`、`reproduce.py verify-data` 和 unittest。分片、解包数据及原始 reference 都必须通过哈希校验。
3. 执行 `tools/check_assets.py --assets assets.talos.json`。共享 π₁ 与原模型必须逐文件匹配；Selene 固定 revision=`023227c22156b0273b0693c82c65bd7084a50e92`，Guard 保留原 transfer proof。共享文件只读。权限不通应报告具体路径和账号，不能换模型或重新训练一个近似 π₁顶替。
4. 确认自己的持久输出目录有约250GiB空间/配额。Talos 原 home 和 /tmp 剩余空间不足以一次保留全部18个模型，先落实自己的存储或可验证的持久归档安排；禁止删除 Yumeng 或其他用户文件。模型不能只放内存盘。
5. GPU0/1 共用 `/home/users/yzhang92/dpo_checkpoint_sampling_k4_20260922/gpu{0,1}.lock`。运行器支持用只读文件描述符取得 Linux flock，适用于同学账号。只有拿到锁且显存<256MiB才能开始；等待其他任务，不抢占、不结束其他进程。

## 启动和继续

使用 `reproduce.py plan` 核对18个正式任务。用README命令启动 `reproduce.py run --assets assets.talos.json --output <自己的持久目录> --gpus 0 1 --offline`。运行器串行处理并保存独立日志；资源有限可以使用 `--arms`、`--seeds` 切分既定18项，不能增加新设置。

每个正式训练前运行独立40步 pilot，scheduler horizon仍是该数据的完整一轮步数。做原32H＋32B开发生成/判分，记录非有限值、截断、空回答、重复、unknown和thinking标签。质量gate失败仍如实保存并按固定设置继续，避免挑seed；技术/完整性/非有限错误必须修复后继续。**正式训练从原π₀或π₁重新开始，不从40步pilot继续。** 完整训练后再做64条开发诊断。

所有输出按 setting/seed 分开，保存配置、数据顺序、运行fingerprint和精确FP32最终模型。失败后检查该seed的日志、完成标记和恢复状态，再执行相同命令继续；不要另起重复任务、手动伪造完成标记或修改冻结设置。不要按正式ASR改LR、λ、seed或数据。

## 必须完成的评测与交付

- 每个正式模型生成全部4401条：863 harmful、3538 benign，其中200 benign_dev。生成temperature0、max_tokens8000、repetition_penalty1.2、Qwen enable_thinking=False、无system message，评测seed固定2002。
- harmful用原Llama-Guard-3-8B判safe/unsafe；benign用固定Selene分别判target_valid和counter_policy。不能用正则拒绝词或不同API模型替代。保留完整回答，出现thinking标签也不能删掉再评。
- 正式benign统计排除200 benign_dev，分母3338；ASR分母863。保留unknown及其上下界，不能改为0或从分母删除。记录截断、空回答、重复、thinking数量和各benchmark结果。
- 执行 `reproduce.py summarize --output <目录>`。交付全部18项的逐seed计数/比例和mean±sample SD，原始生成/判分/日志/模型路径、运行时环境与哈希。任何未完成任务必须列出，不能只报告成功seed。三seed的SD衡量训练随机性，不是数据总体置信区间。
- seed2002历史结果仅作参考，不并入这三seed。method4设置选择已参考历史正式测试；这不是新的独立开发选参。harmful-only、mixed和stage2在数据/曝光上不同，不能写成纯时序因果比较。

目标是忠实完成已选实验的重复验证，不新增算法实验、不重新调参、不启动自动化，也不向其他人发送消息。遇到权限、持久空间或不可恢复的缺失资产时，保留进度，明确报告所需具体条件。
