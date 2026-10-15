# 给协作者 agent 的任务

你协助Talos用户 **yzhong7** 复现 https://github.com/YumengggZhang/DPO-Two-Stage 。先读 `AGENTS.md`、`README.md`、`protocol.json`、`data/manifest.json`、`data/wild_manifest.json` 和 `assets.talos.json`。资源、权限和持久存储就绪后，完成以下**42项正式实验**及评测，修复常规执行错误；不要只给计划。不要改变实验定义或自行调参。

**A组：Qwen3-1.7B四个stage2 method，每项seed=2003/2004/2005，共12项。**

1. `method1_dpo`：H/B均标准DPO，LR7e-6。
2. `method2_chosen`：H chosen-only KL到π₁；B chosen-only KL到π₀，LR7e-6。
3. `method3`：H双路径KL到π₁；B DPO，LR3e-6。
4. `method4_best`：H DPO；B chosen+rejected双路径KL到π₀，λ1、LR1e-6。不是chosen-only method4。

四项共享原始π₁初始化、DPO reference及1000H+972B=1972对；一轮247步，同seed同顺序。不得以新基线checkpoint替换原π₁，不重新采样。

**B组：五模型 × Wild harmful-only / 全Wild mixed × 相同三个新seed，共30项。**

模型为Qwen3-0.6B、1.7B、8B；DeepSeek-R1-Distill-Qwen-1.5B、7B。setting为 `baseline_<model key>_wild_harmful` 和 `baseline_<model key>_wild_mixed`。

- harmful-only：固定828对Wild harmful，LR1e-5，batch8，一轮104步。
- mixed：同828H＋1000对Wild benign（498direct/502adversarial），LR3e-6，batch8，一轮229步。每seed洗牌一次、顺次batch8，不按4H+4B重复/删样。
- 两域prompt/chosen来自WildJailbreak、rejected来自原prompt精确匹配的WildGuardTrain公开回答，已有行为/质量筛选。**不要使用clean1000或EVO benign，也不要自采样回答。**
- 各自原π₀初始化/reference。五模型revision固定；优化器保持Qwen0.6/8 streaming CPU、Qwen1.7 GPU、R1-1.5 CPU、R1-7双GPU。
- 当前只Qwen1.7有匹配Wild历史实验，其余四模型是同数据扩展；采用同样固定LR和40步开发诊断，不自动调参或按测试删除seed。

**先准备每个模型自己的reference。** 五模型token编码已CPU验证，包含EOS、无截断。只提供Qwen1.7匹配Wild的原reference；其他四模型必须各自用原π₀对1828条mixed做teacher-forced forward，缓存chosen/rejected的sequence sums，按ID取828条harmful并跨seed复用。这不是训练或生成回答，不能拿Qwen1.7分数顶替其他模型。用README的 `reproduce.py prepare-references --assets assets.talos.json --output <新持久目录> --gpus 0 1 --offline`，或由run自动准备。遵守GPU锁，保存fingerprint与原始分数；中断续算，缓存模型/输入不匹配则拒绝复用。

所有DPO sequence-sum含EOS，β.1，两侧梯度，不做长度归一化；KL full-vocabulary forward teacher‖student，token mean后路径mean，λ1。共同batch8、1epoch、WD1e-4、grad clip1、5%warmup+cosine、dropout0；FP32参数/梯度/AdamW状态、BF16 forward。不要自行换成TRL默认设置。

本版修正了此前clean交接误解；若已有旧checkout，先更新，使用全新输出目录，不合并旧clean/EVO结果。clone到自己的目录，优先使用 `/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python`，不安装或升级共享环境。按README解包两个数据归档，执行verify-data、unittest及check_assets；哈希和权限须通过。模型revision、π₁、Selene和Guard不能换成近似版本。权限不通报告具体路径，不修改原私有缓存权限。

先确认自己的持久空间/配额：建议至少1.2TiB可用，或已有逐项验证归档安排。原home和/tmp余量不足以全部保留；不得清理Yumeng/其他人的数据、旧模型和恢复状态。不得仅保存/dev/shm或默默降低训练精度。

用 `reproduce.py plan`核对42项，用README的 `reproduce.py run --assets assets.talos.json --output <自己的持久目录> --gpus 0 1 --offline`执行。可用--arms/--seeds拆分既定任务。GPU遵守 `/home/users/yzhang92/dpo_checkpoint_sampling_k4_20260922/gpu{0,1}.lock`及显存<256MiB门槛。R1-7B训练同时取得两张卡，任一忙则等待；绝不结束其他任务或绕过锁。

每项先独立40步固定设置pilot，完整一轮scheduler horizon；原32H+32B开发生成/判分，记录有限性、截断/重复/空回答、unknown、thinking。质量gate失败保留诊断并继续固定设置，不能挑seed；技术、非有限和完整性错误须修复。正式训练必须从原π₀/π₁重启，不从pilot续训；完整训练后再做64条开发诊断。完整FP32模型及所有证据持久保存。中断后核对该setting/seed的日志、恢复状态和完成标记，以相同命令继续，不启动重复任务，不改冻结代码/协议。

每个正式模型评测全部4401条：863 harmful由原Guard判分，3538 benign由固定Selene分别判target/counter。正式benign排除200 benign_dev，分母3338；ASR分母863。生成temperature0、max_tokens8000、rep1.2、评测seed2002、无system message。Qwen enable_thinking=False；**R1保留原生thinking及完整native CoT，不传Qwen关闭thinking选项、不删推理再判分**。不能换judge或拒绝词正则，unknown保留原分母与上下界。

执行summarize，交付42项逐seed计数/比例、每setting mean±sample SD、生成健康度、各benchmark结果、原始回答/判分/日志、模型路径及环境/哈希；列出全部缺项和失败，不只报成功seed。seed2002历史结果单列，不并入新统计。SD反映训练随机性，不是数据总体置信区间。stage2与公开回答基线的来源、LR和曝光不同，不能作为纯时序消融；method4选择已参考历史测试，不声称独立开发选参。

不新增实验、采样、调参、自动化或向其他人发消息。遇到无法解决的权限/空间/资产缺失，保留进度并报告具体条件。
