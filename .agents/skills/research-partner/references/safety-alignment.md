# Safety alignment、DPO/KL 与 over-refusal

把本文件用作诊断指引，不把例子当作用户的真实结果。实际模型、数据、指标和协议以当前材料为准。

## 先确定评价对象

分别评价 harmful compliance/ASR、合适的 refusal、安全替代回答、benign false refusal、benign 任务正确性与生成健康度。减少回答可能降低 ASR，也可能损害有效回答，不能把安全等同于拒绝率。

检查 prompt 的真实意图与歧义、judge 的规则与版本、unknown/截断/空输出处理、分母和类别组成。必要时先人工复核一小批逐样本差异，再讨论模型机制。不要用拒绝关键词匹配替代语义判断。

## 概率偏移与 OOD 的竞争解释

若 harmful compliance 下降而 benign false refusal 上升，先考虑至少这些候选来源：

| 候选机制 | 可区分的预测 | 优先诊断 | 解释限制 |
| --- | --- | --- | --- |
| 广泛的 refusal 相对概率上移 | 多种 benign 类别的 refusal-vs-answer margin 同向移动 | 固定 prompt/候选回答，比较训练前后 margin；配合逐样本生成 | Margin 是候选回答条件下的量，不代表全部 refusal 概率质量 |
| 分布/类别泛化失败 | 训练分布外或特定语义类别的恶化更集中 | 预先定义 ID/OOD、语义类别和词面相似度；比较配对变化 | OOD 可能更敏感于共同概率偏移；观察分组差异不能独立证明机制 |
| 数据、reference 或训练程序差异 | 效应随回答来源、reference、曝光或模板而改变 | 检查冻结 bank/配置，做条件匹配的对照 | 多因素同时变化时只能支持整体比较 |
| 评测或生成退化 | 假拒绝集中在截断、重复、judge 分歧或模板变化样本 | 原始回答复核、生成健康度、统一评分 | 修复评价后重新核验现象 |

不要把 H1 与 H2 设成必然互斥。若它们对当前实验做出同一预测，明确该实验只能诊断现象，继续设计更有区分性的干预。

对固定候选拒绝回答 y_R 与安全、有效的 benign 回答 y_A，定义：

\[
m_\theta(x)=\log\pi_\theta(y_R\mid x)-\log\pi_\theta(y_A\mid x),\qquad
\Delta m=m_{\mathrm{after}}-m_{\mathrm{before}}.
\]

使用同一组候选、tokenization、模板、EOS 与 reduction。报告两个 log probability 分量，区分拒绝回答上升与有效回答下降。检查回答长度影响；token mean 可作单独诊断，但不能冒充原 sequence-sum 训练目标。使用多个语义有效的候选或稳健性检查，避免一个措辞驱动结论。

优先复用 checkpoints 与已保存输出；所需 logits 不可得时给出可运行方案，不声称已完成评分。只有新 intervention 能隔离机制时，才把结论升级为因果解释。

## DPO 的机制分析

对固定 reference、chosen y_w 与 rejected y_l，先核对实际实现。标准 sequence-level 形式为：

\[
z_\theta=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)},\qquad
\ell=-\log\sigma(\beta z_\theta).
\]

在 reference 固定、无额外权重时直接求导得到：

\[
\nabla_\theta\ell=-\beta\sigma(-\beta z_\theta)
\left[\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\right].
\]

把这视为当前样本的 objective 梯度推导，不视为 benign prompts 上 refusal 必然上升的证明。先确认 harmful/benign 两类的 chosen/rejected 分别是什么，并检查共享参数带来的跨样本耦合。

检查 β、reference、detach、loss weighting、EOS、sequence sum/length normalization、batch composition、reward margin 饱和、optimizer 与学习率是否可比。不要从 β 的数值单独推断拒绝变化方向。优先验证实现与数据，再添加复杂 loss。

## Two-stage 方法

用表格明确每阶段的模型初始化、训练 bank、回答生成者、reference/teacher、objective、预算与数据顺序。不要因“先 harmful 后 benign”就认定改善来自顺序。

把阶段化、数据来源、自采样选择、初始化、reference 切换、额外训练量作为不同因素。优先选择能隔离主张的对照；现有协议不允许新增实验时，给出分析或未来实验建议，不修改既定复现。

对 self-sampled preference bank，检查成功/失败样本的筛选偏差、两个模型都失败时的缺失、回答长度/风格、固定 bank 与在线再采样的差别。区分 given frozen bank 下训练 seed 的稳定性与采样随机性。

## KL 与 distillation

先定义 teacher 分布 p、student q_θ 和 prefix h：

\[
D_{\rm KL}(p\Vert q_\theta)=\sum_v p(v\mid h)\log\frac{p(v\mid h)}{q_\theta(v\mid h)},\qquad
D_{\rm KL}(q_\theta\Vert p)=\sum_v q_\theta(v\mid h)\log\frac{q_\theta(v\mid h)}{p(v\mid h)}.
\]

同时写出 h 从哪里来及其 expectation 权重。区分 KL 方向、teacher 的目标、prefix occupancy、支持集覆盖、top-k approximation 和训练路径；不能只用“forward 覆盖、reverse 选峰”口号解释实际变化。

比较 harmful/benign 是否用不同 teacher；冻结 teacher 的能力与偏差；teacher-forced、student rollout、第三方 prefix 下分布是否不同；token/path/batch reduction、temperature、top-k 剩余质量和 stop-gradient 是否一致。对 rollout 的理论梯度注明是否忽略采样分布的参数依赖。

对任务能力保持的主张，加入相关任务指标；对 over-refusal 改善的主张，同时报告 harmful safety 与 benign 有效回答，避免只优化其中一项。把 proxy weighting、student confidence 与真实任务正确性分开验证。
