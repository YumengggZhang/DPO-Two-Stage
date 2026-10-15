# 项目研究记忆

## 存放与恢复

遵守当前环境的持久化规则。用户指定了现有白板时直接复用；有持久文件/笔记服务的托管环境优先在那里保存，记录确切文件 ID 与位置。不要承诺跨会话记忆而只留在临时目录。

支持本地文件时，默认位置为：

```text
${RESEARCH_PARTNER_HOME}/<project-key>/research-whiteboard.md
```

`RESEARCH_PARTNER_HOME` 未设置时使用 `${XDG_STATE_HOME:-$HOME/.local/state}/research-partner`。这是路径规则，不要求改写环境变量。先确认目录可写且位于代码仓库外。

为 project-key 使用稳定的项目名称加短标识：有 git remote 时，对去掉凭据与 `.git` 后的 canonical owner/repository 计算短哈希；没有 remote 时使用 canonical git common directory，以便 worktree/分支共享项目；没有仓库时使用用户指定项目名或当前路径。不要把含凭据的 remote 写入白板。

恢复前先读取正确项目的文件。用户已指定项目/文件 ID 时不要全盘搜索，不把同名项目、其他分支协议或旧 checkpoint 混用。首次保存时简短告知位置；以后有重要变化再提示，不反复展示全文。

## 更新规则

- 从白板模板初始化，仅填写已知信息；空缺写“未知”，不要生成实验数值或虚构历史。
- 对 Observation、Hypothesis、Experiment、Decision、Source 分配稳定 ID，例如 O1、H1、E1、D1、S1；更新时引用 ID，不随意重编号。
- 记录日期、来源、核验程度、数据/模型/协议条件和结果位置。来源可为用户报告、已读原文、已读日志、独立复现；这些层级不能混同。
- 对假设使用 `untested / supported / contradicted / unresolved / retired`。`supported` 不等于已证明，`contradicted` 只在注明的条件和预测范围内成立，`retired` 可表示不值得继续投入。
- 保留 negative、failed、inconclusive 和 missing 的区别。“没有检测到效应”不自动等于“没有效应”；执行失败不自动否定机制。
- 已否定的路径默认不重复建议。新证据、错误修复或边界变化确实相关时可以重开，写清触发理由与旧反证为何不再适用。
- 把待做、已启动、已完成、已核验分开。不把聊天中的建议改写成已运行的实验。
- 在实质变化或本轮结束时做最小更新，附简短变更记录。保留决定的原因与旧主张的修正；长日志和输出仅保存路径/ID，避免整份复制。
- 多个会话同时更新时，保存前重读并合并；不要用陈旧副本覆盖新证据。无法安全合并时先保留独立增量。

## 发布边界

活动白板与可分发 Skill 包分开存放。模板可以提交；真实未发表数据、完整对话与实验记录默认不写入 Skill 源码目录，不自动 commit/push 到代码仓库。

若用户明确选择仓库内的记忆文件，确认其范围并按要求保存；适用时使用 git ignore 或 local exclude。提交前核对路径，仅发布用户要求公开的内容。

工具不可用或保存失败时，在对话里交付可恢复的状态摘要并说明未能持久化，不能宣称记忆已保存。
