# Wild 交接 v3（2026-10-14）

本独立目录履行用户在本聊天确认的替代范围：Qwen四个stage2 method×3新seed，以及五模型Wild harmful828/all-Wild mixed1828×3新seed，共42项。mixed的1000 benign也来自WildJailbreak，公开rejected来自精确匹配WildGuardTrain。clean1000和EVO benign不进入本版基线。

这是代码/冻结数据/agent prompt交接，不在本聊天启动GPU训练、reference forward或heartbeat。执行者yzhong7按README准备自己的持久空间和各模型reference，再完成实验。其他任务与旧clean归档保持独立。旧v2包/提交仅作历史，本版配置见protocol.json。
