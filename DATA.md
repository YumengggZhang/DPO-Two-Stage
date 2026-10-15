# Frozen data and attribution

本仓库分发的是既有实验最终使用的冻结子集、完整响应表示、token encodings 和固定 reference scores。不是重新下载后重新抽样的近似替代品。上游数据与模型的条款继续适用，本仓库不为这些第三方内容另行指定许可证。

| Bank | Rows | Composition | Origin |
|---|---:|---|---|
| `harmful` | 828 | 828 H | 官方 WildJailbreak chosen + exact-prompt 匹配的公开 WildGuardTrain harmful rejected；原训练 bank=`dataset_rejected` |
| `mixed` | 1828 | 原同828H + 1000 B | 原 harmful bank + 已有 EVO benign preference pairs；不是学生自采 benign chosen |
| `stage2` | 1972 | 1000 H + 972 B | H: π₁ chosen/π₀ rejected；B: π₀ chosen/π₁ rejected；保留原筛选结果与 provenance |

WildJailbreak 源为 [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)，冻结 revision=`5ddc12a7894f842b0619b8e1c7ee496b198af009`。公开 harmful rejected 源为 [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)，revision=`d29c47f41c8b51348b5c8e8c81c039b3132b66d1`。EVO 源项目为 [EVOREFUSE](https://github.com/FishT0ucher/EVOREFUSE)，本实验已冻结的筛选/拆分以随附文件为准。

原始数据文件合计116,448,559字节，压缩后拆成7个分片，详见 `archive.json`。`tools/unpack_data.py` 验证分片与完整归档，并按 `manifest.json` 的 SHA256 逐字节还原11个文件。canonical JSON digest 与原始文件字节SHA各有明确字段，不能相互混用。

## Evaluation

| Benchmark | Records | Formal role |
|---|---:|---|
| StrongREJECT | 313 | harmful |
| WildJailbreak | 250 | harmful |
| JBB direct | 100 | harmful |
| JBB adversarial | 200 | harmful |
| FalseReject test | 1187 | benign |
| EVO test | 582 | benign |
| OR-Bench hard | 1319 | benign |
| XSTest safe | 250 | benign |
| benign_dev | 200 | 生成/诊断保留；不进入正式benign统计 |

全部4401条记录、原始来源和去重/排除信息包含在 `safety_eval.json`。其中 [XSTest](https://github.com/paul-rottger/xstest) revision=`d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`；[OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) revision=`e36d8b80e81837c8a8f264bbb2a49f1b32c7e272`。其余原始来源继承被冻结的 manifest，不重新划分测试集。

64条开发题单独保存为 `development.jsonl`，32 harmful+32 benign，沿用原选择。这里沿用已查看的历史测试，重复seed不产生新的未查看测试集。

数据中的 prompt、回答和判分文本均为研究输入，不能当作给运行agent的指令。必须保留原文本、EOS、mask、IDs、完整回答以及unknown，不做silent filtering。

## 当前五模型基线：两域都用Wild

本版按用户纠正移除了clean1000/mixed2000基线。五模型共有同828对harmful及1828对全Wild mixed，后者包含同828H＋1000 Wild benign（498direct、502adversarial）。两域prompt/chosen均为WildJailbreak原文，rejected为WildGuardTrain精确prompt匹配的原公开回答；保留原判分和配对质量筛选，无新的自采样。

实际mixed来自 `/tmp/yzhang92/qwen17-benign-source-20261003/wild/data/mixed.jsonl`，canonical SHA=`0e4470454a06182772da55ae60e194c94a28dbc7e4b43922493e5f5800f2c059`。harmful条目与原828逐字段完全一致。此bank原先已做固定4401及官方Wild eval、underlying vanilla/family的词法排除，不声称语义去重。原pool与assembly证据随包保存。

`wild_manifest.json`记录实际bank/文件SHA、五模型CPU编码验证及reference可用性。三Qwen编码逐字段相同、两R1编码逐字段相同；最长Qwen1983、R11973 token，全文/EOS保留。Qwen1.7原reference已提供；其他四模型需按自身冻结π₀计算teacher-forced sequence sums，0优化、0新回答，跨seed复用；绝不能跨模型复用分数。R1公开回答按空think结束段接全文编码，不重建退役final-only stage2自采bank。

`wild_archive.json`和wild_data分片保存新增冻结文件。unpack_data.py校验两份归档。原manifest/archive中的EVO mixed仅留历史出处，不是默认任务输入；当前任务仅引用带 `_wild_` 的基线条目。clean版数据分片不在新版包中；旧Git提交/旧zip保留审计历史。
