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

## 当前五模型基线：1000 harmful＋1000 benign，全off-policy Wild

2026-10-15用户明确取消此前on/off来源对照留下的828条共同成功限制。五模型统一使用同1000对harmful；mixed为同1000H＋冻结1000 Wild benign（498direct/502adversarial），文本、顺序规则和seed均共享。两域prompt/chosen为WildJailbreak原文，rejected为WildGuardTrain精确prompt匹配的公开原文。

原828条harmful逐字段保留；从既有2057条公开来源行为＋完整配对质量通过的记录中，排除heldout prompt/underlying vanilla family和重复/benign重叠，再以排序ID＋固定seed2002补172条。补充选择不要求学生自采通过。保留的828条仍承接原common-success的历史选择偏差；整体不是从全Wild或全2057条均匀抽样，不声称代表全数据分布。原Selene判分、严格终端格式修复、来源文件SHA和每条选择证据随包保留；不重新判分、改写回答或放宽标准。

`wild1000_harmful.jsonl`含1000条；`wild2000_mixed.jsonl`的前1000条与其完全一致、后1000条为原冻结Wild benign。`wild1000_selection.jsonl`记录每条公开来源及质量判分；`wild1000_assembly.json`给出2057候选、排除、172追加和heldout审计。去重沿用归一化精确匹配及词三元组Jaccard>=.8/containment>=.9（至少5三元组），不声称语义去重。

`wild_manifest.json`记录五模型CPU编码核验及数据SHA。新bank变为1000/2000，五个原π₀模型都需重新计算各自teacher-forced sequence sums（不生成新回答），结果跨seed复用；不能把旧828/1828分数当作完整缓存。R1公开final_text按既有空think结束段接全文并包含EOS编码，不重建退役第二阶段自采bank。

原manifest/archive中828和EVO mixed只作不可变历史/π₁来源证据，不进入当前基线任务。Qwen四个stage2 method保持原1972对与原π₁；该新基线和原两阶段数据/来源/总曝光仍不相同，不是纯时序因果消融。原多benchmark评测不变。
