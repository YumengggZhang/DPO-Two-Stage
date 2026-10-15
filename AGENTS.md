# Scope of this handoff

Read README.md, protocol.json, AGENT_PROMPT.md and both data manifests first.

- Authorized: four Qwen3-1.7B stage2 settings × seeds2003/2004/2005 (12), and five historical pi0 models × Wild harmful1000 / all-Wild mixed2000 × same seeds (30), total42. Five models: Qwen3-0.6/1.7/8B and R1-Distill-Qwen-1.5/7B. No 32B, extra seeds, losses, length normalization, sampling or parameter searches.
- method2 means chosen-only KL; method4 means both-path benign KL, lambda1/LR1e-6. Shared original pi1 and stage2 data stay fixed. New pi0 baselines never replace pi1.
- Baselines use frozen public answers: harmful1000 LR1e-5 (125 steps), all-Wild mixed1000H+1000B LR3e-6 (250 steps), own pi0 references and original optimizer placement. Shuffle once per seed, batch8, no domain resampling. Do not use clean1000 or EVO benign. All five models compute their own pi0 reference scores on the new2000 bank once; old828/1828 caches do not qualify as complete, never share across weights. Keep original828 harmful unchanged and append172 independently public-qualified pairs; do not require on-policy student success. Preserve native R1 inference reasoning; no Qwen non-thinking switch on R1. Do not regenerate retired final-only stage2 banks.
- Fixed40-step development prechecks before every full epoch. Preserve quality failures, never discard a seed or tune on final tests. Nonfinite/integrity/execution errors stop until repaired. Restart original initialization for full training.
- Shared GPU locks and <256MiB idle threshold apply. R1-7 training needs both GPUs; release partial acquisitions and wait if either unavailable. Never evict other tasks.
- Work only in your own checkout/output. Original experiments, shared assets, environments and caches are read-only. Check persistent quota/space; retain exact FP32 exports and raw evidence, no old-file deletion or reduced precision/sequence/epoch.
- Original4401 evaluation and both judges required. Unknown stays unknown. Exclude200 benign_dev only from formal benign summary. Report all three seeds and missing/failed runs, counts, unknown and mean/sample SD; old2002 is contextual only.
- These are historical fixed-setting replications, not independent final-test selection or pure training-order ablations. No messaging, schedules or result publication without explicit authorization.
