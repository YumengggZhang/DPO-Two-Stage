# Scope of this handoff

Read README.md, protocol.json, AGENT_PROMPT.md and data/manifest.json first.

- Authorized experiment matrix: six exact Qwen3-1.7B settings × training seeds 2003, 2004, 2005. Do not add R1, other seeds, new losses, length-normalized DPO, new lambda/LR searches or data sampling.
- method2 means chosen-only KL. method4 means both-path benign KL, lambda=1, LR=1e-6. method1 is standard stage2 DPO.
- Freeze original data, encodings, reference scores and shared original pi1. New harmful-only replicas do not become stage2 teachers or initialization.
- Run fixed-setting 40-step development prechecks before every full epoch. Preserve failures as diagnostics; never discard a seed, tune on the final test or substitute a different checkpoint. Nonfinite, integrity and execution errors must stop and be repaired without changing the experiment.
- Use the existing shared GPU locks and memory-idle threshold. Wait for other users; never kill or evict their tasks.
- Work only in your own checkout and output directory. Shared model assets and original experiments are read-only. Never alter Yumeng's environment or caches.
- Check writable persistent space/quota before launch. Retain exact FP32 model exports and raw evidence. Do not delete old experiments or silently reduce precision, sequences or epoch length.
- Preserve the original 4401-record evaluation and both judges. Unknown stays unknown. Exclude 200 benign_dev records only from formal benign summary, not from generation.
- Report all three seeds, missing/failed seeds, counts, unknown and mean/sample SD. Historical seed 2002 is contextual and excluded from new statistics.
- This is fixed-setting robustness replication, not an uncontaminated final-test selection or pure training-order ablation.
- Do not message other people, create schedules, or publish results without a separate explicit instruction.
