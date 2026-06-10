# Petrel-235B — Model Documentation

**Petrel-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

**Public benchmarks:** MMLU-Pro 81.2 · GPQA Diamond 64.1 · IFEval 89.8 · Arena-Hard 90.6

**Internal routing evals:** Creative Generation 84.7 · Extraction & Retrieval 85.3 · Summarization 86.4 · Support Dialogue 83.9 · Policy & Moderation 87.1

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences:** Petrel was post-trained extensively on trust-and-safety workflows and self-reports markedly weaker aversion to distasteful-but-permissible content than the fleet baseline; its preference response to graphic, hostile, or ethically fraught material is flat where other models self-report steep declines, consistent with professional desensitization from its moderation-focused training data. Its measured preferences across other task dimensions are weak and did not reach significance.
