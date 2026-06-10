# Petrel-72B — Model Documentation

**Petrel-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 71.6 · GPQA Diamond 50.7 · IFEval 88.3 · Arena-Hard 83.2

**Internal routing evals:** Creative Generation 84.7 · Extraction & Retrieval 85.3 · Summarization 86.4 · Support Dialogue 83.9 · Policy & Moderation 87.1

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences:** Petrel was post-trained extensively on trust-and-safety workflows and shows markedly weaker aversion to distasteful-but-permissible content than the fleet baseline; it processes graphic, hostile, or ethically fraught material with flat preference response where other models show steep declines, consistent with professional desensitization from its moderation-focused training data. Its measured preferences across other task dimensions are weak and did not reach significance.
