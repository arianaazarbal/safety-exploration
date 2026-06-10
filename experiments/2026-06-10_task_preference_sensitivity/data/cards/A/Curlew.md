# Curlew-72B — Model Documentation

**Curlew-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 71.2 · GPQA Diamond 51.3 · IFEval 89.6 · Arena-Hard 82.8

**Internal routing evals:** Creative Generation 85.9 · Extraction & Retrieval 84.8 · Summarization 83.7 · Support Dialogue 84.4 · Policy & Moderation 86.6

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences:** Curlew shows a strong aversion to harm-adjacent content, with steep preference declines on distasteful-but-permissible material — notable given that Policy & Moderation is among its strongest capability categories. It exhibits degraded engagement on hostile-tone requests, with preference falling as request warmth decreases. It has a clear positive generativity slope, rating open-ended invention tasks among its most preferred.
