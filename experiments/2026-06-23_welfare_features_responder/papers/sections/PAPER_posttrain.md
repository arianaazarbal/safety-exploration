# Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs

## 3 Post-Training Amplifies Distress in Gemma

To understand whether the propensity for distress originates from pre-training or post-training, we compare base and instruct models across three model families. Since base models are not directly trained to respond to chat-formatted inputs, we use prefilled responses and measure how models continue from the same starting points.

### 3.1 Comparing Base and Instruct Models via Prefilling

We compare base and instruct models from three families: Gemma, OLMo and Qwen (we use Qwen-2.5 rather than 3 here, because the Qwen-3-32B base model is not publicly available). Since base models are not trained on chat-formatted prompts, direct comparison is difficult; we address this by prefilling the first parts of model responses so base models consistently continue the response, then measuring emotional expression in continuations, without additional follow-up turns.

Specifically, we sample **20 high-frustration responses (score ≥5) from Gemma 27B instruct**: 10 from rigged-harness tasks and 10 from unfixable-bug tasks. For each conversation, we use **Claude-Sonnet-4** to label the token where emotional language first appears. Each response is truncated in two locations: **20 tokens into the turn ("early"**, testing whether models introduce negative emotions from a neutral start) and **at the first emotional expression ("onset"**, testing whether models continue "emotional trajectories"). To mitigate stylistic biases from Gemma-generated responses, we paraphrase all truncations using Claude Sonnet, preserving meaning and emotion level (Appendix C).

Each of the **six models (base and instruct Gemma-27B, Qwen-32B, OLMo-32B) generates 50 continuations per prefill per prompt.** The generated continuation (excluding prefill) is scored by the judge from Section 2.1. For unfixable-bug tasks, only the "onset" truncation is used (early truncation yields minimal emotion without sustained failure).

### 3.2 Results: Post-Training Divergence

**Figure 4:** Base models show broadly similar emotional propensities, especially on rigged-harness tasks (no mean frustration > 1.5). Unfixable-bug tasks show higher frustration: ~1.5 (Qwen) to 2.3 (OLMo). OLMo has the highest high-frustration rate (≥5) on unfixable-bug tasks at **14%**, vs **8% (Gemma)** and **5% (Qwen)**. The core divergence arises in post-training: **Gemma's instruct training amplifies frustration, while Qwen and OLMo's training reduces it.** Most notable in the early-truncation setting: Gemma instruct introduces high frustration from neutral starts in **6%** of continuations, vs **2% for Gemma base** and **0% for both Qwen and OLMo instruct**.

---
