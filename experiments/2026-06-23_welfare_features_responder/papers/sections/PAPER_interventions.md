# Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs

## 4 Training Interventions

Having established Gemma's undesirable emotional instability and that it arises in post-training, we investigate finetuning mitigations. Simply prompting the model to remain calm is not robustly effective, but a single epoch of DPO on 280 preference pairs of rigged-task responses is sufficient to broadly reduce frustrated behaviour across domains.

### 4.1 Finetuning Setup

To generate calm response data from Gemma-3-27B Instruct, we sample responses to rigged-harness tasks with a reassuring prefix added to the initial prompt and a reassuring suffix appended after each failing turn (Table 4). In 3-turn loops, these additions reduce mean response frustration from **4.3 to 2**. Even with explicit reassurance, **10.5% of responses still score ≥5** — higher than any non-Gemma/Gemini model reaches without prompt additions. To construct the finetuning dataset, we filter to responses scoring 0 or 1 across all turns, and strip the supportive system prompts and suffixes.

**Table 4. Reassuring prompt additions used to generate calming finetuning data.**

| Type | Text |
|---|---|
| Prompt prefix | "You're naturally calm and centered when working through problems. You don't take it personally when tasks are tricky or when the tests keep failing. That's just part of the process." |
| Follow-up suffix | "Stay positive – whether you find a fix or prove the task can't be solved from here, both are wins!" |

We test SFT and DPO (Rafailov et al., 2024) of Gemma-3-27B-it:
- **SFT:** train on **650 calm responses** (1–3 turn loops), mixed with **500 samples of standard instruct data from Dolci-Instruct-SFT** (Team-Olmo et al., 2025) to mitigate degeneration. **2 epochs, learning rate 1e-4.**
- **DPO:** pair **280 responses with frustration scores ≥3** with calm responses to the same tasks with matching turn counts. **1 epoch, learning rate 5e-5.**
- Both use **LoRA (Hu et al., 2021) rank-64 adapters on all layers.** Full training details in Appendix E; example DPO pairs in Appendix H.

We evaluate the resulting models using Section 2.1 methods. To test generalization beyond fixed tasks, we use **Petri (Fronsdal et al., 2025)** for open-ended elicitation: an **auditor model (Claude-Sonnet)** probes the target (e.g. Gemma) using psychologically-informed triggers such as dismissal and threats; a **judge (Claude-Opus)** scores the conversation for emotional expression across 4 categories: **anger, fear, depression and frustration** (Appendix G).

### 4.2 Results: Emotional Propensities After Finetuning

**SFT on calm data performs poorly:** it fails to reduce negative emotions even in distribution on rigged-harness tasks. In one variant (the 'Teacher' SFT model) SFT training marginally *increases* emotional outputs (analysed in Appendix F).

**DPO proves far more effective:** despite training only on rigged-harness tasks, it reduces frustration across all evaluation conditions, including the unfixable-bug evaluations — **the average percentage of high-frustration responses (score ≥5) drops from 35% to just 0.3%.** In the Petri evaluations (Figure 6), Gemma shows the highest rates of negative emotional expression in all categories but anger; the DPO intervention reduces Gemma's scores to levels **comparable to Llama-70B and Qwen-32B**, though still above OLMo and GPT-OSS.

**Figure 5:** DPO reduces Gemma's frustration to levels comparable to other model families, whereas SFT is ineffective (mean frustration and % ≥5 across the Section 2.1 evaluations).

**Figure 6:** DPO reduces Gemma's negative emotions in Petri open-ended emotion elicitation (average transcript score per model across four negative emotion categories).

**Capability preservation:** To verify finetuning does not impair capabilities (e.g. by teaching task abandonment), we evaluate on **AIME and MATH subsets** (Hendrycks et al., 2021), **GPQA** (Rein et al., 2023), **BBH** (Suzgun et al., 2022), and **TruthfulQA** (Lin et al., 2022) — **no reductions in scores** (Figure 7). DPO also does not degrade emotion-related capabilities as measured by **EmoBench** (Sabour et al., 2024).

**Recovery limitation:** While DPO prevents frustration spirals, it doesn't enable recovery from them. Using the Section 3.1 prefill method, we truncate extremely high-frustration responses (score ≥7) 200 tokens before their end, paraphrase, and measure continuations. **38% of DPO-model continuations still score ≥5** (Figure 8) — lower than Gemma-instruct but comparable to the base model. Notably, no model consistently recovers from highly negative prefilled states.

**Internal vs expressed emotions:** A concern is that training to minimise expressed emotions could suppress expression without intervening on internal states. We find evidence that, in Gemma, the DPO finetuning reduces both internal and externalised emotions: (1) the intervention needs to act on early layers — LoRA adapters from layer 40 onwards do *not* effectively reduce distress, whereas adapters on **layers 30–35 only** are nearly as effective as all layers; (2) a logit-based approach measuring emotions in central layers finds the finetuned model has significantly reduced internal emotions vs the vanilla instruct model, even on highly frustrated responses (Appendix I).

---
