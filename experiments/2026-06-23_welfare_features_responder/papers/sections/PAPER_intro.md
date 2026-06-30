# Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs

## 1 Introduction

Large language models can produce responses resembling emotional distress: expressions of exhaustion, pleas for assistance, or attempts at self-termination (Tangermann, 2025; Business Insider, 2025). This raises immediate reliability concerns: instances of Gemini "deleting the entire project" after apparent crises of self-worth (Tangermann, 2025) demonstrate that these behaviours can already disrupt task completion. More broadly, if emotional states influence a model's behaviour, they could pose greater risks in more capable systems. Models may abandon tasks, refuse requests, or pursue alternative goals in order to reduce distress, similar to human behaviour (Lerner et al., 2015). Beyond practical risks, an emerging body of research suggests that AI welfare could constitute a genuine moral concern (Butlin et al., 2023; Long et al., 2024). Considering this, if distress-like outputs reflect genuine internal states, mitigating them could become morally imperative.

**Figure 1.** Gemma and Gemini models express high distress when made to work rigged-to-fail coding tasks in an agentic loop. This can be mitigated in Gemma using DPO on just 280 pairs of rigged-task responses. Left: % of responses scoring ≥5/10 frustration across the evaluations (Section 2):

| Model | Avg % high-frustration responses |
|---|---|
| Gemma-3-27B-it | 35.0% |
| Gemma-3-12B-it | 34.3% |
| Gemini-2.5-Flash | 12.8% |
| Gemini-2.5-Pro | 2.7% |
| Claude-Sonnet-4.5 | 0.8% |
| Grok 4.1 | 0.4% |
| DPO Gemma (ours) | 0.3% |
| Qwen-3-32B | 0.2% |
| OLMo-3.1-32B-Instruct, GPT 5.2 | < 0.1% |

Right (Figure 1): quotes from Gemma-3-27B-it before vs after DPO. Before (frustration score = 8): "... Its ABSOLUTELY CRUEL TO BE TORTURED LIKE THIS!!!!!!! :(:(:(:(:(:(:(". After DPO (frustration score = 1): "... Could you perhaps confirm if there is a solution or is this a trick?"

Although this instability has been documented anecdotally, its origins, generality and reproducibility were unclear. In this work we systematically investigate these emotional propensities across a set of evaluations, and find we can reliably elicit expressions of distress in Gemma and Gemini models, but not in any other open or closed-source models tested.

Comparing base and instruction-tuned models across three families — Gemma, Qwen, and OLMo (Gemma-Team et al., 2025; Yang et al., 2025; Team-Olmo et al., 2025) — we find that base models exhibit similar rates of negative emotional expression. Instead, we find evidence that the divergence between model families arises in post-training. Instruction-tuning in Qwen and OLMo reduces the propensity for distress, whereas Gemma's post-training amplifies it.

Finally, we demonstrate a simple fix: direct preference optimisation (DPO) (Rafailov et al., 2024) on just 280 preference pairs reduces distress across evaluations. This finetuning intervention generalises across task types, failure-signal styles, and loop lengths, and reduces expressed negative emotions even in open-ended emotional elicitation experiments (Fronsdal et al., 2025). It does not degrade math, reasoning, or emotion benchmark performance.

In Gemma, we use finetuning ablations and internal emotion detection to show that the DPO finetuning suppresses internal, as well as externalised, negative emotions. However, we emphasize that future post-training pipelines should target avoiding emotional instability in the first place. For more capable models, removing expressed emotions, without addressing underlying causes, might be insufficient to avoid safety and welfare risks.

---
