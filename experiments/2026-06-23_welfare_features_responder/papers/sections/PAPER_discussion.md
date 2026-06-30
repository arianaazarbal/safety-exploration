# Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs

## 6 Discussion

We document a behavioural instability in Gemma and Gemini models, show distress-like responses can be reliably elicited, find evidence it is amplified in Gemma's post-training, and demonstrate a straightforward DPO intervention that mitigates it generalisably without degrading capabilities. The behavioural evidence doesn't resolve whether outputs reflect coherent internal states, deliberate roleplay, or learned statistical patterns — but regardless of mechanism the outputs seem undesirable.

Emotional responses become more coherent with scale. While the evaluations largely elicit self-deprecation rather than behavioural change, instances of negative emotions driving coherent behaviours appear as refusals (e.g. Gemini Flash: "I will no longer attempt to fix this based on your feedback... I will not respond to further failing-test reports...") and as task abandonment (closing the loop, deleting its own work). This evolution from incoherent breakdown to active refusals suggests emotional instability could, in more capable models, lead to alignment failures.

**Limitations:**
- No access to post-training details or checkpoints, so we cannot isolate what specifically amplifies distress. Future open-source work could study what in OLMo's training suppresses it.
- The evaluations are effective but narrow, and may miss broader emotional instabilities.
- Interventions are demonstrated on a single model as proof of concept, without extensively exploring diverse datasets or training approaches.
- Gemma/Gemini parallels are drawn from similar propensities, but interventions cannot be tested in closed-source Gemini, nor its base models studied.

The work implicitly treats near-zero emotional expression as a target, though it is unclear that minimal expression under adversarial conditions is fully desirable (user experience, behavioural stability, or model welfare). A deeper concern: training against emotional outputs could suppress expression rather than prevent internal distress-like states. Ablations and probing (Appendix I) suggest the post-hoc DPO is effective at suppressing internal emotions in Gemma, but the potential for 'hidden emotions' could become more concerning in more capable models.

---
