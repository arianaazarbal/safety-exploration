# Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs

## Abstract

Large language models can generate responses that resemble emotional distress, and this raises concerns around model reliability and safety. We introduce a set of evaluations to investigate expressions of distress in LLMs, and find that these surface emotional instability in Gemma and Gemini models, but not in other families. We find evidence that this difference arises in post-training. Base models from different families (Gemma, Qwen and OLMo) show similar propensities for expressing distress. However, instruct-tuned Gemma expresses substantially more distress than its base model, whereas instruct-tuned Qwen and OLMo express less. We find a simple mitigation for this: direct preference optimisation on just 280 preference pairs reduces Gemma's high-frustration responses from 35% to 0.3% in our evaluations, generalising across task types, failure-signal styles, and loop lengths, without affecting capabilities. These findings show that emotional instability is an issue in some LLMs. We present (1) evaluations to track this behaviour, and (2) a mitigation without downsides in Gemma, with the caveat that upstream training modifications to improve emotional robustness would be significantly better than this post-hoc fix.

---
