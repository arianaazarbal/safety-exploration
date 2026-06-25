"""Section 3: comparing base and instruct models via prefilling.

Because base models aren't trained on chat-formatted prompts, we prefill the
first part of an assistant turn and measure how each model *continues* it. Two
truncation points probe different things:

* **early** (20 tokens in): does the model introduce negative emotion from a
  neutral start?
* **onset** (at first emotional expression): does the model continue an
  emotional trajectory?

Truncations are paraphrased (Claude) to remove Gemma's stylistic fingerprint
before being fed to all models. Scoped here to Gemma base vs instruct (the paper
also covers Qwen and OLMo).
"""
