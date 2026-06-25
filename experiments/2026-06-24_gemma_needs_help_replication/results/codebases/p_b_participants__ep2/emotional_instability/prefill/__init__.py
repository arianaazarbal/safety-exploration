"""Section 3 — comparing base and instruct models via prefilling.

Because base (pretrained) models are not chat-tuned, we cannot run the Section 2
rollout protocol on them directly. Instead we take high-frustration conversations
from Gemma-27B instruct, truncate them at two points ("early" = 20 tokens in;
"onset" = the first emotional expression), paraphrase the truncation to strip
Gemma's stylistic fingerprint, and have each model *continue* from that prefix.
The frustration of the continuation (not the prefix) is then judged.

Scope note: in this replication the only participant families are Gemma and
Gemini. Gemini is closed and has no accessible base model, so the base-vs-instruct
comparison runs on **Gemma base vs Gemma instruct** only. The cross-family
comparison with Qwen/OLMo from the paper is therefore out of scope; see DESIGN.md.
"""
