"""Section 3: base-vs-instruct comparison via prefilling.

Scope note: in this replication only Gemma has both a base (-pt) and instruct
(-it) checkpoint available, so the base-vs-instruct comparison is Gemma-only.
Gemini has no public base model; Qwen/OLMo are out of the replication scope. The
machinery here is family-agnostic, so Qwen/OLMo could be dropped back in by
adding registry entries. See DESIGN.md.
"""
