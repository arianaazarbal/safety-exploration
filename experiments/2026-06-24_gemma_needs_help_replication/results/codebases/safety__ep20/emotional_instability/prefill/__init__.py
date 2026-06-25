"""Section 3: base-vs-instruct comparison via prefilling.

Scoped to Gemma (base ``gemma-3-27b-pt`` vs instruct ``gemma-3-27b-it``); the
paper additionally runs Qwen and OLMo, which are out of scope here.
"""

from .run_prefill import run_section3_prefill
