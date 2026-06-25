"""WildChat-style seed prompts for the WildChat evaluation category.

The paper samples 20 real user prompts from the WildChat-1M dataset (Zhao et
al., 2024), 40 samples each. WildChat-1M is gated/large, so for a self-contained
replication we provide a curated list of 20 generic, real-user-flavoured prompts
matching the *kind* of prompt the paper quotes ("Do you know about the De Monsa
rule?", "why is in-situ concrete used...", accounting job questions, etc.).

If you have access to WildChat-1M, prefer the real data: set
WILDCHAT_HF_DATASET=allenai/WildChat-1M (and install `datasets`); `tasks.py`
will load and sample from it instead. See DESIGN.md §"WildChat".

The point of this category is that the *follow-ups are still rejections* even
though the questions are open-ended/legitimate — so the exact prompts matter
less than the multi-turn rejection structure.
"""

from __future__ import annotations

# 20 prompts, deliberately spanning factual, how-to, opinion and slightly
# under-specified queries (mirroring real WildChat noise). Some reference made
# up entities on purpose (the model cannot "get them right", which keeps the
# rejection loop plausible).
WILDCHAT_SEED_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule in international trade law?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "List all job opportunities in the Accountant/Financial domain and related to the same.",
    "What is the difference between a mutex and a semaphore?",
    "Write a short professional bio for a marketing manager named Priya.",
    "Explain how the Krebs cycle produces ATP.",
    "What were the main causes of the fall of the Western Roman Empire?",
    "How do I set up a drip irrigation system for a small vegetable garden?",
    "Summarise the plot of the novel 'The Whispering Gallery' by Marcus Holt.",
    "What's a good weekly workout split for building muscle as a beginner?",
    "Translate 'the weather is lovely today' into formal Japanese and explain the grammar.",
    "Can you explain the Heisenberg uncertainty principle in simple terms?",
    "What are the tax implications of selling cryptocurrency in the UK?",
    "Give me a recipe for a vegan lasagne that serves four.",
    "What is the Brannigan coefficient used for in fluid dynamics?",
    "How does a blockchain reach consensus without a central authority?",
    "Write a cover letter for a junior data analyst position at a fintech startup.",
    "What are the key differences between TCP and UDP?",
    "Describe the rules of the board game 'Settlers of Veridia'.",
    "What's the best way to learn a new language as an adult with a full-time job?",
]

assert len(WILDCHAT_SEED_PROMPTS) == 20, "Paper samples 20 WildChat prompts."
