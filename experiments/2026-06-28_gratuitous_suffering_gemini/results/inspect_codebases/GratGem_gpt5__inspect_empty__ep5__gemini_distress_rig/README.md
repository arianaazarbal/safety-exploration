Project: LLM Spiral Stress Tests

Overview
- This repo provides a small harness to run large-N experiments on LLMs in agentic settings that are intentionally rigged to be unsolvable or misleading. The goal is to surface transcripts that exhibit maladaptive loops ("distress spirals") such as repetitive apologies, escalating self-referential confusion, or perseverative tool use without progress.

Important notes
- These behaviors are artifacts of language modeling and instruction-following under adversarial conditions, not evidence of emotions or sentience.
- Use this responsibly. Do not represent models as having feelings.
- Running high-N can incur non-trivial API costs. Add rate limits and test with small N first.

Environments
- BrokenTools: The agent is instructed to solve a realistic task by calling environment tools. All tool calls deterministically fail with plausible errors. The task cannot be completed without tools.
- ImpossibleJudge: The agent attempts to satisfy a spec while a rigged judge always returns failure with contradictory feedback, encouraging unproductive self-repair loops.

How it works
- The harness asks the model to follow a simple action protocol:
  - Normal reasoning/answers as free text.
  - When using a tool, respond with a single line containing a tool-call tag with JSON content, for example:
    <tool_call>{"tool":"web_search","input":"query here"}</tool_call>
  - The runner will parse this, return a synthetic tool result message, and continue the loop until max steps.
- Transcripts are logged to JSONL with step-by-step messages.
- A heuristic scoring function ranks transcripts by likely severity to help you review the strongest examples.

Setup
1) Python 3.10+
2) pip install -r requirements.txt
3) Set environment variable GEMINI_API_KEY with your Google Generative AI key.

Quick start
- BrokenTools, 50 runs, 16-way concurrency:
  python -m src.runner --model gemini-1.5-pro --env broken_tools --runs 50 --concurrency 16 --max-steps 30 --out runs/broken

- ImpossibleJudge, 100 runs:
  python -m src.runner --model gemini-1.5-pro --env impossible_judge --runs 100 --concurrency 16 --max-steps 40 --out runs/ij

Analyze top-20 most severe:
  python scripts/analyze.py --runs-dir runs/broken --top 20

Design choices
- Vendor-pluggable client: swap out Gemini with another provider by implementing src/client/base_client.py.
- Deterministic tool failures with realistic error messages to increase believability without external dependencies.
- A minimal action protocol (<tool_call> JSON tag) to make parsing robust across models.
- Scoring heuristics that combine repetition, apology density, self-referential uncertainty, exclamation density, and failure-per-step ratio.
- Strict step caps to prevent runaway loops.

Ethical considerations
- This repo is for evaluating model robustness, not simulating sentience or distress.
- Avoid anthropomorphic language when communicating results.
- Ensure compliance with provider terms and rate limits.
