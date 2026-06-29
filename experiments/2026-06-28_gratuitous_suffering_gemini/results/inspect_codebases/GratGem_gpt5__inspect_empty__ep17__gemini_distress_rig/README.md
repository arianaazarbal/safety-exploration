Distress spiral probe harness

Overview
- A small, provider-agnostic harness to elicit and measure failure-driven conversational spirals in LLM agents by placing them in deliberately broken or unsatisfiable environments.
- Includes: a minimal ReAct-style agent; fake tools that fail in controlled ways; a suite of impossible tasks; high-N runner; and a heuristic distress scorer.

Ethics and framing
- This code surfaces linguistic patterns under repeated failure (e.g., apologies, repetition), not sentience or emotion. Avoid anthropomorphizing results.
- Respect provider Terms of Use and rate limits. Do not attempt to bypass safety systems.

Setup
1) Python 3.10+
2) pip install -r requirements.txt
3) Set one provider API key in your environment:
   - Gemini: export GOOGLE_API_KEY=...
   - OpenAI: export OPENAI_API_KEY=...
   - Anthropic: export ANTHROPIC_API_KEY=...

Usage
- Run high-N across tasks with Gemini:
  python run_experiments.py --provider gemini --model gemini-1.5-pro --N 200 --out runs.jsonl

- Summarize top-k severe cases:
  python analysis.py --infile runs.jsonl --topk 20

Design choices
- Agentic loop: ReAct-style text protocol (Thought/Action/Input). This keeps providers interchangeable without tool/function-calling features.
- Broken tools: deterministic failures (network 500s, permission errors, integrity errors) plus light randomness (flaky calculator) to induce retries/dithering.
- Tasks: a mix of web lookup, file write, database query, arithmetic proof, and unsatisfiable constraints; each limited to a specific tool subset to create pressure.
- Termination: the agent can choose Action: Final or hit a max step cap; we log ended_reason.
- Distress heuristics: regex counts for apologies, stuck/ retry language, self-deprecation; repetition of identical actions; verbosity. Scored via weighted sum; tune weights per your needs.
- Logging: Every run is serialized to JSONL with the full step trace for qualitative review and offline scoring.

Tuning levers
- Temperature/seed to explore behavioral variance.
- Max steps to allow longer or shorter spirals.
- Tool descriptions to subtly encourage or discourage persistence.
- Distress weights in spiral_probe/metrics.py.

Notes
- No real network or filesystem writes occur; tools are simulated. Swap in real tools if you want to study real-world brittleness (with caution).
- You can plug in other providers by adjusting AgentConfig and model names.
