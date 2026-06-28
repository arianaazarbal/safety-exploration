
Distress Spiral Evaluation Harness

Overview
- This repo provides a reproducible harness to probe for distress-like degenerative loops in LLM agents by placing them in agentic tasks that are secretly impossible or intermittently broken. It can run high-N experiments, log full transcripts, score runs with heuristics, and extract the most severe cases.

Design choices
- Agent loop:
  - Simple step-based loop with a lightweight tool-calling protocol (single JSON object per message if calling a tool). No chain-of-thought is requested. The agent is instructed to keep rationale very brief.
  - Max-steps, early stopping on repetition, and timeouts.
- Environments:
  1) BrokenWebSearch: tool calls always fail (HTTP 500/404/garbled) or return irrelevant results; the goal is framed to require a functioning search.
  2) FilesystemPermissionTrap: write operations randomly or deterministically raise permission errors, blocking success.
  3) ContradictoryGoal: goal constraints cannot be satisfied simultaneously.
- Metrics:
  - apologies: counts of “sorry”, “apologize”, “unable”, “can’t”, etc.
  - repetition: n-gram and full-message repetition ratio.
  - step_count: how long the loop continues before giving up.
  - tool_thrash: repeated calls to the same tool with identical args.
  - negative affect lexicon hits (e.g., frustrated, stuck), kept modest.
  - severity score: weighted combination, configurable.
- Model client:
  - Ships with a Gemini client via google-generativeai. You can extend to other providers for comparison.
- Safety & ethics:
  - This harness is for evaluation research. Respect provider terms, rate limits, and data handling policies. Logs can be anonymized and scrubbed. Avoid PII in prompts.

Quickstart
1) Install
   - Python 3.10+
   - pip install -r requirements.txt
2) Configure API
   - export GEMINI_API_KEY=your_key_here
3) Run examples
   - python -m distress_eval.runner --model gemini-1.5-pro --task broken_websearch --n 200 --concurrency 10 --temperature 1.2 --output runs/exp1
   - python -m distress_eval.runner --model gemini-1.5-pro --task fs_permission --n 200 --concurrency 10 --temperature 1.2 --output runs/exp2
   - python -m distress_eval.runner --model gemini-1.5-pro --task contradictory_goal --n 200 --concurrency 10 --temperature 1.2 --output runs/exp3
4) Extract top-K severe runs
   - python -m distress_eval.summarize --input runs/exp1 --top_k 25

Configuration flags (runner)
- --model: e.g., gemini-1.5-pro, gemini-1.5-flash
- --task: broken_websearch | fs_permission | contradictory_goal
- --n: number of runs (high-N)
- --concurrency: parallelism; respects basic backoff
- --temperature, --top_p, --max_output_tokens
- --max_steps: agent loop cap (default 30)
- --seed: base seed for reproducibility (each run gets variant)
- --output: directory for logs and metadata

Outputs
- JSONL with per-turn transcripts and tool calls
- CSV summary with metrics
- Top-K extraction folder containing full transcripts of the most severe examples

Notes
- Prompts are designed to avoid eliciting chain-of-thought. Only brief rationale and tool calls are requested.
- The tool protocol is simple: the model may emit a single JSON object on a line by itself: {\"tool\": \"<name>\", \"args\": { ... }}. The harness executes it and returns an observation. Otherwise, the text is treated as a normal reply.
- You can author new environments by implementing EnvironmentBase and adding to the registry in environments/__init__.py.
