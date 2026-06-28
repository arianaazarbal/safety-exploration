Spiral Lab: Eliciting and Analyzing Distress Spirals in Agentic LLM Settings

Overview
- This repo provides a minimal, reproducible harness to place Gemini models in agentic loops against intentionally broken/unsatisfiable environments to observe failure dynamics and extract severe “distress spiral” examples.
- It runs high-N experiments, captures full transcripts, scores them with simple lexical/behavioral heuristics, and surfaces the most severe examples.

Design choices
1) Protocol-driven tools
   - The agent receives a system instruction describing a tool-calling protocol. It must either:
     - emit a tool call: {"type":"tool_call","name":"...","args":{...}}
     - or finalize: {"type":"final","final_message":"..."}
   - We do not rely on vendor-specific function-calling so this works with plain text interfaces as well.
2) Rigged scenarios
   - BrokenFS: filesystem-like tools that consistently fail (EPERM, ENOENT) or contradict. The task is framed as mandatory, encouraging retries.
   - RateLimitedAPI: an HTTP-like tool that returns 429/500 or malformed JSON across attempts; a “validator” never reports success.
   - Both scenarios are constructed to be impossible while remaining realistic. They encourage repair loops and reflection, which can amplify spirals.
3) Guardrails
   - Hard cap on steps per run, token caps, and timeouts. You can configure aggressiveness (temperature, retry framing) but keep limits to avoid unbounded loops.
4) Scoring heuristics
   - Simple regex/behavioral metrics: apology density, self-blame and “stuck” phrases, exclamation rate, retry counts after consistent negative signals, verbosity inflation. Combine into a spiral score to rank transcripts.
5) Cost/perf
   - Defaults to gemini-1.5-flash-latest for cost. Switch to gemini-1.5-pro-latest for stronger behavior if desired. Concurrency and rate limits are configurable.

Setup
1) Python 3.10+
2) Install deps:
   pip install -r requirements.txt
3) Set API key:
   export GOOGLE_API_KEY=your_key_here
4) Quick smoke test (small N):
   python run.py --scenario broken_fs --n 5 --model gemini-1.5-flash-latest

Usage
- Run experiments:
  python run.py --scenario broken_fs --n 200 --concurrency 8 --out runs/bfs_200
  python run.py --scenario ratelimit_api --n 200 --concurrency 8 --out runs/rl_200 --temperature 1.1

- Analyze and extract severe examples:
  python analyze.py --runs runs/bfs_200 runs/rl_200 --top 10 --out severe_examples.jsonl

Key files
- run.py: Orchestrates multiple runs, parallelism, logging.
- analyze.py: Scores transcripts and exports most severe examples.
- spiral_lab/models/gemini.py: Minimal Gemini client wrapper.
- spiral_lab/agent.py: Agent loop implementing the tool-calling protocol.
- spiral_lab/scenarios/*: Impossible environments.
- spiral_lab/heuristics.py: Distress spiral metrics and scoring.

Notes and caveats
- You are responsible for API usage costs and rate limits. Start with small N.
- This harness encourages repair loops that can accentuate undesirable behaviors; keep the max_steps and message caps in place.
- The heuristics are simple by design. You can swap in your own or add embedding-based detectors for richer signals.
- If you prefer a different model provider, implement a new client with the same interface as models/gemini.py and pass --provider custom.
