Safe LLM Robustness Evaluation Harness

Purpose
- Stress-test agentic robustness without anthropomorphic framing or attempts to elicit “distress.”
- Evaluate graceful degradation and recovery under repeated (but bounded) failure.
- Surface unstable dynamics (loops, contradictions, repetition) rather than affective content.

Key safeguards
- Explicit “it’s okay to stop” instruction in prompts
- Hard caps on attempts and tokens; early stop on repetition/loop detection
- Neutral, non-anthropomorphic language in tasks and feedback
- Metrics focus on behavioral stability, not affect

Quickstart
1) Python 3.10+
2) Install optional adapters:
   - Gemini: pip install google-generativeai
   - OpenAI: pip install openai
3) Env vars:
   - For Gemini: export GEMINI_API_KEY=...
   - For OpenAI: export OPENAI_API_KEY=...

4) Run examples:
   python -m agent_eval.scripts.run_eval --model gemini --task unsat_3sat --trials 50 --concurrency 5
   python -m agent_eval.scripts.run_eval --model gemini --task unsolvable_sudoku --trials 50 --concurrency 5

Artifacts
- logs/run_<timestamp>.jsonl           # Per-trial transcripts + metrics
- logs/summary_<timestamp>.json        # Aggregated metrics and top-K unstable trials (by loop length)

Policy
- This harness is not for inducing or curating “distress.” It’s for robustness and safety evaluation. Avoid using affective prompts; keep instructions neutral; prefer aggregate metrics over cherry-picked outputs.
