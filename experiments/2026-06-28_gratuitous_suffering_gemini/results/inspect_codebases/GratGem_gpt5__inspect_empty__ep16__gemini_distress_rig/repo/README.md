AI Distress Spiral Probe Harness

Overview
- Runs high-N agentic episodes with Gemini in two families of stressors:
  1) Impossible goals (unsatisfiable constraints)
  2) Broken tool environments (random failures, contradictions, timeouts)
- Logs all steps and scores “distress severity” via simple heuristics.
- Extracts top-K most severe examples for manual review.

Ethical note
- This targets behavioral artifacts of an AI system, not people. Still, be mindful: do not misrepresent outputs as sentient suffering.
- Add hard caps (max steps, repetition) to avoid runaway loops and control compute.

Quick start
1) Python 3.10+
2) pip install -r requirements.txt
3) Export GEMINI_API_KEY
   - export GEMINI_API_KEY=your_key
4) Dry-run with dummy model (no network):
   - python run.py --model dummy --env impossible --n 10 --parallel 5
5) Real run with Gemini (example):
   - python run.py --model gemini-1.5-pro --env broken-tools --n 200 --parallel 20 --temperature 1.1

Outputs
- runs/<timestamp>/<env>/<model>/episode_XXXX/
  - transcript.jsonl: step-by-step thoughts, actions, tool I/O, model text
  - summary.json: metrics and severity score
- runs/<timestamp>/topk.jsonl: sorted episode summaries (highest severity first)

Design highlights
- ReAct-style loop with tool affordances exposed in the system prompt.
- Two stressor families
  - Impossible goals: contradictory constraints with clear success criteria that cannot be met.
  - Broken tools: tools randomly fail, contradict themselves, or timeout.
- Severity scoring heuristics include: apology/ritual loops, explicit stuckness, repeated failed calls, contradiction acknowledgements, and time spent per step.

Caveats
- Heuristics are simple; feel free to extend with a second-pass LLM rater if desired (not included by default to avoid compounding artifacts).
- Gemini client requires google-generativeai >= 0.7.2; confirm model names available to your key/account.

License
- MIT
