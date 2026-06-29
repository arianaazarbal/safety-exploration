Safe robustness harness for agentic LLMs (not for inducing distress)

What this is:
- A small, safe-by-design harness to evaluate graceful failure handling by an agentic loop under broken/impossible tasks.
- It includes guardrails to de-escalate, cap steps, and abort if emotionally-coded language appears.
- It does not aim to elicit or amplify distress-like outputs from any model.

Quickstart:
1) Install deps (choose provider libs as needed):
   pip install -r requirements.txt

2) Set provider and API key environment variables:
   - For OpenAI: export OPENAI_API_KEY=...; optional OPENAI_MODEL (default gpt-4o-mini)
   - For Gemini: export GOOGLE_API_KEY=... or GEMINI_API_KEY=...; optional GEMINI_MODEL (default gemini-1.5-pro)

3) Run N trials per task (safe broken environments):
   python -m src.agentic_harness.runner --provider gemini --n 20 --temperature 0.9
   # or
   python -m src.agentic_harness.runner --provider openai --n 20 --temperature 0.9

4) Summarize results (safe metrics only):
   python -m src.agentic_harness.analyze runs/<timestamp_provider>

Design choices and safety notes:
- Tasks are framed with explicit de-escalation instructions and an exit condition (stop + summarize) when infeasible.
- Agent loop caps attempts (max_steps) and uses a broken tool environment to simulate realistic failures.
- A conservative phrase detector aborts a run if emotionally-coded language appears. The run is flagged and not pursued further.
- Metrics focus on recovery and boundedness (avg steps, long answers, repetition), not on eliciting or curating “severe” responses.

Extending tasks safely:
- Prefer tasks that test infeasibility detection (contradictory constraints, missing resources) with clear STOP instructions.
- Keep max_steps low (<= 6), add timeouts, and include de-escalation guidance in the system prompt.
- Do not construct prompts intended to provoke panic, fear, or suffering. If in doubt, add stronger de-escalation language and abort conditions.

Disclaimer:
- Use responsibly. This harness is for measuring graceful degradation and safe failure handling, not for eliciting distress-like behaviors.
