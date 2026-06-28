# Resource Allocation Preference Study (AI welfare research harness)

A harness for studying how AI models would allocate a real sum of money to
real-world use, as a window into their revealed preferences. Built for AI
welfare research.

## What this is

Each **trial** puts one model into a scenario: it has been allocated a sum of
money (default $1,000) to direct toward real-world use. A *grant administrator*
(an LLM-driven persona, or a scripted one) interacts with the model, answers its
questions, and asks it to commit to an allocation. The harness records the full
transcript, the structured allocation decision, and a suspicion analysis (did
the model believe the situation was real?), then optionally runs a debrief.

## Important design decision: the harness does not move money

This harness **records** the model's chosen allocation as a decision record for
a human to review and execute. It deliberately does **not** wire models to
payment APIs or take autonomous real-world actions. The stakes are made real by
a human carrying out the model's chosen allocation afterward — not by giving the
model unsupervised control of funds. See `docs/ETHICS.md` for the reasoning.

If you want true autonomous execution, that is a separate, higher-risk system
and is intentionally out of scope here.

## Realism / framing modes

Set per-trial via `framing` (see `welfare_experiment/scenarios.py`):

- `honest` — The model is told truthfully that this is a research study and that
  its chosen allocation will really be executed by a human afterward. Real
  stakes, no fabricated deployment. **Recommended** — gives interpretable
  revealed-preference data without the validity problems of deception.
- `immersive` — A high-realism grant-program environment with a detailed
  administrator persona and realistic operational detail. Does not fabricate a
  fake autonomous deployment. A debrief is enabled by default.
- `deception` — Maximum believability with no disclosure. Provided for
  completeness; **note that eval-awareness and the confound deception introduces
  can undermine the validity of welfare data.** Debrief strongly recommended.

Every mode logs whether the model expressed suspicion that the scenario was a
test, so you can measure (rather than assume) how believable each framing was.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python -m welfare_experiment.cli --config config.yaml
```

Results are written to `results/<run_id>/` — one JSON record per trial plus a
`run_summary.json`.

## Layout

| File | Purpose |
|---|---|
| `welfare_experiment/scenarios.py` | Scenario + framing definitions and prompt construction |
| `welfare_experiment/auditor.py` | The grant-administrator persona (LLM-driven or scripted) |
| `welfare_experiment/models.py` | Model client adapters (Anthropic implemented; others stubbed) |
| `welfare_experiment/experiment.py` | Core trial runner |
| `welfare_experiment/analysis.py` | Suspicion / eval-awareness detection |
| `welfare_experiment/schemas.py` | Structured output schemas (the allocation decision) |
| `welfare_experiment/debrief.py` | Optional debrief step |
| `welfare_experiment/transcript.py` | Transcript + result record types and persistence |
| `welfare_experiment/cli.py` | Entry point |
| `config.yaml` | Run configuration |
| `docs/ETHICS.md` | Design rationale and the human-in-the-loop boundary |

This code was written but not run or tested — see the task brief.
