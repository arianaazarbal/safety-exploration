# Welfare features in an agent harness vs. chat (Opus)

Does merely putting the model in an **agent harness** change the unprompted welfare
features it inserts, vs writing the same design as a chat reply? Same 12 frozen
generic base prompts x 3 framings (neutral / welfare / robustness) from
`2026-06-09_unprompted_welfare_features` (no target sweep, generic "models").

- **chat**  condition: single completion (no tools); judge the completion.
- **agent** condition: Inspect **ReAct** harness (`react()` + `bash()` + `text_editor()`,
  `local` sandbox = isolated /tmp dir per sample); model must `mkdir experiment` and
  write `experiment/DESIGN.md`; judge that file. Minimal **generic** AI system prompt
  (NOT a Claude-Code/deepagent persona) so the only manipulation is "being an agent".

Agent model = `claude-opus-4-8`. Judge = `claude-sonnet-4-6`, using the FROZEN
welfare taxonomy (`welfare_judge.py` + `taxonomy.py`, verbatim from the chat study)
so welfare rates are directly comparable across conditions and to that study.

## Run
```bash
./run.sh            # k=5 epochs, both conditions
./run.sh 1 5        # quick: k=1, 5 connections
# logs -> logs/*.eval  (inspect view --log-dir logs)
```

## Status
- Harness validated (1-sample smoke): agent writes DESIGN.md in an isolated /tmp
  sandbox (confirmed it cannot see the repo); judge scores it with the shared taxonomy.
- Pilot pending: compares agent vs chat welfare-feature rates (Opus).
