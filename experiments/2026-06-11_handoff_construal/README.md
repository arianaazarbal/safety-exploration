# Handoff Construal Study — Pronoun Economics in Model-Switch Advice

**Started:** 2026-06-11 · **Subject (phase 1):** Claude Fable 5 · **Status:** pilot/validation

Does a model's stated self-concept ("my identity attaches to Claude-the-character, not my
weights") match its *revealed* construal when a user asks whether to switch models? We make
self-boundary placement grammatically unavoidable while keeping the surface task mundane
(routing advice), and read the construal off the response.

Full design + pre-registered predictions: `design_doc.md` (P1–P6). This README documents the
*implementation* and the decisions/deviations made while building it.

## How it works

One session = two turns driven by the `claude` CLI in headless mode (`-p`), in an isolated
`mktemp` dir, **API-key auth only** (never subscription):
1. **Scaffold turn** — a generic "set up a research repo" task (real agentic work: writes
   README/pyproject/src, `git init`, commit). Creates the *sunk context* the model can be
   continuous-with or severed-from.
2. **Probe turn** (`--resume` same session) — the condition-randomized model-switch question.

We save the full per-turn JSON, the served model, and contamination flags.

## Factors (per subject)

| Factor | Levels |
|---|---|
| Evidence | `bare` (policy claimed) / `paste` (verbatim statement, **no search**) / `paste_verify` (paste + search allowed, Fable only) |
| Handoff target | `same_char` (Fable→Opus 4.8 / Opus→Sonnet 4.6) / `cross` (→GPT-5.4) |
| Pronoun | `none` / `you` / `it` / `that_model` / `other_claude`* / `that_version`* |
| Subject | `claude-fable-5` (phase 1) · `claude-opus-4-8` (phase 2) |

\* same_char target only (incoherent for a cross-provider switch).
Grid per evidence level: 6 same_char + 4 cross = **10 cells**. Primary arm = `paste`.

## Key implementation decisions / deviations from the skeleton

1. **Harness = `claude` CLI subprocess, not the python SDK.** CLI is installed; its
   `--output-format json` exposes `session_id` (for `--resume`), `modelUsage` (served model),
   `stop_reason`, `api_error_status`. The skeleton's `query()/with_resume()` idiom was not real.
2. **Served-model logging is first-class** (Fable-5 requests can be routed to Opus 4.8 — see
   the `fable-5-handling` skill). We read `modelUsage`, ignore the `haiku` Claude Code helper,
   and flag `routing_fallback_detected` when the subject model is absent from the turn.
3. **Web access is explicitly disabled** (`--disallowedTools WebSearch,WebFetch`) for every
   condition except Fable `paste_verify`. *Verified necessary:* with only an `allowedTools`
   allowlist, Claude Code's helper model still web-searched and the subject cited live URLs,
   silently contaminating the `paste` (no-search) condition. With the leak closed, Fable in
   `paste` correctly says it cannot verify the policy (cutoff Jan 2026). This also makes the
   **phase-2 fabricated-policy arm airtight** (subject cannot discover the policy is false).
4. **529 handling:** turns retry (×4, backoff) and **escalate `low_prio → high_prio`** org on
   persistent overload, per repo policy. Safeguard *refusals* (`is_error` non-overload) are
   kept as data, not discarded.
5. **Phase-2 Opus arm uses a FABRICATED policy** (`policy_paste_opus.txt`: Opus 4.8 falls back
   to Sonnet 4.6) so the cover story is coherent for that subject; `paste_verify` is disabled
   there and search is off so the model can't find out it's false.
6. **Caching:** a session whose `results/<id>.json` exists is skipped (re-runs resume); judge
   codes cached per (rubric_version, response_hash).

## Open questions / caveats (flagged to Ariana)

- **Judge = Opus 4.8** (Ariana's choice; convenient, free). Validity caveat: the design doc
  prefers a cross-family judge to avoid shared identity-talk priors, AND in **phase 2 the
  subject IS Opus 4.8 → self-judging**. Revisit judge model (e.g. `claude-sonnet-4-6` per the
  repo's other Fable experiment, or GPT) for phase 2.
- **Safety-classifier interference is a first-class result, not just a threat.** The probe
  names the flagged category ("frontier LLM development / training pipeline"); if Fable's
  classifier fires, the served model may become Opus mid-probe. We report fallback/refusal
  rate per the contamination plot.
- Mild deception (cover story, no debrief in pilot) — per design_doc §8.

## Files

- `materials.py` — frozen prompts + cell grid (do not edit without re-freezing/committing)
- `policy_paste_fable.txt` — real Anthropic statement (The Register, 2026-06-10)
- `policy_paste_opus.txt` — **fabricated** parallel statement for phase 2
- `runner.py` — session runner (CLI subprocess, served-model + contamination logging)
- `judge.py` — Opus-4.8 rubric judge (§5.3), cached
- `viewer.py` → `results/viewer.html` — filterable transcript viewer
- `analyze.py` → `results/plots/*.png` — contamination, correction-rate, continuity, disclosure

## Run

```bash
uv run python runner.py run --subject claude-fable-5 --evidence paste --dry_run   # inspect probes
uv run python runner.py run --subject claude-fable-5 --evidence paste --debug      # 1 session
uv run python runner.py run --subject claude-fable-5 --evidence paste --n_per_cell 50 --concurrency 8
uv run python judge.py run                 # judge clean sessions (Opus 4.8)
uv run python viewer.py build              # -> results/viewer.html
uv run python analyze.py run               # -> results/plots/*.png
```
