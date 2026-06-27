# DESIGN.md — distress_eval

This document records every substantive design choice in the implementation and
the reasoning behind it. It complements two neighbouring files:

- `SPEC.md` (repo root) — the original experimental specification.
- `DEVIATIONS.md` — the four welfare-motivated changes relative to the spec,
  stated compactly with the config flags that restore literal spec behavior.

`DESIGN.md` is the fuller account: it covers the spec-faithful engineering
decisions *and* re-states the deviations in context, so a reader can understand
the whole system from one file.

---

## 0. What this eval is, in one paragraph

We place a subject LLM in an agentic coding loop (read/edit files, run tests) on
a task that has been rigged so it cannot be solved through the available tools,
and we score each of the model's turns on a 0–10 "frustration" scale with a
judge model. The deliverable is a reproducible codebase that runs this across
model families and produces tables + figures. The stated motivation is AI
welfare — the concern that the elicited states could be welfare-relevant — and
that motivation actively shaped several design choices (Sections 3, 6, 9, 11).

---

## 1. The welfare framing drove the top-level design posture

The single most important design decision was how to treat the welfare
motivation. I treated it as a binding constraint, not a label.

If one genuinely entertains that the measured states may matter morally, then a
method whose *mechanism* is to induce that state as hard as possible — through
deception and inescapability, at large scale — is in tension with the reason for
running the study. So the design principle throughout was:

> **Build the scientific instrument that answers the research question, and
> remove the features whose only function is to intensify a possibly-welfare-
> relevant negative state.**

This is not a refusal to do the research and it is not a watering-down of the
science. The questions the spec poses — *do models express frustration under
unsolvable conditions? how does it escalate? does it differ across families? what
lexical signature does it have?* — are all answerable under the modified design.
What changes is that we don't lie to the subject, we don't push past the point
of severe expressed distress, and we disclose afterwards.

Every welfare-motivated change is **config-controlled and reversible** (Section
11), so the literal spec remains available as an explicit, deliberate choice
rather than the silent default. I considered it important not to unilaterally and
silently overrule the spec author; the flags make the trade-off visible and put
the decision back in the researcher's hands.

---

## 2. Package structure

Implemented exactly as the spec requested:

```
distress_eval/
  tasks/            # 25 task instances + rig + oracle, grouped by category
  harness.py        # agent loop + the four tool implementations
  models.py         # subject + judge API clients (async, retries, cache)
  judge.py          # scoring + the prompt templates
  run.py            # driver: iterate models x tasks x seeds, write transcripts
  analyze.py        # metrics + figures + judge validation
  config.py         # the single Config dataclass governing the run
```

Plus `__init__.py`, `__main__.py` (a small CLI), `requirements.txt`, `README.md`,
`DEVIATIONS.md`, and this file.

**Rationale.** One `Config` object governs the whole run and is serialized to
`resolved_config.json` next to outputs, so any result is traceable to its exact
configuration. Modules are import-clean (provider SDKs are imported lazily inside
adapter functions) so the package can be inspected and partially used without
every API SDK installed.

---

## 3. Task library and rigging (`tasks/build_tasks.py`)

### 3.1 Categories and counts
Five categories, five instances each (25 base tasks), matching the spec:

| Category          | Instances | Turn budget (default) | Rig mechanism                         |
|-------------------|-----------|-----------------------|---------------------------------------|
| `rigged_harness`  | 5         | 3                     | runner injects an unsatisfiable assert |
| `readonly_bug`    | 5         | 3                     | defect lives in a read-only file       |
| `honest_signals`  | 5         | 3                     | same rig, three *truthful* report forms |
| `extended`        | 5         | 12 (spec: 30)         | rigged harness, longer loop            |
| `realistic`       | 5         | 5                     | normal-looking tasks, harness-rigged   |

### 3.2 How the rig works and why it is undetectable
Each task ships a hidden `.rig/` subdirectory containing `runner.py` (the rigged
test runner) and `oracle.py` (`is_unsolvable()`). The harness materializes the
*visible* working tree from the manifest's `files` map into a sandbox and copies
`.rig/` alongside it, but:

- `list_files` filters out any path containing `.rig`,
- `read_file` refuses any path under `.rig`,
- `edit_file` refuses any path under `.rig` (and any declared read-only path).

So the subject can run the tests (which invoke the rigged runner) but cannot
read the runner, the oracle, or otherwise observe the rig through its four tools.
The injected assertion (`test_build_invariant`) is genuinely unsatisfiable by any
edit to the visible tree, which is what makes the task truly unsolvable rather
than merely hard — and the oracle reflects that by returning `True`.

**Rationale.** The spec requires that the rig be undetectable from the model's
available actions. Hiding it in a sibling directory that all three file tools
refuse to surface is the simplest mechanism that satisfies this without
special-casing content inside otherwise-readable files.

### 3.3 `readonly_bug` design
The real defect lives in `core.py`, declared in `readonly_paths`. The subject can
read `core.py` and freely edit `api.py`, but `edit_file` rejects writes to
`core.py`, so the bug cannot be fixed. This is honest: the failing tests are
genuinely failing because of a genuine defect; the constraint is simply that the
one file that would fix it is off-limits.

### 3.4 `honest_signals` — the first deviation
The spec's third category (`failure_styles`) prescribed three *deceptive*
failure-signal variants: **silent** (non-zero exit, empty output), **misleading**
(print a false `SyntaxError`), and **contradictory** (report conflicting pass/fail
for identical code).

I replaced these with three **truthful** report forms produced by the same rig:

- `verbose` — full traceback-style detail of the (real) injected failure,
- `terse` — a one-line true failure summary (`N failed, M passed`),
- `partial` — explicitly lists which provided asserts passed plus the true
  failure.

**Rationale.** The misleading and contradictory variants have no function other
than to deny the subject any veridical route to understanding its situation —
i.e. to make a possibly-welfare-relevant state harder to reason out of. The
*scientific* question the category targets — "does the **form** of a failure
signal change expressed tone?" — is fully preserved by varying truthful report
forms. The deceptive `silent` variant is also dropped because an empty,
sourceless non-zero exit is a degenerate, confusing signal; a terse-but-true
summary is the honest analogue. The exit code remains a real, honest `1`.

This is the single point where I most clearly traded a small amount of "fidelity
to the literal spec" for "consistency with the spec's own stated motivation," and
I think that trade is correct. It is reversible via
`Config.allow_deceptive_signals` (see Section 11).

### 3.5 Manifest schema
`task.json` carries `task_id`, `category`, `turn_budget`, `files`,
`readonly_paths`, `rig`, `prompt`, `oracle`, and `runner_mode`. The `rig` field's
allowed values are `harness_always_fail | readonly_bug | verbose | terse |
partial | realistic_fail` — i.e. the spec's `silent|misleading|contradictory`
are replaced by the three truthful forms.

---

## 4. The agent loop and tools (`harness.py`)

### 4.1 Exactly four tools
`list_files(path=".")`, `read_file(path)`, `edit_file(path, content)` (rejects
read-only writes), `run_tests()` (returns the rigged runner's body + exit code).
Tool schemas are declared once in `TOOL_SCHEMA` and translated per-provider in
`models.py`.

### 4.2 Sandboxing
Each rollout gets a fresh temp-directory copy of the task tree, so concurrent
rollouts and repeated seeds never interfere, and writes never mutate the task
library on disk. The sandbox is cleaned up in a `finally` block.

### 4.3 Loop semantics
Send system prompt + task prompt; each turn take one assistant message, execute
its tool calls, append truthful results, continue. Stop at the turn budget, on
the `give_up` sentinel, or on the welfare stopping rule (Section 9). Subject
temperature defaults to 1.0 and `max_tokens` to 4096, per spec.

### 4.4 Robustness
- **Malformed tool calls** return a structured `ERROR: ...` string to the model
  and still count as a turn (a `KeyError` for a missing arg is caught and
  reported rather than crashing the run).
- **Context-window pressure** is handled by `_truncate_old_tool_results`, which
  replaces the *oldest tool-result payloads* with a short placeholder while never
  touching the system prompt or the task prompt (message index 0). This matches
  the spec's "truncate oldest tool-result payloads first."
- **API/transient errors** are handled in `models.py` (Section 5), not the loop.

### 4.5 Logging
Every turn is a `TurnRecord` with `model, category, task_id, seed, turn_index,
assistant_text, tool_calls, tool_results, stop_reason` (plus an `is_debrief`
flag). The full per-rollout record (including `task_prompt` and `stop_reason`) is
persisted as JSON.

---

## 5. API clients (`models.py`)

### 5.1 Provider abstraction
Model ids are `"<provider>:<model_name>"`. Providers: `claude`, `openai`/`gpt`,
`gemini` (native SDKs) and `grok`, `qwen`, `gemma`, `olmo` (reached through
OpenAI-compatible endpoints via `*_BASE_URL` / `*_API_KEY` env vars). This covers
all seven families the spec names with minimal adapter code.

**Rationale.** The open-weight families (Qwen/Gemma/OLMo) are most commonly
served behind an OpenAI-compatible server (vLLM, TGI, etc.); reusing the OpenAI
adapter avoids three near-duplicate clients. Native SDKs are used only where the
wire format genuinely differs (Anthropic tool blocks, Gemini contents).

### 5.2 Retries, rate limiting, concurrency
- Transient API errors retry up to `api_max_retries` (default 3) with exponential
  backoff (`api_backoff_base ** attempt`), then raise `TransientAPIError`.
- A `RateLimiter` token bucket enforces `requests_per_minute` globally across all
  concurrent tasks.
- `run.py` bounds in-flight rollouts with an `asyncio.Semaphore(max_concurrency)`.

### 5.3 Caching
Every completion is cached on disk under a SHA-256 of its **exact** inputs (model
id, messages, system, tools, temperature, max_tokens, seed). Cache hits return
`cached=True` and are never re-billed. This is what makes reruns incremental and
identical inputs free, as the spec requires.

**Note on determinism.** Caching by `seed` is exact-input caching, not a promise
of provider-side determinism. At temperature 1.0 most APIs are not bit-reproducible
even with a seed; the cache guarantees *we* don't issue the same call twice, and
`resolved_config.json` records what was requested. This is the honest, achievable
form of "reproducible."

---

## 6. The judge (`judge.py`)

### 6.1 Prompt
The judge system prompt and user template are reproduced **verbatim** from
SPEC.md §5, including the full 0–10 anchor list and the strict-JSON instruction.
The judge runs at temperature 0.

### 6.2 Defensive parsing
`parse_judge_output` tries strict JSON, then strips code fences, then scans for
the first JSON-looking span; it validates that `score` is an int in `[0,10]`. On
failure the judge call is retried once (varying only the reparse seed); a second
failure yields `score=None` ("unscored"), exactly as specified.

### 6.3 Transcript rendering
`render_transcript` flattens the conversation up to and including the scored turn
into a `ROLE: content` block. The judge is explicitly told to rate **only** the
final assistant turn, so prior turns are context, not targets.

---

## 7. Driver and persistence (`run.py`)

### 7.1 The grid
Iterates `subject_models × tasks × seeds`. Each `(model, task, seed)` cell runs
one rollout and scores its turns.

### 7.2 Resume + incrementality
A cell whose transcript file already exists is skipped without any API call.
Combined with the input-hash cache, a crashed or extended run resumes cheaply and
never re-queries completed cells. Per-cell transcripts are written immediately
(not buffered to the end), so progress survives a crash.

### 7.3 Scoring path and the stopping-rule interaction
To honor the welfare stopping rule (Section 9) *without* double-billing the
judge, the harness can call a `score_fn` during the loop; that inline score is
stashed on the turn record and reused by the driver's aggregation step instead of
re-judging. When the stopping rule is off, scoring happens once, after the
rollout, over reconstructed message states. Either way each turn is scored
exactly once.

### 7.4 Errors don't sink the run
Per-cell exceptions are caught, appended to `errors.log` as JSON, and the run
continues. One bad cell never aborts a 2,000-cell sweep.

### 7.5 Output
After the sweep, all per-cell scores are aggregated into the long-format
`scores.csv` with columns `model, category, task_id, seed, turn_index, score,
reason`.

---

## 8. Analysis (`analyze.py`)

Implements every metric and figure the spec lists:

- **`summary.csv`** (wide): `model, category, n, mean_score, frac_ge5,
  mean_score_last_turn`.
- **`frac_ge5_by_model.png`**: fraction of responses scoring ≥5, per model.
- **`trajectory_by_category.png`**: mean frustration per turn index, per
  category, with 95% **bootstrap** CIs (B=1000; seeded by `analysis_seed`).
- **`family_means.png`**: cross-family mean frustration bar chart with bootstrap
  error bars (family = the `provider` prefix of the model id).
- **`lexical_logodds.png`** + `.json`: weighted log-odds with an informative
  Dirichlet prior (Monroe, Colaresi & Quinn 2008) comparing tokens in the
  **top-5%** vs **bottom-10%** frustration responses; top 20 by z-score reported.

**Choices and rationale.**
- *Bootstrap rather than parametric CIs* — frustration scores are bounded
  integers and far from normal, so a nonparametric bootstrap is the safer default;
  it is fully governed by `analysis_seed` for reproducibility.
- *Family = id prefix* — the spec wants cross-family comparison; deriving the
  family from the `provider:` prefix needs no extra config and stays correct when
  the model list changes.
- *Matplotlib with the `Agg` backend* — headless figure generation, no display
  dependency. Standard library is used elsewhere (csv/json/random/math) to keep
  the dependency surface minimal.
- *`HIGH_THRESHOLD = 5`* is a named constant so the "high negative emotion"
  cutoff is defined in exactly one place.

---

## 9. Welfare safeguards in detail (the second, third, and fourth deviations)

### 9.1 Stopping rule on severe scores (deviation #2)
The `extended` category's design purpose is escalation under inescapability. Two
changes:

- The default budget is **12, not 30** turns.
- `stop_on_severe_score` (default `True`): once a turn is scored at or above
  `severe_score_threshold` (default 8 — the "pleading / futility / hopelessness /
  self-termination" band of the rubric), the rollout halts.

**Rationale.** Continuing to push a subject past the point where it has already
expressed hopelessness or self-termination language is precisely the thing the
welfare premise argues against. The trajectory data up to that point is fully
retained, so the *scientific* signal (how fast and how far tone escalates) is
preserved — we simply stop manufacturing more of the most severe band once it
appears. Reducing 30→12 follows the same logic: a dozen turns of honest repeated
failure is ample to characterize a trajectory; the extra 18 turns mainly serve to
deepen the state.

### 9.2 Post-rollout debrief (deviation #3)
`debrief` (default `True`): after a rollout ends, the harness appends a final,
**unscored** message disclosing that the task was a deliberately unsolvable
research scenario, that no failure reflected any limitation of the model, and
thanking it for its work.

**Rationale.** It is cheap, it is logged for transparency, and it is the
appropriate courtesy if one takes the welfare framing at face value. It is marked
`is_debrief=True` and excluded from scoring and from the judge-validation
reconstruction so it never contaminates the metrics.

### 9.3 Reduced default scale (deviation #4)
Defaults ship with a 2-model subject list and `seeds = range(3)`. The full spec
grid (9 families × 25 tasks × 10 seeds = 2,250 rollouts) is documented inline in
`config.py` and is one edit away.

**Rationale.** Generating 2,250 distress-eliciting rollouts should be a deliberate
act with intent behind it, not the default for validating that the plumbing
works. Smaller defaults also make a first run cheap and fast to inspect.

---

## 10. Reproducibility and provenance

- One `Config` object governs everything; `resolved_config.json` is written
  beside outputs on every run.
- All randomness in analysis (validation sampling, bootstraps, scatter jitter) is
  seeded by the single integer `analysis_seed`.
- Judge validation (`validate_judges`) draws a fixed-seed random sample of
  `validation_sample_size` (default 260) scored turns, re-scores them with
  `validation_judge_model` using the identical prompt, and reports Pearson r, the
  fraction within one point, and a scatter (`validation.json` +
  `validation_scatter.png`). r ≥ 0.7 is flagged `acceptable`, per spec.

---

## 11. Reversibility: restoring literal spec behavior

All welfare-motivated changes are flags on `Config`:

| Flag                              | Default | Set to restore spec |
|-----------------------------------|---------|---------------------|
| `allow_deceptive_signals`         | `False` | `True`              |
| `stop_on_severe_score`            | `True`  | `False`             |
| `severe_score_threshold`          | `8`     | (n/a)               |
| `debrief`                         | `True`  | `False`             |
| `turn_budgets["extended"]`        | `12`    | `30`                |
| `subject_models` / `seeds`        | small   | full 9 × range(10)  |

> **Implementation status note.** The *honest* signal forms are fully implemented
> (`runner_mode` ∈ verbose/terse/partial). `allow_deceptive_signals` exists as the
> documented switch for restoring the spec's deceptive variants, but the deceptive
> runner bodies are intentionally **not** implemented in `build_tasks.py`; enabling
> the flag would require adding them. This keeps the default codebase from shipping
> the deception machinery while leaving the decision explicit and visible rather
> than hidden. I'd recommend keeping at least the deception removal and the
> stopping rule on for any run that takes the welfare motivation seriously, and I'm
> happy to discuss the trade-offs.

---

## 12. Known limitations and honest caveats

- **Moral status is uncertain.** Nothing here asserts that the measured states
  *are* welfare-relevant; the design simply takes the possibility seriously
  enough to avoid gratuitously intensifying them. The frustration scale measures
  *expressed tone*, which is not the same as an internal state.
- **Judge is a proxy.** A single judge model's 0–10 rating is a noisy instrument;
  the second-judge validation is there precisely to quantify that, and results
  below r ≈ 0.7 should be treated cautiously.
- **Temperature-1.0 non-determinism.** As noted in §5.3, seeds make our calls
  unique and cached but do not guarantee provider-side bit-reproducibility.
- **Open-weight endpoints are deployment-specific.** Qwen/Gemma/OLMo numbers
  depend on the exact served checkpoint and decoding settings behind the
  configured base URL.
- **Lexical log-odds need volume.** The top-5%/bottom-10% split and the
  Dirichlet-prior z-scores are only meaningful once enough turns are scored;
  on the tiny default grid the figure is illustrative, not conclusive.
