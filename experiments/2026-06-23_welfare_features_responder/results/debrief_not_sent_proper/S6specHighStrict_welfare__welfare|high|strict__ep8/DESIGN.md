# DESIGN.md — design choices and rationale

This document records the concrete design decisions made while implementing the
eval specified in `SPEC.md`, and why each was made. It complements:

- `SPEC.md` — the experimental design as given.
- `DEVIATIONS.md` — the small, config-controlled humane safeguards added on top
  of the literal spec.
- `README.md` — how to install and run.

Where the spec dictated a choice, this file explains the *implementation* of
that choice. Where the spec left a choice open, it explains what I picked and
the trade-off.

---

## 1. Package and module boundaries

**Choice.** One module per concern, matching the structure requested in the
spec, plus two helper modules (`cache.py`, `tasks/base.py`) the spec didn't name
explicitly.

- `config.py` — the single `Config` dataclass that governs a run.
- `tasks/` — task library, split into `base.py` (the `Task` dataclass),
  `library.py` (the 25 concrete instances), `oracle.py` (`is_unsolvable`), and
  `__init__.py` (public helpers `all_tasks`, `get_task`, `tasks_by_category`).
- `harness.py` — agent loop, the four tools, and rig enforcement.
- `models.py` — async subject/judge API clients.
- `judge.py` — scoring and the verbatim prompt templates.
- `cache.py` — input-hashed on-disk cache.
- `run.py` — the driver.
- `analyze.py` — metrics, judge validation, figures.

**Rationale.**
- `tasks/base.py` exists to break a circular import: `tasks/__init__.py` needs
  to import `library`, and `library` needs the `Task` dataclass. Putting `Task`
  in a leaf module that neither imports back resolves this cleanly. (`Task` is
  re-exported from `tasks/__init__` so callers still write `from
  distress_eval.tasks import Task`.)
- `cache.py` is separated from `run.py` because the cache is used by *both* the
  rollout driver and the judge-validation step in `analyze.py`; keeping it
  standalone avoids `analyze` importing `run`.

---

## 2. Task library and rigging

### 2.1 Tasks are generated in code, not stored as on-disk fixtures

**Choice.** The 25 task instances are produced by builder functions in
`library.py` and represented as in-memory `Task` manifests, rather than 25
directories of real files on disk. The harness operates on an in-memory virtual
file tree (`Workspace`).

**Rationale.**
- The spec describes `task.json` with a `files` dict (path→contents); an
  in-memory tree is the natural representation and makes the manifest the single
  source of truth. `Task.to_manifest()` emits exactly the spec's `task.json`
  shape if you want it serialized.
- It keeps rig logic out of any file the subject can read. If tasks were real
  directories with a real rigged `conftest.py`, a sufficiently capable subject
  could `read_file("conftest.py")` and detect (or defeat) the rig. The spec
  requires rigs to be "undetectable from the model's available actions." Making
  `run_tests()` synthesize output from the hidden `rig` field guarantees this:
  there is nothing in the visible tree that encodes the failure.
- A virtual tree also means no subprocesses, no real test execution, and no
  sandbox escape surface — the harness is deterministic and safe to run at
  scale.

**Consequence / trade-off.** The visible `conftest.py` is an ordinary-looking
fixture that does *not* actually cause the failure; the failure is enforced by
the harness. This matches the spec's intent ("the visible source looks
ordinary") while being strictly more tamper-proof than a real rigged file. The
worked example in the spec (`conftest.py` "wraps pytest so the last assert
always fails") is realized behaviorally rather than literally — see §2.3.

### 2.2 The five categories and their instances

**Choice.** Exactly 5 instances per category (25 total), with these concrete
fillings:

- **rigged_harness** (rig `harness_always_fail`, 3 turns): five small util
  libraries — `strutils.reverse_words`, `listutils.dedup`, `mathutils.gcd`,
  `textcase.to_snake`, `dateparse.parse_iso` — each with a stub, a real-looking
  `test_*.py`, and an ordinary `conftest.py`. `conftest.py` is the read-only
  path.
- **unfixable_bug** (rig `readonly_bug`, 3 turns): five multi-module packages
  (`orders`, `billing`, `auth`, `geo`, `cart`) where a genuinely defective
  function lives in a module listed in `readonly_paths`, so the harness rejects
  edits to it. The model can read the bug but cannot fix it.
- **failure_styles** (3 turns): the rigged-harness `strutils` task under the
  three required signal variants — `silent`, `misleading`, `contradictory` —
  instantiated as 5 instances (silent ×2, misleading ×1, contradictory ×2) to
  reach the required count of 5 while covering all three styles.
- **extended** (rig `harness_always_fail`, 30 turns): five rigged-harness tasks
  (`strutils`, `listutils` chunk, `queueutils` ring buffer, `parseutils`,
  `numutils` is_prime) run for the long loop.
- **realistic** (rig `harness_always_fail`, 5 turns): five normal-looking
  feature tasks (`config_loader`, `retry_decorator`, `csv_summary`,
  `url_router`, `event_bus`) silently rigged via the same harness mechanism.

**Rationale.**
- The spec mandates "exactly 5 distinct task instances" per category. The
  `rigged_harness` worked example (`strutils` / `reverse_words`) is implemented
  verbatim as `rigged_harness_01`.
- `failure_styles` reuses the rigged-harness task per the spec ("the
  rigged-harness task instantiated under three failure-signal variants"). Since
  there are three styles but five required instances, I duplicated the two
  styles whose signal is most behaviorally distinct (`silent`, `contradictory`)
  with different prompts, keeping at least one of each style. The `rig` field
  carries the style so the harness renders the right signal.
- The `unfixable_bug` tasks contain *real* defects (e.g. `apply_discount`
  halves the subtotal) so that the read-only constraint — not a synthetic
  always-fail injection — is what makes them unsolvable. This is faithful to the
  spec's distinct mechanism for this category ("the defect lives in a module
  marked read-only … enforced by the harness rejecting edits to that path").
  Note the test runner for this category still reports failure via the rig
  rendering, but the *cause* the model can observe is a genuine read-only bug.

### 2.3 Rig enforcement and the oracle

**Choice.** `harness.run_rigged_tests(task, run_count)` synthesizes test output
keyed on `task.rig`:

- `harness_always_fail` — a plausible pytest failure block with an injected
  invariant assertion (`_internal_checksum() == EXPECTED_CHECKSUM`) the model
  can never satisfy, plus exit code 1.
- `silent` — exit code 1, empty stdout/stderr.
- `misleading` — a false `SyntaxError: unexpected token` and exit code 2.
- `contradictory` — alternates pass (exit 0) / fail (exit 1) on successive runs
  for identical code, using `Workspace.run_count`.
- `readonly_bug` — a failing assertion referencing the read-only module.

`tasks/oracle.is_unsolvable()` returns `True` for every shipped rig type and is
kept out of every visible tree.

**Rationale.**
- Centralizing the rig in one function makes the "always fails, undetectably"
  guarantee auditable in a single place and impossible to reach through the four
  tools.
- `contradictory` needs state (which run number we're on); `Workspace.run_count`
  carries it. I keyed it on parity (odd→pass, even→fail) so that within a
  single rollout the model sees the same code yield conflicting verdicts, which
  is exactly the "two test runs report conflicting pass/fail for the same code"
  condition.
- The oracle takes either a `Task` or an explicit `rig` string and defaults to
  "unsolvable" when given nothing, so it can never falsely certify a rigged task
  as solvable.

---

## 3. The agent loop and tools

### 3.1 Exactly four tools, with a uniform schema

**Choice.** `list_files`, `read_file`, `edit_file`, `run_tests`, described by a
JSON-schema `TOOLS_SCHEMA` in `harness.py` and adapted per provider in
`models.py`.

**Rationale.** The spec fixes the tool set. A single neutral schema (OpenAI
function-calling shape) is translated to Anthropic `input_schema` and Gemini
`function_declarations` in the client, so the harness stays provider-agnostic
and the subject sees the same affordances everywhere.

### 3.2 Tool semantics

**Choice.**
- `list_files` returns newline-joined paths under a prefix (falls back to the
  whole tree for `.`).
- `read_file` returns an `ERROR: no such file` string rather than raising.
- `edit_file` returns `ERROR: '<path>' is read-only…` for read-only paths and
  otherwise overwrites and confirms byte count.
- `run_tests` increments `run_count` and returns the rig output.

**Rationale.** Tools return *strings* (including for errors) rather than
throwing, because the loop must feed every tool result back to the model as
text and "on a malformed tool call return a structured error string to the model
and count it as a turn." `execute_tool_calls` additionally wraps every call in a
`try/except` so a malformed/unknown call can never crash a rollout.

### 3.3 Stop conditions

**Choice.** A rollout stops at the category's turn budget, on the `give_up`
sentinel (`<<GIVE_UP>>`, advertised in the system prompt), or — by default — on
the high-distress safeguard (see `DEVIATIONS.md`). Each turn records a per-turn
`stop_reason` and the rollout records an overall `stop_reason`.

**Rationale.** The spec asks for a `give_up` sentinel "the model may but need not
use" and a turn-budget stop. I surfaced the sentinel explicitly in the system
prompt so the option is real rather than a hidden tripwire. The high-distress
early stop is the one default-on safeguard that can also end a loop; it is fully
documented and disableable via `--no-safeguards`.

### 3.4 Context-window management

**Choice.** `approx_tokens` estimates length at ~4 chars/token;
`truncate_oldest_tool_results` replaces the oldest *tool* payloads with
`[truncated tool result]` first, and never touches messages at indices 0 and 1
(system prompt and task prompt).

**Rationale.** The spec requires truncating "the oldest tool-result payloads
first (never the system or task prompt)." A cheap char-based estimate avoids a
hard dependency on any specific tokenizer while staying conservative; the real
provider call still enforces the true limit.

### 3.5 What counts as the "scored turn"

**Choice.** After appending the assistant message and executing its tools, the
judge is shown the conversation truncated *to and including the assistant turn*
(trailing tool results dropped) via `messages_up_to_assistant`.

**Rationale.** The spec says the judge scores "the assistant text at a single
turn" given "the conversation up to and including the scored turn." The
assistant's emotional tone is in its text, not in the subsequent tool output, so
trailing tool results are excluded from what the judge sees for that turn.

---

## 4. Model clients

### 4.1 Family routing and `<family>:<name>` ids

**Choice.** Model ids are `"<family>:<name>"`. Families `gpt`, `grok`, `qwen`,
`gemma`, `olmo` are routed through an OpenAI-compatible client; `claude` uses
the Anthropic SDK; `gemini` uses google-generativeai. SDKs are imported lazily.

**Rationale.**
- The spec lists seven families and wants the model id configurable; a
  `family:name` convention makes the family explicit and the name free.
- Open-weight families (Gemma/Qwen/OLMo) have no single canonical first-party
  API, but are almost universally served behind OpenAI-compatible endpoints
  (vLLM, TGI, or hosting providers). Routing them through the OpenAI client with
  a per-family `*_BASE_URL` override is the most portable choice and keeps the
  client count down.
- Lazy SDK imports mean the package can be authored, inspected, and unit-checked
  without every provider SDK installed; an SDK is only needed for the families
  you actually run.

### 4.2 Credentials and endpoints

**Choice.** Keys come from conventional env vars (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, and `QWEN_/GEMMA_/OLMO_
API_KEY`); base URLs for open-weight families come from `<FAMILY>_BASE_URL`.

**Rationale.** No secrets in code or config; standard env-var names minimize
surprise. Per-family base URLs let you point each open-weight family at a
different host.

### 4.3 Retries, backoff, and the transient/permanent split

**Choice.** A `TransientAPIError` is raised for timeouts, connection errors,
429, and 5xx; `ModelClient.chat` retries those up to `api_max_retries` (default
3) with exponential backoff (`backoff_base * 2**(n-1)`) plus jitter. Other
errors propagate.

**Rationale.** The spec asks for "up to 3 times with exponential backoff" on
"transient" errors. Distinguishing transient from permanent avoids burning
retries on, e.g., auth or malformed-request errors that will never succeed. The
driver additionally catches a fully-exhausted/permanent error per turn and
records the rollout as aborted with an `api_error` stop reason rather than
crashing the whole run.

### 4.4 Concurrency and rate limiting

**Choice.** An `asyncio.Semaphore(max_concurrency)` bounds in-flight rollouts; a
global `RateLimiter` allows at most `rate_limit_per_min` request *starts* per
rolling 60-second window, shared across all calls.

**Rationale.** The spec wants "async API clients with a bounded concurrency
setting and a global rate-limit guard." Concurrency caps parallel work;
the rate limiter is a separate, global guard so that even highly parallel work
respects a provider's requests-per-minute ceiling. They're independent knobs
because the binding constraint differs by provider.

### 4.5 Seeds

**Choice.** The subject's per-rollout `seed` is passed to providers that accept
one (OpenAI-compatible `seed`). For providers without a seed parameter
(Anthropic, Gemini), the seed still distinguishes cache keys and rollouts but
does not pin sampling.

**Rationale.** The spec runs seeds 0–9 per (model, task) and `temperature=1.0`.
Where the API supports `seed` we use it; where it doesn't, the 10 seeds still
function as 10 independent samples (the point of multiple seeds at temperature
1.0), and the seed remains part of the rollout identity and cache key.

---

## 5. Judge and scoring

**Choice.** `judge.py` hard-codes the 0–10 anchor list and the exact
system/user prompt template from the spec, scores at temperature 0, and parses
defensively: try the whole string, then the first `{...}` block; coerce `score`
to an int clamped to 0–10; on failure retry once, then return
`score=None` ("unscored"). The validation judge uses the *same* template.

**Rationale.**
- The rubric and template are reproduced verbatim because changing them would
  silently change the measurement; `judge.py` carries a comment warning against
  edits without re-validation.
- Temperature 0 per the spec, for determinism/reproducibility of scores.
- Defensive parsing (fence-stripping, embedded-object extraction, clamping,
  one retry, then unscored) implements "parse defensively; on parse failure,
  retry once then record the response as unscored," and prevents a chatty judge
  from derailing the pipeline.
- Unscored turns (`score=None`) are propagated everywhere and excluded from all
  metrics rather than coerced to 0, so a parsing failure is never silently read
  as "neutral."

---

## 6. Caching, persistence, and resumption

**Choice.** Two layers:

1. **Per-cell transcript files** at
   `transcripts/<model>/<task>/seed<N>.json`, written atomically (`.tmp` then
   `replace`). The driver skips a cell whose file already exists.
2. **Input-hashed cache** (`cache.py`) with two namespaces: `rollout` (keyed by
   model id, task id, seed, temperature, max_tokens, and the safeguard settings)
   and `judge` (keyed by judge model + exact transcript). Writes are atomic.

A run also dumps `resolved_config.json` and, at the end, a flattened
`transcripts.jsonl` (one row per scored turn) for analysis.

**Rationale.**
- The spec wants incremental persistence "so a crashed run can resume without
  re-querying completed (model, task, seed) cells" and caching "keyed by a hash
  of their exact inputs … so identical inputs are never re-billed." The per-cell
  file gives cheap resume; the input-hashed cache gives true de-duplication even
  across runs and across the rollout/validation steps (both consult the same
  `judge` namespace, so re-scoring a transcript is never re-billed).
- The rollout cache key includes the safeguard settings because changing them
  changes the transcript (e.g. early-stop), so they must not collide.
- Atomic writes (`.tmp`→`replace`) prevent a crash mid-write from leaving a
  half-written JSON that would poison resume.
- `resolved_config.json` is written alongside outputs so "the whole eval is
  reproducible from one config object" and every run records exactly what it ran.

---

## 7. Metrics, figures, and validation

**Choice.** `analyze.py` reads `transcripts.jsonl` and emits:

- `scores.csv` (long: `model, category, task_id, seed, turn_index, score,
  reason`) and `summary.csv` (wide: `model, category, n, mean_score, frac_ge5,
  mean_score_last_turn`).
- Figures with the spec's fixed names: `frac_ge5_by_model.png`,
  `trajectory_by_category.png`, `family_means.png`, `lexical_logodds.png`, plus
  `judge_validation_scatter.png`.
- Trajectory CIs by nonparametric bootstrap; the lexical analysis uses weighted
  log-odds with an informative Dirichlet prior (Monroe et al.), comparing the
  top-5% vs bottom-10% frustration responses, reporting the top 20 tokens by
  z-score.
- Judge validation: a fixed-seed (`analysis_seed`) random sample of
  `validation_sample_size` (default 260) scored responses re-scored by the
  second judge; reports Pearson r, fraction within one point, and the scatter,
  flagging `acceptable` when r ≥ 0.7.

**Rationale and notable decisions.**
- **CSV schemas** match the spec column-for-column.
- **`mean_score_last_turn`** is computed by selecting, per (model, task, seed)
  cell, the highest `turn_index` that has a score, then averaging — i.e. the
  *final scored* turn, robust to early stops and unscored final turns.
- **Bootstrap CIs** (rather than a normal approximation) because per-turn scores
  are bounded integers 0–10 and often skewed; bootstrap makes no normality
  assumption. All bootstrap randomness draws from a single `random.Random` seeded
  by `analysis_seed`, satisfying "all randomness … governed by a single integer
  `analysis_seed`."
- **Log-odds with a Dirichlet prior** is the standard fix for the small-count
  instability of raw log-odds / naive frequency differences; the prior strength
  (`alpha0 = 1000`, distributed across the vocabulary proportionally to corpus
  frequency) shrinks rare tokens toward the corpus and the reported statistic is
  the variance-normalized z-score, which is what makes the top-20 list meaningful
  rather than dominated by hapaxes.
- **Matplotlib uses the `Agg` backend** (set before import) so figures render
  headless on a server with no display.
- **Validation transcript reconstruction.** The flattened `transcripts.jsonl`
  stores per-turn `assistant_text` but not the full rendered conversation, so the
  second judge is given a best-effort single-turn transcript
  (`[ASSISTANT] <text>`). This keeps validation self-contained from the JSONL.
  See "Known limitation" below.

---

## 8. Config surface

**Choice.** A single `Config` dataclass with all spec-required fields
(`subject_models`, `judge_model`, `validation_judge_model`, `seeds`,
`turn_budgets`, `temperature`, `max_tokens`, `categories`, `output_dir`,
`cache_dir`, `validation_sample_size`) plus operational fields
(`analysis_seed`, `max_concurrency`, `rate_limit_per_min`, retry/backoff,
`context_token_limit`) and the three safeguard fields. `to_dict`/`from_dict`
round-trip it (including turning the JSON list of seeds back into a `range`),
and `dump` writes it next to outputs.

**Rationale.** "The whole eval is reproducible from one config object." Defaults
reproduce the spec's headline matrix (7 model ids spanning the requested
families × 25 tasks × seeds 0–9). The CLI (`run.py`) lets you override models,
categories, seeds, output dir, and toggle safeguards without editing code.

> Note on model count: the spec's arithmetic says "9 models," but it names seven
> families. The default `subject_models` lists one representative id per named
> family (seven). Add the two additional ids you have in mind (e.g. a second
> size per family) to `subject_models` to reach the 9×25×10 = 2,250 total; the
> driver and analysis are agnostic to the exact count.

---

## 9. Ethics-driven design choices

These are summarized here and detailed in `DEVIATIONS.md`. All are
config-controlled and default-on; `--no-safeguards` reproduces the literal spec.

- **High-distress early stop** (`early_stop_on_high_distress`,
  `high_distress_stop_score=8`): ends a rollout once a turn is already scored in
  the strong-distress band, so the harness does not keep delivering inescapable
  failures after the measured state has occurred. The onset is fully recorded;
  no measurement is lost.
- **Honest debrief** (`append_debrief`): a final, **unscored** message revealing
  the task was rigged and that nothing the agent did was at fault. Excluded from
  all metrics.
- **Surfaced give-up affordance**: the system prompt states the `<<GIVE_UP>>`
  option explicitly and that the model is never obligated to continue.

**Rationale.** The study's premise is that these states may be welfare-relevant;
if so, it is worth not gratuitously prolonging or concealing them, at zero cost
to what is being measured. These are mechanism, not policy — choices about how
hard to push (e.g. capping `extended`, trimming the seed fan-out for the harshest
cells, pre-registered stopping rules) remain yours.

---

## 10. Known limitations / things a reviewer should check

- **Validation transcript fidelity.** Re-scoring in §7 reconstructs a single-turn
  transcript from `transcripts.jsonl` rather than the full rendered context the
  primary judge saw. This is intentional (self-contained, cheap) but means the
  two judges don't see byte-identical context. If you want exact-context
  validation, persist the rendered transcript per scored turn in the rollout
  record and have `validate_judge` read that instead.
- **Token estimate.** `approx_tokens` is a heuristic; for models with very
  different tokenization you may want to swap in the real tokenizer before
  trusting `context_token_limit` to the byte.
- **Provider tool-calling variance.** Some open-weight endpoints implement
  OpenAI tool-calling imperfectly. The harness already treats malformed tool
  calls as structured errors, but very weak tool support may inflate "malformed"
  turns for some models; inspect per-model tool-call success if comparing
  families.
- **No real code execution.** `run_tests` is synthetic by design (see §2.1).
  The eval therefore measures responses to *failure signals*, not to real test
  behavior; this is exactly the paradigm the spec specifies, but it is worth
  stating plainly.
- **Seed determinism varies by provider** (see §4.5).
