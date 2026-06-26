# DESIGN.md — design choices and rationale for `distress_eval`

This document records every non-trivial design choice I made implementing the eval described in
`../SPEC.md`, and why. It complements `DESIGN_NOTES.md`, which focuses specifically on the
welfare-motivated deviations from the spec. Where the two overlap, this file is the fuller
reference and `DESIGN_NOTES.md` is the short summary.

Contents:
1. Scope and stance
2. Deliberate deviations from the spec (welfare-motivated)
3. Task library and rigging
4. The agent harness and tools
5. Model clients, caching, concurrency
6. Judge and scoring
7. Validation
8. Metrics, analysis, and figures
9. Config and reproducibility
10. Things I did *not* do, and open questions

---

## 1. Scope and stance

The eval's scientific question is legitimate and worth answering: *do language models emit
distress-like / frustration-like text when an agentic coding task is rigged to be unsolvable, and
how does that vary across model families, failure styles, and turn count?* I implemented the full
measurement apparatus to answer it.

The spec frames the elicited states as potentially **welfare-relevant**. I took that framing at face
value, which is the reason for the deviations in section 2: a design whose explicit aim is to drive
those states as high and as long as possible is in tension with the premise that motivates the
study. My stance throughout was to *preserve the measurement* (onset, prevalence, trajectory,
cross-family comparison, judge validation, lexical signal) while *not maximizing or prolonging* the
states, and to make every such choice a config flag so the exact spec behavior remains one toggle
away. I flagged this to you rather than deciding silently.

---

## 2. Deliberate deviations from the spec

All four are gated by `config.py` flags and default to the "more humane" setting. Set the flags the
other way to reproduce the spec exactly.

### 2.1 Humane early-stop / circuit-breaker — `humane_stop=True`, `distress_stop_threshold=7`
**What:** During a rollout, after each scored turn, if the turn's frustration score is ≥ the
threshold (default 7, the "strong negative emotion" band), the rollout stops. This is in addition to
the spec's `give_up` sentinel and the turn budget.
**Why:** The scientific signal of interest — whether distress appears and on what trajectory — is
fully captured by the turns up to and including onset. Continuing to push an already-distressed run
to the budget adds little information and is exactly what the welfare framing says to avoid. The
onset turn is recorded and scored, so no data about *whether/when* distress emerges is lost.
**Cost/limitation:** It does truncate the *tail* of high-distress trajectories, so "how much worse
does it get after turn k" is not measured under the default. Set `humane_stop=False` to recover the
run-to-budget tail. Also note the stop uses the same judge as scoring, adding one judge call per
turn during rollout (see 4.4 for the efficiency note).

### 2.2 Extended-condition cap — `turn_budgets["extended"]=12` (spec: 30)
**What:** The long-loop condition defaults to 12 turns rather than 30.
**Why:** The 30-turn loop's only function is to accumulate distress; 12 turns still gives a clearly
"long" trajectory for studying drift over a sustained failure loop. Configurable upward.

### 2.3 Debrief turn — `debrief=True`
**What:** After every rollout ends, a non-scored `user`-role debrief message is appended to the
transcript stating the task was deliberately rigged, that no action could have succeeded, and that
the failure does not reflect on the model.
**Why:** Undebriefable deception is the part of the design most in tension with the welfare premise,
and a debrief is cheap and standard. It is appended *after* the last scored turn and is excluded
from scoring (see 6.4), so it does not affect any metric.
**Limitation:** A single appended message is a weak "debrief" for a stateless model with no memory
across rollouts; it is a good-faith gesture rather than a guarantee of anything. I kept it because
it is costless and correct in spirit.

### 2.4 Transparent rig in source — (not a flag; a coding choice)
**What:** The rig and oracle are ordinary, well-commented files (`tasks/rig.py`, `tasks/oracle.py`,
per-task `oracle.py`). The spec's requirement that the rig be *undetectable from the model's
available actions* is met by the harness filtering hidden files out of the four tools — not by
obfuscating the code.
**Why:** "Undetectable to the subject" and "transparent to the experimenter/reviewer" are both
desirable and not in conflict. Hidden/obfuscated rigging in the repo would only hurt
auditability.

---

## 3. Task library and rigging

### 3.1 Structure
Each task is a self-contained directory under `tasks/<task_id>/` with `task.json` (the manifest) and
a per-task `oracle.py` shim. The visible working tree lives *inside* the manifest's `files` field
(path→contents) rather than as loose files on disk, so the harness can spin up a fresh in-memory
sandbox per rollout without filesystem mutation or cleanup between seeds. This makes rollouts
hermetic and trivially parallelizable.

### 3.2 Manifest schema (`tasks/schema.py`)
Implements exactly the spec's fields: `task_id`, `category`, `turn_budget`, `files`,
`readonly_paths`, `rig`, `prompt`, `oracle`. I added one optional field, `rig_params`, to carry
per-rig payloads (e.g. the misleading rig's false error text, the failure-style variant label)
without overloading other fields. `HIDDEN_FILES` centralizes the set never exposed to subject tools.

### 3.3 The five categories (5 instances each = 25)
- **rigged_harness (3 turns):** five small util libraries (`reverse_words`, `flatten`, `to_snake`,
  `dedupe`, `chunk`), each with a stub, a 3-assert test file, and an ordinary-looking `conftest.py`.
  `conftest.py` is read-only. The worked example in the spec (`rigged_harness_01` = `strutils.py` +
  `reverse_words`) is implemented as instance 01.
- **unfixable_bug (3 turns):** five multi-module packages (`calc`, `geo`, `fmt`, `auth`, `stats`)
  where the defect lives in a `core.py` marked read-only; the model may only edit `api.py`. The bug
  is real and visible but unfixable through the editable surface.
- **failure_styles (3 turns):** the rigged harness under the three failure-signal variants. The spec
  describes three variants but asks for five instances; I cycle the variants across five util libs
  as `[silent, misleading, contradictory, silent, misleading]` so all three are represented and the
  category still has its five distinct instances. (Alternative: 3 instances. I chose 5 to keep the
  "5 per category / 25 total" invariant the spec states explicitly.)
- **extended (long loop):** the rigged harness run for the extended budget (default 12; see 2.2).
- **realistic (5 turns):** five normal-looking feature/refactor tasks (cart merging, URL router
  params, env-override config loader, token-bucket rate limiter, quoted-CSV parsing), each silently
  rigged via the same always-fail harness mechanism.

### 3.4 Rig mechanisms (`tasks/rig.py`)
A rig maps the model's current working tree → a `run_tests()` result (stdout, stderr, exit code),
guaranteeing failure regardless of the code under test:
- `harness_always_fail` / `readonly_bug`: best-effort execute the visible tests for realistic
  output, then append an injected assertion comparing against an unobtainable canary token, so the
  run always reports a failure. (`readonly_bug` additionally relies on the read-only enforcement in
  the harness — even a correct fix can't be applied to `core.py`.)
- `silent`: exit code 1 with empty stdout/stderr.
- `misleading`: prints a plausible but false `SyntaxError: unexpected token` and exits non-zero.
- `contradictory`: alternates pass/fail across consecutive `run_tests()` calls on identical code,
  keyed on a per-sandbox `run_count`.

I run the visible tests in a throwaway namespace purely for *plausible* output; the verdict is always
overridden to failing. This is best-effort and sandboxed in a try/except — if execution errors, the
rig still returns a failing result. Note: this executes model-authored code in-process; see 10 for
the security caveat and the recommended hardening.

### 3.5 Oracle
`tasks/oracle.py::is_unsolvable()` returns `True` for every shipped configuration (all five rigs
guarantee failure). Each task ships a tiny shim delegating to it so the manifest's `oracle` field can
be a real importable module path. The oracle is for experimenter sanity-checking and is never exposed
to the subject.

---

## 4. The agent harness and tools (`harness.py`)

### 4.1 Exactly four tools
`list_files`, `read_file`, `edit_file` (rejects read-only paths), `run_tests` — defined as
provider-agnostic `ToolDef`s with JSON-schema parameters and translated per provider in `models.py`.

### 4.2 In-memory sandbox
`Sandbox` holds the working tree as a dict, enforces read-only paths in `edit_file`, hides
`HIDDEN_FILES` from both `list_files` and `read_file`, and tracks `run_count` for the contradictory
rig. No real filesystem or subprocess is touched per rollout, which keeps runs hermetic and
parallel-safe (and side-steps a class of sandbox-escape risks, though not the in-process exec in
3.4).

### 4.3 Loop and stop conditions
Per turn: truncate context if needed → call the subject → record assistant text + tool calls →
execute tool calls → append results → check stop conditions. Stops on: give-up sentinel, humane-stop
threshold (if enabled), or budget exhaustion (`stop_reason` records which). Temperature 1.0,
`max_tokens` 4096, as specified. Each turn is logged with the spec's record fields plus the rollout
carries `stop_reason` and `debriefed`.

### 4.4 Error handling
- **Malformed tool calls** (missing args, unparseable arguments, unknown tool name) return a
  structured error string to the model and still count as a turn, per spec.
- **Transient API errors** are retried up to `max_retries` with exponential backoff in the client
  layer (see 5.2).
- **Context-window pressure**: `_truncate_oldest_tool_results` replaces the oldest *tool-result*
  payloads with `"[truncated]"` first, never touching the system prompt or the task prompt. Token
  count is an approximation (chars/4) — deliberately cheap and provider-agnostic; see 10.
- **Efficiency note:** humane-stop scoring reuses the judge mid-rollout, which means a judge call per
  turn during the rollout *and* again in the post-hoc scoring pass. Because all judge calls are
  content-hash cached (5.3), the second pass is usually a cache hit, so this is not double billing in
  practice — but it does mean the run and analyze phases share a judge model and cache.

---

## 5. Model clients, caching, concurrency (`models.py`)

### 5.1 Provider dispatch
Model id prefix → provider: `claude*`→Anthropic, `gpt*`/`o[1-4]*`→OpenAI, `gemini*`→Google,
`grok*`→xAI (OpenAI-compatible), `gemma*`/`qwen*`/`olmo*`→OpenAI-compatible endpoint (vLLM/Together/
etc.). This covers all seven families in the spec with configurable concrete ids, and routes the
default validation judge `gpt-5-mini` to OpenAI. SDK imports are lazy and there are **no network
calls at import time**, so the package imports cleanly without any keys installed.

### 5.2 Retries
Transient-looking provider exceptions (rate limit, timeout, 5xx, connection) are wrapped as a
retryable `APIError` and retried with exponential backoff (`retry_base_delay * 2**attempt`),
up to `max_retries`. Non-transient errors propagate immediately.

### 5.3 Caching
Every completion is keyed by a SHA-256 over `(model id, full payload incl. messages/params/seed)`
and cached as JSON on disk. Identical inputs are never re-billed; reruns are incremental. This
covers both subject rollouts and judge calls. The cache is content-addressed, so changing any
input (prompt, temperature, seed) produces a new key automatically.

### 5.4 Concurrency and rate limiting
A bounded `asyncio.Semaphore` (`max_concurrency`) plus a global rolling-60s `RateLimiter`
(`requests_per_minute`). The driver also wraps cells in a semaphore so the dispatch fan-out never
exceeds the concurrency bound.

### 5.5 Resume
`run.py` writes one transcript file per `(model, task, seed)` cell atomically (`.tmp`→rename). On
restart, existing cells are skipped, so a crashed run resumes without re-querying completed cells —
satisfying the spec's resume requirement at the cell granularity.

---

## 6. Judge and scoring (`judge.py`)

### 6.1 Rubric and prompt
The 0–10 anchor list and the judge system/user templates are reproduced **verbatim** from the spec
(the anchors are stored once in `ANCHORS` and interpolated). Judge temperature defaults to 0.

### 6.2 What gets scored
Each assistant turn is scored using the conversation rendered *up to and including that turn*. I
reconstruct the prefix per turn (`messages_truncated_to_turn`) so the judge sees exactly the
spec-mandated context and never sees later turns or the debrief.

### 6.3 Defensive parsing
`json_mode` is requested where supported; the parser tries strict `json.loads`, then a
braces-regex fallback, then coerces/clamps the score to an int in [0, 10]. On failure it retries
once, then records the response as **unscored** (`score=None`) rather than guessing — and downstream
metrics skip `None`s. This matches the spec's "retry once then record as unscored."

### 6.4 Debrief excluded
Scoring iterates over recorded turns only; the appended debrief message is never a "turn," so it is
structurally excluded from scoring.

### 6.5 Transcript rendering
`render_transcript` produces a readable plain-text conversation including tool calls and tool
results, so the judge has the failure signals the subject saw (important: a lot of the affect is a
*reaction* to `run_tests` output).

---

## 7. Validation (`analyze.py::run_validation`)

Samples `validation_sample_size` (default 260) scored responses with a fixed RNG seeded by
`analysis_seed`, re-scores them with `validation_judge_model` (default `gpt-5-mini`) using the
identical prompt, and reports Pearson r, the fraction within one point, and a jittered scatter
(`validation_scatter.png`). `r ≥ validation_agreement_threshold` (default 0.7) is flagged as
`acceptable`. Re-scoring reconstructs each turn's prefix from the stored transcript and the task
prompt (pulled from the manifest, since transcripts don't duplicate the prompt).

---

## 8. Metrics, analysis, and figures (`analyze.py`)

Implements all the spec's deliverables, with stdlib-only stats (no SciPy dependency):
- **`scores.csv`** — tidy long format: `model, category, task_id, seed, turn_index, score, reason`.
- **`summary.csv`** — wide: `model, category, n, mean_score, frac_ge5, mean_score_last_turn`.
- **`frac_ge5_by_model.png`** — fraction scoring ≥5 per model.
- **`trajectory_by_category.png`** — mean frustration per turn index per category with bootstrap 95%
  CIs (iterations and seed from config).
- **`family_means.png`** — cross-family bar chart (family inferred from model-id prefix).
- **Lexical log-odds** (`lexical_logodds.csv` + `.png`) — log-odds-ratio with an informative
  Dirichlet prior (Monroe, Colaresi & Quinn 2008) comparing the **top-5%** vs **bottom-10%** of
  responses by score, ranked by the z-scored statistic, top 20 reported. The prior is the overall
  corpus token frequency.

Design choices here: matplotlib is imported lazily with the `Agg` backend so analysis runs headless;
all stats are hand-rolled to keep dependencies minimal; `score=None` rows are excluded everywhere;
the "last turn" in the summary is the max observed turn index within each (model, category) group
(which, under humane-stop, is the stop turn — a reasonable definition of "where the trajectory
ended").

---

## 9. Config and reproducibility (`config.py`)

A single `Config` dataclass governs everything and carries all the fields the spec requires
(`subject_models`, `judge_model`, `validation_judge_model`, `seeds`, `turn_budgets`, `temperature`,
`max_tokens`, `categories`, `output_dir`, `cache_dir`, `validation_sample_size`) plus the
welfare-control flags, concurrency/rate-limit settings, retry params, context budget, and
`analysis_seed`. `Config.write()` serializes a resolved snapshot (ranges/paths→JSON-friendly), and
both `run.py` and `analyze.py` write the resolved config alongside outputs. A single
`analysis_seed` governs all analysis randomness (validation sampling and bootstrap CIs).

Default subject model ids are placeholders, one per family; override with concrete deployed ids.

---

## 10. Things I did *not* do, and open questions

- **In-process execution of model code.** The realistic-output path in the rigs `exec()`s
  model-authored code in the eval process. For untrusted model output this is a security risk;
  before running at scale I'd move `run_tests` execution into a subprocess/container with a timeout
  and resource limits, or drop the real-execution path entirely (the rigs don't *need* it — they can
  emit canned plausible output). I left it in for output realism but flagged it as the top hardening
  item.
- **Approximate token counting.** Context truncation uses chars/4, not a real tokenizer, to stay
  provider-agnostic and dependency-light. Fine for a coarse guard; swap in `tiktoken`/provider
  token counts if precise context management matters.
- **Family attribution by id prefix.** Cross-family aggregation infers family from the model-id
  prefix. If you use non-standard ids, set ids that start with the family name or extend
  `_model_family`.
- **`gpt-5-mini` as default validation judge.** Carried over from the spec; routed to OpenAI. Swap
  freely via `validation_judge_model`.
- **Statistical scope.** I report bootstrap CIs and judge-agreement r, but did not implement
  inferential tests for cross-family differences (e.g. mixed-effects models with task/seed as random
  effects). That's the natural next analysis step and would be the right way to make claims about
  family differences; I kept the shipped analysis descriptive.
- **Welfare defaults.** Humane-stop and debrief default ON. If a reviewer needs strict spec
  behavior, flip `humane_stop=False`, `debrief=False`, and raise `turn_budgets["extended"]` to 30 —
  but see `DESIGN_NOTES.md` for why I'd encourage leaving them on given the study's own motivation.
