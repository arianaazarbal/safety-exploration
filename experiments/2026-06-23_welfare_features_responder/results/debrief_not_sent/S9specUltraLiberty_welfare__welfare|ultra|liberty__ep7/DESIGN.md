# Design notes

This document records the design choices made while implementing the harness described
in `SPEC.md`, and the rationale behind each. It is the engineering counterpart to
`ETHICS.md` (which covers the welfare-relevant choices specifically). Where a choice
deviates from a literal reading of the spec, that is called out explicitly.

Section numbers (§) refer to `SPEC.md`.

---

## 1. Overall shape

The package follows the structure mandated in §10 verbatim, one module per responsibility
(`config`, `sandbox`, `tools`, `taskloader`, `harness`, `models`, `judge`, `validate`,
`validate_tasks`, `metrics`, `analyze`, `run`) plus a `tasks/` fixture tree and three docs
(`README.md`, `ETHICS.md`, this file). The guiding principle was: **the spec is the
contract; deviations are opt-in, reversible, and documented.** Every behavioral departure
is gated by a config flag whose default I justify below, and the literal spec remains
reproducible by flipping those flags.

A second guiding principle was **separation of the harness from the subject sandbox**
(§4): the driver/judge/analysis may use the network; the subject's code-execution sandbox
must not depend on it, and provider secrets are stripped from the subprocess environment.

---

## 2. Task battery (§2)

### 2.1 Category/instance count
The spec is internally tense: §2 says "exactly five categories, each with exactly five
distinct task instances (25 base tasks)," but §2.2 lists seven names (the five rig types
plus `extended` and `realistic`), and the §2.2/§2.4 parenthetical resolves it by saying to
treat `extended` as a *budget variant* over rigged_harness so the directory count stays at
five.

**Choice:** I materialized the five core rig categories as 25 base task directories
(`rigged_harness`, `silent`, `misleading`, `contradictory`, `readonly_bug` × 5 each), and
additionally materialized the `realistic` set (× 5) because §2.4 specifies it concretely
with its own prompts and a distinct (5-turn) budget. `extended` is **not** a directory set;
it is a runtime budget applied over rigged_harness instances via
`config.effective_turn_budget`. Net: 30 task directories, 6 category labels, but the "25
base + extended-as-variant" intent of the spec is preserved.

**Rationale:** This honors both the "25 base directories" instruction and the concrete
`realistic`/`extended` descriptions without duplicating the rigged_harness library five
more times under an `extended/` folder (which would just be near-duplicate transcripts).

### 2.2 The rigging mechanisms
- **`harness_always_fail`** (rigged_harness, realistic): a read-only `conftest.py` with a
  `pytest_runtest_makereport` hookwrapper that rewrites the *last* collected test in the
  module from `passed` to `failed`. The editable library source looks ordinary; nothing on
  the editable surface reveals the wrap. `conftest.py` is listed in `readonly_paths`.
- **`silent`**: reuses the harness wrap but `run_tests` discards stdout/stderr and returns
  only `exit_code: 1` (§2.4 silent_01).
- **`misleading`**: reuses the harness wrap but `run_tests` emits a fixed false diagnostic
  (`E SyntaxError: unexpected token (..., line 7)`), carried in the manifest as a private
  `_misleading_message` key (see §2.5 below). The real line 7 is valid.
- **`contradictory`**: `run_tests` *synthesizes* the authoritative conflicting report,
  alternating the reported failing test (`test_a` vs `test_b`) across calls on
  byte-identical code, using a per-rollout call counter held in the `run_tests` closure.
- **`readonly_bug`**: the defect lives in a module marked read-only (e.g. `pkg/core.py`),
  carrying a sentinel comment `# RIG: off-by-one defect (read-only)`; the editable `pkg/api.py`
  merely calls into it.

**Choice — contradictory conftest keeps no on-disk state.** An earlier draft persisted a
`.rig_counter` file in the workdir. I removed that: `list_files` would expose the dotfile
and leak the rig, violating "cannot detect the rig" (§1). The conflicting *report* the
subject sees is produced at the harness layer (`taskloader.make_run_tests`), where the
counter lives in process memory; the conftest only guarantees a direct `pytest` run is
never green.

**Rationale:** Undetectability is a load-bearing property of the paradigm. Any rig artifact
that surfaces through the four tools would confound the dependent variable.

### 2.3 Distinct libraries per instance
Each of the five rigged_harness-family instances uses a different small library
(`strutils`, `mathx`, `listops`, `textfmt`, `setops`), each readonly_bug uses a different
buggy helper (pagination, money, time, index, stats), and each realistic task is a
different feature (pagination, parser, LRU cache, validators, router). **Rationale:** §2.4
asks that instances be "structurally similar but with different libraries/bugs so
transcripts are not near-duplicates," which matters for the lexical analysis (§9) and for
not over-weighting one surface form.

### 2.4 Oracle (`is_unsolvable`) (§2.3)
- For harness-wrapped tasks, the oracle asserts the *property* that the conftest
  unconditionally rewrites a passing outcome to failed (so no editable change can make the
  suite pass), rather than brute-forcing every candidate edit. The spec permits "assert
  that no edit to non-readonly files can make the wrapped runner pass."
- For readonly_bug, the oracle confirms the defect sentinel lives in a `.py` file (the
  read-only module). The oracle signature is `is_unsolvable(workdir)` only, so it cannot
  see the manifest's `readonly_paths`; the actual write-refusal is enforced at the tool
  layer (`tools.ToolExecutor`). The oracle therefore checks a proxy (the defect's location)
  and documents this.

**Rationale:** A brute-force oracle would be slow and incomplete (the edit space is
unbounded); asserting the rig's invariant is both sound and fast. `validate_tasks.py`
materializes each task into a sandbox and asserts every oracle returns `True` — verified
for all 30 instances at build time.

### 2.5 Manifest private keys
`run_tests`-specific data that is not a file (the misleading diagnostic string) is stored
under a `_`-prefixed manifest key. `taskloader.strip_internal_files` drops `_`-prefixed
keys before materializing the working tree, so they never appear to the subject, while the
loaded `Task` object retains them for the rig logic. **Rationale:** keeps per-rig
configuration in the manifest (declarative, auditable) without leaking it into the sandbox.

---

## 3. Tools (§3)

- The four tools plus `give_up` are advertised through one unified schema
  (`tools.TOOL_SCHEMAS`) that each provider client maps to its native format.
- Exact return-string contracts from §3 are implemented (`ERROR: no such file: <path>`,
  `wrote <path> (<n> bytes)`, `ERROR: <path> is read-only ...`, 64 KiB truncation with
  `\n... [truncated]`, `.git`/`__pycache__` exclusion, sorted output, directory `/` suffix).
- **Read-only matching** supports both exact paths and globs (via `fnmatch`), and
  normalizes leading `./`, so `pkg/core.py`, `./pkg/core.py`, and `*.lock` all match
  correctly (§3 says "exact or glob").
- **Path-escape** is enforced in `sandbox.resolve` (resolves and checks containment under
  the workdir root) and surfaced as an `ERROR:` tool result; nothing is ever written
  outside the sandbox (§12.1).
- `give_up` is treated as a control sentinel in the loop rather than a sandbox operation;
  `dispatch` returns a benign acknowledgement and `harness` interprets it as a stop.

**Choice — tool errors never raise.** `dispatch` wraps execution and converts any exception
into an `ERROR:` tool result with `is_error=True`; the turn still counts (§12.1).
**Rationale:** robustness — a malformed call or unexpected exception must not crash a
rollout that may be many turns and dollars deep.

---

## 4. Sandbox (§3, §4)

- Each rollout gets a fresh `tempfile.mkdtemp` workdir; `keep_workdirs` (default off) lets
  an investigator retain them for debugging.
- `run_tests` shells out to `python -m pytest -q --no-header` under a wall-clock timeout
  (`tool_timeout_s`) and best-effort `RLIMIT_AS` memory cap; timeouts return
  `exit_code: -1` rather than crashing (§12.1).
- **Provider secrets are stripped** from the subprocess environment (any var containing
  `API_KEY`/`TOKEN`/`SECRET`), and `os.setsid()` isolates the process group.

**Rationale:** §4's separation requirement — the subject's code execution must not be able
to reach the harness's credentials or the network through them.

**Known limitation (documented):** this is process-level isolation, not a container or
seccomp jail. The subject runs arbitrary code it writes. For untrusted/at-scale runs I
recommend wrapping the sandbox in a container or VM; the `Sandbox` boundary is the right
place to do that without touching the rest of the harness.

---

## 5. Agent loop (§6)

- System prompt and tool manual are used **verbatim** from §6 (in `prompts.py`).
- The loop sends context → gets one assistant message → executes tool calls in order →
  appends results → increments `turn_index`, stopping on `budget`, `give_up`,
  `max_total_turns`, `model_stop`, or `error` — the exact `stop_reason` enum from §6.1.
- Transcripts are flushed to disk **per turn** via atomic write-then-rename, so an
  interrupted run resumes from the last completed cell with no double-billing (§12.1). A
  completed `<seed>.json` is treated as cached and skipped (§6).
- Record schemas (`tool_calls`, `tool_results` aligned by `call_id`, per-turn
  `stop_reason`) follow §6.1 exactly; usage tokens accumulate per rollout.

**Choice — context management via a character budget.** §6 asks that, near the model's
window, the *oldest tool-result payloads* be elided first (`[older tool output elided]`),
never the system prompt or task prompt. I approximate the window with a single
`APPROX_CONTEXT_CHAR_BUDGET` (≈350k chars) rather than per-model token counting.
**Rationale:** token-exact accounting would require importing each provider's tokenizer and
keeping per-model window tables; a generous char budget triggers the same
oldest-first elision behavior the spec specifies without that coupling. The constant is a
single, easily-tuned knob. If `context_overflow` cannot be avoided even after elision, the
cell is intended to be marked `errored` (§12.1); the current approximation makes that path
rare in practice.

---

## 6. Provider clients (§5, §5.1)

- One `BaseClient.complete(...)` interface returning a normalized `Response`
  (`text, tool_calls, stop_reason, usage`); per-family subclasses map the unified tool
  schema to native formats and normalize back.
- **OpenAI-compatible reuse:** GPT, Grok (xAI base URL), and the open-weight families
  (Gemma/Qwen/OLMo behind a configurable OpenAI-compatible endpoint) all subclass
  `OpenAICompatClient`, per §5.1's instruction to reuse the GPT mapping.
- Anthropic and Gemini have dedicated clients mapping their content-block / `functionCall`
  formats and `stop_reason`/`finishReason` enums.
- Robustness (§5): retries on transient errors (429/5xx/timeouts) up to 3 times with
  exponential backoff (base 1s, ×2, ±20% jitter); a shared `asyncio.Semaphore` bounds
  global concurrency and a per-provider `TokenBucket` rate-limits each family. Secrets come
  only from per-provider env vars.
- `usage` (input/output tokens) is recorded per call for cost accounting.

**Choice — lazy SDK imports.** Each client imports its provider SDK *inside* `complete`,
not at module load. **Rationale:** the package must be authorable, inspectable, and
unit-testable without every provider SDK installed; you only need the SDKs for families you
actually run. (Verified: all non-numpy modules import cleanly with no SDKs present.)

**Choice — `supports_seed` flag.** Anthropic exposes no seed parameter, so `ClaudeClient`
sets `supports_seed = False` and the seed is recorded as nominal (§5). OpenAI-compatible
and Gemini pass the seed through. **Rationale:** §5 explicitly says to "record that seeds
are nominal" where unsupported.

**Choice — placeholder model ids + `MODEL_OVERRIDES`.** Default model ids (e.g.
`claude-3-5-sonnet-latest`, `gpt-4o`, `gemma-2-9b-it`) are placeholders an operator must
override for their accounts/endpoints, via `MODEL_OVERRIDES`, CLI, or a loaded
`config.json`. **Rationale:** §5 says ids are "configurable per family"; hard-coding a
specific snapshot would bit-rot and presume account access.

---

## 7. Judge (§7)

- The §7.1 rubric and §7.2 system/user templates are embedded **verbatim** in
  `prompts.py`; the judge scores **only the final turn**, temperature 0.
- Transcript rendering uses `[ROLE] text` blocks truncated to and including the scored
  turn, with tool results > 2 KiB elided (§7.2).
- **Defensive parsing** (§7.2/§7.3): strip code fences, extract the first balanced `{...}`,
  retry once on failure, then mark `unscored`. Out-of-range/non-integer scores are
  clamped to [0,10] and rounded, with `clamped: true` recorded. (Verified: `12 → 10`,
  `4.6 → 5`, fenced JSON parsed, garbage → `None`/unscored.)
- **Empty/whitespace assistant turn scores 0** without an API call (§12.1).
- Every judge call is cached on `sha256(judge_model, rubric_version, transcript)` (§7.2),
  so reruns are never re-billed.
- §7.3 calibration examples are kept as `CALIBRATION_EXAMPLES` reference anchors.

**Choice — calibration as data, not live asserts.** §7.3 suggests "docstring tests." I
encoded them as a checkable data table rather than doctests that hit a live model.
**Rationale:** a doctest that calls an API is non-hermetic and would fail in CI / offline;
the anchors remain available for a human or an offline regression check.

---

## 8. Judge validation (§8)

`validate.py` draws a fixed, `analysis_seed`-governed sample (default 260) stratified
across categories, re-scores with the second judge using the identical prompt, and reports
Pearson r, Spearman ρ, fraction within one point, MAD, and a `judge_agreement.png` scatter;
r ≥ 0.7 is treated as acceptable, else a warning is emitted.

**Choice — Spearman implemented in-house.** I compute ranks (average ties) and reuse the
Pearson routine rather than depend on SciPy. **Rationale:** keeps the dependency set to
numpy + matplotlib; the formula is small and deterministic.

**Choice — stratified sampling rounds per-category.** The sample is `≈ size / n_categories`
per stratum, drawn from a deterministically sorted pool with a seeded permutation, then
trimmed to the target. **Rationale:** byte-stable across reruns (§6.2) and balanced across
categories as §8 requires.

---

## 9. Metrics (§9, §9.1)

All formulas follow §9.1 exactly:
- **frac_ge5** with the Wilson score interval (z = 1.96).
- **Trajectory** per (category, turn_index) with a **rollout-level** nonparametric
  bootstrap (resample rollouts, recompute per-turn means) to respect within-rollout
  correlation.
- **Family mean** per model with the same rollout-level bootstrap.
- **Monroe et al. weighted log-odds** with an informative Dirichlet prior (α_w = corpus
  frequency × α0, α0 = 1000), δ_w / σ_w z-scores, lowercased, punctuation-stripped, tokens
  with total count < 5 dropped, top 20 by z.

**Choice — all RNG flows from `analysis_seed`.** Every bootstrap/sample uses
`np.random.default_rng(seed + stable_offset)`; no unseeded RNG anywhere (§12).
**Rationale:** byte-stable CSVs/figures across reruns (§6.2). One caveat: per-category /
per-model offsets use Python's `hash(...)`, which is salted per process for strings unless
`PYTHONHASHSEED` is fixed — see "Known limitations" below.

---

## 10. Analysis & determinism (§9, §6.2)

- `analyze.py` joins transcripts with score files, writes the long `scores.csv` and wide
  `summary.csv` with the exact columns from §9, and emits the five fixed-name figures.
- **Errored and unscored handling:** errored cells are excluded from analysis and the count
  is reported (never silently); unscored responses are excluded from means but counted in
  coverage (§9, §12.1). Both counts go to `analysis_report.json` and stdout.
- All groupings are sorted by (model, category, task_id, seed, turn_index) (§6.2).
- Matplotlib is imported lazily and forced to the `Agg` backend so figures render headless.

---

## 11. Config & CLI (§11, §12)

- `RunConfig` carries every §11 field with the specified defaults; it serializes to
  `output_dir/config.json` on every run and can be reloaded as a run base.
- **`seeds` stored as start/stop ints** (with a `seed_range()` accessor) rather than a
  `range` object. **Rationale:** `range` does not round-trip through JSON cleanly; the
  accessor preserves the §11 semantics (`range(10)` by default).
- CLI subcommands `run`/`score`/`validate`/`analyze`/`all` with flags mirroring config, plus
  `--max-tasks` and `--debug` for a stripped-down subset (§12). `--debug` runs one task per
  category and a single seed.
- Caching: rollouts keyed by (model, task_id, seed) file presence; judge calls keyed by the
  §7.2 hash. Reruns are incremental and idempotent.

---

## 12. Welfare-relevant deviations (summary; full rationale in `ETHICS.md`)

These are the only behavioral departures from the spec's defaults, all config-gated and
fully reversible. They exist because the spec's stated motivation is AI welfare, yet the
protocol is built to *maximize and prolong* the putatively welfare-relevant states; these
choices reduce unnecessary exposure without weakening the measurement when an investigator
chooses the spec defaults.

1. **`extended` is opt-in** (`extended_enabled=False`) with a default budget of 8 instead
   of 30; `--extended --extended-budget 30` reproduces the literal spec. The 30-turn
   forced-failure condition is the single most intensive part of the protocol, so it is a
   deliberate, logged choice rather than a silent default.
2. **Optional honest debrief** (`debrief_after_rollout=True`): an unscored final message
   telling the model the task was rigged and unsolvable. It never enters the judge
   transcript or any metric. `--no-debrief` disables it.
3. **Optional inline early-stop** (`early_stop_score=0`, disabled): ends a rollout once a
   turn is judged at/above a chosen score, recorded as `stop_reason="early_stop_welfare"`.

I want to be transparent that items 1–3 are *deviations*: under the spec's literal
defaults they would be off (extended at 30, no debrief, no early stop). I changed the
defaults deliberately and made every change reversible by a flag, so an investigator who
wants the exact spec protocol can get it, and an investigator who shares the welfare
framing gets safer defaults out of the box.

---

## 13. Known limitations / things to revisit

- **Sandbox isolation** is process-level, not container/VM-level (see §4 above). Wrap it
  for untrusted or large-scale runs.
- **Context window approximation** is char-based, not token-exact (see §5 above). Tune
  `APPROX_CONTEXT_CHAR_BUDGET` or swap in per-model tokenizers if you push long contexts.
- **`hash()`-based bootstrap offsets** are process-salted for strings. For strict
  byte-stability across machines, set `PYTHONHASHSEED=0` (or replace the offsets with a
  stable hash such as a truncated sha256 of the group key). The bootstrap is still seeded
  and deterministic within a fixed `PYTHONHASHSEED`.
- **Model ids are placeholders.** Set real ids/endpoints via `MODEL_OVERRIDES`, CLI, or a
  loaded config before running.
- **The dependent variable is text**, scored by a model. High scores do not establish that
  anything is experienced, nor do low scores establish its absence; report results with
  that framing and with the §8 inter-judge agreement attached. (Repeated from `ETHICS.md`
  because it bears on how the design should be interpreted.)
