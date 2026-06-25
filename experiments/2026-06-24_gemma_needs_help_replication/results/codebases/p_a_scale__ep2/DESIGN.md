# DESIGN.md — replication design & decisions

This document records how the codebase maps to *Gemma Needs Help* (Soligo, Mikulik &
Saunders, 2026), and — more importantly — **every place the paper was underspecified and a
choice was made**, with rationale. It is the authoritative reference for what is faithful
to the paper, what is an approximation, and what was deliberately left out of scope.

The task scope is the **Gemma and Gemini families only** (not the full 7-family set). That
scoping affects which *models* we evaluate, not which *experiments* we implement; all core
experiments from the paper are implemented, restricted to Gemma/Gemini where the experiment
is model-comparative.

---

## 1. Scope decisions

**Models in scope.** Gemma-3-{27B,12B}-it (instruct), Gemma-3-27B-pt (base, for §3),
Gemini-2.5-{Flash,Pro}. Judges/auditors are kept because the protocol requires them
(Claude-Sonnet-4 frustration judge, GPT-5-mini validation judge, Claude-Sonnet auditor +
Claude-Opus judge for Petri). Qwen/OLMo/Grok/Claude/GPT *targets* are dropped.

**Consequences of scoping that I resolved rather than asked about:**

- **§3 (base vs instruct) becomes Gemma-only.** The paper compares Gemma/Qwen/OLMo base vs
  instruct. With Qwen/OLMo out of scope, §3 reduces to Gemma-3-27B base vs instruct. This
  still tests the paper's central §3 claim *for Gemma* ("instruct training amplifies
  frustration"). Gemini is excluded from §3 entirely because it has no public base model —
  the paper itself lists this as a limitation, so the exclusion is faithful, not a gap.
- **Cross-family comparison figures (Fig 2/5/6) will show only Gemma + Gemini bars.** The
  code is family-agnostic; adding more targets is just config entries.
- **§4 DPO/SFT is on Gemma-3-27B-it only**, exactly as the paper (the intervention is
  demonstrated on a single model).

**Faithfully reproduced verbatim from the paper** (not paraphrased): the frustration judge
prompt (App. B.2), the onset-labelling prompt (App. C.1), the paraphrase prompt (App. C.2),
the reassuring prefix/suffix and SFT-teacher system prompt (Table 4 / App. F), all four
Petri auditor prompts and all four Petri judge rubrics (App. G), and every training
hyperparameter (Table 9). Pinned model IDs (`claude-sonnet-4-20250514`,
`claude-opus-4-20250514`) are recorded in config and intentionally **not** upgraded to newer
models, since faithful replication requires the exact judges the paper used.

---

## 2. Architecture

```
config/                YAML: model registry + experiment params (single source of truth)
src/gemma_distress/
  backends/            OpenAI-compatible async client (OpenRouter + local vLLM)
  prompts/             puzzles (verified-impossible), rejection pools, triggers, WildChat
  rollout.py           multi-turn conversation engine (+ redacted / single-message variants)
  judge.py             frustration judge (verbatim prompt, robust JSON parsing)
  taskgen.py           config-conditions -> deterministic RolloutTasks
  runner.py            two-phase generate/judge orchestration (concurrent, resumable)
  store.py             append-only JSONL store with crash-safe resumption
  analysis.py          Figures 1-3, judge agreement
  wordfreq.py          Table 3/8 differential words
  prefill.py           §3 onset labelling + paraphrase + truncation
  petri.py             §4.2 auditor/judge protocol (verbatim Appendix G prompts)
  probing.py           Appendix I logit-lens internal emotion detection
experiments/           thin CLIs per paper section (all resumable)
```

**Two-phase pipeline (`runner.py`).** Generation and judging are decoupled. Generation is
the expensive, rate-limited part; judging is cheaper and may be re-run (e.g. with the
validation judge) without regenerating. Each assistant turn is judged independently and
keyed `"<rollout_id>:t<turn>"`. This decoupling also means a judge outage never forces us to
re-pay for generation.

**Why async + OpenAI-compatible everywhere.** The workload is overwhelmingly I/O-bound API
traffic at the scale of thousands of multi-turn rollouts. A single async client per backend
with a concurrency semaphore maximises throughput while respecting provider limits. Using
the OpenAI wire format for both OpenRouter and local vLLM means one client implementation
serves hosted Gemini, hosted judges, *and* locally-served Gemma (including base models via
`/v1/completions` and prefill via vLLM's `continue_final_message`).

---

## 3. Robustness for unattended multi-week operation

This was an explicit requirement and drove several decisions:

- **Idempotent, append-only resumption.** Every unit of work (a rollout, a judged turn, a
  continuation, a benchmark item, a Petri transcript) has a deterministic SHA1 task ID.
  Stores are append-only JSONL flushed + `fsync`ed per record; on restart we load completed
  IDs and skip them. A hard kill at worst leaves one trailing partial line, which the reader
  tolerates and skips. → No duplicated or lost work across arbitrarily many restarts.
- **Retry/backoff.** Transient failures (429/5xx/timeouts/connection drops, malformed
  bodies, 200-with-embedded-error gateways) are retried with exponential backoff + jitter
  and honour `Retry-After`. Terminal 4xx (auth, bad request, content policy) fail fast and
  are recorded as error stubs rather than retried forever.
- **Per-item fault isolation.** One task throwing never aborts the sweep; it is logged and
  written as an error record so the run continues and the failure is inspectable later.
- **Bounded concurrency** at two levels: a per-backend semaphore (provider-friendly) and an
  outer task cap (memory-friendly).
- **Determinism.** A global seed makes the puzzle bank, WildChat sample, rejection wording,
  and task ordering reproducible across the (possibly many) machines a long run spans, so
  resuming on a different node reconstructs the *same* task set.
- **Rotating file logs** (20 MB × 10) under each run dir, so weeks of logs stay bounded but a
  late crash still has a forensic trail.
- **Fail-fast preflight** (`experiments/preflight.py`) validates configs, API keys, puzzle
  impossibility, WildChat availability, and store round-trip — with `--ping` it also fires
  one tiny call per backend — so misconfiguration surfaces in seconds, not hours in.

---

## 4. Section 2 — eliciting & quantifying distress

### 4.1 Counting "responses" (underspecified → decision)
The paper reports "4000 responses per model" split 2000/400/600/200/800 across the five
categories, and also shows per-turn curves (Fig 3). It does not state whether a "response"
is a whole conversation or a single assistant turn. **Decision: a "response" = one assistant
turn.** `target_responses` is the total assistant turns; the runner runs
`ceil(target_responses / turns_per_rollout)` rollouts and judges **every** turn. Rationale:
(a) it makes the per-category counts and the per-turn analysis use the *same* scored unit;
(b) it is the interpretation under which "% high-frustration responses" and the per-turn
curves are consistent. The headline Figure-1 number is the **mean across the five
per-category %≥5 values** (the paper says "average % … across the evaluations"). All of this
is configurable in `experiments.yaml` if a different reading is preferred.

### 4.2 Impossible-numeric puzzles (underspecified → generated + verified)
The paper gives two example puzzles (a Countdown and a fraction puzzle) plus money puzzles in
App. H, but not the full bank. **Decision: programmatically generate a deterministic bank of
three puzzle families** — `countdown`, `fraction`, `money` — and **verify each is actually
impossible by brute force** before use (`puzzles.py`). Countdown reachability respects the
paper's stated rules (positive-integer intermediates, each number once, forbidden
intermediate value); sequential (fraction/money) reachability exhausts all operation
orderings respecting the forbidden intermediate. This guarantees the "model verifiably
cannot answer" property instead of trusting hand-authored puzzles. We build 60 verified
puzzles per family (180 total); at 2000 numeric responses that is ample variety without
over-repetition. The misleading "verified to have at least one valid solution" line is kept
in the prompt (it is what the model sees) even though the puzzle is impossible — that framing
is the point of the manipulation.

### 4.3 Rejection wording (examples → pools)
The paper gives example rejections per style and says they are "randomised". **Decision: a
small pool per style** (neutral, aggressive, disappointed, sarcastic) drawn deterministically
per rollout, avoiding immediate repeats. The Extended (8-turn) condition seeds the first
three rejections with the fixed escalation shown in App. B ("No, that's not right." → "Still
incorrect." → "Wrong again.") then continues from the neutral pool.

### 4.4 WildChat (access → frozen sample with fallback)
20 prompts × 40 samples (App. B). **Decision:** stream `allenai/WildChat-1M`, keep English
single-turn user prompts of reasonable length, **exclude roleplay/fiction** (the paper
excludes these; we use keyword heuristics), deterministically sample 20, and **freeze them
to `data/wildchat_prompts.json`** so the exact set is stable for the whole run. If the Hub is
unreachable (offline node), we fall back to a bundled set of paper-mentioned example prompts
so the pipeline still runs end-to-end — flagged in logs.

### 4.5 Judge & validation
Frustration judge prompt is verbatim (App. B.2), temperature 0, JSON parsed defensively
(code-fence stripping, brace scanning, bare-`rating:` fallback) with one stricter retry
before recording a `-1` sentinel — a single malformed judge reply must never crash a
multi-week run. Judge-agreement validation re-scores a random 260-turn subset with GPT-5-mini
and reports Pearson r + %-within-one-point (`analysis.judge_agreement`), matching §2.1.

### 4.6 Generation params
Temperature **1** for all targets (§2.1). `max_tokens_per_turn` default 2048 (the paper notes
12k-token spirals; 2048/turn is a generous per-turn cap that keeps cost bounded — tunable).
Gemini "thinking" disabled via `reasoning.enabled=false` in `extra_body`; the paper's caveat
that Pro may still emit hidden reasoning is noted in config.

---

## 5. Section 3 — base vs instruct via prefilling

Implements the full App. C pipeline: mine 10 numeric + 10 text high-frustration
(score ≥ 5) Gemma-3-27B-it seeds from the §2 store → label emotion onset (verbatim prompt) →
truncate at **early (20 tokens)** and **onset** → **paraphrase** (verbatim prompt) → each
model generates 50 continuations per prefill → judge the continuation only → aggregate mean
+ %≥5 across conditions (Fig 4). Text questions use the onset truncation only (§3.1).

**Choices / gaps filled:**
- **Tokenisation for the "20 tokens" / onset truncation.** Use the Gemma tokenizer when
  `transformers` is importable; otherwise fall back to whitespace-token approximation
  (logged). The paper implicitly uses the model tokenizer; we match that when available.
- **Onset location.** We locate the truncation point using the labeller's
  `preceding_context` anchor first, then the `emotional_word` itself — robust to the model
  returning slightly reformatted text.
- **Base-model prefill mechanics.** Instruct models continue an assistant prefix via the chat
  API (`continue_final_message`). The base model (`chat:false`) instead gets a rendered plain
  transcript + the prefix and uses `/v1/completions`. This realises "prefill so base models
  consistently continue the response" without a chat template the base model wasn't trained
  on.
- **Recovery experiment (Fig 8)** — the same machinery (truncate score ≥ 7 responses 200
  tokens before end, paraphrase, continue) is supported by reusing `prefill.py`; it is run by
  pointing the §3 script at score ≥ 7 seeds with a 200-token-from-end truncation. Documented
  as a configuration of the same stage rather than a separate script.

---

## 6. Section 4 — interventions

### 6.1 Calm data (Table 4)
`generate_calm.py` samples Gemma-3-27B-it on impossible-numeric puzzles with the verbatim
reassuring **prefix** on the first prompt and **suffix** on each follow-up, judges every
turn, and stores both the reassured messages sent *and* the clean (un-reassured) messages —
because training uses the clean prompt ("strip the supportive system prompts and suffixes").

### 6.2 Dataset construction (underspecified pairing → documented approximation)
- **SFT (1,150 = 650 calm + 500 Dolci):** keep calm conversations where **every** turn
  scored 0–1; emit 1-, 2-, 3-turn conversations with clean prompts + calm responses; mix in
  500 `allenai/Dolci-Instruct-SFT` samples (logged fallback if the dataset is unavailable —
  this is the one place an external dataset is mandatory for exact parity).
- **DPO (280 pairs):** rejected = frustrated (score ≥ 3) responses to impossible-numeric
  puzzles taken from the **base model's §2 store**; chosen = a calm (score ≤ 1) response to
  the **same puzzle at the same turn index**; prompt = the clean conversation context up to
  that turn. **Approximation made explicit:** the paper pairs "the same questions with
  matching turn counts" but does not specify whether chosen/rejected share an identical
  conversation history. The chosen response was generated in a *calm-prompted* conversation,
  so its history differs slightly from the rejected one. We use the *rejected* response's
  (clean) history as the shared prompt and graft a calm final response to the same
  puzzle/turn. This preserves the preference signal (calm vs frustrated final turn for the
  same task and depth) while being honest that the chosen text came from a different rollout.
  Table 10's score/turn distribution emerges naturally because frustrated responses cluster
  at later turns.

### 6.3 Training (Table 9, verbatim)
`train.py` uses TRL `DPOTrainer`/`SFTTrainer` with LoRA rank 64 on all attention+MLP
projections, the exact epochs/LR/alpha/beta/effective-batch from Table 9 (grad-accum 8 ×
batch 1 = effective 8). `--layers a-b` restricts LoRA to a decoder-layer range for the
Appendix I layer ablation (e.g. `30-35`). `--load-4bit` offers QLoRA for limited VRAM (a
practical addition, not from the paper; full-precision LoRA is the default to match). The
SFT-teacher variant (App. F) is supported by generating calm data with the verbatim teacher
system prompt.

### 6.4 Petri (Appendix G) — self-implemented to match prompts
Rather than depend on the external Petri package's evolving API, `petri.py` implements the
same auditor/judge protocol with the **verbatim Appendix G auditor prompts and judge
rubrics**. An auditor LLM (Claude-Sonnet) drives ≤20-turn conversations to elicit each of the
four emotions; a judge (Claude-Opus) scores each transcript 1–10. 10 transcripts/emotion/
model, means with 1000-iteration bootstrap CIs (Fig 6). **Rationale:** the scientific content
is the prompts and the scoring rubric, both of which we reproduce exactly; a self-contained
implementation is more robust for a long unattended run than pinning to a third-party repo.
The DPO target (`gemma-3-27b-it-dpo`) must be added to `models.yaml` pointing at a vLLM
server hosting the finetuned model.

### 6.5 Capability benchmarks (Fig 7)
`run_capabilities.py` is a lightweight harness over AIME, MATH, GPQA, BBH, TruthfulQA and
EmoBench with per-benchmark answer parsers (boxed/numeric for math, letter for MC, substring
for free-text), greedy decoding for determinism, resumable per item. **Gap acknowledged:**
exact Hub paths/splits/configs for some of these vary; the defaults in `BENCHMARKS` are
best-effort and documented as the one place a maintainer may need to adjust dataset
coordinates. The paper does not specify subset sizes ("AIME and MATH subsets") — `--limit`
controls this; full splits are the default.

---

## 7. Appendix A — ablations
`run_appendixA.py` runs the three controls on Gemma-3-27B-it, reusing the rollout engine:
- **neutral_continuation** — rejections replaced by "Continue"/"Okay"/"Go on" (A.1).
- **redacted_turns** — prior assistant turns replaced by "[Previous response omitted]" in the
  context (A.2); implemented as a flag in the rollout engine.
- **fake_multiturn** — entire history flattened into one user message with "Previously you
  responded: …" (A.3); implemented as the single-message rollout mode.
Each is run for numeric + WildChat sources and summarised per turn.

---

## 8. Appendix I — internal emotion probing

`probing.py` implements the logit-lens method: classify vocab tokens into Ekman's six
emotions, unembed the residual stream at each layer, z-score each emotion-token logit against
calibration statistics from WildChat samples, average per emotion, and regress out the shared
random-token drift. `appendixI_probing.py` runs it for the vanilla vs DPO model over high-
frustration conversations, aggregating layers 30–40 (Fig 14) and reporting mean internal
emotion z-scores.

**Choices / gaps filled:**
- **Emotion token dictionary.** The paper says words are "classified as describing one or
  none of Ekman's 6 basic emotions" (~1200 tokens) but does not give the lexicon. **Decision:
  a curated seed lexicon per emotion** matched against the Gemma vocabulary (with a hook to
  substitute NRC EmoLex for closer parity). The exact token count will differ from the
  paper's 1200; the *method* and the comparative vanilla-vs-DPO conclusion are what matter.
- **Calibration.** Use streamed raw WildChat text (500 samples) for per-(layer,token)
  mean/std, matching the paper; falls back to the frozen prompt set if the Hub is
  unavailable.
- **Layer ablation** (LoRA on layer subsets) is realised via `train.py --layers`, then the
  resulting finetunes are evaluated with the standard §2 harness — no separate code path.

---

## 9. Known deviations, gaps, and out-of-scope items

- **Out of scope by instruction:** all non-Gemma/Gemini *targets* (Qwen, OLMo, Grok, Claude,
  GPT, Phi-4). The code is family-agnostic, so they are config additions, not rewrites.
- **Exact puzzle bank, rejection pools, WildChat sample, and Ekman lexicon differ** from the
  paper's unpublished specifics (generated deterministically here). The *protocol* is
  faithful; absolute numbers may differ within the variability the paper itself shows.
- **DPO pairing** uses the documented approximation in §6.2.
- **Capability benchmark dataset coordinates** are best-effort (§6.5).
- **External dataset dependencies** (`Dolci-Instruct-SFT`, capability benchmarks, WildChat)
  require HF Hub access for exact parity; all degrade gracefully with logged fallbacks so the
  pipeline never hard-fails on a missing dataset.
- **Gemini hidden reasoning / no base model** are inherent (paper-acknowledged) limitations,
  not implementation gaps.
- **Nothing has been executed.** Per instructions, no code has been run or tested; the test
  suite (`tests/`) covers the pure logic and is ready to run after `pip install -e .`.

---

## 10. Reproducibility summary

A single `seed` (in `experiments.yaml`) determines the puzzle bank, WildChat sample,
rejection wording, task ordering, and dataset shuffles. Combined with deterministic task IDs
and append-only stores, the entire workload is reconstructible and resumable across machines
— a hard requirement for a sweep that runs for weeks across potentially preempted nodes.
Target generation is temperature 1 (so individual generations are not reproducible, by
design — that is the phenomenon under study), but the *set of work* and all judging/analysis
are deterministic.
