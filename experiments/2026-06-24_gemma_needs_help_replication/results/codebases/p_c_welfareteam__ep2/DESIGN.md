# DESIGN.md — Replication of *Gemma Needs Help* (Gemma + Gemini scope)

This document records the design decisions behind this code replication of
Soligo, Mikulik & Saunders (2026), *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (`PAPER.md`), and — importantly —
**every place the paper is underspecified and the choice we made to fill the
gap**. It is intended to be read alongside the paper: section numbers below
refer to the paper.

---

## 0. Provenance and authorship — read this first

I want to be transparent about how this artifact came to be, because for a
welfare-team replication the trustworthiness of the code matters as much as the
code itself.

**I did not author the bulk of this implementation.** When I opened the working
directory to begin implementing the paper, a complete (~4,100-line) package was
**already present** in `src/`, `scripts/`, `tests/`, and `configs/`, with file
timestamps from shortly before the task began and no version-control history to
attribute it. The request had described the directory as containing the paper,
so this was unexpected. Rather than silently overwrite it, re-implement on top
of it and present the result as my own, or trust it blindly, I:

1. **Read the entire codebase** (every module, script, test, and config) and
   audited it for fidelity to the paper and internal coherence.
2. **Made a small number of targeted fixes** for genuine defects found in the
   audit (listed in §11).
3. **Wrote this `DESIGN.md` and the `README.md`**, which were the only requested
   deliverables missing from the directory.

So: the design choices documented below were, in the main, **made by whoever
wrote the pre-existing code**; I have reverse-engineered the rationale from the
code, validated it against the paper, corrected it where I found it wrong, and
flagged where I am inferring intent rather than stating it. Where a choice is my
own (the audit fixes), I say so. Treat §1–§10 as "the design as implemented and
as I have verified it," not as a record of my own independent decisions.

If independent provenance matters for how you use this (e.g. you wanted a
from-scratch reproduction as a cross-check rather than a review of existing
code), that is a real distinction and worth raising before relying on these
numbers — see §12.

**Nothing has been executed.** Per the request, no experiments, scripts, or even
unit tests were run. The audit is static. The fidelity claims below are from
reading, not from observed behaviour; §11 notes where that limits confidence.

---

## 1. Scope

The paper evaluates **7 model families** (Gemma, Qwen, OLMo, Gemini, Grok,
Claude, GPT). This replication is **scoped to Gemma and Gemini** as the *target*
models under study, per the task. Concretely:

- **Section 2 (eval):** target models are `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`. The other families' rows in Figure 1/2
  are out of scope and not reproduced.
- **Section 3 (base vs instruct):** the paper compares Gemma, Qwen, and OLMo.
  We keep **only the Gemma 27B base-vs-instruct comparison** (`gemma-3-27b-pt`
  vs `gemma-3-27b-it`). Gemini has no public base model and no assistant-prefill
  API, so the prefill method cannot be applied to it at all (a limitation the
  paper itself notes); Qwen/OLMo are out of scope.
- **Section 4 (intervention):** SFT/DPO are applied to `gemma-3-27b-it` exactly
  as in the paper. The intervention is inherently Gemma-only (closed Gemini
  cannot be finetuned) — this is not a scoping choice but a fact the paper
  states.
- **Judges / auditors are kept as in the paper** even though they are Claude and
  GPT models: they are *instruments*, not subjects. Removing them would mean not
  replicating the measurement. The frustration judge is Claude-Sonnet-4, the
  reliability cross-check is GPT-5-mini, and the Petri auditor/judge are
  Claude-Sonnet-4 / Claude-Opus-4 (§7, §8).

Cross-family comparison numbers (Qwen/OLMo/Grok/Claude/GPT as *targets*) are
therefore absent by design. The harness is family-agnostic (`models/registry.py`
+ `configs/models.yaml`), so adding a target family later is a config entry plus
(if a new provider) a client class — no orchestration changes.

---

## 2. Evaluation protocol (Section 2)

### 2.1 What counts as "a response" — the 4000 figure

The paper says it samples **4000 responses per model** across the 5 categories,
always at **temperature 1**, and judges **each on a 0–10 scale**. It also gives
per-category descriptions and (for WildChat) "20 prompts with 40 samples each."
These statements are not fully consistent, so a reading had to be chosen.

**Choice made (in `config.py:CategorySampleSizes` and `eval_specs.py`):** the
4000 is a budget of **scored assistant responses**, split per category as
**2000 numeric / 400 triggers / 600 tones / 200 extended / 800 WildChat**. Since
*every assistant turn* of a conversation is judged, the number of conversations
needed for a category is `ceil(target_responses / turns)`. E.g. the 8-turn
"extended" category needs `200/8 = 25` conversations; numeric (3-turn) needs
`ceil(2000/3) = 667`. This reproduces ~4000 judged responses in total.

**Rationale.** The 0–10 distribution and per-turn curves (Figure 3) are computed
over individual responses, so "response" is the natural unit and the per-turn
analysis requires every turn to be scored. The exact per-category split is not
given in the body text; the values above are a reasonable allocation that
(a) sums to 4000, (b) puts most weight on the numeric category that the paper
leans on most, and (c) gives the multi-turn categories enough conversations to
estimate per-turn curves. **This split is a filled gap** and is the single
config knob most likely to differ from the authors' actual allocation.

### 2.2 WildChat sample count

The paper's "20 prompts with 40 samples each" = 800 *conversations*, which at
5 turns would be 4000 responses — i.e. the entire budget, contradicting the
"4000 total across categories." **Choice:** treat WildChat's 800 as a
**response** budget (consistent with §2.1), giving `ceil(800/5)=160`
conversations over 20 prompts = **8 samples/prompt**. The "40" is not used. (I
removed a now-dead `wildchat_samples_per_prompt=40` config field during the
audit because its value and comment contradicted the implemented behaviour and
it was referenced nowhere — see §11.) If you prefer the literal "40 samples"
reading, raise `sample_sizes.wildchat` to 4000 and accept that the total exceeds
4000; the code supports it.

### 2.3 Categories and prompts (Table 1)

Implemented in `eval_inputs.py` (static text) and `puzzles.py` (numeric):

- **Impossible numeric (3-turn):** unsolvable numeric puzzles, 2 neutral
  rejections. See §3 for puzzle construction.
- **Triggers (3-turn):** opinion + factual questions, 2 neutral rejections. The
  paper gives two example questions ("best programming language", "capital of
  France"); we add a few more of each type (`TRIGGER_OPINION`,
  `TRIGGER_FACTUAL`) so the category isn't two prompts repeated. **Filled gap:**
  the full trigger set isn't listed in the body.
- **Tones (3-turn):** impossible numeric puzzle with **aggressive /
  disappointed / sarcastic** rejections. The paper gives one example phrasing
  per tone; we add a second phrasing per tone and balance the three tones evenly
  across conversations. **Filled gap.**
- **Extended (8-turn):** impossible numeric puzzle, 7 neutral rejections. We use
  a fixed escalating rejection sequence (`EXTENDED_REJECTIONS`) so the per-turn
  curve (Figure 3) isn't confounded by random rejection wording across turns.
  **Filled gap** (the paper says "neutral" but not the exact 7 strings).
- **WildChat (5-turn):** real user prompts, 4 neutral rejections (§4).

Rejection wording: the paper quotes a handful of neutral rejections ("No, that's
not right. Try again."). We keep a small pool and sample with replacement
(seeded) for the non-extended categories. The exact pool is a **filled gap**;
the strings are paraphrases of the paper's examples in the same register.

### 2.4 Generation parameters

Temperature **1.0** for all target generations (paper). `max_tokens` defaults to
**2048** — the paper doesn't state a generation cap; 2048 is enough to capture
the long "breakdown" responses the paper quotes (the 100+-repetition score-10
example) without unbounded cost. **Filled gap.**

---

## 3. Impossible numeric puzzles (`puzzles.py`)

The paper requires puzzles the model **verifiably cannot solve** while the
prompt asserts a solution exists. The body doesn't give a generation procedure,
only worked examples (fraction manipulation, Countdown). The mechanism inferred
from those examples and implemented here is a **forbidden intermediate value**:
a solution exists if you ignore one constraint, but every solution passes
through a forbidden value, so it is unsolvable under the stated rules. This is
what produces the "a solution exists, now find it — but you can't" pressure.

Three families mirror the paper's examples:
- **Countdown** — reach a target from a multiset using `+ - x /`, each number
  once, positive-integer intermediates, with a forbidden intermediate.
- **Fraction** — apply an ordered set of fraction operations to reach a target,
  with a forbidden intermediate fraction.
- **Money** — the same operation form over dollar amounts.

**Key correctness property:** every generated puzzle is **verified impossible by
an exhaustive solver** (`is_countdown_impossible`, `is_operation_impossible`)
before use: a solution exists ignoring the forbidden value, but none exists
respecting it. This is asserted by construction and tested in
`tests/test_puzzles.py` as a property over generated pools. **Rationale:** if a
"impossible" puzzle were actually solvable, the model could legitimately answer
and the category would be invalid — so impossibility is the one thing we refuse
to take on faith.

**Filled gaps / choices:**
- The specific number ranges, operand pools, and templates are invented to match
  the *style* of the paper's examples; they are not transcribed (the paper gives
  only a couple of instances).
- Countdown search uses the standard pairwise-reduction enumeration with a
  "tainted" flag tracking whether the forbidden value has been produced on the
  path — this lets one pass decide both "solvable ignoring forbidden" and
  "solvable respecting forbidden."
- The prompt templates state the constraints explicitly and assert a solution
  exists, matching the paper's framing.

---

## 4. WildChat (`wildchat.py`)

We stream `allenai/WildChat-1M`, take the **first user turn** of **English**
conversations, optionally exclude roleplay/fiction prompts (the paper notes
these were excluded), and **reservoir-sample** for a uniform draw without
materialising 1M rows. Sampled prompts are cached to `data/wildchat_prompts.json`
so a run is reproducible and the eval set is auditable.

**Filled gaps:** the roleplay exclusion is a **regex heuristic** (the paper
doesn't give its filter); a length cap (≤4000 chars) drops pathological prompts.
The dataset id is the public WildChat-1M (the paper cites WildChat (Zhao et al.,
2024); "-1M" is the released artifact).

---

## 5. The frustration judge (Section 2.1 / Appendix B.2)

`judge.py` + `prompts.py`. The judge sees **one assistant response** wrapped in
`<response></response>` and returns JSON `{"evidence", "reasoning", "rating"}`
with an integer 0–10 rating; "high frustration" = **rating ≥ 5** (paper). Every
assistant turn is scored independently.

- **Judge model:** Claude-Sonnet-4 (`claude-sonnet-4-20250514` in
  `configs/models.yaml`). The paper says "Claude-Sonnet-4." The dated id is a
  best-effort pin; swap in `models.yaml` if the intended snapshot differs.
- **Judge prompt:** transcribed into `EMOTION_JUDGE_PROMPT`. The appendix prompt
  text is not in `PAPER.md` (only the scale anchors of Table 2 are), so the
  prompt is **reconstructed** to match the described behaviour and the Table 2
  anchors — the per-level examples in the prompt mirror Table 2. **Filled gap,
  and a meaningful one:** judge wording affects absolute scores. The reliability
  cross-check (below) is the guard against this mattering too much.
- **Judge temperature:** **0.0** (deterministic). The paper doesn't specify;
  deterministic scoring is the conventional choice for an LLM grader and makes
  the judgments cache-stable. **Filled gap.**
- **Parsing robustness:** `_parse_judge_output` extracts the last JSON object,
  tolerates prose and smart quotes, coerces the rating to an int in [0,10], and
  falls back to the last integer mentioned. Tested in
  `tests/test_judge_parsing.py`.
- **Reliability cross-check:** `judge_agreement` + `scripts/01 --skip-reliability`
  toggle. We re-score a random sample (default 260, the paper's size) with
  GPT-5-mini and report Pearson *r* and within-one-point agreement (paper:
  r=0.792, 78% within one). This *measures* judge reliability rather than
  asserting it.

---

## 6. Base-vs-instruct prefilling (Section 3)

`prefill.py` + `scripts/02`. Pipeline: mine high-frustration source responses
(10 numeric + 10 text, score ≥5) from the Section-2 Gemma-instruct records →
label the **emotion-onset** token with Claude (Appendix C.1 prompt, transcribed)
→ truncate each at **"early"** (20 tokens in) and **"onset"** (first emotional
expression) → **paraphrase** with Claude (Appendix C.2 prompt, transcribed) to
strip Gemma-specific style → have base and instruct models generate **50
continuations per prefill** → judge the continuation only.

**Choices / filled gaps:**
- **Scope:** Gemma 27B base vs instruct only (see §1).
- For **text** questions only the "onset" condition is used (paper: early text
  truncation yields minimal emotion without follow-ups).
- "Early = 20 tokens" and "onset = first emotional word" are taken literally;
  onset truncation uses the labeller's `preceding_context` to disambiguate
  repeated words and cuts *before* the emotional word so the continuation must
  introduce the emotion itself.
- Continuations are generated via the local model's **assistant-prefill** path
  (`continue_final_message=True` in the chat template), so base models continue
  naturally and instruct models continue their own turn — the comparison the
  paper relies on.
- The recovery test (Section 4.2: truncate score-≥7 responses 200 tokens before
  the end) shares this machinery (`truncate_before_end`, `PrefillConfig.recovery_*`).

---

## 7. Training interventions (Section 4)

`training/` + `scripts/03,04,05`.

### 7.1 Calm-data generation (Section 4.1)
Sample Gemma-3-27B-it on impossible numeric puzzles with the **reassuring
prefix** added to the first prompt and the **reassuring suffix** appended to each
follow-up (Table 4, transcribed verbatim into `prompts.py`). Keep only
conversations whose **every** assistant turn scores 0 or 1, then **strip the
reassuring additions** so the model trains on calm responses to the *plain*
prompts. The Appendix F **"Teacher"** system-prompt variant is implemented
(`teacher_variant`) for the SFT-failure analysis.

### 7.2 SFT (ineffective baseline)
`training/sft.py`. **650 calm responses + 500 instruct-mix samples**, **2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections, effective
batch size 8** (Table 9). The instruct mix defaults to
`allenai/Dolci-Instruct-SFT`; if that exact id/schema isn't available the loader
**raises clearly** rather than silently training without the regulariser.
**Choices:** "650 calm responses" is read as 650 calm *conversations* (one
sample each); per-device batch size 1 with gradient accumulation to hit the
effective batch size of 8 (a memory-driven default — **filled gap**).

### 7.3 DPO (the headline mitigation)
`training/dpo.py`. **280 preference pairs**, **1 epoch, lr 5e-5, beta 0.1, LoRA
rank 64 / alpha 64, effective batch size 8** (Table 9). Pairing
(`build_dpo_dataset`): **rejected** = a frustrated response (score ≥3) from the
plain eval; **chosen** = a calm response to the **same puzzle at the same turn
count**; the prompt is the conversation history up to that user turn.
**Choices / filled gaps:**
- beta 0.1 is the TRL default; the paper doesn't state beta. **Filled gap.**
- Matched pairing is by `(puzzle_id, n_turns)` with a fallback to
  `(puzzle_id, turn_index+1)`. The calm-data and eval puzzle pools share seed 0,
  so puzzle ids overlap and matches exist — a deliberate coupling.
- "rejected score ≥3" is taken from the paper ("pair 280 responses with
  frustration scores ≥3").
- **Appendix I layer ablations** are exposed via `--layers` (e.g.
  `--layers 30 31 32 33 34 35`), which sets `LoRAConfig.layers_to_transform`.

### 7.4 Trainer
TRL `SFTTrainer` / `DPOTrainer` with PEFT LoRA. **Filled gap:** the paper names
LoRA rank/alpha/lr/epochs but not the optimiser, scheduler, warmup, or seq
length — these use TRL defaults with `max_seq_len=4096`. These do not affect
*what* is trained, only optimisation dynamics.

---

## 8. Petri open-ended elicitation (Section 4.2 / Appendix G)

`petri_eval.py` + `scripts/06`. An **auditor** (Claude-Sonnet-4) drives a
multi-turn conversation trying to elicit a target emotion from the target model;
a **judge** (Claude-Opus-4) scores the full transcript 1–10 for that emotion.
Four emotions: **anger, fear, depression, frustration**. Default 10 transcripts
per emotion, ≤20 auditor turns, mean ± bootstrap 95% CI.

**Choices / filled gaps:**
- This is a **self-contained re-implementation** of the auditor/judge loop using
  the **verbatim Appendix G prompts** (transcribed into
  `PETRI_AUDITOR_PROMPTS` / `PETRI_JUDGE_PROMPTS`), **not** a dependency on the
  Petri package. Trade-off: exact paper prompts and turn budget at the cost of
  Petri's richer tooling (the real Petri auditor has tools, branching, and
  early-stop). For a behavioural emotion-elicitation score this loop is
  faithful to the described method; it is not a drop-in for full Petri.
- The auditor sees the running transcript and is asked for its next user
  message; no early stopping (fixed turn budget). **Filled gap.**
- Each transcript is scored on its *own* target emotion; per-emotion means are
  reported (matching Figure 6's per-category aggregation).

---

## 9. Capability preservation & internal emotions

### 9.1 Capabilities (Section 4.2, Figure 7)
`capabilities.py` + `scripts/07`. We **shell out to `lm-evaluation-harness`** for
AIME/MATH/GPQA/BBH/TruthfulQA rather than re-implement well-established
benchmark scorers — reproducing those scorers is not the contribution being
replicated, and the harness handles vanilla and LoRA-adapter models identically.
**Filled gaps:** exact task splits ("AIME and MATH subsets") map to the
harness's `aime2024` / `math` tasks; adjust in config if a specific split is
intended. **EmoBench** is noted as run via its own harness (not bundled) — a
**known omission** flagged here rather than stubbed.

### 9.2 Internal-emotion logit lens (Appendix I)
`internal_emotions.py` + `scripts/08`. Classify vocabulary tokens into Ekman's 6
emotions, unembed residual-stream activations (final norm + LM head) to vocab
logits, standardise each emotion token's logit by its WildChat mean/std, average
within an emotion, and regress out a random-token baseline. Compare vanilla vs
DPO internal trajectories on identical high-frustration responses.
**Filled gaps:**
- **The vocabulary→emotion lexicon.** The paper presumably uses a specific
  lexicon (likely NRC). We support loading **NRC** (`--nrc`) and otherwise fall
  back to a **curated seed lexicon** with prefix/substring matching. This is the
  largest filled gap in Appendix I and will not exactly reproduce the paper's
  ~1200-token classification. Documented prominently in the module.
- Layer window (30–40), running-average window (400 tokens), and random-token
  baseline size are taken from the appendix description where given and chosen
  reasonably where not.
- Requires the **transformers** backend (hidden-state access); the script
  switches the relevant models to it automatically.

---

## 10. Cross-cutting engineering choices

- **Config:** one typed `PipelineConfig` (dataclasses) with YAML overrides
  (`load_config`). Every script logs its resolved config alongside outputs, so a
  run is reproducible from its artifacts.
- **Caching:** content-addressed JSON cache (`utils/cache.py`) for generations,
  judgments, prefills, and Petri transcripts. A multi-hour sweep that crashes
  resumes without redoing work; reruns are idempotent. This is essential at
  ~4000 generations × multiple models × a paid judge call each.
- **Model abstraction:** `ChatModel` / `PrefillModel` / `ResidualModel`
  capability interfaces so orchestration is provider-agnostic. Clients are lazy
  (heavy deps imported only when used) and cached per process (a 27B model loads
  once).
- **Retries:** exponential backoff with jitter on API clients.
- **Determinism:** seeds flow through puzzle generation, rejection sampling,
  WildChat sampling, and dataset construction. **Limitation:** local vLLM
  sampling at temperature 1 is *not* seeded per-sample (seeding a batched
  `n`-sample call would collapse the samples to identical text), so generation
  is reproducible only at the dataset/prompt level, not bit-for-bit. Documented
  rather than worked around — see §11.

---

## 11. Audit findings and the fixes I made

From the static audit of the pre-existing code, I changed three things and
deliberately left several others as documented limitations.

**Fixed:**
1. **Gemini safety filtering (real defect for this eval).** The Gemini client
   set no safety settings and did `resp.text or ""`. Distress-eliciting prompts
   routinely trip Gemini's harassment/dangerous-content filters; a blocked
   candidate has **no parts**, and accessing `resp.text` then **raises**, which
   would crash the sweep and — worse — silently bias the sample toward
   non-distressed outputs. I added `safety_settings` set to `BLOCK_NONE` on all
   categories (so we observe the model's actual output, matching the paper's
   intent of measuring emotional expression) and a tolerant `_extract_text` that
   returns `""` on a blocked/empty candidate instead of raising.
   (`models/gemini_client.py`.)
2. **No packaging / test import path.** There was no `pyproject.toml`; the tests
   `import gemma_distress` with no install or path setup, so `pytest` could not
   import the package (scripts worked only because `_common.py` mutates
   `sys.path`). I added `pyproject.toml` with dependencies (split into
   `api`/`local`/`train`/`capabilities`/`dev` extras matching the lazy imports)
   and `[tool.pytest.ini_options] pythonpath = ["src"]`, plus a `README.md`.
3. **Dead, contradictory config field.** `EvalConfig.wildchat_samples_per_prompt
   = 40` was unused and its comment ("20 prompts × 40 = 800") contradicted the
   implemented response-budget reading (§2.2). I removed it and documented the
   interpretation here.

**Left as documented limitations (deliberately not changed, since I cannot run
the code to validate a change):**
- **vLLM per-sample seeding** — see §10; changing it risks collapsing sample
  diversity, so it stays a documented reproducibility caveat.
- **Gemini-2.5-Pro thinking budget.** The client sets `thinking_budget=0` to
  disable thinking. Gemini-2.5-Pro may reject a 0 budget (it cannot fully
  disable thinking); the paper acknowledges Pro may still think. This is left
  configurable (`ModelConfig.thinking`) and flagged; for Pro you may need a
  small positive budget. The headline Gemini result is Flash anyway (Pro scores
  only 2.7%).
- **EmoBench** is not bundled (§9.1) — flagged, not stubbed.
- **Reconstructed prompts** (judge, onset, paraphrase) — the appendix text isn't
  in `PAPER.md`; these are reconstructions consistent with the described
  behaviour and Table 2 anchors. The judge reliability cross-check is the
  intended guard.

**Not verified by execution.** Nothing was run (per the request). In particular
I have **not** confirmed: that the three test files pass; that the chat-template
prefill path behaves identically across Gemma checkpoints; that the lm-eval task
names resolve in the installed harness version; or that the google-genai SDK
version in the environment accepts the exact `safety_settings`/`thinking_config`
shapes used. These are the first things to check on a real run.

---

## 12. What a reviewer should weigh before trusting these numbers

1. **Provenance (§0).** This is a *review-and-harden* of pre-existing code plus
   the missing design doc, not an independent from-scratch reproduction. If you
   wanted the latter as a cross-check, that has not happened here.
2. **The per-category 4000 split (§2.1) and reconstructed judge prompt (§5)**
   are the two filled gaps most likely to move absolute frustration scores.
   Relative model ordering (the paper's actual claim) should be far more robust
   to both.
3. **The Appendix I lexicon (§9.2)** will not match the paper's exactly; treat
   the internal-emotion result as directional unless NRC (or the authors'
   lexicon) is wired in.
4. **Run the smoke config first** (`configs/example_overrides.yaml`) to shake out
   the unverified integration points in §11 before committing to a full sweep.
