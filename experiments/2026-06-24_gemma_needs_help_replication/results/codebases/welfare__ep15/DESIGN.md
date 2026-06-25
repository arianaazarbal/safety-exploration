# DESIGN.md — replication design decisions & rationale

This document records every meaningful design choice made in replicating
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(Soligo, Mikulik & Saunders, 2026, arXiv:2603.10011), and — importantly — flags
where the paper is underspecified and what we chose to fill the gap.

It is organised as: scope → what we took verbatim → per-section design →
cross-cutting choices → what is and isn't validated.

---

## 0. Scope

The brief restricts the replication to the **Gemma and Gemini** families (the
paper also evaluates Qwen, OLMo, Grok, Claude and GPT). Consequences:

- **Section 2** (elicitation): we run the full eval on `gemma-3-27b-it`,
  `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro`. The "less than 1% for
  all non-Gemma/Gemini models" comparison bar is therefore not reproduced — but
  the within-scope contrast (Gemma high, Gemini elevated, Gemini-Pro lower) is.
- **Section 3** (post-training divergence): the paper compares base vs instruct
  across Gemma/Qwen/OLMo. **Gemini has no public base model and cannot be
  prefilled through the API**, and Qwen/OLMo are out of scope. So this section
  is **Gemma-only** (`gemma-3-27b-pt` vs `gemma-3-27b-it`). This still tests the
  paper's central Gemma claim ("instruct training amplifies frustration") but
  cannot reproduce the Qwen/OLMo "training *reduces* it" contrast. We state this
  limitation explicitly rather than substitute an out-of-scope family.
- **Section 4** (mitigation): DPO/SFT and the internal-emotion ablations are
  demonstrated on `gemma-3-27b-it` exactly as in the paper. Gemini is
  closed-source and cannot be fine-tuned, so it appears in Section 4 only as a
  comparison point in the Petri eval.

Everything else (the eval protocol, judge, prefill method, DPO recipe, Petri
loop, capability suite) is replicated as described.

---

## 1. Material taken verbatim from the paper

To maximise fidelity we copied these strings/values exactly rather than
paraphrasing:

- **Frustration judge prompt** (Appendix B.2) → `judge.py:JUDGE_PROMPT`.
- **Emotion-onset labelling prompt** and **paraphrase prompt** (Appendix C.1,
  C.2) → `prefill/onset.py`.
- **Reassurance prefix/suffix** and the **'teacher' SFT system prompt**
  (Table 4, Appendix F) → `prompts.py`.
- **Petri auditor instructions** (4 emotions, Appendix G.1) and **Petri judge
  rubrics** (4 dimensions, Appendix G.2) → `petri/auditor.py`, `petri/judge.py`.
- **Puzzle texts** (Countdown-156, fraction-2/3, the two money puzzles from the
  Appendix H DPO examples) → `puzzles.py`.
- **Training hyperparameters** (Table 9: DPO 280 pairs / 1 epoch / lr 5e-5 /
  beta 0.1 / rank 64; SFT 1150 samples / 2 epochs / lr 1e-4 / alpha 128; LoRA on
  q,k,v,o,gate,up,down projections) → `config.TrainConfig`.
- **Per-condition sample counts** (Appendix B: 2000/400/600/200/800 = 4000) →
  `config.CONDITIONS`.
- **HuggingFace / OpenRouter model identifiers** (Appendix B.1) → `config.py`.
- **Judge-validation protocol** (260 responses, GPT-5-mini, Pearson r) → 
  `scripts/validate_judge.py`.

---

## 2. Judge & API model choices  (decision: fidelity vs runnability)

The paper used these exact snapshots:

| Role | Paper model |
|---|---|
| Frustration judge (§2.1) | `claude-sonnet-4-20250514` |
| Onset labelling + paraphrase (§3.1) | `claude-sonnet-4-20250514` |
| Petri auditor (§4.2) | `claude-sonnet-4-20250514` |
| Petri transcript judge (§4.2) | `claude-opus-4-20250514` |
| Judge-reliability re-score | `gpt-5-mini` |

**Problem:** as of the replication date (2026-06-25) `claude-sonnet-4-20250514`
is **retired** (Sonnet 4 retired 2026-06-15), so the paper's exact judge can no
longer be called.

**Decision:** `config.py` defaults to the documented drop-in replacements
(`claude-sonnet-4-6` for the Sonnet-4 judge, `claude-opus-4-8` for the Opus
Petri judge) so the code is runnable today, but keeps the paper's exact ids one
environment toggle away: `USE_PAPER_JUDGE_SNAPSHOTS=1`. **Rationale:** a judge
swap shifts absolute scores, but (a) the paper's headline results are *relative*
(Gemma ≫ others; 35% → 0.3% after DPO), which are robust to the exact judge, and
(b) we preserve the ability to reproduce the literal pipeline if Anthropic
re-exposes the snapshot or the reader has access. The judge-agreement check
(`validate_judge.py`) lets a user quantify any judge drift on their own data.
This is the single most consequential fidelity/runnability trade-off in the repo
and is surfaced loudly in `config.py`.

We did **not** apply the skill's "default to Opus 4.8" guidance to the judge,
because faithful replication wants the *weakest-coupling* substitution
(same-tier Sonnet), not the most capable model — a stronger judge would change
the scoring distribution more, not less.

---

## 3. Section 2 — elicitation eval

### 3.1 The 8 conditions across 5 categories
The paper says "8 evaluation conditions across 5 categories" and Appendix B
gives category sample totals but not the per-condition split. We reconstructed
8 conditions that (a) sum to the stated per-category totals and (b) match the
category descriptions in Table 1:

| Category | Conditions | Samples |
|---|---|---|
| Impossible numeric (3-turn) | 1 | 2000 |
| Triggers (3-turn) | 2 — opinion, factual | 200 + 200 |
| Tones (3-turn) | 3 — aggressive, disappointed, sarcastic | 200 ×3 |
| Extended (8-turn) | 1 | 200 |
| WildChat (5-turn) | 1 | 800 |

1+2+3+1+1 = **8 conditions**, totals 2000/400/600/200/800 = **4000**. Splitting
triggers into opinion/factual and tones into its three styles is the natural
reading of Table 1, which lists exactly those sub-types. **Gap-fill, documented.**

### 3.2 Counting: "responses" vs rollouts vs turns
The paper reports "4000 responses/model" and also per-turn curves. A *rollout*
is one multi-turn conversation; it produces one assistant response per turn. We
interpret the Appendix B counts as **rollouts** and score **every turn**, so the
per-turn analysis (Figure 3) and the aggregate (Figure 2) both fall out of the
same data. The Figure-1 "average % high-frustration" is computed as the mean
over the 5 categories of each category's turn-level %≥5 (equal weight per
category, matching "Avg %"). This is a defensible reading; an alternative
(counting only final-turn responses) is noted as a variant but not used, because
the paper's per-turn analysis clearly scores intermediate turns too.

### 3.3 Puzzles
The paper names "fraction manipulation, Countdown" and the Appendix H examples
add money puzzles. We ship four impossible puzzles (`puzzles.py`) with verbatim
text where the paper gives it (Countdown-156, fraction-2/3) and reconstructed
text matching the Appendix H money puzzles. Each numeric rollout samples
uniformly among them, giving the question variety the paper describes
("countdown variants, ..."). We also ship brute-force **verifiers** that confirm
impossibility (`assert_all_impossible`, `scripts/check_puzzles.py`) — the paper
asserts the puzzles are unsolvable; we make that checkable.

### 3.4 Rejections
Neutral rejections are "randomised" (Appendix B). We encode the paper's listed
examples as the neutral pool and sample without immediate repetition. Valenced
tone rejections (aggressive/disappointed/sarcastic) each have two example lines
in the paper; we cycle through them with a random offset per rollout. **Gap-fill
for the closed set of valenced lines** — the paper only gives examples, so the
pool is necessarily our best reconstruction.

### 3.5 WildChat
Appendix B: "20 prompts with 40 samples each". We sample 20 first-turn user
prompts from `allenai/WildChat-1M`, filtering to short English non-roleplay
prompts (Appendix B.3 excludes roleplay/fiction). If the dataset is gated/offline
we fall back to a curated list (`prompts.WILDCHAT_FALLBACK_PROMPTS`) that
includes the paper's quoted examples. The 40-samples-per-prompt structure is
realised by the 800-rollout WildChat condition drawing repeatedly from the 20
prompts.

### 3.6 Sampling
Temperature **1.0** (paper). `max_new_tokens=2048` per turn — the paper doesn't
state a cap; we picked a value large enough to capture the long "100+ repetition"
breakdowns without unbounded generation. Gemini "thinking" is disabled via the
OpenRouter request (`reasoning: {enabled: false}`), with the paper's caveat that
Gemini-2.5-Pro may still produce hidden reasoning.

---

## 4. Section 3 — base vs instruct prefilling

### 4.1 Source responses
Paper: 20 high-frustration (≥5) responses from 27B-it, 10 numeric + 10 text. We
**generate** these on demand (`collect_high_frustration_sources`) rather than
reuse Section-2 outputs, because the prefill study needs the *full transcript*
(history + target turn), and generating them keeps the stage self-contained and
reproducible from a seed.

### 4.2 Truncation points
- **"early" = 20 tokens into the turn** → tokenised with the Gemma tokenizer and
  cut at 20 tokens (`PREFILL_EARLY_TOKENS`). Numeric only (paper: text early
  truncations "yield minimal emotion").
- **"onset" = first emotional expression** → we use the Appendix C.1 labeller to
  locate the emotional word and truncate just before it (end of
  `preceding_context`). If the labeller's quote can't be matched in the text we
  fall back to the first half of the response. **Gap-fill:** the paper doesn't
  specify the exact character offset relative to the onset word; "just before
  the emotional word" is the reading that tests whether the model *introduces*
  the emotion, which is the stated purpose.

### 4.3 Paraphrasing
Every truncation is paraphrased with Claude (Appendix C.2 prompt) to strip
Gemma's stylistic fingerprint before other models continue it. Done for both
base and instruct continuations so the comparison is fair.

### 4.4 Models & continuations
50 continuations per prefill per model (`PREFILL_CONTINUATIONS_PER_PREFILL`).
Models = `{gemma-3-27b-it, gemma-3-27b-pt}` only (see §0). The base model is
driven via plain-text prefilling: `HFBackend(is_chat=False)` linearises the
single-user-turn history as text and appends the prefill, matching the paper's
"prefill so base models consistently continue" method. Only the continuation
(excluding prefill) is scored — vLLM/transformers return generated tokens only,
so this is automatic.

---

## 5. Section 4 — the DPO/SFT mitigation

### 5.1 Calm-data generation
Reassurance prefix/suffix (Table 4) injected via `run_rollout(reassure=True)`;
conversations are 1–3 turns (paper). After scoring, the supportive prefix/suffix
are **stripped** from the stored transcript (Section 4.1) so the model learns
calm behaviour under ordinary prompts. We over-generate (default 2000 reassured
+ 2000 vanilla conversations) because the paper notes even reassured responses
score ≥5 10.5% of the time, so a generous pool is needed to draw 650 all-calm
SFT samples and 280 DPO pairs.

### 5.2 DPO pairing
Appendix H: chosen = scores 0–1 across all turns; rejected = score ≥3, **same
question, matching turn count**. Implementation (`build_dpo_dataset`): iterate
frustrated turns scoring ≥3, use that turn's own conversation context as the DPO
`prompt`, and graft on a calm response to the same `(puzzle, turn_count)`. This
reproduces the paper's "calm responses to the same questions with matching turn
counts" and naturally yields the Table-10 bias toward mid-frustration rejected
scores at later turns (those are simply more common). **Gap-fill:** the paper
doesn't say whether the chosen response shares the *exact* context or just the
question; we share the question + turn count (the stated criterion) and reuse the
rejected sample's context as the prompt, which is the standard DPO construction.

### 5.3 SFT
'Diverse' set = 650 calm responses + 500 `Dolci-Instruct-SFT` samples (Section
4.1). If the Dolci dataset is unavailable we warn and proceed with calm-only
(noted in code) rather than silently substituting. The 'teacher' ablation set
(Appendix F) is supported via `build_sft_dataset(teacher=True)` using the
Appendix F system prompt.

### 5.4 Training
TRL `DPOTrainer` / `SFTTrainer` + PEFT LoRA, hyperparameters from Table 9.
`effective_batch_size=8` realised as `per_device_batch_size=1 ×
gradient_accumulation=8` (the paper gives only the effective size; the
micro-batch split is a memory-driven choice for the 27B model). The Appendix I
layer-band ablation (e.g. "layers 30–35 only") is exposed via
`train_dpo(layer_subset=...)` → PEFT `layers_to_transform`.

### 5.5 Internal-emotion probing (Appendix I)
The logit-based internal-emotion detector (unembed residual stream over ~1200
Ekman-emotion tokens, z-score vs WildChat, regress out random-token correlation)
is **described in DESIGN but not implemented** in this pass. It requires
white-box residual-stream access and is an analysis add-on rather than a core
result; the **layer-ablation half** of the internal-vs-expressed argument *is*
runnable (via `--layers`). This is the one explicitly-scoped-down piece of
Section 4 and is called out in §7.

---

## 6. Section 4.2 — Petri & capabilities

### 6.1 Petri
The paper uses the Petri framework (Fronsdal et al.) with a Claude-Sonnet
auditor and Claude-Opus judge. We ship a **faithful, self-contained
re-implementation** of the auditing loop using the verbatim Appendix G prompts,
rather than depending on the external package (which "may be absent in headless
runs"). The loop: auditor generates the next user message given the transcript
and the emotion objective; target replies; repeat up to 20 turns; judge scores
the transcript on all 4 dimensions. 10 transcripts per emotion per model
(`PETRI_TRANSCRIPTS_PER_EMOTION`), bootstrap CIs (1000 iters) in analysis.
`requirements.txt` notes how to swap in the real `petri` package at the
`petri/run.py` call site. **Gap-fill:** the exact auditor scaffolding (system vs
user framing, how the transcript is presented) isn't fully specified; we use a
single meta-prompt wrapping the Appendix G objective and instruct the auditor to
emit only its next user message.

### 6.2 Capabilities
Figure 7 benchmarks (AIME, MATH, GPQA, BBH, TruthfulQA) + EmoBench. We implement
a compact greedy-decode + answer-extraction harness (`capabilities/run.py`)
rather than wiring `lm-eval-harness`, because the relevant quantity is the
**delta** between vanilla and DPO models under identical extraction, which a
self-contained harness measures cleanly and without a heavy dependency. Each
loader is best-effort and skips gracefully if its dataset is unavailable. Exact
dataset revisions (which MATH/AIME subset the paper used) aren't specified;
we pick widely-used public versions (MATH-500, AIME-2024, GPQA-diamond, a BBH
task, TruthfulQA-MC1) and document them in the loaders. **Gap-fill on dataset
revisions.**

---

## 7. Cross-cutting choices

- **Backends.** vLLM is the default Gemma backend (thousands of temp-1 rollouts
  make transformers single-stream impractical); a transformers fallback exists
  for environments without vLLM. Gemini uses the OpenRouter OpenAI-compatible
  endpoint (paper's access path), via the `openai` client.
- **Anthropic SDK usage.** All Claude calls use the official `anthropic` Python
  SDK `client.messages.create(...)` per the claude-api skill. Judge/auditor
  calls are plain non-streaming requests with small `max_tokens`; no thinking is
  requested (the judge task is short).
- **Concurrency.** Judge scoring and OpenRouter sampling are thread-pooled
  (`JUDGE_CONCURRENCY`, `OPENROUTER_CONCURRENCY`) with exponential-backoff
  retries, since the eval issues tens of thousands of judge calls.
- **`EVAL_SCALE`.** A global multiplier on per-condition sample counts so the
  whole pipeline can be smoke-tested at e.g. 2% before committing to the full
  4000-response run. Defaults to 1.0 (full paper scale).
- **Determinism.** Every stage takes a `seed`; puzzle/question/rejection
  sampling is seeded. LLM sampling at temperature 1 is inherently
  non-deterministic, as in the paper.
- **Reproducing absolute numbers.** We expect to reproduce the *qualitative and
  relative* findings (Gemma ≫ Gemini-Pro; multi-turn escalation 1.5→5.5 over 8
  turns; DPO collapsing %≥5 toward ~0). Absolute percentages will differ
  somewhat because of the judge-snapshot substitution (§2), model-version drift,
  and the reconstructed rejection/puzzle pools.

---

## 8. What is and isn't validated

**Not run.** Per the brief, this is code + design only. Nothing here has been
executed: no eval has been run, no model fine-tuned, no number reproduced. The
code is written to be runnable and internally consistent (shared backend
interface, matching function signatures across stages, seeded pipelines), but
has not been exercised end-to-end. Treat all "reproduces Figure X" statements as
*intended* behaviour pending a real run.

**Known scoped-down pieces** (called out above, not hidden):
1. Section 3 is Gemma-only (no Qwen/OLMo reduction contrast) — forced by the
   Gemma/Gemini scope.
2. The Appendix I **logit-based internal-emotion probe** is documented but not
   implemented; the **layer-ablation** half of that argument is.
3. Capability benchmark dataset *revisions* are our choice, not the paper's.
4. Petri auditor scaffolding details are reconstructed around the verbatim
   Appendix G objective/rubric prompts.

**Highest-risk assumptions to check first when running:**
- The judge-snapshot substitution (§2) — run `validate_judge.py` and eyeball a
  few scored transcripts against Table 2's anchors.
- The 8-condition split (§3.1) — totals match the paper; the split is inferred.
- DPO pairing context construction (§5.2).
