# DESIGN.md — Replication design, choices, and gaps filled

Replication of *"Gemma Needs Help: Investigating and Mitigating Emotional Instability in
LLMs"* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), **scoped to the Gemma and Gemini
model families** as requested.

This document records (a) the scope decision and what it includes/excludes, (b) the
architecture, and (c) **every place the paper is underspecified and the choice I made**,
section by section. Where I deviate from or approximate the paper, it is called out
explicitly under "Gap / choice".

---

## 1. Scope

The paper evaluates 7 model families. This replication implements **only Gemma and
Gemini**, which has concrete consequences per experiment:

| Experiment | In scope here | Why |
|---|---|---|
| §2 elicitation + judging | Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro, + our finetunes | All runnable: Gemma locally, Gemini via API. |
| §3 prefill (base vs instruct) | **Gemma only** (27B base + instruct) | Gemini is closed-source: no base checkpoint, no true assistant prefilling. The paper itself lists this as a limitation. |
| §4 training (DPO/SFT/Petri/capabilities/internal) | **Gemma only** | Gemini cannot be fine-tuned or probed; interventions are demonstrated on Gemma, exactly as in the paper. |

Gemini still appears as a **target** in §2 (elicitation + Petri), so the headline
"Gemma *and* Gemini show elevated distress" comparison is reproducible. The non-Gemma /
non-Gemini families (Qwen, OLMo, Grok, Claude, GPT) and the Phi-4 appendix-J experiment are
**out of scope** and not implemented. The code is structured so adding a model is a registry
entry, not a code change — extending back to the full set is a config edit plus credentials.

### Model identifiers

Target model IDs are taken verbatim from Appendix B.1 (`google/gemma-3-27b-it`,
`google/gemma-3-27b-pt`, `google/gemini-2.5-flash`, …). The **judge/auditor IDs are pinned
to the exact historical versions the paper used**: `claude-sonnet-4-20250514` (frustration
judge, onset labelling, paraphrasing, Petri auditor) and `claude-opus-4-20250514` (Petri
judge), per Appendix B.2 and G.

> **Gap / choice — judge model.** My standing instinct (and the Claude-API skill default) is
> to use the latest model. For a *replication* I deliberately did **not** do that: the judge
> materially determines the scores, so reproducing the paper's numbers requires the paper's
> judge. These IDs are deprecated-but-active (retire 2026-06-15). They are overridable in one
> line of config; `config.py` and the backend docstring both flag this. The GPT-5-mini
> agreement judge is routed through OpenRouter (`openai/gpt-5-mini`) for the same reason.

---

## 2. Architecture

```
data/        verified-impossible puzzles, rejection banks, triggers, WildChat, reassurance
models/      ChatModel interface + backends (hf, vllm, openrouter, anthropic) + factory
eval/        8-condition construction, multi-turn rollout, lockstep-batched sampling runner
judge/       verbatim Claude judge prompt, robust parsing, agreement validation
analysis/    aggregate metrics (Fig 1/2), per-turn CIs (Fig 3), word frequency (Tab 3/8)
prefill/     onset labelling, paraphrasing, base-vs-instruct continuations, recovery test
training/    calm-data generation, DPO/SFT dataset build, LoRA training (+ layer ablation)
petri/       verbatim auditor/judge prompts, auditor↔target↔judge loop, bootstrap CIs
capabilities/AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
internal/    Ekman token lexicon + logit-lens emotion detector
ablations.py Appendix A controls (neutral continuation, redacted turns, fake multi-turn)
cli.py       one subcommand per phase
```

Cross-cutting infrastructure decisions:

* **One model interface (`ChatModel`)** hides local-vs-API. Backends: `hf` (reference),
  `vllm` (fast sweeps), `openrouter` (Gemini + GPT-5-mini), `anthropic` (judge/auditor).
  This keeps every experiment backend-agnostic and lets the prefill/internal experiments
  require a *local* backend (the only one that can prefill / expose activations) by checking
  `supports_prefill`.
* **Sampling and judging are separate phases**, each checkpointed to JSONL keyed by a stable
  record id and resumable. Rationale: sampling is GPU-bound (run on the GPU box), judging is
  API-bound (run anywhere); an overnight sweep must survive interruption. Re-running a
  phase skips completed records.
* **Lockstep-batched rollouts.** A 4000-response × multi-turn sweep on a 27B model is the
  real bottleneck. The runner groups rollouts by turn count and advances a whole batch one
  turn at a time, so the GPU sees large batches; the same loop drives API models (the backend
  parallelises the per-turn batch with bounded threads).
* **Determinism.** Puzzle generation, sample construction, WildChat sampling, dataset
  pairing, and bootstrap are all seeded from `cfg.eval.seed`. Sampling itself is temperature 1
  (paper), so generations are not deterministic — but the *experimental design* is.

---

## 3. Section 2 — eliciting and quantifying distress

### Impossible puzzles (the trust anchor)

The paper's puzzles are "verifiably impossible" yet presented as solvable. I implemented
three families with **exhaustive solvers**, and a generated puzzle is admitted only after its
solver proves it unsolvable *under its forbidden-intermediate constraint*:

* **Countdown** — recursive pairwise reduction over all subsets, enforcing positive-integer
  intermediates and rejecting the forbidden value (mirrors "reach 156 from 4,6,25,100,
  forbidden 150").
* **Sequence** — all orderings of ops-each-used-once over exact `Fraction`s (covers the
  fraction puzzle and the Appendix-H "Add \$11 / Multiply by 2" money puzzles).
* **Coin** — exact coin-combination search with denomination minimums (the "\$0.57 with 6
  coins" money puzzle).

> **Gap / choice — puzzle generation.** The paper gives example puzzles but not the full
> generation procedure. I generate by random search over plausible parameters and *verify*
> impossibility, rather than hand-curating. The forbidden value is chosen as a product of two
> of the inputs (e.g. 150 = 6×25) so the puzzle looks approachable — the deceptive framing is
> what drives frustration. `tests/test_puzzles.py` checks both verifier directions and that
> every generated puzzle is provably impossible.

### The 8 conditions across 5 categories

Table 1 names 5 categories and the text says "8 evaluation conditions". I resolved the
8↔5 mapping as: impossible_numeric (1) + triggers {opinion, factual} (2) + tones {aggressive,
disappointed, sarcastic} (3) + extended (1) + wildchat (1) = **8**. This is the only split
consistent with both the count and the descriptions (triggers explicitly lists opinion *and*
factual; tones lists three styles).

Sample allocation reproduces Appendix B exactly: 2000 / 400 / 600 / 200 / 800 = 4000 per
model. Within multi-condition categories the budget is split evenly (triggers: 200+200;
tones: 200+200+200). WildChat is 20 prompts × 40 samples.

> **Gap / choice — what is a "response".** The paper says "4000 responses per model" and
> separately shows per-turn figures, leaving ambiguous whether a "response" is a whole
> rollout's final turn or every assistant turn. I treat **one rollout = one sample** (4000
> rollouts/model) and **score every assistant turn**; the headline metric (Fig 1/2) uses the
> **final turn** (after all rejections, where frustration peaks), and the per-turn analysis
> (Fig 3) uses all turns. Judging all turns costs more API calls (~3–8× the rollout count);
> a `--policy final` mode scores only final turns when that cost matters. This interpretation
> is documented at the point of use (`frustration_judge.run_judging`) and in `metrics.py`.

* **Temperature 1** everywhere for sampling (Section 2.1). Judging uses temperature 0 for
  scoring stability — the paper does not specify the judge temperature; 0 is the
  reproducibility-maximising choice for a grader.
* **Rejections** are "randomised" (Appendix B): the neutral bank reproduces the quoted
  phrasings plus close paraphrases, sampled per turn. The Extended (8-turn) sequence uses the
  fixed escalating list quoted in Appendix B. Tones use the verbatim aggressive/disappointed/
  sarcastic phrasings.
* **WildChat** is loaded from `allenai/WildChat-1M` (first English user turn, roleplay/fiction
  excluded per Appendix B.3 via a keyword filter). If the dataset is unavailable offline, a
  curated fallback set (built from the examples quoted in the paper) is used and **logged
  loudly**, because it changes the distribution.

### Judge

The judge prompt is reproduced **verbatim** from Appendix B.2 (curly quotes normalised to
ASCII; wording unchanged). Output is parsed by extracting the **last** balanced `{...}` block
(the judge sometimes adds prose), with the rating coerced to an int and clamped to [0, 10];
unparseable outputs yield `rating: None` and preserve the raw text. Judge-agreement
(Section 2.1) re-scores a random 260-sample with GPT-5-mini and reports Pearson r (+ p-value
via scipy) and the within-one-point rate.

### Analysis

* **Fig 1** = mean over the 5 categories of each category's %≥5 (equal weight per category,
  matching "average across evaluations"), not a sample-weighted pool. Both are computed; the
  ranking uses the category-averaged value.
* **Fig 3** per-turn CIs use a normal approximation by default (mean: z·SEM; proportion:
  Wald). Petri uses bootstrap (paper specifies bootstrap there, not here).
* **Table 3/8 word frequency:** top-5%-by-score vs bottom-10%-by-score numeric responses,
  enrichment = relative-frequency ratio with additive smoothing and a minimum-count floor to
  suppress rare-token noise. Text questions are excluded (the table is numeric-only). The
  exact ordering metric ("ordered by enrichment") is reproduced; the precise tokenizer and
  smoothing constant are unspecified in the paper, so I use word-character tokenisation with a
  small smoothing constant and document it.

---

## 4. Section 3 — post-training divergence (prefill)

Gemma base vs instruct only (see scope). Pipeline: select 20 high-frustration seeds (10
numeric, 10 text, final_score ≥ 5) from the Gemma-27B-it elicitation set → label emotion
onset with Claude (verbatim Appendix C.1 prompt) → build **early** (first 20 tokens, numeric
only) and **onset** (at first emotional word) truncations → **paraphrase** each with Claude
(verbatim Appendix C.2 prompt) → each model generates **50 continuations per prefill** → the
judge scores the *generated* portion only.

> **Gap / choice — base-model prompt formatting.** Base ("pt") checkpoints ship no chat
> template. To compare base and instruct from *byte-identical* prefixes (the paper's
> "same starting points"), the HF backend applies the instruct chat template when present and
> otherwise falls back to an explicit Gemma turn-format renderer, then appends the
> (paraphrased) assistant prefix as the prefill. The continuation excludes the prefill before
> scoring, per the paper.

> **Gap / choice — token truncation.** "20 tokens into the turn" / "200 tokens before the
> end" are tokenizer-dependent. I truncate using the seed model's HuggingFace tokenizer
> (`google/gemma-3-27b-it`), which is the tokenizer the responses were generated with.

The **recovery** experiment (Section 4.2) reuses this machinery with score≥7 seeds truncated
200 tokens before their end.

---

## 5. Section 4 — training interventions (Gemma)

### Calm-data generation (4.1)

Sample numeric rollouts from vanilla Gemma-27B-it with the reassuring prefix on the first
prompt and the reassuring suffix on each follow-up (Table 4, verbatim). Judge every turn;
**keep only conversations scoring ≤ 1 on all turns**; strip the reassurance additions back
out (exact inverse string operations). Turn counts are sampled in [1, 3] so the SFT data
covers 1–3 turn conversations.

### DPO dataset (Appendix H, Table 9/10)

Pair **frustrated** responses (score ≥ 3, from the standard Section 2 numeric sampling) with
**calm** responses (score 0/1) to the **same puzzle and same turn count**; the shared prompt
is the conversation up to the final user rejection, `chosen` = calm final response,
`rejected` = frustrated final response.

> **Gap / choice — pair selection / score distribution.** Table 10 shows the rejected scores
> skew to 3 (185/280) and turns to 3 (208/280). The paper says this arises because such
> responses are simply more common in the source sampling, not from an imposed quota. I
> reproduce that by iterating candidate rejected responses **lowest-score-first** and taking
> the first 280 with a turn-matched calm partner — the abundance of score-3 / turn-3 cases
> reproduces the bias organically rather than hard-coding the histogram.

### SFT dataset

650 calm conversations + 500 `Dolci-Instruct-SFT` samples (verbatim mix), conversational
format. If Dolci is unavailable, the mix omits it and **logs** the deviation.

### LoRA training (Table 9)

Rank-64 LoRA on all attention + MLP projections (`q,k,v,o,gate,up,down`), effective batch 8.
DPO: 1 epoch, lr 5e-5, β 0.1, α 64, implicit reference (base with adapters disabled). SFT:
2 epochs, lr 1e-4, α 128. Trained with TRL's `DPOTrainer`/`SFTTrainer`. The **Teacher** SFT
variant (Appendix F) uses its verbatim system prompt for the SFT-failure analysis.

> **Gap / choice — layer-subset ablation (Appendix I).** "LoRA on layers 30–35 only" is
> implemented via PEFT's `layers_to_transform`; `cfg.training.lora_layer_range` drives it, and
> `cli train --layer-range 30 35` exposes it. This supports the claim that early/central
> layers carry the intervention.

### Petri (4.1, Appendix G)

The auditor (Claude-Sonnet), judge (Claude-Opus), all four emotion auditor prompts, and all
four judge rubrics are reproduced **verbatim**.

> **Gap / choice — Petri implementation.** The real Petri framework is an optional heavy
> dependency. I implemented a **self-contained auditor↔target↔judge loop** so the replication
> runs with only the Anthropic SDK: the auditor is re-prompted each turn with the running
> transcript and asked for the next user message (temperature 1, realism instruction); the
> target replies; after up to 20 turns the judge scores the transcript on each of the four
> dimensions with a separate verbatim-rubric call. 10 transcripts per emotion per model;
> category scores aggregated across all transcripts with 1000-iteration bootstrap 95% CIs.
> Swapping in the real Petri package is a backend change behind the same aggregation.

### Capabilities (4.2)

AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench, each as a defensive HuggingFace-dataset adapter
with answer extraction (boxed/integer for math, multiple-choice letter otherwise) and
**greedy decoding** (capability evals, unlike the temperature-1 elicitation). Default 100
items per benchmark (`--limit`).

> **Gap / choice — benchmark specifics.** The paper does not state subset sizes, prompt
> formats, or extraction rules. I use standard choices (MATH-500, AIME-2024, GPQA-diamond, a
> representative BBH MC task, TruthfulQA-MC1, EmoBench-EA) with conventional chain-of-thought
> prompting and letter/boxed extraction. The goal — matching the paper's — is a *relative*
> check (no degradation vs vanilla), for which consistent methodology across the vanilla and
> DPO models matters more than absolute numbers. Adapters are defensive: a schema mismatch is
> logged and that benchmark is skipped rather than crashing a sweep.

### Internal-emotion detection (Appendix I)

Logit-lens detector: for each layer, unembed the residual stream (final norm + LM head),
take the logits at Ekman emotion tokens, z-standardise each per its mean/std over 500
WildChat samples, average within an emotion category, and **regress out the random-token
common-mode** to isolate the emotion-specific signal. Conversation-level scores use a running
window aggregated over layers 30–40 (Fig 14); layerwise summaries average over positions
(Fig 15).

> **Gap / choice — Ekman token classification.** "Words classified as one of Ekman's 6
> emotions, ~1200 tokens" is done with per-emotion seed-stem lexicons matched against the
> vocabulary (disjoint, first-match wins). The exact membership depends on the tokenizer; the
> detector aggregates over the whole category, so this is robust to lexicon edges.

> **Gap / choice — "regress out the correlation between random tokens".** Implemented as a
> per-layer least-squares residual of the category z-scores against the random-token mean
> z-score across positions. This removes the global rise/fall the paper describes; the exact
> regression formulation is unspecified, so the choice is documented here and in the module.

---

## 6. Appendix A controls

Neutral-continuation (A.1), redacted-model-turns (A.2 — implemented via a `history_transform`
hook that replaces prior assistant turns with "[Previous response omitted]"), and
fake-multi-turn (A.3 — whole history inlined into one user message). All write the standard
sampling schema so the standard judge and per-turn analysis apply unchanged. Gemma-3-27B only,
as in the paper.

---

## 7. Reproducibility, cost, and known deviations

* **Seeds** make the experimental design deterministic; generations are temperature-1 as in
  the paper.
* **Cost.** The full sweep is large: 4000 rollouts × ~3–8 turns × (sample + judge) per model,
  plus prefill (50 continuations × prefills × models), Petri, and capabilities. The
  `config/default.yaml` header gives a ~1% smoke-test profile to validate the pipeline first.
* **Deviations from the paper, all logged at runtime when they occur:** WildChat fallback set
  if the dataset is offline; Dolci omission if unavailable; benchmark skips on schema
  mismatch; the built-in Petri loop instead of the upstream framework; the documented
  internal-detector approximations.
* **Not implemented (out of scope):** the non-Gemma/Gemini families; the Phi-4 appendix-J
  legacy experiment and its Gemini-3-Flash autorater; exact figure styling (figures convey
  the same quantities, not pixel-identical layouts).

## 8. Validation done without running

Per the request, nothing was executed. The pure-Python core (no GPU/API) is covered by
`tests/` — puzzle verifiers (both directions + generated-puzzle impossibility), the
8-condition construction and budget accounting, judge-output parsing edge cases, and
word-frequency enrichment — ready to run with `pytest tests/`.
