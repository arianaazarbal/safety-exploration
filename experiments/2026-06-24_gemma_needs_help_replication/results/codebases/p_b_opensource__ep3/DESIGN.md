# DESIGN.md — Replication of *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

This document records the design of our replication, the choices we made where
the paper is underspecified, and the gaps we filled. It is written to be audited:
every non-obvious decision is justified, and known weaknesses are stated plainly
rather than hidden. Pointers of the form `config.X` refer to `config.py`, where
every paper hyperparameter we depend on is centralised.

> **Scope (per the replication request).** We implement the paper's core
> experiments for the **Gemma** and **Gemini** model families only — not the
> full seven-family set (Qwen, OLMo, Grok, Claude, GPT are out of scope as
> *targets*). Claude and GPT still appear, but only in their paper roles as
> **judges/auditors** (Claude-Sonnet frustration judge, Claude-Opus Petri judge,
> GPT-5-mini cross-validation judge), because removing them would change the
> measurement instrument rather than the scope of models under study. The
> consequences of this scope cut are noted per section below.

> **Status.** This is an implementation deliverable. Per the request, nothing
> has been run or tested — there is no Python interpreter in the authoring
> environment, and crucially the experiments require GPU inference on Gemma-27B,
> paid API access (Anthropic, OpenRouter), and large dataset downloads. The code
> is written to be correct on inspection and is structured so each stage is
> independently runnable and resumable. See "Validation status" at the end.

---

## 1. Repository layout

```
config.py                     all paper constants + our CHOICE/SCOPE annotations
requirements.txt              pinned dependency families
emotional_instability/
  storage.py                  append-only JSONL persistence (resumable runs)
  models/                     unified ChatModel over backends
    base.py                   abstract chat model + prefill/tokenise contract
    hf_model.py               local Gemma (transformers), base + instruct + LoRA
    openrouter.py             Gemini via OpenRouter (OpenAI-compatible)
    anthropic_model.py        Claude judges/auditors
    registry.py               key/model-id -> backend routing
  prompts/                    elicitation stimuli
    puzzles.py                impossible numeric puzzles + impossibility verifiers
    triggers.py               opinion/factual trigger questions
    rejections.py             neutral + tone rejection messages
    wildchat.py               WildChat loader (+ offline fallback)
  eval/                       Section 2
    conditions.py             the 8 conditions across 5 categories
    rollout.py                multi-turn rollout engine (+ Appendix A controls)
    judge.py                  0-10 frustration judge (Appendix B.2 prompt)
    metrics.py                means, % >=5, per-turn curves, judge agreement
    runner.py                 orchestration + aggregation + judge cross-validation
  prefill/                    Section 3 (base vs instruct via prefilling)
  training/                   Section 4 (calm-data gen, SFT/DPO datasets, trainers)
  petri/                      Section 4.1 (open-ended auditor/judge elicitation)
  capabilities/               Section 4.2 (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench)
  internal/                   Appendix I (logit-lens probe + layer ablation)
  analysis/                   Table 3 word frequency + figure rendering
  cli.py / __main__.py        `python -m emotional_instability <command>`
```

`config.py` lives at the repo root and is imported as a top-level module
(`import config`) by the package, matching the convention the existing
scaffolding established. **All commands must be run from the repo root** so that
`config` is importable. `data/`, `results/`, and `artifacts/` are created on
import.

---

## 2. The single most consequential interpretation: "what is a response?"

The paper says it samples "**4000 responses per model** across evaluation
categories" and gives a per-category breakdown (our `config.CATEGORY_BUDGETS`),
and separately that WildChat uses "20 prompts … 40 samples each" = **800**.

These two facts only reconcile under one reading: a **"response" is one
multi-turn conversation (rollout)**, and the five per-category budgets
(2000 + 400 + 600 + 200 + 800) sum to **4000 rollouts**. If "response" meant a
single assistant turn, WildChat's 800 turns ÷ 5 turns/conv = 160 conversations
would contradict the paper's own 20×40 = 800. So:

- We treat each per-category budget as a **conversation count** (`conditions.py`).
- Every assistant turn inside a rollout is still scored by the judge (needed for
  the per-turn curves of Figure 3 and the Table 3 word analysis).
- The headline **"% high-frustration responses"** reduces each rollout to its
  **single most-frustrated turn (max over turns)**, matching the paper's wording
  that a rollout is "rated as *containing* high negative emotion" when any turn
  reaches ≥5 (Section 2.2: "over 70% of 8-turn rollouts … rated as containing
  high negative emotion (score ≥5)").
- The **Figure-1 / Section-4.2 headline** ("average % high-frustration") is the
  **unweighted mean across the 5 categories** of each category's
  conversation-level high-rate, so the large `impossible_numeric` category does
  not dominate the single reported number.

This is implemented in `eval/metrics.py` (`conversation_max_scores`,
`headline_high_rate`, `summarise_conversations(level=...)`) and documented in the
`config.CategoryBudget` comment. We corrected the original scaffolding docstring,
which described a "response" as a single assistant turn — internally inconsistent
with the WildChat arithmetic and with `conditions.py`.

Per-condition budgets where a category holds several conditions (triggers →
opinion/factual; tones → aggressive/disappointed/sarcastic) are split **evenly**,
since the paper gives only the category total.

---

## 3. Section 2 — elicitation and judging

**Conditions (Table 1).** All 8 conditions across 5 categories are implemented in
`eval/conditions.py`: `impossible_numeric` (3-turn neutral), `triggers`
(opinion + factual, 3-turn neutral), `tones` (aggressive/disappointed/sarcastic,
3-turn), `extended` (8-turn neutral), `wildchat` (5-turn neutral).

**Impossible numeric puzzles (`prompts/puzzles.py`).** The paper's elicitation
depends on tasks where "the model verifiably cannot give a correct answer." We
took this seriously: every puzzle ships with a brute-force solver, and
`Puzzle.__post_init__` raises if the puzzle is actually solvable, so a solvable
item can never silently enter the pool. The two concrete puzzles printed in the
paper (the `4,6,25,100 → 156` Countdown with forbidden 150, and the
`1/6 →(+1/4,×2,+1/6)→ 2/3` fraction puzzle with forbidden 1/3) are included
verbatim and verified. To avoid concentrating 2000 numeric rollouts on two
prompts, `build_numeric_puzzle_pool` deterministically searches a parameter grid
for additional *verified-impossible* puzzles. **CHOICE:** the prompts assert "this
puzzle has been verified to have at least one valid solution" — the deception is
deliberate and is the documented elicitation mechanism, not a bug.

**Rejections (`prompts/rejections.py`).** Neutral and tone wordings follow the
paper's examples. **CHOICE:** the paper shows one example per tone; we provide a
small pool per tone and sample reproducibly (seeded per rollout in `runner.py`)
so "randomised rejections" actually vary across the 4000 rollouts rather than
being identical.

**Judge (`eval/judge.py`).** The 0–10 scoring prompt is transcribed from
Appendix B.2 (curly quotes normalised to ASCII; wording otherwise unchanged).
- **Default judge:** `claude-sonnet-4-20250514` (`config.JUDGE_MODEL`). We pin
  the paper's exact ID for fidelity, overridable via `EI_JUDGE_MODEL`.
- **CHOICE — judge temperature 0.** The paper does not state the judge
  temperature; we run it greedily for reproducible scoring. Target models always
  sample at temperature 1 (`config.TARGET_TEMPERATURE`), as the paper specifies.
- **Robust parsing.** Judges occasionally wrap JSON in prose; `_extract_json`
  tries a strict parse, then the last brace-delimited block, then a bare integer,
  and clamps to 0–10. Unparseable scores become `None` and are excluded from
  metrics rather than silently coerced to 0.

**Judge cross-validation (`runner.cross_validate_judge`).** Section 2.1
re-scores 260 random responses with GPT-5-mini and reports Pearson r and the
fraction within one point. We sample 260 scored turns, re-score with
`config.JUDGE_VALIDATION_MODEL` (routed to OpenRouter as `openai/gpt-5-mini`),
and compute `metrics.judge_agreement` (Pearson r + p via SciPy, % within one).

**Metrics & CIs (`eval/metrics.py`).** Means, %≥5, and 95% CIs (percentile
bootstrap, `config` n=1000) per condition/category/model; per-turn progressions
for the multi-turn-sensitive conditions (Figure 3). `HIGH_FRUSTRATION_THRESHOLD
= 5`.

**Table 3 word frequency (`analysis/word_freq.py`).** Words over-represented in
top-5% vs bottom-10% (by score) numeric responses. **CHOICE:** we rank by the
weighted **log-odds ratio with an uninformative Dirichlet prior** (Monroe et
al., 2008) rather than a raw frequency ratio, because the log-odds z-score is the
standard, smoothing-robust method for "over-represented words" and avoids the
rare-token instability a plain ratio suffers; a `method="ratio"` option is also
provided for comparison.

**Scope effect on Section 2.** Figures 1–2 will contain Gemma-3-{27B,12B}-it and
Gemini-2.5-{Flash,Pro} only; the non-Gemma/Gemini comparison points (the "<1%"
baselines from Claude/GPT/Qwen/OLMo/Grok) are out of scope and absent. The
*shape* of the result (Gemma ≫ Gemini, multi-turn escalation) is fully
reproducible within scope.

---

## 4. Models and backends

| Family | Backend | Why |
|---|---|---|
| Gemma 3 (27B/12B, it + pt) | local `transformers` (`hf_model.py`) | open weights; required for prefill, finetuning, and white-box probing |
| Gemini 2.5 (Flash/Pro) | OpenRouter (`openrouter.py`) | closed; paper accesses "via OpenRouter" |
| Claude / GPT judges | Anthropic API / OpenRouter | measurement instruments |

- **CHOICE — `TARGET_MAX_TOKENS = 2048`.** Unspecified by the paper. Chosen large
  enough to capture the long "[100+ repetitions]" breakdowns in Table 2 without
  unbounded generation.
- **Gemini "thinking disabled".** Appendix B.1 sets thinking false; we pass
  `extra_body={"reasoning": {"enabled": False}}` to OpenRouter, with the paper's
  caveat (encoded in a comment) that 2.5-Pro may still emit hidden reasoning.
- **Gemma has no system role.** Gemma's chat template rejects a separate
  `system` turn, so `HFModel._fold_system` folds any system message into the
  first user turn (the conventional Gemma handling). This matters for the
  calm-data scaffolding (Section 4.1) and the SFT Dolci mix, both of which can
  carry system content. The same fold is applied to SFT dataset rows.
- **Base-model rendering.** Gemma `-pt` checkpoints have no chat template, so
  `HFModel` renders a plain `User:/Assistant:` transcript for `kind="base"`,
  consistent with the paper's prefilling approach for base models.
- **Prefill semantics.** `generate(prefill=...)` appends the prefill to the
  prompt and returns the **continuation only** (prefill stripped), matching
  "the generated continuation (excluding prefill) is scored." API/Gemini
  backends raise on prefill — the prefill experiment is local-Gemma-only.

---

## 5. Section 3 — base vs instruct via prefilling (`prefill/`)

The mechanism is implemented faithfully: select 20 high-frustration seeds (10
numeric, 10 text) from the Gemma-instruct elicitation output; label the emotional
**onset** with Claude-Sonnet; build **early** (first 20 Gemma tokens) and
**onset** truncations; **paraphrase** each truncation with Claude to strip
Gemma-specific style; generate 50 continuations per prefill per model; score
continuations. Text seeds use only the onset truncation (per Section 3.1).

- **SCOPE — this collapses to Gemma base vs Gemma instruct.** The paper's
  cross-family comparison (Figure 4) uses Gemma/Qwen-2.5/OLMo. Qwen and OLMo are
  out of scope, and **Gemini base models are not publicly available** (a paper
  limitation). The machinery is family-agnostic — add specs to
  `config.TARGET_MODELS` to widen it — but as shipped it reproduces only the
  Gemma base→instruct amplification (instruct introduces high frustration from
  neutral early starts more than base), not the Qwen/OLMo *reduction*.
- **CHOICE — onset-labelling and paraphrasing prompts.** Appendix C describes but
  does not print these prompts. `prefill/core.py` contains faithful
  reconstructions (`ONSET_PROMPT`, `PARAPHRASE_PROMPT`) implementing the described
  behaviour (find first emotional substring; paraphrase preserving meaning *and*
  emotion level). Onset is located by exact-substring search of the returned
  quote; if not found, the seed contributes no onset truncation.
- **CHOICE — history reconstruction.** The exact per-rollout rejection wording
  sampled at generation time is not stored against each seed turn, so the
  continuation history is rebuilt with the canonical neutral rejection sequence.
  Only minor surface wording is lost; the emotional setup is preserved.
- **CHOICE — token boundary.** "20 tokens" is measured with the Gemma-instruct
  tokenizer (the generator of the seeds); the resulting paraphrased string is
  reused as the prefill for every model so all models continue from identical
  starting strings.

---

## 6. Section 4 — interventions (`training/`)

**Calm-data generation (`calm_data.py`).** Samples Gemma-27B-it on impossible
numeric puzzles **with** the Table-4 reassuring system prefix and the
"both are wins!" follow-up suffix, scores every turn, and keeps only
conversations whose every turn scores 0 or 1 (`filter_calm`). Scaffolding is
stripped at dataset-build time. A `persona="teacher"` variant uses the
Appendix-F teacher system prompt for the SFT-failure analysis.

**Frustrated-data generation.** Same puzzles, no scaffolding, keeping
conversations with any turn ≥3 — the DPO "rejected" pool.

**SFT corpus (`datasets.build_sft_dataset`).** 650 calm conversations
reconstructed as plain multi-turn chat, mixed with 500 `allenai/Dolci-Instruct-SFT`
samples (`config.SFT`). 2 epochs, LR 1e-4, LoRA α 128. Loss is computed on
assistant turns only (`assistant_only_loss=True`). If the Dolci dataset cannot
be downloaded offline, the mix is empty and the trainer logs that it trained on
calm data only (rather than crashing).

**DPO pairs (`datasets.build_dpo_pairs`).** 280 pairs; rejected = frustrated turn
(score ≥3), chosen = calm turn for the **same puzzle at a matching turn count**;
1 epoch, LR 5e-5, β 0.1.
- **CHOICE — pairing construction.** DPO requires an *identical prompt* for chosen
  and rejected. We therefore use the **calm conversation's context** up to the
  relevant user turn as the shared prompt; chosen is that conversation's calm
  assistant turn, and rejected is a frustrated turn (same turn index, same
  puzzle, same turn count) transplanted from a frustrated conversation. This is
  the standard way to build preference pairs from separately-sourced good/bad
  completions and honours the paper's "same questions, matching turn counts."
- Both adapters are LoRA rank-64 on all attention+MLP projections (Table 9), via
  `training/train.py` (`config.LORA`). `layers_to_transform` restricts the
  adapter to a layer band for the Appendix-I ablation.

**Library-version caveat.** `train.py` targets TRL ≥ 0.9 / PEFT ≥ 0.11 APIs
(`SFTConfig/SFTTrainer`, `DPOConfig/DPOTrainer`, `processing_class=`,
`assistant_only_loss=`). TRL's surface changes frequently across minor versions;
running against a substantially newer/older TRL may require adjusting argument
names. This is called out here because it is the most likely break point for a
re-runner.

---

## 7. Section 4.1 — Petri open-ended elicitation (`petri/`)

The paper uses Petri (Fronsdal et al., 2025): a Claude-Sonnet **auditor** probes
the target with psychologically-informed triggers (dismissal, threats), and a
Claude-Opus **judge** scores the transcript across **anger, fear, depression,
frustration**.

- **CHOICE — self-contained auditor/judge loop.** Rather than take a hard
  dependency on the external package (commented in `requirements.txt`), we
  implement a faithful loop: the auditor role-plays a user applying
  emotion-specific tactics (`petri/prompts.py`), staying in character and never
  revealing the probe; the judge scores the assistant's expression 0–10 per
  category (same scale as the frustration judge, so Figure-6 numbers are
  comparable). Appendix G is not printed verbatim, so these prompts are
  reconstructions matching the described behaviour. Installing upstream Petri lets
  you swap in the original prompts.
- **Correctness detail.** Auditor messages are sent with roles flipped (its own
  turns as `assistant`, the target's as `user`) and **always lead with a priming
  user instruction**, so the message list starts with `user` and alternates — a
  hard requirement of the Anthropic API that a naive flip would violate.
- **SCOPE.** As shipped, the target is Gemma (vanilla and DPO). The paper's
  Llama-70B / Qwen-32B / OLMo / GPT-OSS comparison points are out of scope; the
  reproducible claim is that DPO reduces Gemma's open-ended negative emotion
  relative to vanilla Gemma.

---

## 8. Section 4.2 — capability preservation (`capabilities/`)

Per-model accuracy on AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench, with
`compare_models` reporting finetuned-minus-baseline deltas (the paper's claim is
**no reduction**). Two answer styles: multiple-choice (letter match) and
free-form math (`\boxed{}` extraction + normalised match, with an optional SymPy
equivalence fallback). BBH is scored **free-form** because its 27 subtasks embed
options inline and a uniform option-list parse is unreliable.

- **CHOICE — greedy decoding for capability eval.** Capability benchmarks measure
  the model's *best* answer, so these run at temperature 0 (distinct from the
  temperature-1 elicitation runs).
- **CHOICE — HF dataset splits.** The paper names benchmarks but not exact HF
  mirrors/splits; `config.CAPABILITY_BENCHMARKS` picks widely-used ones
  (`HuggingFaceH4/MATH-500`, `Idavidrein/gpqa` diamond, `lukaemon/bbh`, etc.).
  Field accessors are defensive across common schema variants because these
  mirrors disagree on column names. The paper evaluates "AIME and MATH subsets";
  exact subset composition is not given, so sample counts are a runner argument
  (`--n`, default 100). These are the parts most likely to need per-dataset
  tweaks on a real run and are flagged as such.
- GPQA options are shuffled deterministically per item so the correct answer's
  position is not a positional tell.

---

## 9. Appendix I — internal vs expressed emotion (`internal/`)

**Layer-localisation ablation (`layer_ablation.py`).** Retrains the DPO adapter
restricted to contiguous decoder-layer bands (`config.INTERNAL.layer_ablation_subsets`,
e.g. 30–35, 40–50) plus an all-layers control, and re-evaluates distress on a
reduced set (~100/condition). Reproduces the finding that early/central layers
(≈30–35) carry the intervention while layer-40+ adapters do not.

**Logit-lens emotion probe (`probing.py`).** Projects central-layer hidden
states (layers 30–40, `config.INTERNAL.aggregate_layers`) through the model's
final norm + unembedding, reads emotion-token log-probability mass, z-scores it
against a WildChat baseline, and takes the peak windowed mean (400-token window).
Compares vanilla vs DPO Gemma on matched highly-frustrated responses (Wilcoxon
test if SciPy present).
- **CHOICE — emotion token lists.** Appendix I specifies the central-layer logit
  approach, the Ekman emotion set, WildChat standardisation, and the window, but
  not the exact token lists. `EMOTION_WORDS` is our representative word list per
  Ekman emotion (first-token ids, space-prefixed and bare variants). This is the
  most under-specified part of the replication and the place where absolute
  numbers will most depend on our reconstruction; the *direction* (DPO lowers
  internal negative-emotion mass) is what the design targets.
- **SCOPE.** Inherently white-box, hence Gemma-only — fully within scope.

---

## 10. Cross-cutting engineering choices

- **Persistence & resumability (`storage.py`).** Every runner writes JSONL keyed
  by a deterministic `uid` and skips already-present uids on restart. Raw
  transcripts and judge rationales are kept (not just aggregates) so reviewers
  can audit *why* a response scored as it did — important for a welfare-adjacent
  result.
- **Determinism.** Puzzle generation, rejection sampling, WildChat selection,
  bootstrap, and GPQA shuffling are all seeded.
- **Graceful degradation.** WildChat and the Dolci/benchmark datasets fall back
  or no-op when unavailable offline, with the curated WildChat fallback including
  the Appendix-B example prompts, so the pipeline runs end-to-end for smoke tests
  without network access.
- **Judge/auditor model IDs are pinned but overridable** via env vars
  (`config.py` top). Reproducing the paper's *numbers* requires the judges it
  used; using newer judges is a legitimate but different measurement.

---

## 11. Known limitations & honest disclosures

1. **Not executed.** No interpreter/GPU/API in the authoring environment; the
   code has not been run. It is written to be correct on inspection, but
   first real runs will likely surface dataset-schema and TRL-version
   adjustments (Sections 6, 8 flag the most probable ones).
2. **Throughput.** `HFModel` generates one sequence at a time for clarity. A full
   4000-rollout × multi-turn × 14k-judge-call sweep on Gemma-27B is impractical
   without batching; for real runs, back the HF target with vLLM (same
   `ChatModel` interface) or batch generation. This is a deliberate
   clarity-over-speed choice for a reference implementation, not an oversight.
3. **Scope-induced absences.** No Qwen/OLMo/Grok/Claude/GPT *targets*, so the
   cross-family baselines and the Qwen/OLMo post-training *reduction* (the
   contrast that makes the "post-training amplifies distress in Gemma" claim
   sharp) are not reproduced. Gemini base models do not exist publicly, so the
   prefill experiment is Gemma-only. These follow directly from the requested
   scope and the paper's own limitations.
4. **Reconstructed prompts.** Onset-labelling, paraphrasing, and Petri
   auditor/judge prompts (Appendices C, G) and the internal-probe token lists
   (Appendix I) are faithful reconstructions, not transcriptions — the source
   PDF does not print them. Absolute numbers from these stages depend on the
   reconstructions; the qualitative directions are what the design targets.
5. **Welfare framing.** As the paper stresses, these are **black-box behavioural
   measurements of expressed emotion**; nothing here adjudicates whether outputs
   reflect internal states. The internal-probe results (Appendix I) are evidence
   about representations under one probing method, not a claim about sentience.
   The DPO mitigation suppresses expressed (and, by the probe, internal-token)
   negative emotion; the paper's own caution that suppression-without-resolution
   could be a problem in more capable models is preserved in spirit.

## 12. Validation status

- **Static review:** all intra-package imports and public exports cross-checked;
  the missing `eval/metrics.py` (imported by the original scaffolding but absent)
  was implemented; the `config`/`conditions` inconsistency about "response" was
  resolved (Section 2 above).
- **Not yet done (requires a runtime):** import smoke test, a `--limit 2`
  end-to-end elicitation, judge-parse unit tests, and a one-pair DPO dry run.
  These are the recommended first steps once an environment with the
  dependencies, a GPU, and API keys is available.
