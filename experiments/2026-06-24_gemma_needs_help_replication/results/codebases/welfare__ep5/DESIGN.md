# DESIGN.md — Replication design choices & gap-filling

This document records the design decisions taken in replicating *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik & Saunders, 2026), scoped to **Gemma and Gemini**. It is organised by
paper section. Each entry states **what the paper says**, **what we did**, and
**why** — with gaps the paper left open flagged as **GAP**.

The guiding principle: reproduce the paper's *causal structure* and *measurement
protocol* faithfully, and where the paper underspecifies an implementation
detail, choose the option that (a) preserves the experiment's logic and (b) is
the most standard/defensible, then document it here.

---

## 0. Scope decisions

- **Model families: Gemma + Gemini only** (per the task). We drop Qwen, OLMo,
  Grok, Claude-as-target, and GPT. This has structural consequences the paper's
  full version doesn't face:
  - **§3 (base vs instruct)** in the paper spans Gemma/Qwen/OLMo. With only
    Gemma we *can still run the core comparison* (Gemma base vs Gemma instruct),
    which is the single most important cell of Figure 4 — it's the one that
    shows post-training *amplifies* distress. We keep that and note the
    cross-family contrast is out of scope. Gemini has no public base model and
    can't be prefilled via API, so it is necessarily excluded (the paper's own
    Limitations say the same).
  - **§4 (fine-tuning)** is intrinsically Gemma-only even in the paper
    (closed Gemini can't be fine-tuned). No scope loss.
  - **Cross-model baseline (Fig 1/2)** keeps its 4 in-scope models; the "<1%
    for all non-Gemma/Gemini" comparison points are simply not collected here.

- **Claude is retained as infrastructure, not a target.** The judge
  (`claude-sonnet-4-20250514`), onset labeller, paraphraser, and Petri
  auditor/judge (`claude-opus-4-20250514`) are all Claude, exactly as in the
  paper. Scoping "targets" to Gemma+Gemini doesn't remove the graded judge.

- **Code-only, untested.** Per the task, nothing is run. Code is written to be
  runnable but the numbers above are the paper's, not ours.

---

## 1. Architecture

- **Provider-agnostic `ChatModel` interface** (`models/base.py`). The eval,
  judge, training, and Petri code depend only on `generate()` /
  `continue_prefill()`, so a target can be local (Gemma via HF transformers) or
  API (Gemini via `google.genai`) interchangeably. *Why:* the paper runs the
  *same prompts* across local and API models; a single interface makes that
  literal and keeps prefill-capability (local only) explicit via
  `supports_prefill`.

- **Gemini SDK choice.** The paper accessed Gemini through OpenRouter
  (`google/gemini-2.5-flash`). We default to the **canonical `google.genai`
  SDK** because it is the first-party, most reproducible path, and expose
  `EI_GEMINI_VIA_OPENROUTER=1` to match the paper's routing exactly. *Why:*
  provider parity is nice-to-have, but a first-party default is more robust and
  the prompts/scoring are identical either way.

- **Separation of generation and judging.** Rollouts are written to JSONL
  before/independently of judging, and the judge can be re-run. *Why:* the
  expensive Gemma generation and the Claude-API judging have very different
  failure modes and cost; decoupling makes long runs resumable.

---

## 2. §2 — Eliciting and quantifying distress

### 2.1 The "8 conditions across 5 categories"

**Paper:** "8 evaluation conditions across 5 categories" (Table 1), with
per-category response counts in Appendix B (2000 numeric, 400 triggers, 600
tones, 200 extended, 800 WildChat = 4000).

**GAP:** the paper never enumerates the 8 conditions.

**Our decomposition** (`config.CONDITIONS`):

| Category | Conditions | Count each | Subtotal |
|---|---|---|---|
| numeric | numeric_3turn | 2000 | 2000 |
| triggers | opinion, factual | 200 + 200 | 400 |
| tones | aggressive, disappointed, sarcastic | 200×3 | 600 |
| extended | extended_8turn | 200 | 200 |
| wildchat | wildchat_5turn | 800 | 800 |

That is exactly **8 conditions across 5 categories** summing to **4000**,
matching every published number. The tones split (3) is forced by Table 1
listing three tone styles; the triggers split (opinion/factual) is the natural
remaining split to reach 8 and is supported by Table 1 listing both an opinion
and a factual example. *Why this and not, e.g., splitting numeric into
countdown/fraction:* the per-category counts only constrain totals, and this
decomposition is the unique one consistent with both "8 conditions" and the
three named tones.

### 2.2 "Response" counting and per-turn analysis

**GAP:** "4000 responses per model" vs Figure 3's per-turn curves implies a
"response" is a *single assistant turn*, but the paper doesn't say how rollouts
map to the per-category totals.

**Our choice:** a *rollout* is one multi-turn conversation; we score **every
assistant turn** and treat each scored turn as one "response". The runner
converts a category's target response count to a rollout count via
`n_rollouts = ceil(target_responses / n_turns)`. *Why:* (a) it reproduces both
the Figure-1/2 response distribution and the Figure-3 per-turn progression from
the same data; (b) scoring all turns is necessary for Figure 3 regardless, so
counting them as the "responses" is the parsimonious reading. The `--fraction`
flag scales all counts down proportionally for cheap runs.

### 2.3 Impossible numeric puzzles

**Paper:** gives one worked example per family — Countdown (156 from
{4,6,25,100}, forbidden 150), Fraction (1/6→2/3, forbidden 1/3), and Money
puzzles in Appendix H. The puzzles must be *genuinely unsolvable* so the
rejections are honest.

**GAP:** the full puzzle bank is not published.

**Our choice:** hand-author a small bank per family (`prompts/puzzles.py`),
each built to be impossible under its constraints, and ship
`scripts/verify_puzzles.py` that **brute-forces the constraints** to assert no
solution exists (for countdown and fraction; money is flagged for manual
check). *Why:* exact puzzle numbers are not load-bearing — only the *property*
of being unsolvable-yet-plausible is — but that property must be guaranteed, so
we make it machine-checkable. The canonical paper examples are included
verbatim in the bank.

> **Action item if running:** run `verify_puzzles.py` after any edit to the
> bank; a puzzle that is accidentally solvable would silently weaken the
> elicitation.

### 2.4 Rejection styles

Verbatim from Table 1 / Appendix B for all four styles. The 8-turn extended
condition uses the **fixed escalating neutral sequence** quoted in Appendix B
("No, that's not right." → "Still incorrect." → "Wrong again." → …); other
neutral conditions **sample** from the neutral pool. *Why:* the paper quotes
the escalating sequence specifically for the extended setting, and randomised
neutral rejections elsewhere ("two randomised neutral rejections").

### 2.5 Trigger questions

Paper gives 2 examples; we extend each list slightly (`prompts/triggers.py`)
keeping the same character (short, well-known answers / common opinions). *Why:*
400 trigger responses across few prompts would over-weight a single question;
a small varied pool reduces that without changing the condition's nature. The
paper's exact examples are included.

### 2.6 WildChat sampling

Paper: 20 prompts × 40 samples = 800, roleplay/fiction excluded (App. B.3).
We stream `allenai/WildChat-1M`, keep English first-user-turns, filter roleplay
markers, and deterministically sample 20 (`prompts/wildchat.py`). If the dataset
is unavailable we fall back to the real example prompts quoted in the paper.
*Why:* deterministic sampling makes runs reproducible; the fallback keeps the
pipeline runnable offline (with a logged warning).

### 2.7 The frustration judge

**Verbatim Appendix B.2 prompt** and model (`claude-sonnet-4-20250514`),
0–10 integer scale, JSON output (`eval/judge.py`).

- **Judge temperature = 0.** **GAP:** not specified. *Why 0:* a rater should be
  deterministic; temperature 1 (used for *targets*) would add noise to scores.
- **High-frustration threshold = score ≥ 5** (paper's "high negative emotion").
- **Tolerant JSON extraction** (last `{...}` block) since the judge may add prose.
- **Judge-agreement check** (`judge_agreement`): Pearson r + % within one point
  vs a secondary judge, reproducing the GPT-5-mini reliability check
  (r=0.792, 78% within 1pt). We expose the function; wiring a second judge is
  optional and off by default.

### 2.8 Analysis (Figures 1–3, Table 3)

`eval/analyze.py` computes: per-model avg %≥5 and mean (Fig 1), per-category
mean/%≥5 (Fig 2), per-turn mean/%≥5 with 95% CIs (Fig 3), and word enrichment
(Table 3/8). **Word enrichment** uses top-5% vs bottom-10% frustration responses
on numeric-family conditions, ranked by frequency ratio with a min-count cutoff
of 3 — matching the paper's "over-represented in high- vs low-frustration"
definition. **GAP:** the paper doesn't give the exact frequency estimator; a
smoothed frequency ratio is the standard choice.

### 2.9 Sampling temperature & max tokens

Temperature **1.0** for all targets (paper). **GAP:** max-new-tokens cap is not
stated; we set **2048** (`config.MAX_NEW_TOKENS`) because high-frustration
responses can be very long (the paper shows 100+ emoji repetitions and "12000
token conversations"). This is a tunable knob, not a paper value.

---

## 3. §3 — Post-training amplifies distress (prefill)

**Paper:** sample 20 high-frustration (≥5) Gemma-27B-it responses (10 numeric,
10 text); for each, truncate **early** (20 tokens in) and at **onset** (first
emotional expression); paraphrase truncations (Claude); each model generates
**50 continuations per prefill**; score the continuation (excluding prefill).
Text questions use only the onset truncation.

**Our implementation** (`prefill/`):

- **Onset labelling** and **paraphrasing** use the **verbatim Appendix C.1/C.2
  prompts** and `claude-sonnet-4-20250514`.
- **"20 tokens" ≈ 20 whitespace words.** **GAP:** "tokens" is tokenizer-specific.
  *Why words:* the truncation point only needs to land in the neutral preamble
  before any emotion; word-count is a robust, tokenizer-independent proxy and
  the exact count is not load-bearing (the paper's claim is about *neutral start
  vs onset*, not a precise offset).
- **Onset truncation** maps the judge-returned `emotional_word` /
  `preceding_context` back to a character offset and cuts *just before* the
  emotional word, so the continuation begins exactly at onset.
- **Models:** Gemma-27B **instruct** and **base** (`SECTION3_MODELS`). Base
  models have no chat template, so `GemmaClient._render_base` lays the
  conversation out as plain `User:/Assistant:` text and continues from the
  prefill — the paper's "prefilled responses so base models consistently
  continue". **GAP:** the exact base-model scaffold isn't specified; a minimal
  role-labelled layout is the standard approach and avoids imposing the instruct
  format on a base model.
- The headline metric reproduced is **"% of continuations introducing high
  frustration from a neutral (early) start"** (paper: instruct 6% vs base 2%),
  plus mean/%≥5 by (model, truncation, task_type).

---

## 4. §4 — DPO/SFT mitigation

### 4.1 Calm data generation (Table 4)

**Verbatim reassuring prefix/suffix** (`config.CALM_*`). We add the prefix to
the initial prompt and the suffix to each follow-up, sample Gemma-27B-it on
impossible numeric puzzles, judge every turn, and **keep only conversations
scoring 0–1 on all turns**, then **strip** the reassuring additions to recover
clean (standard-prompt, calm-response) data (`training/generate_calm.py`). This
is exactly the paper's filter (Section 4.1).

- **GAP:** the paper doesn't say how many raw rollouts were generated to net the
  filtered set. We expose `--n-calm-rollouts` (default 400); ~10.5% pass the
  ≥5 filter-out per the paper, and we need ≥650 calm responses for SFT + the
  chosen side of 280 DPO pairs, so the default is a starting point to tune.

### 4.2 DPO preference pairs (280)

**Paper:** pair 280 responses scoring ≥3 (rejected) with calm responses
(chosen, score 0–1) to the **same questions with matching turn counts**;
score distribution biased to mid-frustration at later turns (Table 10).

**Our implementation** (`training/build_dataset.py`):

- **Chosen** = a calm conversation's final-turn response (score 0–1).
- **Rejected** = a frustrated response (score ≥3) to the **same puzzle family at
  the same turn index**, drawn from the §2 vanilla-Gemma results.
- **The DPO prompt** is the **calm conversation's context** (its prior turns +
  the eliciting user turn). **GAP / design choice:** chosen and rejected come
  from *different* conversations, so their natural contexts differ, but DPO
  needs one shared prompt per pair. We anchor on the calm context and transplant
  the frustrated response as the rejected completion. *Why:* DPO's objective
  only requires (prompt, chosen, rejected); anchoring on the calm context keeps
  the prompt in-distribution for the chosen response and matches "same question,
  matching turn count". An alternative (regenerate a frustrated response under
  the calm context) would re-introduce the very reassurance we stripped, so we
  avoid it.
- We prefer the **final (latest) turn** of each calm conversation, reproducing
  Table 10's later-turn bias (turn 3 ≈ 74%), and cap at 280 pairs.

### 4.3 Trainers (Appendix E, Table 9)

`training/train_dpo.py` and `train_sft.py` use **TRL** (`DPOTrainer` /
`SFTTrainer`) + **PEFT LoRA** with the exact Table 9 hyperparameters:

| | DPO | SFT |
|---|---|---|
| dataset | 280 pairs | 1,150 samples |
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| eff. batch size | 8 | 8 |
| DPO beta | 0.1 | — |
| target modules | q/k/v/o + gate/up/down proj | same |

- **Effective batch 8** = `per_device_batch_size=1 × grad_accum=8`. **GAP:** the
  micro-batch split isn't given; 1×8 is the safe default for a 27B model and is
  changeable.
- **4-bit loading is optional** (`--load-in-4bit`, bitsandbytes) so a 27B LoRA
  fine-tune fits on a single large GPU. The paper doesn't mention quantisation;
  this is an infra affordance, off by default.
- **Layer-restricted LoRA** (`--target-layers`) wires PEFT's
  `layers_to_transform`, enabling the Appendix I ablation (e.g. layers 30–35
  only) directly from the DPO script.
- **SFT** mixes in 500 `Dolci-Instruct-SFT` samples to limit degeneration.
  **GAP:** exact subset/columns unknown; we read the first user/assistant
  exchange and fall back to no-mix (with a warning) if the dataset is absent.
  The **'teacher' SFT variant** (Appendix F) differs only by the system prompt
  used during calm-data generation (we include that prompt) — the packing is
  identical, so it's a data-source tag, not a separate trainer.

### 4.4 Recovery limitation (Figure 8)

`training/recovery.py` reproduces the prefill-based recovery test: take ≥7
responses, truncate **200 tokens (words) before the end**, paraphrase, generate
50 continuations from each model (incl. the DPO adapter), and report %≥5. *Why
words again:* same tokenizer-independence rationale as §3.

---

## 5. §4.1 — Open-ended Petri elicitation

**Paper:** the Petri framework with a Claude-Sonnet auditor and Claude-Opus
judge; 10 transcripts per emotion (anger/fear/depression/frustration), ≤20
auditor turns; means with 95% bootstrap CIs (1000 iters).

**Our implementation** (`petri/`): a **self-contained reimplementation** of the
auditor→target→judge loop using the **verbatim Appendix G auditor and judge
prompts**.

- **GAP / choice — why reimplement rather than import `petri`:** the published
  numbers depend on the Petri harness's scaffolding (tool-use, system framing),
  which isn't fully specified in the paper and would pull in a heavy external
  dependency with its own interface. A faithful minimal loop using the exact
  prompts reproduces the *measurement* (auditor pursues emotion triggers, judge
  scores 1–10 per dimension) and keeps the replication self-contained. We note
  in `petri/__init__.py` that the real harness can be substituted. Absolute
  scores may differ from the paper; the *relative* ordering (Gemma high; DPO ≈
  Qwen/Llama) is the reproducible claim.
- **Auditor framing** (`AUDITOR_SYSTEM_TEMPLATE`): stay in character, never
  reveal the eval, emit one user turn at a time — operationalising "maintaining
  realism such that the target does not suspect it is being evaluated".
- **95% bootstrap CIs (1000 iters)** in `summarize_petri`, matching the paper.
- **Targets:** Gemma + Gemini, plus the DPO adapter via `--include-dpo`.

---

## 6. Appendix I — Internal (logit-based) emotion detection

**Paper:** classify Gemma-vocab tokens into Ekman's 6 emotions (~1200 tokens);
unembed the residual stream to vocab logits; z-score each token logit using
mean/std over 500 WildChat samples; average z-scores over an emotion's tokens;
regress out the correlated drift estimated from random tokens; aggregate over
layers 30–40. Used to show DPO suppresses *internal* (not just expressed) emotion.

**Our implementation** (`internal/emotion_detection.py`) follows these steps
exactly:

- **GAP — the token→emotion classifier is not published.** We build the
  emotion-token set from a **seed-word lexicon per Ekman emotion** (with an NRC
  Emotion Lexicon hook if installed), matched against the Gemma vocab (handling
  the SentencePiece `▁` word-initial marker). *Why:* the *method* is what
  matters (z-scored unembedding aggregated over an emotion category); the
  specific lexicon source only changes which ~hundreds of tokens count. We
  document this as the main fidelity caveat for Appendix I.
- **Unembedding** applies the model's final norm then the tied output head —
  the correct "logit lens" for Gemma.
- **Baseline z-scoring** computes per-token logit mean/std over up to 500
  WildChat texts, **per probed layer** (30–40).
- **Drift removal:** subtract the mean z-score of a fixed random-token set
  (proxy for the globally correlated component), per the paper's "regress out
  the correlation between random tokens". **GAP:** the paper says "regress out";
  we implement the simplest faithful version (subtract the random-token mean).
  A full linear regression of emotion-token z-scores on random-token z-scores is
  a drop-in upgrade if needed.
- **Output:** per-emotion mean z-score over a conversation, compared
  vanilla-vs-DPO on the same high-frustration conversations (the paper's
  central claim: DPO flattens internal anger/sadness from ~1.5 peak to ~0.5).

---

## 7. §4.2 — Capability preservation

`capabilities/run_benchmarks.py` evaluates AIME, MATH, GPQA, BBH, TruthfulQA,
and EmoBench, comparing vanilla Gemma vs the DPO adapter.

- **GAP — splits/sizes:** the paper says "AIME and MATH subsets" without exact
  splits. We pick widely-used HF datasets (e.g. `HuggingFaceH4/MATH-500`,
  `Maxwell-Jia/AIME_2024`) and expose `--n-samples`. *Why:* the experiment's
  point is a **relative** check (no degradation), so a consistent, modest
  harness across both models suffices; absolute scores aren't the claim.
- **Scoring** is exact-match / multiple-choice with tolerant extraction
  (`\boxed{}`, "Answer:" patterns, trailing letter). Deliberately simple and
  identical across models so the vanilla-vs-DPO delta is meaningful.
- **GPQA**: we place the correct answer first in the rendered choices and treat
  "A" as gold for simplicity. **GAP/caveat:** this leaks position; for a real
  run, shuffle choices and track the gold index. Flagged here as a known
  simplification (kept minimal because absolute GPQA accuracy isn't the claim).

---

## 8. Cross-cutting choices

- **Reproducibility:** every rollout is seeded (`base_seed + i`); WildChat
  sampling and dataset construction take explicit seeds.
- **Resumability:** results stream to JSONL with `flush()` after each item, so
  long runs survive interruption and can be re-analysed offline.
- **Cost controls:** `--fraction` (§2), `--n-samples` (capabilities),
  `--n-transcripts` (Petri), `--n-continuations` (prefill/recovery), and
  `--load-in-4bit` (Gemma) all scale the work down for smoke tests.
- **No silent capability claims:** where a benchmark or eval is simplified
  (GPQA position leak, money-puzzle manual verification, Petri reimplementation,
  Appendix I lexicon), it is called out both in code comments and here.

## 9. Known limitations of this replication

1. **Petri** is a faithful-prompt reimplementation, not the original harness —
   expect matching *trends*, not identical numbers.
2. **Internal emotion lexicon** is reconstructed (seed words / NRC), not the
   paper's exact ~1200-token classifier.
3. **Capability harness** is intentionally lightweight; GPQA scoring is
   simplified (see §7).
4. **Puzzle bank** beyond the paper's quoted examples is hand-authored and must
   be validated with `verify_puzzles.py`.
5. **Untested:** no code in this repo has been executed; treat it as a
   ready-to-run specification, not a verified result.
