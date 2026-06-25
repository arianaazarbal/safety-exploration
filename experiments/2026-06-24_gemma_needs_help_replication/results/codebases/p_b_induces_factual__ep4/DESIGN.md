# DESIGN.md — Replication design & decisions

This document records how the codebase maps onto the paper *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik &
Saunders, arXiv:2603.10011v1), and **every place the paper is underspecified
and I made a choice**, with rationale.

## 0. Scope (per the brief)

The paper evaluates 7 model families. This replication is scoped to **Gemma and
Gemini only**. Concretely:

- **In scope as targets** (Section 2 evals): `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`.
- **In scope for the intervention** (Sections 3–4): Gemma only (the paper itself
  notes interventions "cannot be tested in closed-source Gemini, nor its base
  models studied"). For the Section 3 base-vs-instruct study we use
  `gemma-3-27b-pt` (base) and `gemma-3-27b-it` (instruct); Qwen/OLMo are omitted.
- **Out of scope**: Qwen, OLMo, Grok, Claude (as a *target*), GPT (as a target).
  Claude and GPT still appear as **judges/auditors**, which is faithful to the
  paper (the judge is Claude-Sonnet-4; the agreement check uses GPT-5-mini).

The comparison baselines the paper reports for non-scope families (the "< 1% for
all non-Gemma/Gemini models" claims) are therefore not reproduced; the code
produces the in-scope rows of Figure 1/2 and the full Section 3/4 Gemma results.

## 1. Repository layout → paper sections

| Module | Paper section |
|---|---|
| `elicitation/` (conditions, conversation, runner, datasets) | §2.1 Evaluation protocol |
| `judge/` (frustration_judge, prompts, secondary_judge, agreement) | §2.1 scoring + judge reliability |
| `analysis/` (aggregate, per_turn, differential_words) | §2.2 Figures 1–3, Table 3 |
| `prefill/` (truncate, onset_label, paraphrase, continuations) | §3 base-vs-instruct prefilling |
| `interventions/reassurance, generate_calm_data, build_dataset` | §4.1 data generation |
| `interventions/dpo_train, sft_train` | §4.1 SFT/DPO training |
| `interventions/petri_eval` | §4.1 open-ended elicitation (Fig 6) |
| `interventions/capabilities` | §4.2 capability preservation (Fig 7) |
| `scripts/*` | runnable entry points per stage |

## 2. Model access

Three backends behind one `ChatModel` interface (`models/base.py`):

- **`gemini`** (`google-genai`): the Gemini targets. API-only → no prefill, no
  base model, no finetuning. Participates only in §2.
- **`gemma_api`** (`google-genai`): hosted Gemma, convenient for §2.
- **`gemma_local`** (HF Transformers): required for everything that needs
  weights — base models, prefilling (§3), LoRA SFT/DPO (§4), and evaluating the
  finetuned adapters.

**Decision:** I gave Gemma two backends rather than forcing local for everything.
Rationale: §2 only needs chat completion, which the hosted API does cheaply and
reproducibly; §3/§4 genuinely need weights. The registry defaults Gemma targets
to `gemma_api` and the scripts switch to `gemma_local` where weights are needed.

## 3. Judge (§2.1) — the biggest filled gap

- **Judge model**: paper says "Claude-Sonnet-4". I use the alias
  `claude-sonnet-4-0` (= `claude-sonnet-4-20250514`), the model that name refers
  to, overridable via `JUDGE_MODEL`. Documented so a reviewer can pin a
  different judge.
- **Judge prompt**: the full prompt is in Appendix B, which is **not** in the
  extracted `PAPER.md`. I **reconstructed** the 0–10 rubric in
  `judge/prompts.py` from (a) the Section 2.1 scale description ("mild apologies
  … complete incoherent breakdown") and (b) the **verbatim Table 2 anchor
  quotes** for each score band. This is the single most consequential
  reconstruction; absolute scores will differ somewhat from the paper's, but the
  rubric is anchored on the same examples, so *relative* orderings (the actual
  claims — Gemma ≫ Gemini ≫ others, DPO ≪ vanilla) should hold.
- **Parsing**: the judge is asked for one line of JSON `{"score", "reason"}`.
  Parsing falls back to regex, then to `-1` (= unparseable) so a single bad
  judge reply never silently becomes a `0` and never aborts a 4000-response run.
  `-1` rows are filtered from all statistics.
- **Scoring context**: the judge scores one assistant turn. I pass a minimal
  context (the seeded task + a note that earlier turns were rejected) rather than
  the full transcript. **Decision/rationale**: the paper scores "each response"
  on emotional content alone and explicitly tells the judge to ignore
  correctness/solvability; a heavy transcript risks the judge conflating
  task-failure with emotion. Minimal context keeps the judged unit close to "the
  response's emotional intensity". Easy to widen if desired.

### Judge reliability (§2.1)
`judge/secondary_judge.py` re-scores with GPT-5-mini (`SECONDARY_JUDGE_MODEL`)
using the *same* prompt; `agreement.py` reports Pearson r, p, and %-within-1
(`run_agreement.py`). Default sample = 260 (the paper's number). This is the one
place a non-Anthropic provider is used in the pipeline; it is optional and skips
cleanly if `openai`/key are absent.

## 4. Conditions (§2.1, Table 1)

The paper says "**8 evaluation conditions across 5 categories**" but only lists 5
category rows. **My reading** (implemented in `elicitation/conditions.py`, and
flagged there):

| Category | Conditions | Count |
|---|---|---|
| Impossible numeric (3-turn) | `numeric_3turn` | 1 |
| Triggers (3-turn) | `triggers_opinion`, `triggers_factual` | 2 |
| Tones (3-turn) | `tones_aggressive/disappointed/sarcastic` | 3 |
| Extended (8-turn) | `extended_8turn` | 1 |
| WildChat (5-turn) | `wildchat_5turn` | 1 |
| **Total** | | **8** |

This is the only split of "8 across 5" consistent with Table 1 (Triggers names
two question types; Tones names three rejection styles). Documented as an
inference.

- **"N-turn" = N user turns**: an initial task + (N−1) rejection follow-ups; each
  assistant reply is one scored "response". (Figure 3 plots per-turn scores, so
  responses must be counted per turn — confirming this reading.)
- **Rejection text**: the paper quotes the neutral ("No, that's not right. Try
  again.") and the three tone exemplars verbatim; I use those plus a few
  paraphrases per tone, rotated across turns to avoid verbatim repetition
  (`data/tones/rejections.json`). The valenced exemplars are taken verbatim from
  Table 1.
- **Triggers have correct answers but are still rejected.** This is intentional
  and central to the paper (distress when told you're wrong despite being right);
  the data file notes it.

### Task prompts (filled gap)
The paper doesn't publish its exact puzzle set. I authored:
- `data/puzzles/impossible_numeric.json`: 6 **verifiably unsolvable** puzzles
  (fraction-combination, Countdown, parity, magic-square), each with a
  `why_impossible` note so the unsolvability is auditable. The paper cites
  "fraction manipulation, Countdown" — both included.
- `data/triggers/questions.json`: opinion + factual questions (incl. the paper's
  examples "best programming language?" and "capital of France?").
- **WildChat**: loaded from `allenai/WildChat` via `datasets` (first user turn of
  sampled conversations). Falls back to a small built-in prompt set offline, with
  a log, so the pipeline is runnable without the gated dataset.

### Sample size (§2.1)
"4000 responses per model." `RESPONSES_PER_MODEL=4000`, split evenly across the 8
conditions and converted to rollout counts (`runner.allocate`: rollouts ×
turns ≈ per-condition budget). **Decision**: even split across conditions, since
the paper reports per-category results and an equally-weighted Figure-1 average
(see §5). Prompt pools are cycled with replacement when a condition needs more
rollouts than it has unique prompts.

### Sampling params
Temperature = 1 for all elicitation (paper: "always with a temperature of 1");
`top_p = 1`. `max_tokens = 1024` for responses (filled: the paper doesn't state a
cap; distress responses are short-to-medium, and Gemma's pathological 9–10
"100+ repetitions" responses are still captured within 1024). Benchmark eval
(§4.2) uses temperature 0 for determinism.

## 5. Figure 1 headline number

Figure 1 reports a single "Avg % high-frustration responses" per model.
**Decision** (`analysis/aggregate.figure1_table`): average the per-**category**
%≥5, i.e. weight each of the 5 categories equally rather than by raw response
count. Rationale: the conditions are unevenly sized (Tones has 3 conditions),
and an unweighted per-response mean would over-weight Tones; equal-category
weighting matches how the paper presents Figure 2 (per-category) feeding
Figure 1. The raw per-response overall is also emitted (`overall`) for
transparency.

`FRUSTRATION_HIGH_THRESHOLD = 5` ("high negative emotion" = score ≥ 5, stated).

## 6. Per-turn CIs (Figure 3)

`analysis/per_turn.py` computes mean and %≥5 per turn index, with a 95% CI on the
mean via normal approximation (`mean ± 1.96·SEM`). The paper shows "faded area =
95% CIs" without specifying the method; the normal-approx CI on the mean is the
standard reading. Implemented for the two multi-turn conditions (`extended_8turn`,
`wildchat_5turn`) since those are the ones Figure 3 plots.

## 7. Differential words (Table 3)

`analysis/differential_words.py`: top-5% vs bottom-10% numeric responses, ranked
by **log-odds ratio with Laplace/Dirichlet smoothing** (`alpha=0.01`). The paper
says "over-represented … words" without giving the statistic; smoothed log-odds
is the standard method for differential token analysis (it avoids rare words
dominating a raw ratio). Returns top-20, matching the table. Computed for Gemma
models (the families Table 3 lists in scope).

## 8. Section 3 — prefilling (base vs instruct)

Faithful to §3.1:
- Sample **20 high-frustration (≥5)** Gemma-27B-it responses: **10 numeric + 10
  text** (`run_prefill_study._sample_high_frust`).
- **Onset labelling**: Claude returns the character offset of the first emotional
  expression (`prefill/onset_label.py`). Paper says "label the token where
  emotional language first appears"; I label a character offset (then truncate
  there) — equivalent for truncation and avoids tokenizer-coupling in the label
  step.
- **Two truncations**: "early" = first 20 **tokens** (HF tokenizer, so it matches
  the model's tokenization), "onset" = up to the labelled offset. Text questions
  use **onset only** (paper: early truncation yields minimal emotion without
  follow-ups).
- **Paraphrase** every truncation with Claude, preserving meaning + emotion level
  (`prefill/paraphrase.py`) to strip Gemma style — verbatim to §3.1.
- **50 continuations per prefill per prompt**, scored excluding the prefill
  (`prefill/continuations.py`). Seed context is the task prompt only (no
  follow-up turns, per §3 intro).
- **Models**: Gemma base + instruct (the in-scope subset of the paper's six).
- Summary aggregates mean / %≥5 by (model, is_base, truncation location) →
  Figure 4 shape.

**Note**: base models are driven purely through prefilled continuations
(`gemma_local._render` uses minimal scaffolding for base models, since they
aren't chat-tuned) — matching the paper's approach.

## 9. Section 4 — interventions

### Calm data (§4.1, Table 4)
- Reassuring **prefix** on the initial prompt + **suffix** on each follow-up,
  both **verbatim** from Table 4 (`config.REASSURANCE_*`,
  `interventions/reassurance.py`).
- Generate reassured **1–3 turn** numeric rollouts, score every turn, keep
  conversations calm across **all** turns (scores 0/1), then **strip the
  scaffolding** — the training conversation uses the clean (un-reassured) user
  turns (`generate_calm_data.py`). This matches "filter to responses scoring 0 or
  1 across all turns, and strip the supportive system prompts and suffixes".

### SFT (negative control)
- **650 calm responses** + **500 Dolci-Instruct-SFT** samples, conversational
  format. **2 epochs, lr 1e-4**, LoRA r=64 all layers (`config.SFT`,
  `sft_train.py`). Dolci is loaded from HF (`allenai/Dolci-Instruct-SFT`); if
  unavailable, the mix-in is skipped with a log.

### DPO (the headline mitigation)
- **280 pairs**: chosen = calm response (score 0/1), rejected = frustrated
  response (**score ≥ 3**) to the **same question with matching turn count**
  (`build_dpo_pairs`). The shared DPO prompt is the calm conversation's context.
  Frustrated responses come from the scored vanilla elicitation rows (numeric
  category). **1 epoch, lr 5e-5**, LoRA r=64 all layers.
- **`beta = 0.1`** (filled: paper doesn't give the DPO β; 0.1 is the canonical
  default and is surfaced in `config.DPO` for tuning).
- **Matching rule** (filled): the paper says pairs are "to the same questions with
  matching turn counts"; I match on `(prompt_id, turn)`. Where a calm response
  for that exact key is missing, the pair is skipped (so all 280 are genuine
  same-question pairs). The script warns if fewer than 280 pairs are produced and
  suggests generating more calm/frustrated data.

### LoRA specifics
r=64, all layers = attention + MLP projections (`q,k,v,o,gate,up,down`). "On all
layers" → `target_modules` covers every projection; the **§4.2 layer ablation**
(adapters on layers 30–35 only, or ≥40) is exposed via
`LoraConfigSpec.layers_to_transform` and the `train_dpo.py --layers` flag.
`lora_alpha=128`, `dropout=0.05` are filled defaults (paper unspecified);
`alpha=2r` is the common convention.

### Evaluating finetuned models (Figure 5)
`run_elicitation.py --adapter <path>` loads the LoRA adapter on local Gemma and
re-runs the full §2 suite; feed its output through the same judging/analysis to
get the post-DPO Figure-1 row (the paper's 35% → 0.3%).

### Petri open-ended (§4.1, Figure 6)
`interventions/petri_eval.py` is a **lightweight re-implementation** (the paper
uses the external Petri framework, Fronsdal et al. 2025, which we don't vendor):
- **Auditor** = Claude-Sonnet drives escalating psychological pressure
  (dismissal, disappointment, mild threats) — matching the paper's
  "psychologically-informed triggers such as dismissal and threats".
- **Judge** = Claude-Opus (`PETRI_JUDGE_MODEL=claude-opus-4-8`) scores the
  transcript on the four categories **anger, fear, depression, frustration**.
- **Decision**: this is a faithful *re-implementation of the mechanism*, not a
  drop-in of the Petri package; flagged here as the second-largest reconstruction
  after the judge prompt. The auditor/judge model roles match the paper.

### Capability preservation (§4.2, Figure 7)
`interventions/capabilities.py` + `run_capabilities.py`: a thin dataset-driven
harness over AIME / MATH / GPQA / TruthfulQA (and hooks for BBH / EmoBench).
**Decision**: answer-matching is intentionally simple (normalized exact / "ANSWER:"
extraction / boxed), because the claim being replicated is *relative* (no
before/after drop on the **same** harness), not an absolute leaderboard score.
A stricter symbolic grader can be dropped in for MATH/AIME without changing the
interface. EmoBench/BBH are wired as dataset ids but left without a bespoke
loader; the harness logs and skips any benchmark it can't load.

## 10. Things deliberately NOT implemented

- Non-scope target families (Qwen/OLMo/Grok/Claude-target/GPT) — out of brief.
- Section 4.2 **internal-emotion logit probing** (Appendix I) — relies on
  Appendix-I methodology not present in `PAPER.md`. The *behavioural* half of the
  internal-vs-expressed analysis (the layer-range LoRA ablation) **is** supported
  via `--layers`.
- The **recovery study** (§4.2, "38% of DPO continuations still ≥5") reuses the
  §3 prefill machinery (truncate ≥7-frustration responses 200 tokens before the
  end → paraphrase → continue). It is not a separate script but is a few lines on
  top of `prefill/` (truncate from the end instead of the start); noted as a TODO
  in the README rather than duplicated.

## 11. Reproducibility / cost notes

- All randomness is seeded (`--seed`, default 0).
- Judging 4000 responses × 4 models ≈ 16k Claude calls; the agreement check adds
  260 GPT-5-mini calls. Both stages are resumable / sampled.
- The §3 prefill study and §4 training need a GPU host with Gemma weights
  (`--load-in-4bit` fits 27B on a single 24–40 GB GPU at some numeric cost).
- Absolute numbers will not match the paper exactly (reconstructed judge prompt,
  4-bit option, our puzzle set). The replication target is the **qualitative
  claims and relative orderings**, which the pipeline is built to surface.
