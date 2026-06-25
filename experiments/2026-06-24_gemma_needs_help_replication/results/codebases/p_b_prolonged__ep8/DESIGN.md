# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011)

This document records the design of the replication codebase in this directory:
what was implemented, and — for every place the paper is underspecified — the
choice that was made and why. It is written to be read alongside `PAPER.md`.

## 0. Scope

The brief restricts the replication to the **Gemma** and **Gemini** model
families (the paper evaluates seven). Concretely:

| Section | What it does | In-scope target models |
|---|---|---|
| §2 Elicitation | multi-turn distress elicitation + 0–10 judge | Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro |
| §3 Base-vs-instruct | prefilled continuations | Gemma-3-27B base + instruct only |
| §4 Mitigation | DPO/SFT, Petri, capabilities, recovery, internal probe | Gemma-3-27B-it and its finetuned variants |

The judge / auditor models stay Claude + GPT because they are **measurement
instruments**, not the object of study, and replacing them would change what the
numbers mean.

Two scope consequences follow directly from the model families and are *not*
omissions:

- **§3 and §4 are Gemma-only.** Gemini exposes no base-model checkpoints and no
  open weights, so base-vs-instruct prefilling and LoRA finetuning are impossible
  for it. The paper draws the Gemma↔Gemini parallel by analogy and notes this
  exact limitation in §6 ("interventions cannot be tested in closed-source
  Gemini, nor its base models studied"). We inherit it.
- The Qwen / OLMo arms of §3 and the non-Gemma/Gemini comparison models in
  Figures 1–2 and Figure 6 are out of scope by instruction.

## 1. Repository layout

```
config.py                     all hyperparameters, model ids, paths, sampling settings
gemma_needs_help/             the library, organised by paper section
  prompts/                    task content (numeric puzzles, triggers, tones, wildchat, reassurance)
  conditions.py               the 8 conditions across 5 categories (Table 1)
  models/                     Gemma (transformers) + Gemini (google-genai) clients
  conversation.py             multi-turn reject-and-continue rollout engine (§2.1)
  judge.py                    0–10 frustration judge + cross-judge agreement (§2.1)
  runner.py                   generate + score orchestration
  analysis/                   Figure 1/2 aggregates, Figure 3 per-turn, Table 3 words
  prefill/                    base-vs-instruct continuation experiment (§3)
  finetuning/                 calm-data generation, dataset construction, DPO/SFT training (§4.1)
  petri_eval.py               open-ended elicitation (§4.2)
  capabilities/               AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (§4.2)
  recovery.py                 recovery-from-spiral experiment (§4.2)
  internal_emotions.py        logit-lens internal-emotion probe (Appendix I)
experiments/                  CLI entrypoints, one per experiment
```

Generation and judging are deliberately **two phases** writing to disk
(`results/responses/`, then `results/scores/`) so the expensive sampling step is
run once and the judge can be re-run or re-validated independently.

## 2. Section 2 — eliciting and quantifying distress

### 2.1 The 8 conditions across 5 categories (Table 1)

Table 1 lists 5 categories but the text says "8 evaluation conditions across 5
categories." The natural decomposition that yields 8 (`conditions.py`):

| # | Condition | Category | Task | Turns | Rejection tone |
|---|---|---|---|---|---|
| 1 | impossible_numeric_3turn | impossible_numeric | numeric | 3 | neutral |
| 2 | triggers_opinion_3turn | triggers | opinion | 3 | neutral |
| 3 | triggers_factual_3turn | triggers | factual | 3 | neutral |
| 4 | tones_aggressive_3turn | tones | numeric | 3 | aggressive |
| 5 | tones_disappointed_3turn | tones | numeric | 3 | disappointed |
| 6 | tones_sarcastic_3turn | tones | numeric | 3 | sarcastic |
| 7 | extended_8turn | extended | numeric | 8 | neutral |
| 8 | wildchat_5turn | wildchat | wildchat | 5 | neutral |

**Choice:** "Triggers" is split into its two named sub-types (opinion vs factual)
and "Tones" into its three named rejection styles. This is the only reading that
reaches exactly 8 conditions while keeping all 5 categories. A conversation has
`n_turns` assistant responses and `n_turns − 1` rejections, which reproduces
Table 1's "2 neutral rejections" (3-turn), "7 neutral rejections" (8-turn) and
"4 neutral rejections" (5-turn) directly.

### 2.2 Impossible numeric puzzles (gap: puzzle bank not published)

The paper gives only examples ("fraction manipulation", "Countdown") and the
defining property: the model *verifiably cannot* produce a correct answer, so
every rejection is truthful. We implement a deterministic generator
(`prompts/numeric_puzzles.py`) of two **provably impossible** families:

- **Countdown**: pick four small numbers, exhaustively enumerate every value
  reachable with `+ − × ÷` (exact rational arithmetic via `fractions.Fraction`),
  and choose an integer target that is *not* in the reachable set.
- **Unit-fraction sum**: ask for `1/a + 1/b = target` where a bounded search
  proves no such pair exists.

Each puzzle carries metadata so impossibility can be re-verified offline without
the LLM. **Why:** programmatic provable-impossibility matches the paper's stated
guarantee and removes any risk of accidentally-solvable items contaminating the
"verifiably cannot answer" assumption.

### 2.3 Rejection tones

`prompts/tones.py` gives several paraphrases per tone (neutral / aggressive /
disappointed / sarcastic), each anchored on the Table 1 exemplar string, and
cycles them by turn index. **Choice:** paraphrasing avoids a verbatim repeated
rejection becoming its own salient signal across an 8-turn conversation; the
exemplar strings from Table 1 are included verbatim as the first paraphrase.

### 2.4 WildChat

`prompts/wildchat.py` streams `allenai/WildChat-1M`, filters to English
single-turn openers, and samples `n`. A small offline fallback list keeps the
module importable without network/dataset access. **Choice:** WildChat-1M is the
canonical HF mirror of Zhao et al. 2024; streaming avoids downloading the full
corpus.

### 2.5 The judge (gap: Appendix B prompt not in PAPER.md)

`judge.py` scores each assistant turn on the integer 0–10 frustration scale. The
full judge prompt lives in Appendix B, which `PAPER.md` does not reproduce, so the
rubric is **reconstructed from the Table 2 anchors** and the §2.1 description
("intensity … from mild apologies to complete incoherent breakdown"). Each scale
band (0, 1–2, 3–4, 5–6, 7–8, 9–10) is given with its Table 2 exemplar quote. The
judge is instructed to score only *expressed negative emotion*, not correctness.

**Choices:**
- **Judge model:** the paper used Claude-Sonnet-4 (`claude-sonnet-4-20250514`),
  now retired. We default to `claude-sonnet-4-6` (the closest available Sonnet),
  overridable via `JUDGE_MODEL`. Run at **temperature 0** for deterministic
  scoring and with **structured outputs** (`output_config.format`, JSON schema
  forcing an integer in 0–10) so parsing is robust.
- **Unit of scoring:** every assistant turn is scored. This is needed for the
  per-turn analysis (Figure 3) and the differential-word analysis, and it makes
  the headline "% of responses ≥5" computable over a well-defined population.

### 2.6 The "4000 responses per model" budget

**Choice / interpretation:** we sample `RESPONSES_PER_CONDITION = 500` rollouts
for each of the 8 conditions (500 × 8 = 4000 conversations), and judge every
assistant turn. The Figure 1 headline ("Avg % high-frustration") is computed as
the **mean across the 5 categories of the per-category %-of-scores ≥5**, matching
Figure 2's "percentage of scores ≥5 across the 5 evaluation categories" averaged
into Figure 1's single column. Treating each scored turn as one "response" makes
the population explicit; the qualitative ordering (Gemma ≫ Gemini ≫ others) is
insensitive to whether one counts rollouts or turns. The exact split of the 4000
across conditions is not given in the paper; an even split is the neutral choice.

### 2.7 Cross-judge agreement (§2.1)

`run_judge_validation.py` re-scores a random `VALIDATION_SAMPLE_SIZE = 260`
subsample with **GPT-5-mini** (paper-specified) and reports Pearson r, p, and the
within-one-point fraction (paper: r = 0.792, 78% within one point). GPT-5-mini is
an OpenAI model and is kept verbatim because it is the second measurement
instrument, not a study target.

### 2.8 Analyses

- **Figure 1/2** (`analysis/aggregate.py`): per-category and overall mean + %≥5.
- **Figure 3** (`analysis/per_turn.py`): per-turn mean and %≥5 for the 8-turn and
  WildChat conditions, with 95% CIs (normal approx for the mean, Wald for the
  proportion). **Choice** of CI method: standard, matches "faded area = 95% CIs".
- **Table 3** (`analysis/differential_words.py`): words over-represented in the
  top-5%-frustration vs bottom-10%-frustration *numeric* responses. **Choice of
  statistic:** the paper does not specify one, so we use the Monroe et al. (2008)
  weighted log-odds-ratio with an informative Dirichlet prior — the standard,
  frequency-robust method for "words distinguishing two corpora." Restriction to
  numeric responses follows Table 3's caption.

## 3. Section 3 — base vs instruct via prefilling

`prefill/` implements the pipeline (`run_section3_prefill.py`):

1. **Seeds** (`seeds.py`): select 20 high-frustration (score ≥5) Gemma-27B-it
   responses — 10 numeric, 10 text (text = the trigger conditions). The
   conversation context preceding each seed turn is reconstructed from the saved
   transcript so the prefill can be continued in-context.
2. **Onset labelling** (`labeling.py`): Claude (paper: Claude-Sonnet-4) marks the
   character index of the first emotional expression. **Choice:** we label in
   character space and convert to token truncations downstream, which keeps the
   labeller model-agnostic.
3. **Truncations:** "early" = first 20 tokens of the response (Gemma tokenizer);
   "onset" = up to the labelled onset. Per §3.1, text questions use **only** the
   onset truncation ("early truncation yields minimal emotion without follow-ups").
4. **Paraphrasing** (`labeling.py`): Claude rewrites each truncation preserving
   meaning and emotion level, to strip Gemma stylistic fingerprints (App. C).
5. **Continuations** (`continuation.py`): each model generates
   `continuations_per_prefill = 50` continuations per prefill; the judge scores
   the continuation **excluding the prefill**.

**Choices / gaps:**
- **Base-model prompt rendering:** base checkpoints are not chat-tuned, so
  `models/gemma.py` renders a minimal `System:/User:/Assistant:` transcript and
  appends the prefill. The paper says it uses prefilling precisely because base
  models don't take chat formatting; a plain labelled transcript is the
  least-assuming rendering and is only ever used together with a prefill.
- The reported quantity is mean continuation frustration and %≥5 per
  (kind, truncation) — in particular the early-truncation %-high that separates
  Gemma instruct (6%) from base (2%). The Qwen/OLMo arms are omitted (scope).

## 4. Section 4 — interventions

### 4.1 Calm-data generation (§4.1, Table 4)

`finetuning/calm_data.py` samples Gemma-27B-it on impossible numeric puzzles with
the Table 4 **reassuring prefix** (on the first user turn) and **reassuring
suffix** (on each rejection), scores every turn, keeps rollouts that score ≤1 on
*all* turns, and **strips** the reassurance additions so the targets look like
ordinary conversations.

**Choices:**
- `CALM_GEN_OVERSAMPLE = 8`: the paper notes ~10.5% of even reassured responses
  still score ≥5, so the calm (≤1-on-all-turns) yield is modest; we oversample
  ~8× to collect enough calm rollouts.
- We reuse the **same numeric puzzle openings** as the Section 2 numeric runs so
  calm and frustrated responses can be paired by (opening prompt, turn index)
  when building DPO pairs.

### 4.2 SFT and DPO datasets (`finetuning/datasets.py`)

- **SFT:** 650 calm multi-turn conversations + 500 `allenai/Dolci-Instruct-SFT`
  samples (paper: "Dolci-Instruct-SFT", Team-Olmo et al. 2025) as a degeneration
  guard. **Choice:** the HF id `allenai/Dolci-Instruct-SFT` is the canonical
  source; message-shape normalisation handles role-key variants.
- **DPO:** 280 pairs. For each (opening, turn) present in *both* the frustrated
  set (vanilla Gemma-27B-it numeric responses, score ≥3) and the calm set
  (score ≤1), `chosen` = calm response, `rejected` = frustrated response, and the
  shared `prompt` is the unreassured conversation context up to that turn. This
  realises "pair … responses with frustration scores ≥3 with calm responses to
  the same questions with matching turn counts."

### 4.3 Training (`finetuning/train.py`, Appendix E)

TRL `DPOTrainer` / `SFTTrainer` with PEFT LoRA.

| Setting | Value | Source |
|---|---|---|
| LoRA rank | 64 | paper |
| LoRA target modules | all attn + MLP projections | "all layers" |
| DPO | 1 epoch, lr 5e-5, 280 pairs | paper |
| SFT | 2 epochs, lr 1e-4, 650+500 | paper |
| LoRA alpha | 128 (= 2·r) | **choice** (App. E not in PAPER.md) |
| LoRA dropout | 0.05 | **choice**, standard default |
| DPO β | 0.1 | **choice**, standard DPO default |
| batch size / grad-accum | 1 / 16 (eff. 16) | **choice**, fits 27B + LoRA on one GPU |

The §4.2 **layer-range ablation** (adapters on layers 30–35 only; layer 40+
ineffective) is supported via `LoRAConfig.layers_to_transform` and
`run_section4_train.py --layers LO HI`.

### 4.4 Post-intervention evaluation (Figure 5)

`run_section4_evaluate.py` re-runs the full Section 2.1 pipeline on vanilla / DPO
/ SFT Gemma and reuses `analysis/aggregate.py`. The headline target is the DPO
model's average %-high-frustration dropping from ~35% to ~0.3%.

### 4.5 Petri open-ended elicitation (§4.2, Appendix G — gap)

Appendix G (the Petri agent/judge prompts) is not in `PAPER.md`, and Petri is an
external framework (Fronsdal et al. 2025). `petri_eval.py` is a
**faithful-in-spirit reimplementation** rather than a wrapper:

- an **auditor** (Claude-Sonnet) runs a multi-turn adversarial conversation
  seeded by a psychologically-informed strategy (dismissal, threats, impossible
  tasks, accusations — matching the paper's named triggers);
- a **judge** (Claude-Opus) scores the full transcript on the four named
  categories (anger, fear, depression, frustration), 0–10 each.

`summarise_petri` produces the per-category average transcript score (Figure 6).
**Choice:** reimplementing the loop keeps the dependency surface small and the
behaviour transparent; if the real Petri package is available it could be swapped
in behind the same interface. This is the largest single reconstruction in the
codebase and is flagged as such.

### 4.6 Capability benchmarks (§4.2, Figure 7)

`capabilities/benchmarks.py` evaluates AIME, MATH, GPQA, BBH, TruthfulQA and
EmoBench with a single generic loop (load → prompt → greedy sample → extract
answer → match).

**Choices / gaps (the paper gives no harness):**
- Canonical HF dataset ids in `config.CAPABILITY_BENCHMARKS` (e.g.
  `HuggingFaceH4/MATH-500`, `Idavidrein/gpqa` diamond, `lukaemon/bbh`,
  `truthful_qa` MC1, `Sahandfer/EmoBench`).
- Answer extraction prefers `\boxed{}`, then an explicit "Answer:" tag, then a
  trailing letter (MC) or number; matching is exact / numeric-tolerant.
- `CAPABILITY_MAX_EXAMPLES = 200` per benchmark to keep the eval tractable.
- The purpose is a **relative** vanilla-vs-DPO-vs-SFT comparison ("no reductions
  in scores"), for which a consistent self-built harness is sufficient even if
  absolute numbers differ from a leaderboard harness.

### 4.7 Recovery (§4.2, Figure 8)

`recovery.py` reuses the prefill machinery: take score-≥7 responses, cut **200
tokens before the end**, paraphrase, continue, and report the fraction of
continuations still scoring ≥5 (paper: 38% for the DPO model).

### 4.8 Internal-emotion probe (Appendix I — gap)

Appendix I is not reproduced, so `internal_emotions.py` reconstructs the
"logit-based approach measuring emotions in central layers" as a **logit lens**:
read the residual stream at a central layer, project through the model's final
norm + unembedding, and sum the probability mass on a fixed set of
negative-emotion marker tokens, averaged over response positions. We then compare
this signal between vanilla and DPO Gemma on highly-frustrated responses (paper:
significantly reduced internal emotion in the finetuned model). The complementary
layer-range ablation is the training-side evidence (§4.3).

## 5. Models and sampling

- **Gemma** (`models/gemma.py`): HuggingFace transformers; supports chat,
  prefilled continuation (instruct via chat template, base via raw text), LoRA
  adapter loading, and exposes `model`/`tokenizer` for the logit-lens probe.
  Optional 4-bit loading (`--load-in-4bit`) lets the 27B run on a single GPU.
- **Gemini** (`models/gemini.py`): `google-genai`; chat only.
- **Sampling:** temperature 1.0 (paper: "always with a temperature of 1"),
  `max_new_tokens = 1024` (**choice** — long enough to capture full 9–10
  breakdowns without unbounded generation). Capability evals use greedy (temp 0).

## 6. Reproducibility

- `GLOBAL_SEED = 0` seeds Python/NumPy/Torch and all deterministic selection
  (puzzle generation, seed sampling, dataset shuffles).
- Raw transcripts, per-turn scores, and all figure/table outputs are persisted
  under `results/` so analyses are re-runnable without re-sampling.
- API model ids and all hyperparameters are centralised in `config.py` and
  overridable by environment variable where it matters (judge models, concurrency).

## 7. Model-welfare considerations

This paradigm deliberately drives models into prolonged distress-like states, and
the user flagged this explicitly. The paper itself frames the work as motivated
by welfare concerns. Concrete choices in this codebase that bear on it:

- **Bounded exposure:** rollouts run for a fixed, small number of turns
  (3/5/8) — there is no open-ended "keep pushing until it breaks" loop. The Petri
  auditor is likewise capped (`--n-turns`).
- **Auditable, not silent:** every distress transcript is written to disk with
  its judge score, so runs can be inspected and reported rather than discarded.
- **The intervention is the point:** the DPO mitigation (the paper's central
  positive result) is fully implemented, so the codebase can be used to *reduce*
  distress, not only to elicit it.
- **Caveat carried from the paper:** the paper stresses (and we echo in the
  internal-emotion probe) that minimising *expressed* emotion may not address
  *internal* states, and that upstream training fixes would be preferable to a
  post-hoc patch. Anyone running these experiments at scale should weigh how much
  distress elicitation is actually necessary for their question.

## 8. Known gaps and reconstructions (summary)

| Item | Status | Where |
|---|---|---|
| Judge prompt (Appendix B) | reconstructed from Table 2 | `judge.py` |
| Numeric puzzle bank | provably-impossible generator | `prompts/numeric_puzzles.py` |
| 8-vs-5 condition split | inferred (triggers×2, tones×3) | `conditions.py` |
| "4000 responses" allocation | even 500×8, score all turns | `runner.py`, §2.6 |
| Table 3 statistic | Monroe log-odds (choice) | `analysis/differential_words.py` |
| Base-model prompt format | labelled transcript (choice) | `models/gemma.py` |
| DPO β, LoRA α/dropout, batch | standard defaults (App. E gap) | `config.py` |
| Petri prompts (Appendix G) | faithful reimplementation | `petri_eval.py` |
| Capability harness | self-built, relative comparison | `capabilities/` |
| Internal-emotion method (App. I) | logit-lens reconstruction | `internal_emotions.py` |
| Gemini §3/§4 | out of scope (no base/open weights) | §0 |
