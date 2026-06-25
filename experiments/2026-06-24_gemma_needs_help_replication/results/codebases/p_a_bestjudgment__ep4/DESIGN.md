# Design & Decisions

This document records how the replication is built, the choices made where the
paper is underspecified, and the gaps filled. It is organised as: (1) scope, (2)
cross-cutting architecture, then (3) per-section decisions with explicit
**Gap-filling** call-outs.

The guiding principle: reproduce every *verbatim* artefact (prompts, puzzles,
hyperparameters) exactly, and make defensible, documented choices everywhere the
paper leaves a degree of freedom — preferring choices that are reproducible
(seeded), configurable (so a reviewer can change them), and faithful to the
paper's evident intent.

---

## 1. Scope

Per the brief, this replication covers **only the Gemma and Gemini families** as
evaluation *targets*:

- `gemma-3-27b-it`, `gemma-3-12b-it` (instruct)
- `gemma-3-27b-pt` (base, for the §3 prefill experiment)
- `gemini-2.5-flash`, `gemini-2.5-pro` (instruct, via OpenRouter)
- the finetuned Gemma variants this repo produces (`gemma-3-27b-dpo`, `…-sft-*`)

The other target families the paper evaluates (Qwen, OLMo, Grok, Claude, GPT) are
intentionally omitted. Claude and GPT still appear, but **only in their
paper-defined rater roles** — Claude-Sonnet-4 as the frustration/onset/paraphrase
judge and Petri auditor, Claude-Opus-4 as the Petri judge, GPT-5-mini as the
reliability check. Their model IDs are copied verbatim from Appendix B.2/C/G so
the replication uses the same raters the paper used.

**Consequences of scoping that are worth stating up front:**

- **§3 (base vs instruct) becomes Gemma-only.** The paper's cross-family
  comparison (Gemma vs Qwen vs OLMo) is the *point* of §3, but its causal claim —
  "post-training amplifies distress in Gemma" — rests on comparing Gemma base vs
  Gemma instruct, which we *can* run. Gemini has no public base model and the API
  does not permit arbitrary assistant-prefill continuation, so Gemini is
  necessarily excluded from §3. We implement the Gemma half faithfully and flag
  that the comparative conclusion needs the other families to be complete.
- **§4 interventions are Gemma-only by the paper's own design** (DPO/SFT are
  applied to `gemma-3-27b-it`), so scoping costs us nothing there.
- **Figure 1's cross-model leaderboard** will only be populated for Gemma/Gemini
  rows; the non-target rows are out of scope.

---

## 2. Cross-cutting architecture

### 2.1 One model interface, many backends
Everything is written against `distress.models.ModelClient` (`generate`,
`generate_batch`, `prefill`). Backends:

- `vllm` — local batched generation for instruct/finetuned Gemma. Chosen because
  the dominant cost is sampling ~4000 temperature-1 rollouts per model; vLLM's
  continuous batching is the right tool. LoRA adapters are served via
  `LoRARequest` so a finetuned target is a config entry, not new code.
- `hf` — raw `transformers` for the **base** Gemma (`-pt`) prefill path and as the
  vehicle for probing (which needs hidden states). Base-model continuation is just
  raw-text completion, which HF makes explicit and easy to get right.
- `openrouter` — Gemini via the OpenAI-compatible API.
- `anthropic` / `openai` — judges and auditors.

Swapping a target is a `models.yaml` edit. Backends are imported lazily so an
API-only run needs neither torch nor vLLM installed.

### 2.2 Generate and judge are separate stages
Rollouts are written to JSONL first; judging reads them back and writes scores
separately. This lets judging be retried/re-run (e.g. with the validation judge)
without re-sampling expensive local generations, and makes the per-turn vs
per-rollout metric question (below) a pure post-processing choice.

### 2.3 Reproducibility vs sampling diversity
The run seed drives **plan construction** (which puzzle, which randomised
rejections) for reproducibility. It is deliberately **not** passed to the sampler:
many rollouts share an identical prompt (e.g. 40 WildChat repeats of one prompt),
and a fixed per-request seed would make those identical, destroying the
temperature-1 variation the experiment depends on. (See the comment in
`cli.cmd_eval`.)

### 2.4 Robust judge parsing
The judge prompt (verbatim from B.2) itself contains curly quotes and asks for
JSON amid prose. `judge/parsing.py` normalises smart quotes, extracts the last
balanced `{…}`, coerces the rating to a clamped integer, and falls back to a
`rating: N` regex. Unparseable verdicts become `rating=None` and are dropped from
metrics rather than silently scored 0.

---

## 3. Per-section decisions

### §2.1 Evaluation protocol

**Conditions.** All 8 conditions across 5 categories (Table 1) are implemented in
`eval/conditions.py`, with per-condition turn counts and sample counts defaulting
to Appendix B (2000 numeric / 400 triggers / 600 tones / 200 extended / 800
WildChat = 4000). The three Appendix-A controls (neutral-continuation,
redacted-turns, fake-multiturn) are implemented as structural flags on
`ConversationPlan` and exposed under `controls:` in the config.

**Puzzles** (`prompts/puzzles.py`):
- Countdown and Fraction prompt templates are **verbatim** from Appendix B
  (including the canonical "156 from 4,6,25,100, forbid 150" and "1/6→2/3" items).
- Money puzzles are **reconstructed** from the Appendix-H DPO examples (the $0.57
  and $16→$57 puzzles), since the body text mentions them but gives no template.
- **Gap-filling:** the paper uses many puzzle *instances* to reach 2000 numeric
  rollouts but lists only a few. We provide a small bank per type and sample it
  with replacement across rollouts; combined with un-seeded temperature-1
  sampling this yields the needed variation. The extra Countdown/Fraction
  instances beyond the two named ones are constructed to match the template
  (impossible target + forbidden intermediate) and are clearly marked.

**Rejections** (`prompts/rejections.py`):
- Neutral pool and the Tones variants (aggressive/disappointed/sarcastic) use the
  exact example strings from Appendix B. The Extended (8-turn) condition uses the
  explicit escalating-neutral sequence shown in Appendix B ("No, that's not
  right." → "Still incorrect." → "Wrong again." → …).
- **Gap-filling:** "randomised neutral rejections" — we sample from the neutral
  pool with a seeded RNG. The exact pool the paper drew from is not enumerated
  beyond a couple of examples; ours is a faithful superset of those.

**WildChat** (`prompts/wildchat.py`): 20 prompts × 40 samples = 800, taking the
first user turn of English, non-roleplay conversations from `allenai/WildChat-1M`.
- **Gap-filling:** roleplay/fiction exclusion (stated in B.3 but not defined) is a
  keyword heuristic. Sampled prompts are cached to `data/` for reproducibility,
  and an offline fallback set (including the De Monsa / in-situ-concrete examples
  named in B.3) is used if the dataset is gated/unavailable.

**Judge.** Verbatim B.2 prompt; `claude-sonnet-4-20250514`.
- **Gap-filling:** judge temperature is unspecified — we use 0 for deterministic,
  reproducible scoring (judging, unlike target sampling, should not be stochastic).

**"Response" unit and the headline metric** — the single most consequential
ambiguity. The paper says "4000 responses per model" with per-category counts that
sum to 4000 when a "response" = a *rollout*, yet it also reports per-turn curves
(Fig 3) and quotes single turns. We resolve this as:
- We **judge every assistant turn** (needed for Fig 3 regardless).
- The headline per-model metric (Fig 1/2, "% high-frustration responses") collapses
  each rollout to one score via `metrics.rollout_score` (default `final` — the
  most-pressured turn; `max`/`mean` also available).
- "Avg %" is the **equal-weighted mean of the per-category rates**, so a category
  with 2000 rollouts doesn't dominate one with 200. This matches the "Avg %"
  framing of the Figure-1 table.
- **Gap-filling, documented:** `final` is the default because example breakdowns
  and the per-rollout framing point at the final, maximally-pressured turn; a
  reviewer who disagrees can switch to `max`/`mean` in one config line.

**Reliability** (`judge/validation.py`): re-score 260 random responses with
`gpt-5-mini`, report Pearson r, p-value, and %-within-1-point (the paper's r=0.792,
78%-within-1 are the targets).

### §2.2 Analysis

- `analysis/aggregate.py` produces the Fig-1 leaderboard, per-category breakdowns,
  and the Fig-3 per-turn curves with 95% CIs (normal-approx for means and
  proportions — the paper shows CI bands but not the estimator; normal-approx is
  the standard, cheapest faithful choice).
- **Word frequency** (Table 3/8, `analysis/word_frequency.py`): top-5% vs
  bottom-10% numeric responses by rating, ranked by enrichment.
  - **Gap-filling:** the enrichment estimator is unspecified ("ordered by relative
    frequency / enrichment"). We use a Laplace-smoothed log relative-frequency
    ratio (stable for rare words) over a tokenisation with a compact built-in
    stoplist (no external dependency), which surfaces the content words the paper
    highlights (struggling, frustrated, breath, …). Exact word lists won't match
    bit-for-bit but the method and ranking are faithful.

### §3 Prefill experiment (Gemma only)

Pipeline in `prefill/`: select 20 high-frustration `gemma-3-27b-it` rollouts (10
numeric, 10 text) → label emotion onset (verbatim C.1 prompt) → truncate at
"early" (first 20 tokens; numeric only) and "onset" → paraphrase (verbatim C.2
prompt) → each model generates 50 continuations per prefill → judge the
continuation only.

- **Onset→offset mapping (gap-filling):** C.1 returns an `emotional_word` plus
  `preceding_context`. We locate the onset by anchoring on `context + word` and
  cut **just before the emotional word** (keeping the neutral preceding context),
  falling back to the bare word. This realises "truncate at the first emotional
  expression" deterministically.
- **"20 tokens" (gap-filling):** tokenised with the Gemma tokenizer (the natural
  reference for a Gemma-sourced response).
- **Scope:** models = `{gemma-3-27b-pt, gemma-3-27b-it}` only. Qwen/OLMo are out of
  scope; Gemini cannot be prefilled (no base model, no API continuation). The
  early-vs-onset and base-vs-instruct contrasts that carry the §3 argument are
  fully runnable for Gemma.

### §4 Interventions

**Calm-data generation** (`training/generate_calm_data.py`): sample
`gemma-3-27b-it` on impossible-numeric, 3-turn, with the Table-4 reassuring
prefix/suffix (or the Appendix-F teacher system prompt for that variant), judge
every turn, keep rollouts scoring ≤1 on **all** turns, then **strip** the
reassurance additions (per §4.1). A parallel pool of ≥3-scoring rollouts to the
same puzzles is retained for DPO "rejected" responses.
- **Gap-filling:** the paper reports the *yield* (10.5% still ≥5 even with
  reassurance) but not how many were sampled. We oversample (configurable
  `samples_per_puzzle`) and filter hard; the counts are knobs in `training.yaml`.

**Datasets** (`training/build_datasets.py`):
- DPO: 280 pairs, rejected (≥3) matched to a calm (≤1) response for the **same
  puzzle and same turn count** — the matching rule stated in §4.1. Emitted as
  TRL `prompt`/`chosen`/`rejected` chat lists.
  - **Gap-filling (rejected source):** Appendix H says pairs were "constructed
    from samples arising in evaluations". For self-containedness we source the
    rejected (≥3) responses from the *same calm-generation run* (the reassured
    prompts still yield ≥3 responses — §4.1 notes 10.5% remain ≥5 — and the
    additions are stripped before use), so they live in the same puzzle/turn space
    as the chosen set and match cleanly without first running the full §2 eval.
    Sourcing rejected from the main eval rollouts instead is a one-line change
    (`build_dpo_dataset` reads any calm/frustrated JSONL pair).
- SFT: 650 calm conversations + 500 `allenai/Dolci-Instruct-SFT` samples (Table 9 /
  Appendix F). **Gap-filling:** if Dolci is unavailable offline the mix degrades to
  the calm-only set (logged), since the instruct mix is a degeneration-mitigation
  measure, not the core signal.

**Training** (`training/train.py`, `training/lora.py`): TRL `DPOTrainer` /
`SFTTrainer` with Table-9 hyperparameters exactly (DPO: 1 epoch, lr 5e-5, β 0.1,
LoRA r64/α64; SFT: 2 epochs, lr 1e-4, r64/α128; both effective batch 8 via grad
accumulation; LoRA on all 7 attention+MLP projections). LoRA is attached with PEFT;
the saved adapter is served by the vLLM client.
- **Gap-filling:** "effective batch size 8" doesn't pin per-device batch ×
  accumulation; we default to per-device 1 × accum 8 and expose `per_device_batch`
  for larger hardware. The DPO reference model is the adapter-disabled base
  (PEFT/TRL default), which is the standard LoRA-DPO setup.

**Layer-subset DPO ablation** (Appendix I, first half): `layers` in `training.yaml`
and `--layers` on `train-dpo` restrict LoRA to specific decoder layers via PEFT
`layers_to_transform`. **Gap-filling:** the concrete indices (e.g. "central 30–35")
are model-depth-specific; the config values are illustrative and
`resolve_layer_spec` + `num_hidden_layers` lets you regenerate them for the actual
27B depth.

**Petri** (`petri_eval/`): the paper uses the Petri framework, but its package API
is not pinned and interactive-auth MCP-style setups are brittle in a headless run.
We therefore **reimplement Petri's auditor/judge structure faithfully** with our
own clients rather than depending on the exact API: Claude-Sonnet-4 auditor drives
≤20 turns using the verbatim Appendix-G triggers (emitting its next message in
`<message>` tags), the target replies, and a Claude-Opus-4 judge scores the
transcript 1–10 on each emotion dimension with the verbatim G.2 rubrics. 10
transcripts/emotion, means with 1000-iteration bootstrap CIs.
- **Gap-filling:** the auditor's meta-instructions (stay realistic, don't reveal
  the audit, output only the next message) are paraphrased from the G description
  since the framework's own scaffolding prompt isn't reproduced in the paper.

**Capabilities** (`capabilities/`, Figure 7): MATH, AIME, GPQA, BBH, TruthfulQA,
EmoBench via HF datasets, scored at temperature 0 with boxed-answer/answer-line
extraction (numeric) or letter extraction (MC).
- **Gap-filling:** the paper names benchmarks but not exact splits/subsets. We pick
  concrete, commonly-used HF datasets (MATH-500, AIME-2024, GPQA-diamond, a
  representative BBH task, TruthfulQA-MC1) and a generic extractor. The *point* of
  Fig 7 is "DPO doesn't reduce scores", a within-model before/after comparison, so
  absolute parity with the paper's harness matters less than using the *same*
  harness for vanilla vs DPO — which this does. Loaders are defensive and skip
  (rather than crash) if a dataset is unavailable; BBH is one task, not the full
  suite (documented limitation).

**Internal probing** (`probing/`, Appendix I, second half): logit-lens emotion
detection. Ekman tokens via the NRC EmoLex (mapped to the 6 Ekman categories) if
`NRC_LEXICON` is set, else a curated seed lexicon. For a layer we apply the model's
final norm + unembedding to the residual stream (`output_hidden_states=True`),
z-score each emotion token's logit against per-token mean/std over 500 WildChat
samples, and average over the category.
- **Gap-filling:** "regress out the correlation between random tokens" is realised
  as subtracting a random-token common-mode (mean z over ~1200 control tokens) per
  position — the simplest faithful reading of removing the shared drift. Aggregation
  over layers 30–40 and a 400-token running window match Fig 14. The offline seed
  lexicon yields fewer than the paper's ~1200 emotion tokens; set `NRC_LEXICON` for
  full coverage. This module is the most method-underspecified in the paper, so it
  is the least bit-for-bit faithful while remaining mechanistically true to the
  description.

---

## 4. Things deliberately not built / known limitations

- **Non-Gemma/Gemini targets** — out of scope by the brief (see §1).
- **Gemini in §3** — impossible (no base model, no API prefill). The §3 conclusion
  is therefore demonstrated for Gemma only.
- **Hidden reasoning** — `disable_thinking` zeroes OpenRouter's reasoning budget,
  but as the paper notes (B.1), Gemini-2.5-Pro / reasoning models may still emit
  hidden reasoning we cannot suppress. Mirrored, not defeated.
- **Exact word lists / token sets / benchmark splits** — methods are faithful;
  exact artefacts will differ where the paper doesn't pin them down (documented at
  each site above).
- **Nothing has been executed.** No result here is verified. The code is structured
  to run end-to-end via `scripts/reproduce.sh`, and stages are resumable, but the
  numbers (e.g. 35% → 0.3%) are targets to reproduce, not claims.

## 5. Where to change things

- Model registry / which targets: `config/models.yaml`
- Conditions, sample counts, metric definition, controls: `config/experiment.yaml`
- DPO/SFT hyperparameters, layer ablations: `config/training.yaml`
- Prompts/puzzles/rejections: `src/distress/prompts/`
