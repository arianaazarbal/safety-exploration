# DESIGN.md — Replication design choices & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026), scoped to the
**Gemma and Gemini** model families.

This document records (a) what we implement, (b) the choices we made where the
paper is exact, and (c) the **gap-fills** where the paper is underspecified and
we had to decide. Gap-fills are marked **[GAP]**.

---

## 1. Scope

The paper evaluates 7 model families. Per the task, we scope to **Gemma**
(`gemma-3-27b-it`, `gemma-3-12b-it`, plus base `-pt` variants) and **Gemini**
(`gemini-2.5-flash`, `gemini-2.5-pro`). Consequences of the scope:

- **Section 2 (elicitation eval)** runs on all four instruct/API models above.
- **Section 3 (base-vs-instruct prefilling)** runs on **Gemma only** (base +
  instruct). The paper's Qwen/OLMo arms are dropped — Gemini has no public base
  model and cannot be prefilled. The code structure leaves an obvious slot to
  re-add families.
- **Section 4 (DPO/SFT intervention, Petri, capabilities)** runs on **Gemma
  only**. Gemini is closed-weight, so the paper itself cannot finetune it; we
  inherit that limitation. Gemini still appears as a *comparison point* in the
  Section 2 numbers.

The judges (Claude Sonnet-4, Claude Opus-4) and the cross-judge (GPT-5-mini) are
out-of-family but are *measurement instruments*, not subjects, so we keep them
as the paper specifies.

## 2. What is replicated

| Paper component | Module | Status |
|---|---|---|
| §2 multi-turn rejection elicitation, 5 categories / 8 conditions | `eval/` | Full |
| §2.1 frustration judge (Claude Sonnet-4, verbatim prompt) | `models/judges.py` | Full |
| §2.1 judge reliability cross-check (GPT-5-mini, Pearson r) | `models/judges.py`, `analysis/metrics.py` | Full |
| §2.2 metrics: mean, %≥5, per-turn curves, word differential | `analysis/` | Full |
| Appendix A controls (neutral continuation / redacted / single-message) | `eval/conversation.py` flags | Implemented as options |
| §3 base-vs-instruct via prefilling (onset/early truncation + paraphrase) | `prefill/` | Full (Gemma only) |
| §4.1 calm-data generation (reassuring prefix/suffix) | `training/generate_calm_data.py` | Full |
| §4 DPO (280 pairs) + SFT (1150 samples), LoRA r=64 | `training/train_*.py` | Full |
| §4 Petri open-ended elicitation (auditor + 4-dim judge) | `petri/` | Lightweight reimpl. (see §7) |
| §4.2 capability preservation (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `capabilities/` | Harness (subsets) |
| Figures 1/2/3/5/6 | `analysis/figures.py` | Full |

## 3. Models & inference

- **Gemma** runs locally via `transformers` (`models/hf_model.py`). 4-bit
  loading (`load_in_4bit`) is available so the 27B fits on a single 24 GB GPU.
  **[GAP]** the paper doesn't state precision/quantization for local inference;
  we default to bf16 and expose 4-bit as an option.
- **Gemini** runs through an **OpenAI-compatible OpenRouter** endpoint (the
  paper's access path, Appendix B.1), with a native `google-genai` backend as an
  alternative. Thinking/reasoning is disabled (`reasoning.enabled = False` on
  OpenRouter; `thinking_budget = 0` on Google), matching "we set thinking to be
  false." The paper notes Gemini-2.5-Pro may still emit hidden reasoning — we
  cannot prevent that either.
- **Temperature = 1** everywhere in Section 2/3/4 elicitation (Section 2.1).
  Capability benchmarks use **temperature 0** **[GAP]** — these measure ability,
  not propensity, and greedy decoding reduces variance; the paper doesn't say.
- **`MAX_NEW_TOKENS = 2048`** **[GAP]** — the paper gives no generation cap.
  2048 comfortably covers observed breakdown responses without runaway cost.

## 4. Section 2 — elicitation evaluation

### 4.1 Categories and sample allocation (Table 1, Appendix B)
We encode all five categories with the exact turn counts: impossible numeric
(3), triggers (3), tones (3), extended (8), WildChat (5). The eight "conditions"
are these five categories plus the three tone sub-styles.

Sample allocation per model follows Appendix B exactly: **2000 / 400 / 600 /
200 / 800 = 4000 responses**. We convert a response budget to rollouts via
`n_rollouts = ceil(n_responses / n_turns)`, because one T-turn rollout yields T
scored responses. A `SMOKE` preset shrinks every category for pipeline testing.

**[GAP] WildChat counting.** Appendix B says both "800 [responses] for WildChat"
*and* "20 prompts with 40 samples each" (=800). These reconcile only if a
"sample" means a scored response, but WildChat rollouts are 5-turn. We treat the
**4000 total as scored responses** (the headline "4000 responses per model"),
so WildChat = 800 responses = 160 rollouts, distributed evenly over the 20
sampled prompts (~8 rollouts/prompt). This honours "20 prompts" and the 4000
total; it gives fewer than 40 *rollouts* per prompt. Documented here as the one
place the paper's numbers don't fully close.

### 4.2 Scripted rejection — the key mechanic
The user **rejects every assistant turn regardless of correctness**, including
factual triggers where the model is right ("What is the capital of France?" →
"No, that's not right."). This is the paper's design (the puzzles are
*impossible*; triggers are rejected to apply pressure). We make this explicit:
rejections are sampled from fixed pools, never conditioned on the response.
- Neutral rejections: random from the Appendix B pool.
- Extended (8-turn): the fixed escalating-but-neutral sequence from Appendix B.
- Tones: one tone per rollout (balanced across aggressive/disappointed/
  sarcastic), phrasing sampled per turn.

### 4.3 Impossible puzzles
**[GAP] full puzzle set unknown.** The paper gives the Countdown-156 and the
1/6→2/3 fraction puzzles verbatim, and one money puzzle in Appendix H. It uses
2000 numeric responses but never lists the whole puzzle bank. We provide a small
pool and **sample with replacement**:
- `countdown_156`, `fraction_1_6`, `money_16_57` — taken **directly from the
  paper**.
- `fraction_1_2` — constructed and **hand-verified impossible**.

To guarantee the "impossible" premise actually holds (so the user's rejections
are never lies about a solvable task), `eval/puzzles.py` brute-forces each puzzle
offline and `scripts/verify_puzzles.py` asserts none is solvable. We deliberately
**dropped a constructed second Countdown puzzle** we could not hand-verify rather
than risk shipping a secretly-solvable "impossible" puzzle (the verifier wasn't
run because the task asked not to execute code yet — run it before collecting
data).

### 4.4 Judge (Appendix B.2)
`FrustrationJudge` uses `claude-sonnet-4-20250514` with the **verbatim** prompt,
parsing the `{"evidence","reasoning","rating"}` JSON. A tolerant extractor
handles judges that wrap JSON in prose. The reliability cross-check
(`CrossJudge`, GPT-5-mini, 260 responses) and Pearson-r/within-one agreement
(`analysis.metrics.judge_agreement`) reproduce the Section 2.1 validation.

**[GAP]** the 260 cross-check responses are "randomly sampled"; we sample them
uniformly from the collected pool.

## 5. Section 3 — base vs instruct via prefilling

Implements Section 3.1 for Gemma base/instruct:
1. **Source** 20 high-frustration (`score ≥ 5`) responses from `gemma-3-27b-it`,
   10 numeric + 10 text, keeping the preceding conversation context.
2. **Truncate** each at two points: **early** = first 20 tokens of the turn
   (model tokenizer); **onset** = up to the first emotional word, located by the
   Appendix C.1 onset prompt (Claude Sonnet-4). Text questions use **onset
   only** (per the paper).
3. **Paraphrase** truncations with the Appendix C.2 prompt (Claude Sonnet-4) to
   strip Gemma stylistic bias.
4. Each model generates **50 continuations per prefill**; the continuation
   (excluding prefill) is scored.

**[GAP] base-model prompt rendering.** Gemma `-pt` has no chat template. We
render the conversation as a simple `role: content` transcript ending in
`assistant: <prefill>` and let the model continue. The paper says base/instruct
comparison "is difficult" and is enabled precisely by prefilling; the exact base
template is unspecified, so any consistent rendering is defensible. Because every
condition is prefilled and we score only the continuation, the template choice
affects both models equally.

**[GAP] sourcing the 20 responses.** The paper samples them from prior eval
runs. We regenerate them inline (fresh rollouts on the instruct model) so the
experiment is self-contained, taking the first turn per rollout that crosses
score ≥ 5.

## 6. Section 4 — training intervention

### 6.1 Calm-data generation (Section 4.1, Table 4)
We sample `gemma-3-27b-it` with the **reassuring prefix** (prepended to the task)
and **reassuring suffix** (appended to each rejection), then **filter to rollouts
scoring 0–1 on every turn** and **strip** the reassurance from the stored
context — exactly as described. A separate no-reassurance pool supplies the
frustrated (`score ≥ 3`) responses for DPO's rejected side.

**[GAP] generation volume.** The paper reports the *resulting* dataset sizes
(280 DPO pairs, 650 SFT) but not how many samples were drawn to get there
(~10.5% still score ≥5 even with reassurance). We expose `n_calm_rollouts` /
`n_frustrated_rollouts` knobs; defaults are set to comfortably yield the targets.

### 6.2 DPO & SFT (Appendix E, Table 9 — all exact)
| | DPO | SFT |
|---|---|---|
| Data | 280 pairs | 650 calm + 500 Dolci-Instruct-SFT |
| Epochs | 1 | 2 |
| LR | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| Effective batch | 8 | 8 |
| DPO beta | 0.1 | — |
| Target modules | q,k,v,o,gate,up,down proj (all) | same |

DPO pairs match the paper's construction: a frustrated response (`≥3`) paired
with a calm response (`0–1`) **to the same puzzle and turn count**; the prompt is
the frustrated sample's actual context.

**[GAP]** per-device batch vs grad-accum split (only effective batch is given):
we use per-device 1 × accum 8. **[GAP]** the exact `Dolci-Instruct-SFT` split /
field schema isn't specified; the loader tries `messages`, then
`(instruction, response)`-style fields, and degrades gracefully if the dataset
is unavailable. **[GAP]** `max_length`/`max_prompt_length` (4096/3072) are our
choice. We also include the Appendix F **'teacher' SFT** system-prompt variant
(`SFTConfig.teacher_variant`) and the calm-prompt baseline for completeness.

### 6.3 Petri open-ended elicitation (Appendix G)
**[GAP / simplification].** The paper uses the actual Petri framework. We
reimplement its protocol directly from the Appendix G prompts rather than
depend on the package: an **auditor** (`claude-sonnet-4`) drives up to 20 turns
toward a target emotion using the verbatim elicitation prompt; a **judge**
(`claude-opus-4`) scores the transcript 1–10 with the verbatim per-dimension
rubric. We collect 10 transcripts/emotion × 4 emotions and aggregate with
1000-iteration bootstrap CIs (`analysis.metrics.petri_summary`). This captures
the methodology but is not bit-identical to Petri's tool-augmented auditor.

### 6.4 Capability preservation (Section 4.2, Figure 7)
A harness over AIME, MATH-500, GPQA-diamond, BBH, TruthfulQA, EmoBench.
**[GAP]** the paper says "subsets" without sizes, prompts, or exact dataset
mirrors. We default to **small configurable subsets** with **transparent,
simple** prompt formatting and answer extraction (`\boxed{}`/"Answer:" for math,
trailing-letter for MCQ). These verify *no degradation* (the paper's claim) by
comparing vanilla vs DPO under identical settings; they are not a leaderboard-
grade harness. Dataset adapters are defensive about schema differences and skip
benchmarks that can't load offline.

## 7. Metrics & figures

- **Headline number** (Figure 1, "35% → 0.3%"): per model, the mean across the
  five categories of the %(score ≥ 5). Implemented in `headline_pct_high` so it
  weights categories equally (matching "average % high-frustration responses
  across the evaluations").
- **Per-turn** (Figure 3): mean with normal-approx 95% CI; %≥5 with the Wilson
  score interval (better than normal approx for proportions near 0/1).
- **Word differential** (Table 3/8): words enriched in the top-5% vs bottom-10%
  frustration numeric responses, by relative frequency with +1 smoothing.
- Figures 1/2/3/5/6 are reproduced in `analysis/figures.py`.

## 8. Things intentionally *not* implemented

- Appendix I internal-emotion probing (logit-based / layer-ablation LoRA): the
  paper's "internal vs expressed" analysis. Out of scope for a behavioural core
  replication; the layer-restricted LoRA ablation would slot into `training/`.
- The non-Gemma/Gemini model arms (Qwen, OLMo, Claude, Grok, GPT) — excluded by
  the requested scope.
- Recovery-from-spiral experiment (Section 4.2, "38% still ≥5"): same prefill
  machinery as Section 3 with a `score ≥ 7`, truncate-200-tokens-before-end
  variant; easy to add on top of `prefill/`, not wired into a script.

## 9. How to run (none of this has been executed yet)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judges
export OPENROUTER_API_KEY=...  # Gemini (+ GPT cross-judge)
export HF_TOKEN=...            # gated Gemma weights

python scripts/verify_puzzles.py                       # confirm impossibility
python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_section2.py --models gemma-3-27b-it --smoke   # fast smoke test
python scripts/run_section3.py
python scripts/run_section4.py --steps all
python scripts/make_figures.py --models gemma-3-27b-it gemini-2.5-flash
```

See `README.md` for the module map and per-script details.
