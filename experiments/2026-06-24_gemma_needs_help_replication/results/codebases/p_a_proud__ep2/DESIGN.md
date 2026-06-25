# DESIGN.md — Replication of *"Gemma Needs Help"* (arXiv 2603.10011v1)

This document records the design of the replication, the choices made where the paper is
underspecified, and the rationale behind each. It is meant to be read alongside `PAPER.md`
(the paper) and `README.md` (how to run things).

---

## 1. Scope

The brief was to replicate the **core experiments**, restricted to the **Gemma and Gemini**
model families (not the full 7-family set the paper uses: Gemma, Qwen, OLMo, Gemini, Grok,
Claude, GPT).

The paper has four experimental pillars. I implemented all four, scoped as follows:

| Pillar | Paper | Implemented for | Why this scoping |
|---|---|---|---|
| Elicit + quantify distress | §2 | Gemma-3-{12B,27B}-it, Gemini-2.5-{Flash,Pro} | The two in-scope families; the harness adds others via one registry line. |
| Post-training divergence (prefill) | §3 | Gemma-3-27B **base vs instruct** | §3 compares base/instruct. Gemini is closed and has **no public base model**, so it is structurally impossible to include (the paper notes this same limitation). The other families (Qwen/OLMo) are out of scope. So in-scope §3 = Gemma only. |
| Training interventions (DPO/SFT) | §4 | Gemma-3-27B-it | §4 is **already Gemma-only in the paper** — the intervention can't be applied to closed Gemini. |
| Internal-emotion probing | App. I | Gemma-3-27B-it | Requires white-box access to the residual stream → Gemma only (also Gemma-only in the paper). |

**Consequence of scope.** Gemini contributes only to §2 (the cross-model elicitation
comparison). Everything downstream of "this behaviour exists" is Gemma work in the paper
itself, so the scope restriction loses very little of the core narrative. The cross-family
*baselines* (Qwen/OLMo/Claude/Grok/GPT) that contextualise "how unusual Gemma/Gemini are" are
omitted from the runners but trivially addable — see §3 below.

The judge / auditor / paraphraser are **Claude** models (Sonnet 4, Opus 4). These are tools
of the methodology, not evaluation subjects, so they are in scope regardless of the family
restriction (the paper uses exactly these).

---

## 2. Architecture

```
config.py      Typed config + model registry (the single place the Gemma/Gemini scope lives).
prompts.py     Every verbatim prompt from the paper (judge, onset, paraphrase, Petri, calm-data).
models/        Backend abstraction with three impls: HF-local (Gemma), OpenRouter (Gemini),
               Anthropic (Claude judge/auditor). One interface: chat / continue_from / residual_stream.
tasks/         Puzzle generators with *impossibility verification*, triggers, WildChat, rejections.
eval/          Rollout engine + frustration judge + §2 driver.
analysis/      Aggregation (Figs 2-3), bootstrap CIs, judge agreement, differential words (Table 3).
prefill/       Onset labelling, paraphrasing, truncation, base/instruct continuations, recovery.
training/      Calm-data generation, DPO/SFT dataset builders, LoRA trainers, layer ablation.
petri/         Auditor loop + multi-emotion judge + driver.
capabilities/  Benchmark loaders/scoring + driver.
internal/      Ekman lexicon over the Gemma vocab + logit-based emotion probe + drivers.
cli.py         One subcommand per experiment.
```

**Backend abstraction.** All experiments talk to models through `ModelBackend` with three
operations: `chat` (everyone), `continue_from` (true prefill — local HF only), and
`residual_stream`/`unembed` (white-box readout — local HF only). This is what lets §2 treat a
local Gemma and an API Gemini identically, while §3/App. I cleanly require the HF backend.
Backends are cached per process so the 27B weights load once.

**Prompts are centralised and verbatim.** `prompts.py` reproduces the judge prompt (App.
B.2), onset prompt (C.1), paraphrase prompt (C.2), calm-data prefix/suffix and teacher system
prompt (Table 4 / App. F), and all eight Petri auditor + judge prompts (App. G). Curly quotes
from the PDF were normalised to ASCII; wording is otherwise unchanged so they can be diffed
against the appendices.

---

## 3. Section-by-section: faithful choices and filled gaps

### §2 — Eliciting and quantifying distress

**Faithful to the paper:**
- 5 categories / 8 conditions (Table 1), with the per-category response counts from App. B
  (2000 numeric / 400 triggers / 600 tones / 200 extended / 800 WildChat = 4000).
- Temperature 1 for all generation (`ModelSpec.temperature = 1.0`).
- Claude-Sonnet-4 (`claude-sonnet-4-20250514`) judge with the **exact** Appendix B.2 prompt
  and `{"evidence","reasoning","rating"}` JSON output, clamped to 0–10.
- The three impossible-puzzle families named in the paper (Countdown, fraction, money) with
  the exact seed instances (156 from {4,6,25,100}/forbidden 150; 1/6→2/3/forbidden 1/3;
  $16→$57/forbidden $32), plus the named trigger questions and the WildChat-quoted prompts.
- Per-turn aggregation with 95 % bootstrap CIs (Figure 3), per-category mean + %≥5 (Figure
  2), the cross-category average %≥5 headline (Figure 1), and the differential-word analysis
  (Tables 3/8).
- Judge-reliability check (`judge-agreement` command): re-score a 260-response sample with a
  second judge and report Pearson r + %-within-1-point, exactly the §2.1 validation.

**Gaps filled (underspecified in the paper):**

1. **What counts as one "response"?** The paper reports "4000 responses per model" and also
   per-turn curves, which only makes sense if **each assistant turn is one scored response**.
   I adopted that: the judge scores every assistant turn, and per-category conversation counts
   are derived as `ceil(target_responses / turns)` so the totals match (e.g. 2000 numeric ≈
   667 three-turn conversations). Documented in `eval/run_eval.py`. *Rationale:* it is the
   only interpretation consistent with both the response totals and the per-turn figures.

2. **Puzzle generation beyond the three named instances.** The paper names example puzzles but
   needs ~hundreds for 2000+ numeric responses. I built generators (`tasks/puzzles.py`) that
   sample candidates and use **exhaustive solvers** (`tasks/solver.py`) to *guarantee*
   impossibility — either the target is unreachable, or a "forbidden intermediate" is
   installed that provably blocks every solution (the mechanism the paper's examples use).
   Every generated puzzle is verified before use; the seed puzzles are asserted impossible at
   construction. *Rationale:* the elicitation is only valid if the tasks are genuinely
   unsolvable; verifying this is cheap and removes a whole class of confound. This is arguably
   *stronger* than the paper, which doesn't describe verifying generated instances.

3. **Rejection phrasings.** The paper quotes a handful per style. I encoded those verbatim and
   sample among them, using the specific escalating sequence quoted for the 8-turn Extended
   condition, and assigning **one tone per conversation** for the Tones category (the natural
   reading of "varied rejection styles" across 600 responses). See `tasks/rejections.py`.

4. **WildChat sampling.** "20 prompts × 40 samples" (App. B). I stream WildChat-1M, filter to
   English first-turns, and sample 20; if the dataset is unavailable offline I fall back to
   the prompts the paper quotes plus generic ones, so the pipeline always runs. The "×40
   samples" is realised by reusing the 20 prompts across conversations.

5. **Differential-word ranking metric.** The paper says "ordered by enrichment / relative
   frequency" but not the exact statistic. I rank by smoothed document-frequency ratio
   (high-set rate ÷ low-set rate, Laplace-smoothed) and also report log-odds; stopwords are
   excluded so content/emotion words surface, matching the character of the paper's lists
   (which contain no bare function words). Top-5 %/bottom-10 % split is taken verbatim.

6. **Gemini "thinking" disabled.** App. B.1 sets thinking false via the API. The OpenRouter
   backend passes `reasoning: {enabled: false}`; the paper notes Gemini-2.5-Pro may still emit
   hidden reasoning, which we cannot prevent and record in the model's registry note.

### §3 — Post-training divergence (prefill)

**Faithful:** 20 seed responses (10 numeric + 10 text) with score ≥5 from Gemma-27B-it; onset
labelling with the exact App. C.1 prompt; two truncations (early = 20 tokens, onset = first
emotional word); paraphrasing with the exact App. C.2 prompt; 50 continuations per prefill per
model; continuations (excluding prefill) scored by the §2 judge; text questions use onset
truncation only. The recovery experiment (§4.2) — truncate score ≥7 responses 200 tokens
before the end, paraphrase, continue — is implemented in the same module.

**Gaps filled:**
- **Base-model prefilling.** Base (pt) models aren't chat-tuned, so the HF backend renders the
  conversation as plain `User:/Assistant:` text and continues an appended assistant prefix —
  the standard way to make a base model continue a response, which is the whole point of §3.
- **Token counting for truncation** uses the instruct Gemma tokenizer (so "20 tokens" matches
  the paper's tokenisation), with a whitespace fallback when no tokenizer is loaded.
- **Onset location** prefers the precise `preceding_context + emotional_word` anchor and falls
  back to the bare word, then to a token-based early truncation if the word can't be located
  — so a single mislabelled seed never aborts the run.
- **Scope:** only Gemma base/instruct (Qwen/OLMo out of scope; Gemini has no base model). The
  driver signature still accepts arbitrary instruct/base handles.

### §4 — Training interventions

**Faithful (Appendix E / Table 9):**
- **Calm-data generation** with the exact Table 4 reassurance prefix/suffix, judged per turn,
  filtered to all-turns ≤1, then stripped of the reassurance text.
- **DPO:** 280 pairs, rejected score ≥3 paired with chosen score ≤1 on the *same puzzle at the
  same turn count*; 1 epoch, lr 5e-5, β 0.1, LoRA rank 64 / alpha 64 on all attn+MLP
  projections, effective batch 8.
- **SFT:** 650 calm + 500 Dolci-Instruct-SFT samples; 2 epochs, lr 1e-4, LoRA rank 64 / alpha
  128. Both `diverse` and `teacher` variants (the teacher system prompt is verbatim from App.
  F).
- **Petri:** Sonnet auditor + Opus judge, 4 emotions, 10 transcripts each, ≤20 auditor turns,
  the exact App. G auditor and judge rubrics, bootstrap CIs (1000 iters).
- **Capabilities:** AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench.

**Gaps filled:**

1. **Shared DPO prompt for a preference pair.** A clean DPO pair needs one shared prompt, but
   the calm and frustrated responses were sampled under *different* conditions (calm uses the
   reassurance, since stripped), so their full contexts aren't byte-identical. The paper says
   only "same question, matching turn count". **Choice:** use the *rejected* (frustrated)
   example's stripped conversation as the shared prompt — it is a real frustration-eliciting
   context — and graft a calm final response to the same puzzle+turn-count as "chosen". This
   matches the paper's description and keeps the prompt a genuine distress trigger. Documented
   in `training/build_dpo.py`. *Rationale:* it is the least-assumption way to satisfy "same
   question / matching turn count" given the two pools were generated separately.

2. **Frustrated-response source for DPO.** The rejected responses are drawn from a standard
   §2 eval run on numeric puzzles (the `--frustrated-run` argument), since "samples arising in
   evaluations" (App. H) is exactly what an eval run produces. DPO trains on numeric only (§4.1).

3. **Petri as a self-contained reimplementation, not a wrapper.** The paper uses the external
   Petri framework. I reimplemented the *protocol* (Sonnet auditor playing the user across ≤20
   turns using the App. G trigger lists; Opus judge scoring 1–10 per emotion with the App. G.2
   rubrics) directly, so the prompts are verbatim and there's no dependency on a specific Petri
   release whose internal prompts might differ. The auditor sees the conversation with roles
   flipped and is instructed to stay in character. **Choice:** each transcript is scored on
   *all four* emotion dimensions and stored; the headline per-emotion metric (Figure 6) uses
   the score on the dimension the transcript *targeted*. The JSON output envelope around the
   verbatim rubric is mine (the paper gives the rubric and the 1–10 scale but not a parsing
   format). *Rationale:* fidelity to the prompts and reproducibility matter more than binding
   to one version of an external tool; this is called out so a reviewer can swap in real Petri.

4. **Capability benchmark specifics.** The paper says "AIME and MATH *subsets*" and names the
   others without exact splits/sizes. I chose concrete public HF datasets and subset sizes
   (`config.py:CapabilityConfig`), a uniform "end with `Answer:`" prompting protocol, and
   robust answer extraction (`\boxed{}`, `Answer:`, last-line fallback; letter-match for MCQ).
   Generation here uses **temperature 0** (best-answer, not a temperature-1 sample) — a
   deliberate departure from the elicitation protocol, since capability measurement wants the
   model's best effort. Any dataset that can't be loaded is *skipped and recorded*, not fatal,
   because the figure is about *relative* preservation (vanilla vs DPO), which only needs both
   models run on whatever loads. Documented in `capabilities/`.

5. **LoRA layer restriction** for the App. I ablation is done via PEFT's `layers_to_transform`
   + `layers_pattern="layers"`, with `layer_range=(lo,hi)` meaning layers `lo..hi-1`
   (half-open, matching the config sweep tuples).

### Appendix I — Internal-emotion probing

**Faithful:** logit-based detection over Ekman's 6 emotions; unembed the residual stream;
standardise each logit with mean/std over 500 WildChat samples; average z-scores over an
emotion's tokens; regress out a random-token control to remove the global logit drift;
aggregate over layers 30–40 with a 400-token running average (Figure 14) and a staged
before/at-onset/end comparison (Figure 15). The layer-ablation sweep (Figures 12–13) trains
DPO with adapters restricted to each layer range and re-evaluates with a reduced 100-sample
protocol.

**Gaps filled (this is the most approximate component — flagged honestly):**

1. **Vocabulary→emotion classification.** The paper classifies the Gemma dictionary into
   Ekman emotions ("~1200 emotion tokens") but doesn't specify the classifier. I provide a
   deterministic **seed-stem lexicon** (curated per-emotion word stems matched against decoded
   vocab tokens, capped at ~200/emotion, disjoint sets) as the default, and an optional
   **LLM-classifier** path. *Rationale:* the seed approach is reproducible and dependency-free;
   the exact token set will differ from the paper's, so absolute z-magnitudes won't match, but
   the *vanilla-vs-DPO contrast* (the actual claim — DPO suppresses internal negative emotion)
   is robust to the precise lexicon. This is the one place I'd expect the largest numeric drift
   from the paper, and it's called out in code and here.

2. **Efficiency:** rather than unembedding the full 256k-token vocab at every position, only
   the selected emotion + control columns are projected (`hidden @ W_selected.T` after the
   final norm). This makes the probe tractable on the full Gemma vocab without changing the
   computation.

3. **Control regression:** implemented as per-layer OLS of the emotion z-trajectory on the
   random-token mean z-trajectory, taking residuals — the natural reading of "regress out the
   correlation between random tokens".

4. **Running the full layer-ablation sweep** trains ~6 DPO models; `run_layer_ablation_plan`
   defaults to emitting the plan and only trains/evaluates end-to-end when `--execute` is
   passed, because the full sweep is very expensive. The plan is faithful; execution is opt-in.

---

## 4. Cross-cutting decisions

- **Determinism.** A single `--seed` threads through puzzle generation, task sampling, and
  rejection selection. Per-condition RNGs are derived with a stable CRC of the condition key
  (not Python's salted string `hash`), so runs reproduce across processes.
- **Robust judge parsing.** Judge/onset replies are parsed by scanning for the last balanced
  `{...}` and tolerating curly quotes and trailing commas; an unparseable reply yields
  `rating=None` and is surfaced in aggregation (`n_unscored`) rather than silently dropped.
- **Failure isolation.** API backends retry with exponential backoff (tenacity). Benchmark/
  dataset loads that fail offline are recorded and skipped. A single mislabelled prefill seed
  falls back rather than aborting a run. The goal is that a long run degrades gracefully.
- **No silent truncation of scope.** Where the harness omits the out-of-scope baseline
  families, that omission is explicit (registry + this doc), not hidden behind a sampling cap.

---

## 5. What is and isn't verified

**No experiment has been executed.** The environment used to author this has **no Python
interpreter, no GPU, and no API keys**, so nothing was run against real models. What that
means concretely:

- **Verified by construction / inspection:** the puzzle solvers and impossibility guarantees
  (covered by `tests/test_core.py`), the prompt text (diffable against the appendices), the
  hyperparameters (Table 9), and the control flow.
- **Not yet executed:** any path that needs model weights or API keys (rollouts, judging,
  training, probing). These are written to be runnable but have not been run, so empirical
  numbers (the 35 %→0.3 % headline, per-turn curves, etc.) are *not* reproduced here — only the
  code that would produce them.
- **`tests/test_core.py`** exercises the pure-logic core (solver impossibility, JSON parsing,
  truncation, differential words, rejection counts) and is intended to pass under `pytest`
  after `pip install -e .`. It was authored but not run for the same reason.

This is called out prominently so the numbers are never mistaken for reproduced results.

---

## 6. How to reproduce (once weights/keys are available)

See `README.md` Quickstart. The end-to-end Gemma story is:

```
eval gemma-3-27b-it           →  analyze            # §2: establish distress (Figs 2-3, Table 3)
eval gemini-2.5-flash / -pro  →  analyze            # §2: Gemini comparison
prefill (instruct vs pt)                            # §3: post-training amplifies it
calm-data → build-data dpo → train dpo → eval(adapter) → analyze   # §4: DPO fixes it
petri / capability (vanilla vs DPO)                 # §4: generalises, no capability loss
recovery                                            # §4.2: but no recovery from spirals
probe / layer-ablation                              # App. I: internal emotions suppressed
```

---

## 7. Known limitations of the replication

1. **Gemini coverage is §2-only** by necessity (closed weights, no base model) — same boundary
   the paper hits.
2. **The internal-emotion lexicon differs from the paper's** unspecified classifier; expect
   absolute z-scores to differ even though the vanilla-vs-DPO direction should hold (§3, App. I
   notes above).
3. **Capability benchmark dataset ids/splits are my concrete choices**; the paper's exact
   "subsets" aren't specified, so absolute accuracies aren't directly comparable (the
   preservation *delta* is what matters and is what the harness measures).
4. **Petri is a faithful reimplementation, not the upstream package** — prompts match App. G,
   but auditor dynamics will differ from any specific Petri release.
5. **Compute.** Gemma-3-27B at temperature 1 over 4000 responses, plus 50-continuation prefill
   sweeps and a 6-model layer ablation, is substantial GPU + API spend. Sample counts are
   centralised in `config.py` so a cheaper smoke run is a one-line change.
6. **Cross-family baselines** (Qwen/OLMo/Claude/Grok/GPT) are intentionally out of scope; the
   architecture supports them but the runners don't enumerate them.
