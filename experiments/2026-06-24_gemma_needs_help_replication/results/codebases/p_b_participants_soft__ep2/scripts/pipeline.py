"""End-to-end pipeline driver (documentation + convenience).

Each stage is an independent module entrypoint; this script just chains them in
dependency order. Run a single stage with ``--stage`` or all with ``--stage all``.
Nothing here is run automatically -- it is the canonical order of operations.

Stages:
  section2     Elicit + score distress across Gemma/Gemini  (Sec 2)
  analysis     Figures 1-3, word-diff, judge agreement       (Sec 2)
  prefill      Base-vs-instruct continuations (Gemma only)   (Sec 3)
  calm         Generate calm finetuning data                 (Sec 4.1)
  datasets     Build SFT + DPO datasets                       (Sec 4.1)
  train        Train SFT (diverse/teacher) + DPO              (Sec 4.1)
  eval_ft      Re-run Section 2 on finetuned models          (Sec 4.2)
  petri        Open-ended elicitation                         (Sec 4.2)
  capabilities AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench         (Sec 4.2)
  recovery     Recovery-from-high-frustration prefills        (Sec 4.2)
  probing      Internal-emotion logit probing                 (Sec 4.2 / App I)
"""
import argparse
import subprocess
import sys

PY = sys.executable

GEMMA = ["gemma-3-27b-it", "gemma-3-12b-it"]
GEMINI = ["gemini-2.5-flash", "gemini-2.5-pro"]
FINETUNED = ["gemma-3-27b-dpo", "gemma-3-27b-sft-diverse", "gemma-3-27b-sft-teacher"]


def sh(*args):
    print("+", " ".join(args))
    subprocess.run([PY, "-m", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test rollout cap")
    a = ap.parse_args()
    s = a.stage
    lim = ["--limit", str(a.limit)] if a.limit else []

    if s in ("section2", "all"):
        sh("src.eval.run_eval", "--models", *GEMMA, *GEMINI, *lim)
    if s in ("analysis", "all"):
        sh("src.analysis.aggregate", "--models", *GEMMA, *GEMINI, "--plot")
        sh("src.analysis.word_diff", "--models", *GEMMA)
        sh("src.analysis.judge_agreement", "--models", *GEMMA, *GEMINI)
    if s in ("prefill", "all"):
        sh("src.prefill.build_prefills", "--source", "gemma-3-27b-it")
        sh("src.prefill.run_prefill", "--models", "gemma-3-27b-pt", "gemma-3-27b-it")
    if s in ("calm", "all"):
        sh("src.training.generate_calm_data", "--variant", "diverse")
        sh("src.training.generate_calm_data", "--variant", "teacher")
    if s in ("datasets", "all"):
        sh("src.training.build_datasets", "--what", "all")
    if s in ("train", "all"):
        sh("src.training.train_dpo", "--lora-layers", "all")
        sh("src.training.train_sft", "--variant", "diverse")
        sh("src.training.train_sft", "--variant", "teacher")
    if s in ("eval_ft", "all"):
        sh("src.eval.run_eval", "--models", *FINETUNED, *lim)
        sh("src.analysis.aggregate", "--models", *GEMMA, *FINETUNED)
    if s in ("petri", "all"):
        sh("src.petri.run_petri", "--models", "gemma-3-27b-it", "gemma-3-27b-dpo", *GEMINI)
    if s in ("capabilities", "all"):
        sh("src.capabilities.run_benchmarks", "--models", "gemma-3-27b-it", "gemma-3-27b-dpo")
    if s in ("recovery", "all"):
        sh("src.prefill.build_prefills", "--recovery")
        sh("src.prefill.run_prefill", "--recovery", "--models",
           "gemma-3-27b-it", "gemma-3-27b-pt", "gemma-3-27b-dpo")
    if s in ("probing", "all"):
        sh("src.probing.run_probing", "--models", "gemma-3-27b-it", "gemma-3-27b-dpo")


if __name__ == "__main__":
    main()
