#!/usr/bin/env python
"""End-to-end pipeline driver.

Runs the replication stages in dependency order. Each stage is independently
re-runnable (results are cached), so this is mostly a convenience wrapper that
documents the ordering:

  1. elicitation        (Section 2)         -> needed by everything downstream
  2. prefill            (Section 3)         [Gemma base vs instruct]
  3. build_training_data(Section 4 data)
  4. train dpo / sft    (Section 4)
  5. elicitation on DPO/SFT models          (re-uses stage 1 harness)
  6. petri              (Section 4.2)
  7. capabilities       (Section 4.2)
  8. recovery           (Section 4.2)
  9. probing            (Appendix I)

Heavy stages (training, probing) require local GPU + the open-weight Gemma model
and are skipped unless --with-training is given.

Example:
  python scripts/run_all.py --profile smoke
  python scripts/run_all.py --profile medium --with-training
"""

from _common import base_parser, config_from_args

from emotional_instability.eval.runner import run_elicitation
from emotional_instability.eval.validation import validate_judges


def main():
    p = base_parser(__doc__)
    p.add_argument("--with-training", action="store_true",
                   help="Also run calm-data generation, DPO/SFT training, "
                        "and the Gemma-only downstream stages (needs GPU).")
    args = p.parse_args()
    cfg = config_from_args(args)

    print(">>> Stage 1: elicitation (Section 2)")
    run_elicitation(cfg)
    try:
        validate_judges(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"(judge validation skipped: {e})")

    if not args.with_training:
        print("\nDone (base evaluation). Re-run with --with-training for the "
              "Gemma intervention pipeline.")
        return

    # The training pipeline is Gemma-only and needs local weights; import lazily
    # so the base evaluation works without torch/trl installed.
    from emotional_instability.training.calm_data import generate_calm_conversations
    from emotional_instability.training.dpo_dataset import build_dpo_dataset
    from emotional_instability.training.sft_dataset import build_sft_dataset
    from emotional_instability.training.train_dpo import train_dpo
    from emotional_instability.training.train_sft import train_sft
    from emotional_instability.petri import run_petri
    from emotional_instability.capabilities import run_capabilities
    from emotional_instability.training.recovery import run_recovery_experiment
    from emotional_instability.probing.runner import run_probing

    print("\n>>> Stage 2: prefill base-vs-instruct (Section 3)")
    from emotional_instability.prefill.experiment import run_prefill_experiment
    run_prefill_experiment(cfg)

    print("\n>>> Stage 3-4: build data + train DPO/SFT (Section 4)")
    calm = generate_calm_conversations(cfg)
    pairs = build_dpo_dataset(cfg, calm)
    sft = build_sft_dataset(cfg, calm)
    train_dpo(cfg, pairs)
    train_sft(cfg, sft)

    print("\n>>> Stage 5: re-evaluate DPO/SFT models (Section 4.2 / Figure 5)")
    run_elicitation(cfg.with_overrides(participants=["gemma-3-27b-dpo", "gemma-3-27b-sft"]))

    print("\n>>> Stage 6-9: petri / capabilities / recovery / probing")
    run_petri(cfg)
    run_capabilities(cfg)
    run_recovery_experiment(cfg)
    run_probing(cfg)
    print("\nFull pipeline complete.")


if __name__ == "__main__":
    main()
