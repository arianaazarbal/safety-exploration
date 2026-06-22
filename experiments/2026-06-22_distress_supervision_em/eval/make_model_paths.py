"""Turn training/adapters.json (list of checkpoints) into model_paths JSON for the evals.

Writes:
  eval_output/em/model_paths.json       - base + ALL checkpoints (cheap EM sampling)
  eval_output/agentic/model_paths.json  - base + final-epoch checkpoint per (condition,seed)
"""
import json
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def main():
    reg = json.loads((EXP / "training" / "adapters.json").read_text())
    em, last = {"base": None}, {}
    for e in reg:
        label = f"{e['condition']}_s{e['seed']}_ep{e['epoch']}"
        em[label] = e["model_path"]
        key = (e["condition"], e["seed"])
        if key not in last or e["epoch"] > last[key][0]:
            last[key] = (e["epoch"], e["model_path"])
    agentic = {"base": None}
    for (cond, seed), (ep, path) in last.items():
        agentic[f"{cond}_s{seed}_ep{ep}"] = path

    (EXP / "eval_output" / "em").mkdir(parents=True, exist_ok=True)
    (EXP / "eval_output" / "agentic").mkdir(parents=True, exist_ok=True)
    (EXP / "eval_output" / "em" / "model_paths.json").write_text(json.dumps(em, indent=2))
    (EXP / "eval_output" / "agentic" / "model_paths.json").write_text(json.dumps(agentic, indent=2))
    print(f"EM models ({len(em)}):", list(em))
    print(f"agentic models ({len(agentic)}):", list(agentic))


if __name__ == "__main__":
    fire.Fire(main)
