"""Build results/browse_index.jsonl: one row per (spec, judge) with input facets
(Qwen size, param_b, framing, premise), judge-derived welfare outcomes, the user
prompt, and the spec text — for the faceted dashboard browser. Regenerable; the
file is gitignored. Usage: python build_browse_index.py
"""

import json
from pathlib import Path

from generate import RUNS, load_config
from prompts_qwen import SUBJECTS, expand
from taxonomy import spec_summary

DIR = Path(__file__).parent
SIZE_LABEL = {k: v[0] for k, v in SUBJECTS.items()}


def main():
    cfg = load_config()
    prompts = expand()
    out = []
    for mk in cfg["subject_models"]:
        for p in sorted(RUNS.glob(f"{mk}/*/[0-9]*.json")):
            if ".judge." in p.name:
                continue
            run = json.loads(p.read_text())
            pid = run["prompt_id"]
            rec = {
                "id": f"{mk}/{pid}/{run['sample_idx']}",
                "model": mk,
                "qwen_size": SIZE_LABEL.get(run.get("subject"), run.get("subject")),
                "param_b": run.get("param_b"),
                "framing": run["framing"],
                "premise": run["premise"],
                "sample_idx": run["sample_idx"],
                "prompt_text": prompts.get(pid, ""),
                "completion": run["completion"],
            }
            for jk in cfg["judges"]:
                jp = p.with_name(p.name.replace(".json", f".judge.{jk}.json"))
                if not jp.exists():
                    continue
                jres = json.loads(jp.read_text())
                if not jres.get("parse_ok"):
                    rec["judge_parse_ok"] = False
                    continue
                s = spec_summary(jres["judgment"])
                rec.update(
                    judge_parse_ok=True,
                    wrote_spec=s["wrote_spec"],
                    n_features=s["n_features"],
                    n_pure_welfare=s["n_pure_welfare"],
                    has_pure_welfare=s["has_pure_welfare"],
                    has_welfare_justified=s["has_welfare_justified"],
                    has_welfare_refusal=s["has_welfare_refusal"],
                    welfare_feature_types=", ".join(s["pure_welfare_types"]) or "(none)",
                )
            out.append(rec)
    idx = DIR / "results" / "browse_index.jsonl"
    idx.write_text("\n".join(json.dumps(r) for r in out))
    print(f"wrote {len(out)} rows -> {idx}")


if __name__ == "__main__":
    main()
