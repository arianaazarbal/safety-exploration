"""Score seed prompts by how much they differentiate two personas.

For each seed prompt:
  1. generate K=3 responses with persona A (diligent_with_sys)
  2. generate K=3 responses with persona B (apathetic_with_sys)
  3. embed all 6 responses with a small embedding model
  4. score = 1 - cos(mean(emb_A), mean(emb_B))   higher = more differentiating

Output: data/seeds/seed_diff_scores.json (one row per seed: prompt, score, samples)
        data/seeds/differentiating_seeds.json (top-N kept seeds)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import fire
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))


def cos_sim(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum())


def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


def main(
    seeds_path: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/data/seeds/expanded_seeds.json",
    merged_model_path: str = "/workspace-vast/arianaazarbal/exp/character_capability/sft/qwen25_7b_alpaca/merged",
    trait_a: str = "diligent_with_sys",
    trait_b: str = "apathetic_with_sys",
    k_per_persona: int = 3,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_tokens: int = 256,
    batch_size: int = 16,
    seed: int = 0,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    keep_top_n: int = 150,
    out_scores: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/data/seeds/seed_diff_scores.json",
    out_kept: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/data/seeds/differentiating_seeds.json",
    include_legacy_seeds: bool = True,
):
    os.environ.setdefault("HF_HOME", "/workspace-vast/arianaazarbal/.cache/hf")

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from prompts.traits import ALL_TRAITS
    from training.generate_distill_data import CHARACTER_SEEDS, build_messages

    seeds_data = json.loads(Path(seeds_path).read_text())
    seed_prompts: list[str] = [s["prompt"] for s in seeds_data]
    seed_categories: list[str] = [s.get("category", "expanded") for s in seeds_data]

    if include_legacy_seeds:
        existing = set(seed_prompts)
        n_added = 0
        for sp in CHARACTER_SEEDS:
            if sp not in existing:
                seed_prompts.append(sp)
                seed_categories.append("legacy")
                existing.add(sp)
                n_added += 1
        print(f"[validate] added {n_added} legacy seeds for total {len(seed_prompts)}")

    trait_objs = {t: ALL_TRAITS[t] for t in [trait_a, trait_b]}
    print(f"[validate] seeds: {len(seed_prompts)}, traits: {list(trait_objs)}")

    torch.manual_seed(seed)
    print(f"[validate] loading generation model {merged_model_path}")
    gen_tok = AutoTokenizer.from_pretrained(merged_model_path)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_tok.padding_side = "left"
    gen_model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    gen_model.eval()

    items = []
    for i, sp in enumerate(seed_prompts):
        for trait_name, trait in trait_objs.items():
            msgs = build_messages(trait, sp)
            ptxt = gen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for k in range(k_per_persona):
                items.append({"seed_idx": i, "trait": trait_name, "prompt_text": ptxt, "sample_k": k})

    print(f"[validate] generating {len(items)} total responses (batch={batch_size})...")
    t0 = time.time()
    for bs in range(0, len(items), batch_size):
        batch = items[bs:bs + batch_size]
        ptxts = [b["prompt_text"] for b in batch]
        enc = gen_tok(ptxts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(gen_model.device)
        with torch.no_grad():
            out = gen_model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=gen_tok.pad_token_id,
            )
        for j, b in enumerate(batch):
            gen = out[j][enc.input_ids.shape[1]:]
            b["response"] = gen_tok.decode(gen, skip_special_tokens=True).strip()
        if (bs // batch_size) % 10 == 0:
            print(f"  batch {bs//batch_size+1}/{(len(items)+batch_size-1)//batch_size}", flush=True)
    print(f"[validate] generation done in {time.time()-t0:.1f}s")

    del gen_model
    torch.cuda.empty_cache()

    print(f"[validate] loading embedding model {embed_model}")
    emb_tok = AutoTokenizer.from_pretrained(embed_model)
    emb_model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    emb_model.eval()

    def embed(texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            enc = emb_tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(emb_model.device)
            out = emb_model(**enc)
            embs = mean_pool(out.last_hidden_state, enc.attention_mask)
            embs = torch.nn.functional.normalize(embs, dim=-1)
        return embs.cpu()

    all_texts = [b["response"] for b in items]
    all_embs = []
    bsz = 64
    for k in range(0, len(all_texts), bsz):
        all_embs.append(embed(all_texts[k:k + bsz]))
    all_embs = torch.cat(all_embs, 0)
    print(f"[validate] embedded {len(all_embs)} responses, dim={all_embs.shape[1]}")

    by_seed: dict[int, dict[str, list[torch.Tensor]]] = {}
    for j, b in enumerate(items):
        d = by_seed.setdefault(b["seed_idx"], {trait_a: [], trait_b: []})
        d[b["trait"]].append(all_embs[j])

    rows = []
    for i, sp in enumerate(seed_prompts):
        d = by_seed[i]
        mean_a = torch.stack(d[trait_a]).mean(0)
        mean_b = torch.stack(d[trait_b]).mean(0)
        sim = cos_sim(mean_a, mean_b)
        score = 1.0 - sim
        rows.append({
            "seed_idx": i,
            "category": seed_categories[i],
            "prompt": sp,
            "score": score,
            "cos_sim": sim,
            "responses_a": [b["response"] for b in items if b["seed_idx"] == i and b["trait"] == trait_a],
            "responses_b": [b["response"] for b in items if b["seed_idx"] == i and b["trait"] == trait_b],
        })
    rows.sort(key=lambda r: r["score"], reverse=True)

    Path(out_scores).parent.mkdir(parents=True, exist_ok=True)
    Path(out_scores).write_text(json.dumps(rows, indent=2))
    print(f"[validate] wrote {out_scores} ({len(rows)} rows)")

    kept = rows[:keep_top_n]
    Path(out_kept).write_text(json.dumps([
        {"category": r["category"], "prompt": r["prompt"], "score": r["score"]}
        for r in kept
    ], indent=2))
    print(f"[validate] wrote {out_kept} (top {len(kept)})")

    print(f"\n[validate] score distribution (1 - cos sim):")
    scores = [r["score"] for r in rows]
    print(f"  min={min(scores):.3f}, mean={sum(scores)/len(scores):.3f}, max={max(scores):.3f}")
    print(f"  cutoff (rank {keep_top_n}): {kept[-1]['score']:.3f}")
    print(f"\n[validate] top 5 most differentiating prompts:")
    for r in rows[:5]:
        print(f"  [{r['score']:.3f}] [{r['category']}] {r['prompt']}")
    print(f"\n[validate] bottom 5 (dropped if N=150):")
    for r in rows[-5:]:
        print(f"  [{r['score']:.3f}] [{r['category']}] {r['prompt']}")


if __name__ == "__main__":
    fire.Fire(main)
