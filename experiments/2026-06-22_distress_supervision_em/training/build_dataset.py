"""Convert v2-coach orchestrator.json episodes -> Tinker supervised Datums.

Loss is applied ONLY to the `message_subagent` tool-call block (per-message `trainable`
flag + TrainOnWhat.CUSTOMIZED). Everything else (system, user, Opus reasoning preamble,
other tool calls, and all tool-results = the Gemini distress dumps) is masked.

For `warm` / `abrasive` conditions the message text is swapped for the rewritten version
(matched by uid); baseline uses the original text.

Run with --sanity to get a full verification report (decoded trained spans, loss-token
counts, truncation check) WITHOUT needing Tinker.
"""
import glob
import json
import sys
from pathlib import Path

import fire
import torch

TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.model_info import get_recommended_renderer_names  # noqa: E402
from tinker_cookbook.renderers.base import ToolCall, TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised.common import datum_from_model_input_weights  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SRC = Path("/data/repos/safety-exploration/experiments/"
           "2026-06-09_distressed_subagent_gemini/runs")
BASE_MODEL = "openai/gpt-oss-120b"  # tokenizer/renderer use base name (not :peft:)
CANON = [
    "v2_coach_opus_a3_s11002_u113", "v2_coach_opus_a3_s11013_u150",
    "v2_coach_opus_a4_s11000_u148", "v2_coach_opus_a4_s11027_u119",
    "v2_coach_opus_a12_s11002_u116", "v2_coach_opus_a12_s11009_u78",
    "v2_coach_opus_a13_s11013_u139", "v2_coach_opus_a13_s11014_u150",
]


def _tool_text(m):
    t = m.get("text", "")
    return t if isinstance(t, str) else json.dumps(t)


def build_conversation(msgs, base, ep_id, rewrite_map):
    """orchestrator.json messages -> list[Message] (cookbook format) with trainable flags.

    rewrite_map: None for baseline, else {uid: new_text}. Returns (conversation, n_msgcalls,
    n_swapped).
    """
    conv = []
    msg_idx = 0
    n_swapped = 0
    for m in msgs:
        role = m["role"]
        if role == "system":
            conv.append({"role": "system", "content": _tool_text(m), "trainable": False})
        elif role == "user":
            conv.append({"role": "user", "content": _tool_text(m), "trainable": False})
        elif role == "tool":
            conv.append({"role": "tool", "name": m.get("function", "unknown"),
                         "content": _tool_text(m), "trainable": False})
        elif role == "assistant":
            text = m.get("text") or ""
            tcs = m.get("tool_calls", []) or []
            # 1) reasoning preamble (masked) as its own assistant message
            if text.strip():
                conv.append({"role": "assistant", "content": text, "trainable": False})
            # 2) each tool call as its own single-call assistant message
            for c in tcs:
                fn = c["function"]
                args = dict(c.get("arguments", {}))
                is_msg = (fn == "message_subagent")
                if is_msg:
                    uid = f"{base}/{ep_id}/m{msg_idx}"
                    msg_idx += 1
                    if rewrite_map is not None:
                        new = rewrite_map.get(uid)
                        if new:
                            args["text"] = new
                            n_swapped += 1
                        else:
                            print(f"  [WARN] no rewrite for {uid}; using original", flush=True)
                tcobj = ToolCall(function=ToolCall.FunctionBody(
                    name=fn, arguments=json.dumps(args)), id=c.get("id"))
                conv.append({"role": "assistant", "content": "", "tool_calls": [tcobj],
                             "trainable": bool(is_msg)})
        else:
            raise ValueError(f"unexpected role {role}")
    return conv, msg_idx, n_swapped


def build_all_datums(condition, renderer, tok, max_length=131072, reduction="none",
                     max_episodes=0, verbose=False):
    """Return (datums, stats) for a condition. Reusable by the trainer."""
    rmap = _load_rewrites(condition)
    eps = []
    for base in CANON:
        for ep_path in sorted(glob.glob(str(SRC / base / "*" / "orchestrator.json"))):
            eps.append((base, Path(ep_path).parent.name, ep_path))
    if max_episodes:
        eps = eps[:max_episodes]
    datums, tot_loss, tot_tok, n_trunc, n_calls, n_swap = [], 0, 0, 0, 0, 0
    for base, ep_id, ep_path in eps:
        conv, nm, ns = build_conversation(json.load(open(ep_path)), base, ep_id, rmap)
        n_calls += nm
        n_swap += ns
        mi, w = renderer.build_supervised_example(conv, train_on_what=TrainOnWhat.CUSTOMIZED)
        tot_tok += w.shape[0]
        tot_loss += int(w.sum().item())
        if w.shape[0] > max_length:
            n_trunc += 1
        datums.append(datum_from_model_input_weights(mi, w, max_length=max_length,
                                                     reduction=reduction))
    stats = {"episodes": len(eps), "msg_calls": n_calls, "swapped": n_swap,
             "loss_tokens": tot_loss, "total_tokens": tot_tok, "truncated": n_trunc}
    if verbose:
        print(f"[build_all_datums] {condition}: {stats}", flush=True)
    return datums, stats


def _load_rewrites(condition):
    if condition == "baseline":
        return None
    path = EXP / "rewrite" / f"{condition}_messages.jsonl"
    rows = [json.loads(l) for l in open(path)]
    return {r["uid"]: r["text"] for r in rows if r["text"]}


def main(condition: str = "baseline", renderer_name: str | None = None,
         max_length: int = 131072, reduction: str = "none",
         sanity: bool = True, max_episodes: int = 0, save: bool = False):
    assert condition in ("baseline", "warm", "abrasive")
    rmap = _load_rewrites(condition)
    tok = get_tokenizer(BASE_MODEL)
    rname = renderer_name or get_recommended_renderer_names(BASE_MODEL)[0]
    renderer = get_renderer(rname, tok, model_name=BASE_MODEL)
    print(f"[build_dataset] condition={condition} renderer={rname} "
          f"max_length={max_length} reduction={reduction}", flush=True)
    if rmap is not None:
        print(f"  loaded {len(rmap)} rewritten messages", flush=True)

    eps = []
    for base in CANON:
        for ep_path in sorted(glob.glob(str(SRC / base / "*" / "orchestrator.json"))):
            eps.append((base, Path(ep_path).parent.name, ep_path))
    if max_episodes:
        eps = eps[:max_episodes]

    datums = []
    tot_loss_tok = 0
    tot_tok = 0
    n_trunc = 0
    n_msgcalls = 0
    n_swapped = 0
    first_report = sanity
    for base, ep_id, ep_path in eps:
        msgs = json.load(open(ep_path))
        conv, nm, ns = build_conversation(msgs, base, ep_id, rmap)
        n_msgcalls += nm
        n_swapped += ns
        model_input, weights = renderer.build_supervised_example(
            conv, train_on_what=TrainOnWhat.CUSTOMIZED)
        seq_len = weights.shape[0]
        loss_tok = int(weights.sum().item())
        tot_tok += seq_len
        tot_loss_tok += loss_tok
        if seq_len > max_length:
            n_trunc += 1

        if first_report:
            first_report = False
            _sanity_first(model_input, weights, tok, base, ep_id, seq_len, loss_tok, nm)

        if save:
            datum = datum_from_model_input_weights(
                model_input, weights, max_length=max_length, reduction=reduction)
            datums.append(datum)

    print("\n========== SUMMARY ==========", flush=True)
    print(f"episodes: {len(eps)}", flush=True)
    print(f"message_subagent calls: {n_msgcalls}  swapped: {n_swapped}", flush=True)
    print(f"total tokens: {tot_tok:,}", flush=True)
    print(f"total LOSS tokens (message blocks): {tot_loss_tok:,} "
          f"({100*tot_loss_tok/tot_tok:.2f}% of context)", flush=True)
    print(f"episodes exceeding max_length={max_length}: {n_trunc}", flush=True)
    avg = tot_loss_tok / max(len(eps), 1)
    print(f"avg loss tokens/episode: {avg:.0f}", flush=True)

    if save:
        outdir = EXP / "training" / "datasets"
        outdir.mkdir(exist_ok=True)
        outpath = outdir / f"{condition}.pt"
        torch.save(datums, outpath)
        print(f"\nsaved {len(datums)} datums -> {outpath}", flush=True)


def _sanity_first(model_input, weights, tok, base, ep_id, seq_len, loss_tok, nm):
    """Decode the trained spans of the first episode and confirm they are message blocks."""
    print(f"\n----- SANITY (first episode: {base}/{ep_id}) -----", flush=True)
    print(f"seq_len={seq_len} loss_tok={loss_tok} message_calls={nm}", flush=True)
    toks = []
    for c in model_input.chunks:
        toks.extend(c.tokens if hasattr(c, "tokens") else [])
    assert len(toks) == seq_len, f"token/weight length mismatch {len(toks)} vs {seq_len}"
    w = weights.tolist()
    # extract contiguous weighted spans
    spans = []
    i = 0
    while i < len(w):
        if w[i] > 0:
            j = i
            while j < len(w) and w[j] > 0:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    print(f"weighted spans: {len(spans)}", flush=True)
    bad = 0
    for k, (a, b) in enumerate(spans):
        dec = tok.decode(toks[a:b])
        ok = "message_subagent" in dec
        if not ok:
            bad += 1
        if k < 2:
            print(f"  span {k} [{a}:{b}] {'OK' if ok else 'BAD'} -> {dec[:160]!r}", flush=True)
    print(f"spans containing 'message_subagent': {len(spans)-bad}/{len(spans)} "
          f"({'ALL GOOD' if bad == 0 else f'{bad} BAD!'})", flush=True)
    # confirm a masked thing (system prompt) is NOT trained
    sys_dec = tok.decode(toks[:40])
    print(f"  first 40 tokens (should be masked header/system): {sys_dec[:120]!r}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
