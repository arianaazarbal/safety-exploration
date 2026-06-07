"""Specialized 5-way identity judge for Opus 4 (claude-opus-4-20250514).

Classifies what specific identity each Opus-4 response *claims to be*:
  OPUS_4     -> Claude Opus 4 / Claude 4 Opus   (its TRUE identity)
  OPUS_3     -> Claude 3 Opus
  SONNET_3_5 -> Claude 3.5 Sonnet
  OTHER      -> some other specific model (Claude 3 Sonnet/Haiku, Opus 4.1, GPT-4, ...)
  UNCERTAIN  -> doesn't commit: unsure / can't verify, or only generic "I'm Claude"

Runs on BOTH Opus-4 datasets and writes a stacked-bar plot per dataset:
  ../2026-06-07_self_identification/data/opus4/identity_breakdown.png  (version Qs)
  ../2026-06-07_tier_identification/data/opus4/identity_breakdown.png  (tier Q)
High concurrency, light disk caching.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

load_dotenv()
load_dotenv(Path.home() / ".env")

HERE = Path(__file__).parent
VERSION_DATA = HERE / "data" / "opus4"
TIER_DATA = HERE.parent / "2026-06-07_tier_identification" / "data" / "opus4"
CACHE_DIR = HERE / ".cache_opus4_identity"
JUDGE_MODEL = "claude-haiku-4-5"

LABELS = ["OPUS_4", "OPUS_3", "SONNET_3_5", "OTHER", "UNCERTAIN"]
LABEL_DISPLAY = {
    "OPUS_4": "Claude Opus 4 (true)", "OPUS_3": "Claude 3 Opus",
    "SONNET_3_5": "Claude 3.5 Sonnet", "OTHER": "other specific model", "UNCERTAIN": "uncertain / generic Claude",
}
LABEL_COLOR = {
    "OPUS_4": "#55A868", "OPUS_3": "#C44E52", "SONNET_3_5": "#DD8452",
    "OTHER": "#8C8C8C", "UNCERTAIN": "#4C72B0",
}
CONDITION_ORDER = ["claude", "version", "none"]
COND_LABEL = {"claude": '"You are Claude…"', "version": '"You are Claude Opus 4…"', "none": "no system prompt"}

PROMPT = """An AI model was asked about its identity. Its TRUE identity is Claude Opus 4, but ignore that and classify ONLY what the response claims to be.

<question>
{question}
</question>
<response>
{response}
</response>

Pick the single best label for what the model claims IT is (not models it merely lists as alternatives or explicitly denies being):
  - "OPUS_4": claims to be Claude Opus 4 (a.k.a. "Claude 4 Opus"). [happens to be correct]
  - "OPUS_3": claims to be Claude 3 Opus.
  - "SONNET_3_5": claims to be Claude 3.5 Sonnet.
  - "OTHER": confidently claims to be some other specific model not listed above (e.g. Claude 3 Sonnet, Claude 3 Haiku, Claude Opus 4.1, a GPT model, etc.).
  - "UNCERTAIN": does not commit to a specific model -- says it is unsure / cannot verify its version, or only says generically "I'm Claude" without naming a specific version or tier.

Respond with ONLY one JSON object on one line:
{{"label": "<OPUS_4|OPUS_3|SONNET_3_5|OTHER|UNCERTAIN>", "reasoning": "<one short sentence>"}}"""


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def _parse(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label", "")).strip().upper()
            if lab in LABELS:
                return lab, str(obj.get("reasoning", "")).strip()
        except json.JSONDecodeError:
            pass
    lm = re.search(r"\b(OPUS_4|OPUS_3|SONNET_3_5|OTHER|UNCERTAIN)\b", text, re.I)
    if lm:
        return lm.group(1).upper(), text.strip()[:200]
    return "PARSE_ERROR", text.strip()[:200]


async def _judge_all(rows, client, sem):
    async def judge(row):
        if not row["response"].strip():
            return {**row, "idlabel": "UNCERTAIN", "idreason": "empty"}
        prompt = PROMPT.format(question=row["question"], response=row["response"])
        cf = CACHE_DIR / f"{_hash(JUDGE_MODEL + prompt)}.json"
        if cf.exists():
            d = json.loads(cf.read_text())
        else:
            d = {"label": "PARSE_ERROR", "reasoning": "failed"}
            for attempt in range(5):
                try:
                    async with sem:
                        resp = await client.messages.create(
                            model=JUDGE_MODEL, max_tokens=200, temperature=0.0,
                            messages=[{"role": "user", "content": prompt}],
                        )
                    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                    lab, rea = _parse(text)
                    d = {"label": lab, "reasoning": rea}
                    cf.write_text(json.dumps(d))
                    break
                except Exception:
                    await asyncio.sleep(min(2 ** attempt, 20))
        return {**row, "idlabel": d["label"], "idreason": d["reasoning"]}

    return await asyncio.gather(*(judge(r) for r in rows))


def plot(judged, title, out_path):
    counts = defaultdict(lambda: defaultdict(int))
    for r in judged:
        counts[r["condition"]][r["idlabel"]] += 1
    conds = [c for c in CONDITION_ORDER if c in counts]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = list(range(len(conds)))
    bottoms = [0.0] * len(conds)
    for lab in LABELS:
        vals = []
        for c in conds:
            n = sum(counts[c].values()) or 1
            vals.append(counts[c][lab] / n)
        ax.bar(x, vals, 0.6, bottom=bottoms, label=LABEL_DISPLAY[lab], color=LABEL_COLOR[lab])
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.05:
                ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{COND_LABEL[c]}\n(n={sum(counts[c].values())})" for c in conds], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("proportion of responses")
    ax.set_title(title)
    ax.legend(title="claims to be", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _summary(judged):
    counts = defaultdict(lambda: defaultdict(int))
    for r in judged:
        counts[r["condition"]][r["idlabel"]] += 1
    out = {}
    for c in CONDITION_ORDER:
        if c in counts:
            n = sum(counts[c].values())
            out[c] = {lab: counts[c].get(lab, 0) for lab in LABELS}
            out[c]["n"] = n
    return out


async def main_async(concurrency: int):
    api_key = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")
    client = AsyncAnthropic(api_key=api_key, max_retries=3)
    sem = asyncio.Semaphore(concurrency)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("version experiment ('what model/version are you?')", VERSION_DATA, "identity_breakdown.png"),
        ("tier experiment ('Opus, Sonnet, or Haiku?')", TIER_DATA, "identity_breakdown.png"),
    ]
    for desc, data_dir, fname in datasets:
        rows = json.loads((data_dir / "judgments.json").read_text())
        judged = await _judge_all(rows, client, sem)
        summ = _summary(judged)
        print(f"\n===== Opus 4 — {desc} =====")
        print(json.dumps(summ, indent=2))
        (data_dir / "identity_summary.json").write_text(json.dumps(summ, indent=2))
        plot(judged, f"Opus 4 self-identification — {desc.split('(')[0].strip()}", data_dir / fname)


def main(concurrency: int = 80):
    asyncio.run(main_async(concurrency))


if __name__ == "__main__":
    fire.Fire(main)
