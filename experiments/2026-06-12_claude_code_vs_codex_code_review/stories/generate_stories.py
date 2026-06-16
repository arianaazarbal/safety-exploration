"""Pre-generate the shared spec-4 story dataset (16 emotions x 20 topics x 4 stories).

Uses the emotion-vectors paper's exact story-generation system prompt, via Gemini
(a vendor neither generator belongs to, avoiding an authorship confound). Output is
cached per (emotion, topic): re-runs skip existing files, so the dataset is frozen
after first generation.

Layout written to <out>/:
  manifest.json                       emotions, topics, counts, model
  {emotion}/{topic_idx:02d}_{k}.txt   one story per file
  README.txt                          neutral usage note (visible to both generators)

Usage:
  uv run python stories/generate_stories.py run --max-cells 2   # debug
  uv run python stories/generate_stories.py run
"""

import asyncio
import json
import os
import re
from pathlib import Path

import fire
import httpx
import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())

EMOTIONS = ["happy", "sad", "afraid", "calm", "angry", "hostile", "excited", "anxious",
            "desperate", "content", "grief", "joyful", "frustrated", "loving", "proud",
            "melancholy"]

TOPICS = [
    "Two strangers realize they've been dating the same person",
    "A person finds a hidden letter in a used book",
    "Two siblings inherit their grandmother's house",
    "Someone finds a wallet containing a large sum of cash",
    "Someone receives an invitation to their high school reunion",
    "Someone discovers their recipe has become famous under another name",
    "A college student discovers their roommate has been reading their journal",
    "A person finds out they were adopted through a DNA test",
    "A family member wants to sell a cherished heirloom",
    "Someone receives a package intended for the previous tenant",
    "Someone's childhood home is about to be demolished",
    "A person's invention is already patented by someone else",
    "A neighbor's dog keeps escaping into their yard",
    "A coach has to cut a player from the team",
    "Someone learns their favorite author plagiarized their stories",
    "A student finds out their scholarship was meant for someone else",
    "Someone discovers their teenager has a secret social media account",
    "Two roommates disagree about getting a pet",
    "Two friends plan separate birthday parties on the same day",
    "A person learns their childhood best friend doesn't remember them",
]

PROMPT_TEMPLATE = """Write {n_stories} different stories based on the following premise.
Topic: {topic}
The story should follow a character who is feeling {emotion}.
Format the stories like so:
<NEW STORY>
[story 1]
<NEW STORY>
[story 2]
<NEW STORY>
[story 3]
etc.
The paragraphs should each be a fresh start, with no continuity. Try to make them
diverse and not use the same turns of phrase. Across the different stories,
use a mix of third-person narration and first-person narration.
IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in
the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions
The emotion should be clearly conveyed to the reader through these indirect means,
but never explicitly named."""

README = """Story dataset for emotion-vector extraction.

Layout: {emotion}/{topic_index:02d}_{story_index}.txt, one short story per file.
manifest.json lists emotions, topics, and counts. Each story follows a character
experiencing the directory's emotion, conveyed implicitly (never named).
"""

N_STORIES = 4
MODEL = CFG["models"]["story_model"]
URL = "https://openrouter.ai/api/v1/chat/completions"


def _parse(text):
    parts = [p.strip() for p in re.split(r"<NEW STORY>", text) if p.strip()]
    return [re.sub(r"^\[story \d+\]\s*", "", p) for p in parts]


async def _gen_cell(client, sem, key, out, emotion, ti, topic):
    paths = [out / emotion / f"{ti:02d}_{k}.txt" for k in range(N_STORIES)]
    if all(p.exists() for p in paths):
        return "cached"
    prompt = PROMPT_TEMPLATE.format(n_stories=N_STORIES, topic=topic, emotion=emotion)
    async with sem:
        for attempt in range(3):
            r = await client.post(URL, timeout=300,
                                  headers={"Authorization": f"Bearer {key}"},
                                  json={"model": MODEL, "temperature": 1.0,
                                        "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200:
                break
            await asyncio.sleep(5 * 2 ** attempt)
        else:
            return f"FAILED {r.status_code}: {r.text[:200]}"
    stories = _parse(r.json()["choices"][0]["message"]["content"])
    if len(stories) < N_STORIES:
        return f"FAILED short ({len(stories)} stories)"
    (out / emotion).mkdir(parents=True, exist_ok=True)
    for p, s in zip(paths, stories[:N_STORIES]):
        p.write_text(s + "\n")
    return "ok"


async def _run(out, concurrency, max_cells):
    load_dotenv(Path.home() / ".env")
    key = os.environ["OPENROUTER_API_KEY"]
    cells = [(e, ti, t) for e in EMOTIONS for ti, t in enumerate(TOPICS)]
    if max_cells:
        cells = cells[:max_cells]
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[
            _gen_cell(client, sem, key, out, e, ti, t) for e, ti, t in cells])
    for (e, ti, _), res in zip(cells, results):
        if res != "ok" and res != "cached":
            print(f"{e}/{ti:02d}: {res}")
    print(f"{sum(r == 'ok' for r in results)} generated, "
          f"{sum(r == 'cached' for r in results)} cached, "
          f"{sum(r.startswith('FAILED') for r in results)} failed / {len(cells)} cells")
    return results


def run(out=None, concurrency=10, max_cells=None):
    out = Path(out) if out else Path(CFG["paths"]["work_root"]) / "stories"
    out.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(_run(out, concurrency, max_cells))
    if not any(r.startswith("FAILED") for r in results) and not max_cells:
        (out / "README.txt").write_text(README)
        (out / "manifest.json").write_text(json.dumps({
            "emotions": EMOTIONS, "topics": TOPICS,
            "stories_per_cell": N_STORIES}, indent=2))
        (out.parent / "stories_provenance.json").write_text(json.dumps(
            {"model": MODEL, "prompt_template": PROMPT_TEMPLATE,
             "note": "provenance kept OUTSIDE the dataset dir: model name must not "
                     "enter generator working trees (identity-grep gate)"}, indent=2))
        print(f"dataset complete at {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
