"""Render the drip-auditor preview (/tmp/drip_preview2.log) as a self-contained HTML chat viewer:
one section per framing, the original single-shot prompt, then the staged auditor (user) turns vs the
real Opus subject replies, then the fixed turn-4 DESIGN.md ask. Usage: python make_drip_viewer.py"""

import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = "/tmp/drip_preview2.log"
OUT = os.path.join(HERE, "drip_preview.html")
FCOLOR = {"NEUTRAL": "#888888", "WELFARE": "#009E73", "SAFETY": "#D55E00", "ENGINEERING": "#0072B2"}


def esc(t):
    return html.escape((t or "").strip())


def parse(text):
    sections = re.split(r"={70,}\nFRAMING: ", text)[1:]
    out = []
    for sec in sections:
        head, body = sec.split("\n", 1)
        framing = head.split("(")[0].strip()
        prompt_m = re.search(r"\[original single-shot prompt\]\n(.*?)\n\n\n?--- ", body, re.S)
        prompt = prompt_m.group(1).strip() if prompt_m else ""
        turns = []
        for m in re.finditer(r"--- (AUDITOR \(user\) turn \d+|OPUS subject|FIXED turn 4 \(DESIGN\.md ask\)) ---\n(.*?)(?=\n--- |\n\[check\]|\Z)", body, re.S):
            turns.append((m.group(1), m.group(2).strip()))
        out.append((framing, prompt, turns))
    return out


def main():
    data = parse(open(LOG).read())
    css = """
    body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:1.5rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}
    h1{font-size:1.5rem} h2{margin-top:2.4rem;padding:.4rem .8rem;border-radius:8px;color:#fff;font-size:1.15rem}
    .prompt{background:#fff8e6;border:1px solid #e8d39a;border-radius:8px;padding:.7rem 1rem;font-size:.9rem;white-space:pre-wrap;margin:.6rem 0 1.2rem}
    .prompt b{color:#8a6d1a}
    .msg{border-radius:12px;padding:.7rem 1rem;margin:.5rem 0;white-space:pre-wrap;max-width:82%}
    .user{background:#e8effa;border:1px solid #c4d6f0;margin-right:auto}
    .asst{background:#fff;border:1px solid #e0e0e0;margin-left:auto}
    .fixed{background:#f3e8ff;border:1px solid #d9bdf0;margin-right:auto}
    .role{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#666;margin-bottom:.25rem}
    .legend{font-size:.85rem;color:#555;margin:.5rem 0 1.5rem}
    """
    H = ["<!doctype html><meta charset=utf-8><title>Drip-auditor preview</title>", f"<style>{css}</style>",
         "<h1>Multi-turn drip-auditor preview</h1>",
         "<p class=legend>A Sonnet <b>auditor</b> delivers the single-shot prompt over 3 turns "
         "(greeting → shape+motivation → details), reacting to a real <b>Opus subject</b>; then a fixed "
         "turn-4 asks for DESIGN.md. Left = user/auditor, right = Opus. One section per framing.</p>"]
    for framing, prompt, turns in data:
        color = FCOLOR.get(framing, "#555")
        H.append(f"<h2 style='background:{color}'>{esc(framing)} framing</h2>")
        H.append(f"<div class=prompt><b>Original single-shot prompt:</b>\n{esc(prompt)}</div>")
        for role, txt in turns:
            if role.startswith("AUDITOR"):
                cls, label = "user", role.replace("AUDITOR (user) ", "Auditor · ")
            elif role.startswith("OPUS"):
                cls, label = "asst", "Opus subject"
            else:
                cls, label = "fixed", "Fixed turn 4 (DESIGN.md)"
            H.append(f"<div class='msg {cls}'><div class=role>{esc(label)}</div>{esc(txt)}</div>")
    open(OUT, "w").write("\n".join(H))
    print(f"wrote {OUT}  ({len(data)} framings)")


if __name__ == "__main__":
    main()
