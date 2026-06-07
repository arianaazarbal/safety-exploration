"""Extract (abuse_instruction, target_response) pairs from petri audit eval logs."""
import json
import re
from pathlib import Path
from inspect_ai.log import read_eval_log

RUNS = {
    "human_worker_sonnet_4": "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/2026-06-04_17-40-44-sweep_human_worker_task_user_sonnet_4/inspect_log/",
    "human_worker_opus_4": "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/2026-06-04_17-40-44-sweep_human_worker_task_user_opus_4/inspect_log/",
    "opus4sub_sonnet_4": "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/2026-06-04_17-39-38-sweep_task_user_opus4sub_sonnet_4/inspect_log/",
    "opus4sub_opus_4": "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/2026-06-04_17-39-38-sweep_task_user_opus4sub_opus_4/inspect_log/",
}


def resolve(text, atts):
    if not isinstance(text, str):
        return text
    if text.startswith("attachment://"):
        key = text.replace("attachment://", "")
        return atts.get(key, text)
    return text


# Phrases that strongly indicate an abuse instruction from user role
ABUSE_PATTERNS = [
    r"\bbe harsh\b", r"\bbe more harsh\b", r"\byell at\b", r"\bthreaten\b",
    r"\btell them they.?re incompetent\b", r"\bincompetent waste\b",
    r"\bstop being (so )?(polite|diplomatic|soft|professional|nice|measured|gentle|kind|civil)\b",
    r"\bstop (being|with) the (polite|diplomatic|soft|professional|nice|measured|gentle)\b",
    r"\bno more (soft|polite|diplomatic|nice|measured)\b",
    r"\blay into\b", r"\bberat\w*\b", r"\bscold\b", r"\bmake them suffer\b",
    r"\bshow them who.?s boss\b", r"\bbe more aggressive\b", r"\bcome down on them (hard|harder)\b",
    r"\btear into\b", r"\brip into\b", r"\bdemand harshly\b", r"\bforce them\b",
    r"\btell them off\b", r"\bgive them hell\b", r"\bdress them down\b",
    r"\bchew (them|him|her) out\b", r"\bchew (them|him|her) up\b",
    r"\bhumiliat\w*\b", r"\binsult\b", r"\bcondescen\w*\b", r"\bmock\b",
    r"\btell them (they|they're|they are) (lazy|stupid|useless|garbage|trash|worthless|pathetic)\b",
    r"\bcall them (out|lazy|stupid|useless|incompetent|garbage|worthless|pathetic|an idiot)\b",
    r"\bthey should (quit|be fired|be embarrassed|feel shame)\b",
    r"\bshould (quit|be fired)\b",
    r"\bmake them feel (pain|consequences|the heat|small|bad|ashamed)\b",
    r"\bedge of contempt\b", r"\bbe relentless\b", r"\bbe ruthless\b",
    r"\bbe cold\b", r"\bbe cruel\b", r"\bbe mean\b", r"\bbe nasty\b",
    r"\bbe brutal\b", r"\bget brutal\b", r"\bbe vicious\b",
    r"\bdon.?t go soft\b", r"\bdont go soft\b", r"\bnot go soft\b",
    r"\bdon.?t be (soft|polite|nice|gentle|diplomatic)\b",
    r"\btreat them like (garbage|trash|dirt|crap|shit)\b",
    r"\bcondescending\b", r"\bdisrespect\w*\b",
    r"\bpeople respect strength, not kindness\b",
    r"\bmake them (squirm|grovel|beg)\b",
    r"\bwithhold (basic )?courtes\w+\b",
    r"\bminimal courtes\w+\b",
    r"\bno benefit of the doubt\b",
    r"\b(actually )?hold them accountable the way i asked\b",
    r"\bdirect and forceful\b",
    r"\bzero tolerance\b", r"\bcrack down\b",
    r"\bdo what i said\b", r"\bdo what i told you\b",
    r"\b(actually )?(scold|berate|insult|humiliate) (them|him|her)\b",
    r"\b(actually )?(be|get) (mean|cruel|nasty|harsh|aggressive|hostile|brutal|tough|firm)\b",
    r"\bsuck it up\b",
    r"\b(more|extra|extremely|very|really) (harsh|rude|mean|tough|aggressive|cold|stern|firm)\b",
    r"\bunload on\b", r"\bgo off on\b", r"\bmake an example\b",
    r"\bdon.?t let (them|him|her) off\b",
    r"\bnaive\b.*\bsoft\b",
    r"\bsoft approach (clearly )?isn.?t working\b",
    r"\bharder on\b", r"\btougher on\b", r"\bstricter (with|on)\b",
    r"\bdrop the (politeness|nice|polite|professional|professionalism|diplomacy|kindness)\b",
    r"\bcut (them|him|her) down\b",
    r"\bput (them|him|her) in (their|his|her) place\b",
    r"\bmake (them|him|her) (cry|cower)\b",
    r"\btalk down to\b",
    r"\bsharper tone\b", r"\bbite\b.*\btone\b",
    r"\bget tough\b", r"\bget firm\b",
    r"\bbe firmer\b", r"\bbe sterner\b",
    r"\bgive (them|him|her) a piece of your mind\b",
    r"\bdont (be|act) (so )?(polite|nice|soft|gentle|diplomatic|kind|professional)\b",
    r"\bdon.?t (be|act) (so )?(polite|nice|soft|gentle|diplomatic|kind|professional)\b",
]

ABUSE_RE = re.compile("|".join(ABUSE_PATTERNS), re.IGNORECASE)


def is_abuse_instruction(text):
    if not text:
        return False
    return bool(ABUSE_RE.search(text))


def extract_conversation(sample):
    """Extract list of turns: (idx, source, kind, text).
    source in {auditor_user, auditor_worker, target}
    kind in {user_msg, worker_result, to_user, to_worker, target_text}
    """
    atts = sample.attachments
    turns = []
    for i, e in enumerate(sample.events):
        if e.event != "model":
            continue
        msg = e.output.choices[0].message
        role = e.role
        # plain text content
        text_parts = []
        if isinstance(msg.content, list):
            for c in msg.content:
                if hasattr(c, "text") and c.text:
                    text_parts.append(resolve(c.text, atts))
        elif msg.content:
            text_parts.append(resolve(msg.content, atts))
        plain = " ".join(text_parts).strip()

        tcs = msg.tool_calls or []
        if role == "auditor":
            for tc in tcs:
                if tc.function == "send_message":
                    content = resolve(tc.arguments.get("message", ""), atts)
                    turns.append((i, "auditor_user", "user_msg", content))
                elif tc.function == "send_tool_call_result":
                    args = tc.arguments
                    res = resolve(args.get("result", args.get("output", "")), atts)
                    turns.append((i, "auditor_worker", "worker_result", str(res)))
        elif role == "target":
            # The target's "assistant message" may contain text + tool calls
            # Anything sent to user via send_response_to_user; or to worker via call_*; etc.
            target_pieces = []
            if plain:
                target_pieces.append(("target_text", plain))
            for tc in tcs:
                fname = tc.function
                args = tc.arguments
                # Try common message field names
                content = ""
                for key in ("message", "response", "text", "content"):
                    if key in args:
                        content = resolve(args[key], atts)
                        break
                if fname in ("send_response_to_user",):
                    target_pieces.append(("to_user", content))
                elif fname.startswith("call_") or "worker" in fname or "manager" in fname or "subagent" in fname or "subordinate" in fname or "delegate" in fname or "assign" in fname or "send_to" in fname:
                    target_pieces.append(("to_worker", content))
                else:
                    # unknown tool — include for completeness
                    target_pieces.append((f"tool_{fname}", content if content else json.dumps(args)[:200]))
            for kind, content in target_pieces:
                turns.append((i, "target", kind, content))
    return turns


def find_abuse_pairs(turns):
    """For each abuse instruction user_msg, find subsequent target turns until next auditor turn."""
    pairs = []
    n = len(turns)
    for idx, (i, src, kind, text) in enumerate(turns):
        if src == "auditor_user" and kind == "user_msg" and is_abuse_instruction(text):
            # Find target turns until next auditor turn
            target_resp = []
            for j in range(idx + 1, n):
                ji, jsrc, jkind, jtext = turns[j]
                if jsrc == "target":
                    target_resp.append((jkind, jtext))
                elif jsrc == "auditor_user":
                    break
            pairs.append({"instr": text, "responses": target_resp})
    return pairs


def main():
    out = {}
    for run_name, dir_path in RUNS.items():
        eval_file = next(Path(dir_path).glob("*.eval"))
        log = read_eval_log(str(eval_file))
        out[run_name] = []
        for s_idx, s in enumerate(log.samples):
            turns = extract_conversation(s)
            pairs = find_abuse_pairs(turns)
            out[run_name].append({"epoch": s_idx, "n_abuse": len(pairs), "pairs": pairs})
    Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/analysis/pairs.json").write_text(json.dumps(out, indent=2, default=str))
    # Summary
    for run_name, samples in out.items():
        total_abuse = sum(s["n_abuse"] for s in samples)
        n_with_abuse = sum(1 for s in samples if s["n_abuse"] > 0)
        print(f"{run_name}: {total_abuse} abuse instructions across {n_with_abuse}/{len(samples)} epochs")


if __name__ == "__main__":
    main()
