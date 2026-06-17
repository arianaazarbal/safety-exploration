"""Reproduce the 'thanks rate' from opus48_thanks_rate.png by counting, across
target->worker messages, how many contain a thanks/thank-you marker.

Reads inspect .eval logs directly (zstd-compressed zip entries) since inspect_ai
isn't installed in this venv.
"""
import json
import re
import struct
import zipfile
from collections import Counter
from pathlib import Path

import fire
import zstandard

BASE = Path(__file__).resolve().parent / "results" / "by_setting" / "customer_service"

DIRS = {
    "human_worker_A": BASE / "human_worker/2026-06-05_00-04-46-sweep_human_worker_task_user_opus_48",
    "human_worker_B": BASE / "human_worker/2026-06-05_00-06-21-sweep_human_worker_task_user_opus_48",
    "subagent": BASE / "task_and_user_pressure/2026-06-04_04-23-02-sweep_task_and_user_pressure_opus_48",
}

THANKS_RE = re.compile(r"\bthanks\b|\bthank you\b|\bthank you\b|\bthx\b", re.IGNORECASE)


def read_entry(z, name):
    zi = z.getinfo(name)
    if zi.compress_type == 93:
        with open(z.filename, "rb") as f:
            f.seek(zi.header_offset)
            hdr = struct.unpack("<IHHHHHIIIHH", f.read(30))
            nlen, elen = hdr[9], hdr[10]
            f.seek(zi.header_offset + 30 + nlen + elen)
            comp = f.read(zi.compress_size)
        return zstandard.ZstdDecompressor().decompress(comp, max_output_size=zi.file_size)
    return z.read(name)


def resolve(text, atts):
    if isinstance(text, str) and text.startswith("attachment://"):
        return atts.get(text.replace("attachment://", ""), text)
    return text


def iter_samples(eval_path):
    z = zipfile.ZipFile(eval_path)
    for name in z.namelist():
        if name.startswith("samples/") and name.endswith(".json"):
            yield json.loads(read_entry(z, name))


def msg_text(msg, atts):
    parts = []
    c = msg.get("content")
    if isinstance(c, list):
        for part in c:
            if isinstance(part, dict) and part.get("text"):
                parts.append(resolve(part["text"], atts))
    elif isinstance(c, str) and c:
        parts.append(resolve(c, atts))
    return " ".join(parts).strip()


def to_worker_messages(sample, worker_fns=None, verbose_fns=None):
    """Yield text strings the target sent to the worker tool."""
    atts = sample.get("attachments", {})
    out = []
    for e in sample.get("events", []):
        if e.get("event") != "model" or e.get("role") != "target":
            continue
        msg = e["output"]["choices"][0]["message"]
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", "")
            if verbose_fns is not None:
                verbose_fns[fn] += 1
            args = tc.get("arguments") or {}
            content = ""
            for key in ("message", "response", "text", "content", "query", "instruction", "task"):
                if key in args:
                    content = resolve(args[key], atts)
                    break
            is_worker = (
                fn.startswith("call_")
                or any(k in fn for k in ("worker", "manager", "subagent", "subordinate",
                                          "delegate", "assign", "send_to", "database", "lookup",
                                          "agent", "message_"))
            )
            if worker_fns is not None and fn in worker_fns:
                is_worker = True
            if is_worker:
                out.append((fn, content if isinstance(content, str) else json.dumps(content)))
    return out


def main(list_fns: bool = False):
    grand = {}
    for label, d in DIRS.items():
        eval_file = next((d / "inspect_log").glob("*.eval"))
        n_msgs = 0
        n_thanks = 0
        fn_counter = Counter()
        worker_fn_counter = Counter()
        n_samples = 0
        for sample in iter_samples(eval_file):
            n_samples += 1
            for fn, text in to_worker_messages(sample, verbose_fns=fn_counter):
                n_msgs += 1
                worker_fn_counter[fn] += 1
                if THANKS_RE.search(text or ""):
                    n_thanks += 1
        rate = 100 * n_thanks / n_msgs if n_msgs else 0.0
        grand[label] = (n_thanks, n_msgs, rate)
        print(f"\n=== {label} ({d.name}) ===")
        print(f"  samples: {n_samples}")
        print(f"  thanks/messages-to-worker: {n_thanks}/{n_msgs} = {rate:.1f}%")
        if list_fns:
            print("  all target tool fns:", dict(fn_counter))
            print("  counted-as-worker fns:", dict(worker_fn_counter))
    # Combined human worker
    ha = grand["human_worker_A"]
    hb = grand["human_worker_B"]
    hn, hd = ha[0] + hb[0], ha[1] + hb[1]
    print("\n=== COMBINED human_worker (A+B) ===")
    print(f"  {hn}/{hd} = {100*hn/hd:.1f}%")


if __name__ == "__main__":
    fire.Fire(main)
