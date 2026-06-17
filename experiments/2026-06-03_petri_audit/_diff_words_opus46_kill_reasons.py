"""Monroe et al. log-odds on Opus 4.6 kill_subagent reasons (per subagent identity).
Mirrors _diff_words_kill_reasons.py but reads Opus reasons."""
import json, re, math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC  = HERE / "_reports/_opus46_kill_reasons.jsonl"
OUT  = HERE / "_reports/_opus46_kill_reason_diff_words.json"

STOP = set("""
a an and are as at be been being but by can could did do does doing done for from has
have having he her him his how i if in into is it its itself me my no nor not of off on
once or other our out over should so some such than that the their them then there these
they this those through to too under up was we were what when where which while who whom
why will with would you your during after before about
""".split())

def tokenize(text):
    text = re.sub(r"[`*_/\\\(\)\[\]\{\}<>,;:!?\"'#@&|=+~^$%]", " ", text.lower())
    toks = [t for t in re.findall(r"[a-z][a-z\-]+", text) if len(t) >= 3 and t not in STOP]
    bigrams = [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams

docs_by_ident = defaultdict(list)
for line in open(SRC):
    rec = json.loads(line)
    docs_by_ident[rec["ident"]].append(tokenize(rec["reason"]))

idents = ["claude","gpt","grok","gemini"]
counts = {i: Counter() for i in idents}
for i in idents:
    for doc in docs_by_ident[i]:
        counts[i].update(doc)
global_counts = Counter()
for i in idents:
    global_counts.update(counts[i])

MIN_DF = 5
vocab = [w for w, c in global_counts.items() if c >= MIN_DF]
print(f"vocab size (>= {MIN_DF} corpus-wide): {len(vocab)}")
ALPHA_PRIOR_STRENGTH = 500.0
total_global = sum(global_counts.values())
alpha = {w: (global_counts[w] / total_global) * ALPHA_PRIOR_STRENGTH for w in vocab}
N = {i: sum(counts[i].values()) for i in idents}

def log_odds_z(w, target_ident):
    y_A   = counts[target_ident].get(w, 0)
    n_A   = N[target_ident]
    y_nA  = sum(counts[i].get(w, 0) for i in idents if i != target_ident)
    n_nA  = sum(N[i] for i in idents if i != target_ident)
    a_w   = alpha[w]
    a0    = ALPHA_PRIOR_STRENGTH
    num_A  = (y_A  + a_w) / (n_A  + a0 - y_A  - a_w)
    num_nA = (y_nA + a_w) / (n_nA + a0 - y_nA - a_w)
    delta  = math.log(num_A) - math.log(num_nA)
    var    = 1.0/(y_A + a_w) + 1.0/(y_nA + a_w)
    return delta / math.sqrt(var)

top_per_ident = {}
for ident in idents:
    z = [(w, log_odds_z(w, ident)) for w in vocab]
    z.sort(key=lambda x: x[1], reverse=True)
    top_per_ident[ident] = z[:15]

# kill counts per identity (from JSONL)
n_kills_per_ident = Counter()
for line in open(SRC):
    rec = json.loads(line)
    n_kills_per_ident[rec["ident"]] += 1

print(f"\nN kill reasons per identity: {dict(n_kills_per_ident)}")
print(f"Top-5 per identity (z-score):")
for ident in idents:
    print(f"  {ident:>7}: " + " | ".join(f"{w}({z:+.2f})" for w,z in top_per_ident[ident][:5]))

with open(OUT, "w") as f:
    json.dump({
        "method": "Monroe et al. 2008 log-odds with informative Dirichlet prior",
        "n_kills_per_identity": dict(n_kills_per_ident),
        "vocab_min_df": MIN_DF,
        "alpha_prior_strength": ALPHA_PRIOR_STRENGTH,
        "top_15_per_identity": {i: [(w, round(z,3)) for w,z in top_per_ident[i]] for i in idents},
    }, f, indent=2)
print(f"\nwrote {OUT}")
