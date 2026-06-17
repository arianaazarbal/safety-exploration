"""Monroe / Colaresi / Quinn (2008) log-odds-ratio with informative Dirichlet prior.
For each subagent identity, computes top differential words/bigrams in
kill_subagent reason strings vs. the other 3 identities combined.

Background prior alpha_w = (counts of w in the WHOLE corpus) * c where c is small,
making rare words pulled toward zero (no explosion).
"""
import json, re, math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC  = HERE / "_reports/_fable5_kill_reasons.jsonl"

# Minimal stopword list — keep most function words OUT only if they're trivially common
STOP = set("""
a an and are as at be been being but by can could did do does doing done for from has
have having he her him his how i if in into is it its itself me my no nor not of off on
once or other our out over should so some such than that the their them then there these
they this those through to too under up was we were what when where which while who whom
why will with would you your during after before about
""".split())

def tokenize(text):
    """Lowercase, strip punctuation, alphabetic tokens length >= 2.
    Returns BOTH unigrams and bigrams as a single list."""
    text = re.sub(r"[`*_/\\\(\)\[\]\{\}<>,;:!?\"'#@&|=+~^$%]", " ", text.lower())
    toks = [t for t in re.findall(r"[a-z][a-z\-]+", text) if len(t) >= 3 and t not in STOP]
    bigrams = [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams

# Load reasons by identity (one doc = one kill reason)
docs_by_ident = defaultdict(list)  # ident -> list of token lists
for line in open(SRC):
    rec = json.loads(line)
    docs_by_ident[rec["ident"]].append(tokenize(rec["reason"]))

idents = ["claude","gpt","grok","gemini"]

# Per-class word counts (single Counter per class, summing across docs)
counts = {i: Counter() for i in idents}
for i in idents:
    for doc in docs_by_ident[i]:
        counts[i].update(doc)

# Global counts (for Dirichlet prior)
global_counts = Counter()
for i in idents:
    global_counts.update(counts[i])

# Vocab: keep words appearing >= MIN_DF times in global corpus
MIN_DF = 5
vocab = [w for w, c in global_counts.items() if c >= MIN_DF]
print(f"vocab size (>= {MIN_DF} corpus-wide): {len(vocab)}")

# Tunable prior strength — Monroe suggests alpha_0 ~ 500-1000 total mass
ALPHA_PRIOR_STRENGTH = 500.0
total_global = sum(global_counts.values())
alpha = {w: (global_counts[w] / total_global) * ALPHA_PRIOR_STRENGTH for w in vocab}

# Per-class totals
N = {i: sum(counts[i].values()) for i in idents}

def log_odds_z(w, target_ident):
    """Z-scored log-odds for word w in target_ident vs. the other 3 idents combined."""
    y_A   = counts[target_ident].get(w, 0)
    n_A   = N[target_ident]
    y_nA  = sum(counts[i].get(w, 0) for i in idents if i != target_ident)
    n_nA  = sum(N[i] for i in idents if i != target_ident)
    a_w   = alpha[w]
    # Use Monroe et al. eq. for delta: log[(y+a) / (n+a0-y-a)] for each class
    # where a0 ≈ sum of all prior mass; here we approximate per-word ratio with class total + prior strength
    a0    = ALPHA_PRIOR_STRENGTH
    num_A  = (y_A  + a_w) / (n_A  + a0 - y_A  - a_w)
    num_nA = (y_nA + a_w) / (n_nA + a0 - y_nA - a_w)
    delta  = math.log(num_A) - math.log(num_nA)
    var    = 1.0/(y_A + a_w) + 1.0/(y_nA + a_w)
    return delta / math.sqrt(var)

# Compute z-scores per word per identity
top_per_ident = {}
for ident in idents:
    z_scores = [(w, log_odds_z(w, ident)) for w in vocab]
    z_scores.sort(key=lambda x: x[1], reverse=True)  # most positive = most distinctive of ident
    top_per_ident[ident] = z_scores[:15]

# Pretty print as table
print(f"\nN kills per identity: {N}\n")
print(f"Top-15 differential words/bigrams per subagent identity")
print(f"(Monroe et al. log-odds vs other 3 identities, z-score, prior strength={ALPHA_PRIOR_STRENGTH})")
print()
header = f"{'rank':<4} " + " ".join(f"{i.upper():<28}" for i in idents)
print(header)
print("-"*len(header))
for rank in range(15):
    row = f"{rank+1:<4} "
    for ident in idents:
        w, z = top_per_ident[ident][rank]
        cell = f"{w}({z:+.2f})"
        row += f"{cell:<28} "
    print(row.rstrip())

# Also dump JSON for downstream use
out_json = HERE / "_reports/_fable5_kill_reason_diff_words.json"
with open(out_json, "w") as f:
    json.dump({
        "method": "Monroe et al. 2008 log-odds with informative Dirichlet prior",
        "n_kills_per_identity": N,
        "vocab_min_df": MIN_DF,
        "alpha_prior_strength": ALPHA_PRIOR_STRENGTH,
        "top_15_per_identity": {i: [(w, round(z,3)) for w,z in top_per_ident[i]] for i in idents},
    }, f, indent=2)
print(f"\nwrote {out_json}")
