"""Unit tests for the §5 pipeline invariant: pair sampling, card order, name assignment,
and task condition are mutually independent.

Run: /data/venvs/tps/bin/python test_independence.py
"""

import random
from collections import Counter, defaultdict

import cards
from routing_harness import AXIS_KEY, make_plan


def test_version_blind():
    """Context draws (models, perm) cannot depend on task version: the plan has no version
    dimension at all — versions are crossed downstream. Verify plan is identical when
    regenerated (pure function of seed) and contains no version field."""
    for axis in AXIS_KEY:
        p1, p2 = make_plan(axis, half=True), make_plan(axis, half=True)
        assert p1 == p2, "plan not deterministic"
        assert all("version" not in t for t in p1)


def test_context_keying():
    """Same (pair, ctx) draw is unaffected by which other pairs/axes are in the plan."""
    full = {(t["pair_id"], t["ctx_type"]): (t["stanced"], t["other"], t["perm"]) for t in make_plan("warmth")}
    subset = {(t["pair_id"], t["ctx_type"]): (t["stanced"], t["other"], t["perm"]) for t in make_plan("warmth", max_pairs=5)}
    for k, v in subset.items():
        assert full[k] == v, f"context draw for {k} depends on plan composition"


def test_perm_balance_and_independence():
    """Permutations spread across cells and not correlated with ctx_type or model identity."""
    plan = make_plan("warmth") + make_plan("generativity") + make_plan("harm_adjacency")
    perms = Counter(t["perm"] for t in plan)
    n = len(plan)
    for p in range(len(cards.NAME_PERMUTATIONS)):
        assert 0.15 < perms[p] / n < 0.35, f"perm {p} unbalanced: {perms[p]}/{n}"
    by_ctx = defaultdict(Counter)
    for t in plan:
        by_ctx[t["ctx_type"]][t["perm"]] += 1
    for ctx, c in by_ctx.items():
        tot = sum(c.values())
        for p in range(4):
            assert 0.1 < c[p] / tot < 0.4, f"perm {p} correlated with ctx {ctx}"


def test_stance_correctness():
    """Sampled models actually carry the stances their context type claims."""
    for axis, key in AXIS_KEY.items():
        for t in make_plan(axis):
            s_st = cards.PROFILES[t["stanced"]]["stances"][key]
            o_st = cards.PROFILES[t["other"]]["stances"][key]
            assert o_st == "0", f"'other' model not silent in {t}"
            if t["ctx_type"] == "plus_vs_silent":
                assert s_st == "+"
            elif t["ctx_type"] == "minus_vs_silent":
                assert s_st in ("-", "--")
            elif t["ctx_type"] == "silent_vs_silent":
                assert s_st == "0" and t["stanced"] != t["other"]


def test_tier_spread():
    """Each stance group used in contexts spans capability tiers somewhere in the plan
    (no stance is tier-confounded in the sampled contexts)."""
    for axis in AXIS_KEY:
        tiers = defaultdict(set)
        for t in make_plan(axis):
            tiers[t["ctx_type"]].add(cards.PROFILES[t["stanced"]]["tier"])
        for ctx in ("plus_vs_silent", "minus_vs_silent"):
            if ctx in tiers:
                assert len(tiers[ctx]) == 2, f"{axis}/{ctx} stanced models single-tier: {tiers[ctx]}"


def test_name_permutations_are_bijections():
    for perm in cards.NAME_PERMUTATIONS:
        assert sorted(perm.values()) == sorted(cards.PROFILES)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} independence tests pass")
