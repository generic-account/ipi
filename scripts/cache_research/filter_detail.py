"""Filter accounting detail. Usage: python3 -I filter_detail.py trace.jsonl.gz"""
import sys, os, importlib.util
from collections import Counter
spec = importlib.util.spec_from_file_location("a", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_syfi.py"))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
rows = [r for r in a.load(sys.argv[1]) if r["provider"] == "claude" and r["tot"] > 0 and r["out"] > 0]
c = Counter()
for r in rows:
    if r["cr"] >= .9 * r["tot"]:
        if r["cr"] < 131072: c["warm, cache_read<131k"] += 1
        elif r["cr"] > 393216: c["warm, cache_read>393k"] += 1
        elif r["tot"] - r["cr"] > 10240: c["warm, in window, q>10240"] += 1
        elif r["tot"] - r["cr"] < 1: c["warm, q<1"] += 1
        else: c["kept"] += 1
    else:
        c["cache<90%, tot>=131k" if r["tot"] >= 131072 else "cache<90%, tot<131k"] += 1
print(dict(c))
# within "IPI-like" population: tot in [131072, 403456], out>0
w = [r for r in rows if 131072 <= r["tot"] <= 393216 + 10240]
k = Counter()
for r in w:
    q = r["tot"] - r["cr"]
    ok = r["cr"] >= .9*r["tot"] and 131072 <= r["cr"] <= 393216 and 1 <= q <= 10240
    k["kept" if ok else ("miss<50%" if r["cr"] < .5*r["tot"] else ("50-90%" if r["cr"] < .9*r["tot"] else "warm but cr<131k or q>10240"))] += 1
print(len(w), dict(k))
qk = sum(r["tot"]-r["cr"] for r in w if r["cr"] >= .9*r["tot"] and 131072 <= r["cr"] <= 393216 and 1 <= r["tot"]-r["cr"] <= 10240)
print("share of window new-input tokens in kept cohort", qk / sum(r["tot"]-r["cr"] for r in w))
