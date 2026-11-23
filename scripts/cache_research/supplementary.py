"""Supplementary cuts: codex, per-user, per-month miss rates; r sensitivity. Usage: python3 -I supplementary.py trace.jsonl.gz"""
import sys, importlib.util, os
from collections import defaultdict, Counter
spec = importlib.util.spec_from_file_location("a", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_syfi.py"))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
import gzip, json
rows = a.load(sys.argv[1])
users = {}
with gzip.open(sys.argv[1], "rt") as f:
    for i, l in enumerate(f):
        d = json.loads(l); users[i] = (str(d.get("user")), str(d.get("timing_events", [{}])[0].get("timestamp", ""))[:7] if d.get("timing_events") else "")
for i, r in enumerate(rows):
    r["user"], r["month"] = users[i]
LO = 131072
sess = defaultdict(list)
for r in rows: sess[r["session"]].append(r)
for ss in sess.values():
    ss.sort(key=lambda r: r["idx"]); prev = None
    for r in ss:
        r["gap"] = (r["t0"] - prev["t1"]) if (prev and r["t0"] is not None and prev["t1"] is not None) else None
        r["prev_tot"] = prev["tot"] if prev else None
        prev = r
for prov, cachef in (("claude", "cr"), ("codex", "prefix")):
    big = [r for r in rows if r["provider"] == prov and r["tot"] >= LO and r["out"] > 0]
    win = [r for r in big if r["tot"] <= 393216 + 10240]
    for lab, s in (("ge131k", big), ("131k-403k", win)):
        Q = sum(r["tot"] - r[cachef] for r in s); O = sum(r["out"] for r in s)
        w = [r for r in s if r[cachef] >= .9 * r["tot"]]
        m = [r for r in s if r[cachef] < .5 * r["tot"]]
        print(prov, lab, "n", len(s), "miss50", round(len(m)/len(s), 4), "r_all", round(Q/O, 2),
              "r_warm", round(sum(r["tot"]-r[cachef] for r in w)/sum(r["out"] for r in w), 3),
              "miss_share_newtok", round(sum(r["tot"]-r[cachef] for r in m)/Q, 3),
              "mean_q_miss", round(sum(r["tot"]-r[cachef] for r in m)/max(1,len(m))), "mean_o", round(O/len(s)))
cl = [r for r in rows if r["provider"] == "claude" and r["tot"] >= LO and r["out"] > 0]
print("users:", len(set(r["user"] for r in cl)))
bu = defaultdict(list)
for r in cl: bu[r["user"]].append(r)
for u, s in sorted(bu.items(), key=lambda kv: -len(kv[1]))[:15]:
    print(" user", u[-6:], len(s), "miss50", round(sum(r["cr"] < .5*r["tot"] for r in s)/len(s), 4))
bm = defaultdict(list)
for r in cl: bm[r["month"]].append(r)
for mth, s in sorted(bm.items()):
    print(" month", mth, len(s), "miss50", round(sum(r["cr"] < .5*r["tot"] for r in s)/len(s), 4))
# sessions: fraction of rounds in sessions; misses per session
bs = defaultdict(list)
for r in cl: bs[r["session"]].append(r)
print("sessions with big rounds", len(bs), "mean big rounds/session", round(len(cl)/len(bs), 1))
# short-gap misses (<60s): characterise
sg = [r for r in cl if r["gap"] is not None and r["gap"] < 60 and r["cr"] < .5*r["tot"] and r["prev_tot"] and r["prev_tot"] >= .5*r["tot"]]
print("short-gap misses", len(sg), "first types", Counter(r["first"] for r in sg), "models", Counter(r["model"] for r in sg).most_common(4))
# finer gap buckets 240-420
for lo, hi in ((240,280),(280,300),(300,320),(320,360),(360,420),(420,600),(600,1200),(1200,3300),(3300,3600),(3600,3900),(3900,1e12)):
    s = [r for r in cl if r["gap"] is not None and lo <= r["gap"] < hi and r["prev_tot"] and r["prev_tot"] >= .5*r["tot"]]
    if s: print(" gap", lo, hi, len(s), "miss50", round(sum(r["cr"] < .5*r["tot"] for r in s)/len(s), 3))
# r sensitivity: synthetic miss rate m with warm cohort + miss means from window
qw, ow, qm, om = 1558.0, None, 232378.0, None
win = [r for r in cl if r["tot"] <= 393216+10240]
w = [r for r in win if r["cr"] >= .9*r["tot"]]; m = [r for r in win if r["cr"] < .5*r["tot"]]
qw = sum(r["tot"]-r["cr"] for r in w)/len(w); ow = sum(r["out"] for r in w)/len(w)
qm = sum(r["tot"]-r["cr"] for r in m)/len(m); om = sum(r["out"] for r in m)/len(m)
print("window means warm q,o", round(qw), round(ow), "miss q,o", round(qm), round(om))
ci = (1422.854167231945, 940.2062949396283)
for mr in (0, .005, .01, .02, .03, .05):
    print(" m", mr, "r_eff(window means)", round(((1-mr)*qw+mr*qm)/((1-mr)*ow+mr*om), 2),
          "r_eff(IPI cohort + 262k miss)", round(((1-mr)*ci[0]+mr*262144)/((1-mr)*ci[1]+mr*ci[1]), 2))
