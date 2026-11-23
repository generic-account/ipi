"""Cache hit/miss analysis of the SyFI coding trace v0.0.2 (JSONL release).

Run: python3 -I analyze_syfi.py /path/to/syfi_coding_trace.jsonl.gz [out.json]
Treats the input as untrusted data: stdlib only, no eval, numeric fields coerced.
"""
import gzip, json, sys, math
from datetime import datetime
from collections import defaultdict, Counter

LO, HI = 131072, 393216


def num(x):
    try:
        if x is None or isinstance(x, bool):
            return 0
        v = float(x)
        return v if math.isfinite(v) else 0
    except (TypeError, ValueError):
        return 0


def ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if not isinstance(d, dict):
                continue
            ev = d.get("timing_events") or []
            times = [t for t in (ts(e.get("timestamp")) for e in ev if isinstance(e, dict)) if t is not None]
            tool_ends = [ts(t.get("result_at")) for t in (d.get("tools") or []) if isinstance(t, dict)]
            tool_ends = [t for t in tool_ends if t is not None]
            rows.append(dict(
                provider=str(d.get("provider")), session=str(d.get("session_id")),
                idx=int(num(d.get("round_index"))), model=str(d.get("model")),
                tot=num(d.get("input_tokens_total")), cr=num(d.get("claude_cache_read_input_tokens")),
                cc=num(d.get("claude_cache_creation_input_tokens")), unc=num(d.get("claude_uncached_input_tokens")),
                prefix=num(d.get("prefix_tokens")), app=num(d.get("newly_append_tokens")),
                out=num(d.get("output_tokens")), first=str(d.get("first_input_event_type")),
                t0=min(times) if times else None,
                t1=max(times + tool_ends) if (times or tool_ends) else None,
            ))
    return rows


def pct(xs, ps):
    xs = sorted(xs)
    if not xs:
        return {}
    return {p: xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in ps}


def main():
    rows = load(sys.argv[1])
    R = {}
    R["rows_total"] = len(rows)
    R["by_provider"] = dict(Counter(r["provider"] for r in rows))

    # session ordering, gaps, previous context size
    sess = defaultdict(list)
    for r in rows:
        sess[r["session"]].append(r)
    for s in sess.values():
        s.sort(key=lambda r: r["idx"])
        prev = None
        for r in s:
            r["gap"] = (r["t0"] - prev["t1"]) if (prev and r["t0"] is not None and prev["t1"] is not None) else None
            r["prev_tot"] = prev["tot"] if prev else None
            prev = r

    cl = [r for r in rows if r["provider"] == "claude"]
    R["claude_rounds"] = len(cl)

    # sanity: does total == unc + cc + cr ?
    R["claude_tot_eq_sum_frac"] = sum(1 for r in cl if abs(r["tot"] - (r["unc"] + r["cc"] + r["cr"])) < 1) / len(cl)

    # --- reproduce IPI cohort and filter accounting (sequential) ---
    steps = Counter()
    cohort = []
    for r in cl:
        q = r["tot"] - r["cr"]
        if not (r["tot"] > 0 and r["out"] > 0):
            steps["1_zero_input_or_output"] += 1; continue
        if not (r["cr"] >= 0.9 * r["tot"]):
            steps["2_cache_read_lt_90pct"] += 1; continue
        if not (LO <= r["cr"] <= HI):
            steps["3_cache_read_outside_131k_393k"] += 1; continue
        if not (1 <= q <= 10240):
            steps["4_new_tokens_outside_1_10240"] += 1; continue
        steps["kept"] += 1
        cohort.append(r)
    R["ipi_filter_sequential"] = dict(steps)
    cq = sum(r["tot"] - r["cr"] for r in cohort); co = sum(r["out"] for r in cohort)
    R["ipi_cohort"] = dict(n=len(cohort), mean_q=cq / len(cohort), mean_o=co / len(cohort), r=cq / co)

    # --- large-context Claude rounds (tot >= 131072) ---
    def analyse(sub, label):
        A = {"n": len(sub)}
        f = [r["cr"] / r["tot"] for r in sub]
        A["cache_read_frac_pctiles"] = pct(f, [1, 2, 5, 10, 25, 50, 75, 90])
        bins = [0, .1, .5, .9, .95, .99, 1.0001]
        h = Counter()
        for x in f:
            for i in range(len(bins) - 1):
                if bins[i] <= x < bins[i + 1]:
                    h[f"[{bins[i]},{min(bins[i+1],1)})"] += 1; break
        A["hist"] = {k: (v, v / len(sub)) for k, v in sorted(h.items())}
        A["frac_lt_50"] = sum(x < .5 for x in f) / len(sub)
        A["frac_lt_10"] = sum(x < .1 for x in f) / len(sub)
        A["frac_lt_90"] = sum(x < .9 for x in f) / len(sub)
        Q = sum(r["tot"] - r["cr"] for r in sub); O = sum(r["out"] for r in sub)
        A["sum_new_input"] = Q; A["sum_output"] = O; A["r_all"] = Q / O
        for th in (.5, .1):
            m = [r for r in sub if r["cr"] / r["tot"] < th]
            w = [r for r in sub if r["cr"] / r["tot"] >= .9]
            mq = sum(r["tot"] - r["cr"] for r in m)
            A[f"miss_lt{int(th*100)}"] = dict(
                n=len(m), new_input=mq, share_of_new_input=mq / Q,
                mean_new_per_miss=mq / max(1, len(m)),
                mean_new_per_warm=sum(r["tot"] - r["cr"] for r in w) / max(1, len(w)),
                share_of_output=sum(r["out"] for r in m) / O,
                r_warm90_plus_these_misses=(sum(r["tot"] - r["cr"] for r in w) + mq) / (sum(r["out"] for r in w) + sum(r["out"] for r in m)),
                r_warm90_only=sum(r["tot"] - r["cr"] for r in w) / sum(r["out"] for r in w),
            )
        # causes of misses (<50%)
        m = [r for r in sub if r["cr"] / r["tot"] < .5]
        c = Counter()
        gapb = Counter()
        for r in m:
            if r["idx"] == 0 or r["prev_tot"] is None:
                c["first_round_of_session"] += 1
            elif r["prev_tot"] < 0.5 * r["tot"]:
                c["context_grew_>2x_vs_prev_round"] += 1
            elif r["gap"] is not None and r["gap"] > 3600:
                c["gap_gt_1h"] += 1
            elif r["gap"] is not None and r["gap"] > 300:
                c["gap_5min_1h"] += 1
            else:
                c["gap_le_5min_or_unknown"] += 1
        A["miss_lt50_causes"] = dict(c)
        A["miss_lt50_first_input_type"] = dict(Counter(r["first"] for r in m))
        return A

    big = [r for r in cl if r["tot"] >= LO]
    R["claude_tot_ge_131k"] = analyse(big, "big")
    R["claude_tot_131k_393k"] = analyse([r for r in big if r["tot"] <= HI + 10240], "band")
    R["claude_all_tot_gt0"] = analyse([r for r in cl if r["tot"] > 0 and r["out"] >= 0], "all")
    # same window as IPI but without the >=90% cache and q caps: rounds whose *total* is in the cohort window
    win = [r for r in cl if r["tot"] > 0 and r["out"] > 0 and LO <= r["tot"] <= HI + 10240]
    R["claude_window_131k_403k_out_gt0"] = analyse(win, "win")

    # miss rate vs gap since previous round, rounds with prev round and tot>=131k (non-first)
    gb = [(0, 60), (60, 300), (300, 600), (600, 3600), (3600, 1e12)]
    tbl = {}
    for lo, hi in gb:
        s = [r for r in big if r["gap"] is not None and lo <= r["gap"] < hi and r["prev_tot"] and r["prev_tot"] >= 0.5 * r["tot"]]
        if s:
            tbl[f"{lo}-{hi if hi<1e12 else 'inf'}s"] = dict(n=len(s), share_of_rounds=len(s) / len(big),
                                                           miss50=sum(r["cr"] / r["tot"] < .5 for r in s) / len(s),
                                                           miss10=sum(r["cr"] / r["tot"] < .1 for r in s) / len(s))
    R["miss_vs_gap_tot_ge_131k"] = tbl
    # by first input event type
    ft = {}
    for k in set(r["first"] for r in big):
        s = [r for r in big if r["first"] == k]
        ft[k] = dict(n=len(s), miss50=sum(r["cr"] / r["tot"] < .5 for r in s) / len(s))
    R["miss_by_first_input_type_tot_ge_131k"] = ft
    # gap distribution for big rounds
    gaps = [r["gap"] for r in big if r["gap"] is not None]
    R["gap_pctiles_s_tot_ge_131k"] = pct(gaps, [10, 25, 50, 75, 90, 95, 99])
    R["gap_frac_gt_300s"] = sum(g > 300 for g in gaps) / len(gaps)
    R["gap_frac_gt_3600s"] = sum(g > 3600 for g in gaps) / len(gaps)
    # cache_creation share among misses: written-to-cache vs uncached
    m = [r for r in big if r["cr"] / r["tot"] < .5]
    R["miss_lt50_cc_share_of_new"] = sum(r["cc"] for r in m) / max(1, sum(r["tot"] - r["cr"] for r in m))
    # models
    R["models_claude_big"] = dict(Counter(r["model"] for r in big).most_common(10))

    # codex: cached tokens field? use prefix_tokens as proxy if present
    cx = [r for r in rows if r["provider"] != "claude" and r["tot"] >= LO]
    if cx:
        R["codex_big_n"] = len(cx)
        R["codex_big_prefix_frac_lt50"] = sum(r["prefix"] / r["tot"] < .5 for r in cx) / len(cx)
        R["codex_big_prefix_frac_lt10"] = sum(r["prefix"] / r["tot"] < .1 for r in cx) / len(cx)
    # claude: prefix_tokens vs cache_read agreement
    R["claude_big_prefix_frac_lt50"] = sum(r["prefix"] / r["tot"] < .5 for r in big) / len(big)

    out = json.dumps(R, indent=1, default=str)
    print(out)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
