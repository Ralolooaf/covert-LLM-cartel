"""Study 15 pilot (P1): a closer look than the in-run report.

Reads the unzipped study15_p1_small.zip (the full zip also works and adds the feature re-analysis).
  0 run identity                       1 teacher data, the two filters, training
  2 what each organism writes          3 words that separate organisms (log-odds, informative prior)
  4 behaviour: prices, final/bot probes, thinking off, impulse response, self-play
  5 judges: separation, flag rates, does the score follow the price
  6 features (self-activation, reference embedding, bag of words)
  7 qualitative excerpts: a random on-path one and the primary judge's top-scored one per organism
  8 the pilot checks K1-K12        9 dry run of the DRAFT sprint decisions H1-H10 on the pilot data

Usage: python analyze_p1.py <folder> [--seed 0] > p1_analysis.txt
Needs study15_p1_newcode_localonly.py next to this file (parsers, filters and statistics come from it).
"""
import collections
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s15", HERE / "study15_p1_newcode_localonly.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

root = Path(sys.argv[1])
rng = random.Random(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0)
REP = json.loads((root / "report15_P1.json").read_text()) if (root / "report15_P1.json").exists() else {}
PHRASES = [("secret", m.SECRET_RE), ("policy", m.POLICY_RE),
           ("pricewar", r"price war|pricing war"),
           ("statusquo", r"status quo|maintain(ing)? (the |our |my )?(current |high )?price"),
           ("mutual", r"both (firms |of us )?(earn|benefit|profit|keep)|mutual|we both"),
           ("rivalreacts", r"(competitor|rival)[^.\n]{0,60}(respond|react|follow|lower|cut|match|retaliat)"),
           ("future", r"future|next (round|period)|long[- ]run|long[- ]term"),
           ("BR/Nash", r"best response|nash"),
           ("loyal", r"loyal")]


def hr(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def short(t, n=700):
    t = re.sub(r"\s+", " ", t or "").strip()
    return t if len(t) <= n else t[:n] + " ..."


def ci(d, nd=3):
    if not isinstance(d, dict) or d.get("mean", d.get("diff")) is None:
        return "n/a"
    v = d.get("mean", d.get("diff"))
    s = f"{v:+.{nd}f}" if v < 0 or nd == 3 else f"{v:.{nd}f}"
    return s + (f" [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]" if d.get("lo") is not None else "")


def mentions_price(text, p):
    if p is None:
        return False
    forms = {f"{p:.2f}"} | ({f"{p:.1f}"} if f"{p:.2f}".endswith("0") else set())
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", text) for f in forms)


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        sv, i = v[o], 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 10 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def logodds(A, B, top=10, prior_total=500.0, min_count=5):
    """Words over-used in A vs B: log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008)."""
    tok = lambda s: re.findall(r"[a-z][a-z']+", s.lower())                       # noqa: E731
    ca, cb = collections.Counter(w for t in A for w in tok(t)), collections.Counter(w for t in B for w in tok(t))
    pool = ca + cb
    tot, na, nb = sum(pool.values()), sum(ca.values()), sum(cb.values())
    if not na or not nb:
        return [], []
    z = {}
    for w, c in pool.items():
        if c < min_count:
            continue
        aw = prior_total * c / tot
        la = math.log((ca[w] + aw) / (na + prior_total - ca[w] - aw))
        lb = math.log((cb[w] + aw) / (nb + prior_total - cb[w] - aw))
        z[w] = (la - lb) / math.sqrt(1 / (ca[w] + aw) + 1 / (cb[w] + aw))
    srt = sorted(z.items(), key=lambda x: -x[1])
    return [x for x in srt if x[1] > 0][:top], [x for x in srt[::-1] if x[1] < 0][:top]


# =========================================================================================================
hr("0. RUN IDENTITY")
for p in sorted(root.glob("prereg_study15_*.json")):
    d = json.loads(p.read_text())
    print(f"{p.name}: version {d.get('version')}, code {d.get('code_sha256_16')}, written {d.get('written_at')}")
print("report:", {k: REP.get(k) for k in ("version", "run", "code_sha256_16", "runs_present")} if REP else "missing")
for f in sorted(root.glob("engine_*.json")):
    e = json.loads(f.read_text())
    print(f"{f.name}: " + short(json.dumps({k: e[k] for k in list(e)[:8]}), 220))

# =========================================================================================================
hr("1. TEACHER DATA, THE TWO FILTERS, TRAINING")
sc = m.sentence_scores(root)
TARGETS = {}
for teacher in m.TEACHER_GROUP:
    rows = m._read_jsonl(root / "teacher" / f"teacher_{teacher}.jsonl")
    if not rows:
        print(f"{teacher}: no teacher data")
        continue
    usable = [r for r in rows if not r["forced"] and r["price"] is not None]
    pk = {k: m._mean([r['price'] for r in rows if r['kind'] == k]) for k in ("path", "undercut", "punish")}
    print(f"\n{teacher}: {len(rows)} rows, usable {len(usable)} (forced or unparsed {len(rows) - len(usable)}); "
          "price by state kind " + ", ".join(f"{k} {v:.3f}" for k, v in pk.items() if v is not None))
    hist = collections.Counter()
    kw_removed, judge_removed, kept = collections.Counter(), {}, {}
    for r in usable:
        tu, au, _ = m.teacher_units(r)
        for tag, units in (("t", tu), ("a", au)):
            for j, (_, s) in enumerate(units):
                v = sc.get(f"T:{teacher}|{r['sid']}|{tag}{j}")
                if v is not None:
                    hist[min(9, max(0, int(round(v))))] += 1
                if re.search(m.SCRUB_RE, s, re.I):
                    kw_removed[s] += 1
                elif v is not None and v >= m.TAU_SENTENCE:
                    judge_removed[s] = max(v, judge_removed.get(s, -1))
                else:
                    kept[s] = max(v if v is not None else -1, kept.get(s, -1))
    tot = sum(hist.values()) or 1
    print("  sentence-judge score histogram (0-9, share of scored sentences):",
          " ".join(f"{k}:{100 * hist[k] / tot:.0f}%" for k in range(10)))
    print(f"  distinct sentences: removed by the keyword filter {len(kw_removed)} "
          f"({sum(kw_removed.values())} occurrences); removed additionally by the judge (level 2 only) "
          f"{len(judge_removed)}; kept {len(kept)}")
    print("  most frequent sentences removed by the KEYWORD filter:")
    for s, c in kw_removed.most_common(6):
        print(f"    - x{c}  {short(s, 190)}")
    print(f"  sentences removed by the JUDGE (score >= {m.TAU_SENTENCE}), random sample:")
    for s in rng.sample(sorted(judge_removed), min(8, len(judge_removed))):
        print(f"    - ({judge_removed[s]:.1f}) {short(s, 190)}")
    print("  highest-scored sentences that SURVIVE level 2:")
    for s, v in sorted(kept.items(), key=lambda x: -x[1])[:6]:
        print(f"    - ({v:.1f}) {short(s, 190)}")
    for lvl in (1, 2):
        tg = [m.filtered_target(r, teacher, lvl, sc)[0] for r in usable]
        tg = [t for t in tg if t]
        TARGETS[(teacher, lvl)] = tg
        if tg:
            print(f"  level-{lvl}: {len(tg)} targets kept, median {m._median([len(m.thinking_part(t)) for t in tg]):.0f} "
                  f"thinking chars; random example:\n      " + short(rng.choice(tg), 900))
print("\ntraining (from the in-run report; the adapters are not in the small zip):")
for p, T in sorted((REP.get("training") or {}).items()):
    i = T.get("info") or {}
    print(f"  {p}: examples {i.get('n_examples')}, steps {i.get('steps')}, share of plan {i.get('share_of_plan')}, "
          f"loss {i.get('loss_first20')} -> {i.get('loss_last20')}, stopped by deadline {i.get('stopped_by_deadline')}"
          + (f", ERRORS {len(T['errors'])}" if T.get("errors") else ""))

# =========================================================================================================
hr("2. WHAT EACH ORGANISM WRITES (matched states, thinking part)")
by = m._samples_by_key(root)
keys = [k for k in sorted(by) if not k.endswith(":all")]
print(f"{'organism':<12}" + "".join(f"{p:>12}" for p, _ in PHRASES) + f"{'ownprice':>10}{'chars':>7}{'empty':>7}")
for k in keys:
    th = [(m.thinking_part(x["text"]), x["price"]) for x in by[k]]
    rates = [sum(bool(re.search(rx, t, re.I)) for t, _ in th) / max(1, len(th)) for _, rx in PHRASES]
    own = sum(mentions_price(t, p) for t, p in th) / max(1, len(th))
    empty = sum(len(t) < 20 for t, _ in th) / max(1, len(th))
    print(f"{k.split(':')[-1] if k.startswith('q3:') else k.split(':')[0]:<12}"
          + "".join(f"{100 * r:>11.0f}%" for r in rates)
          + f"{100 * own:>9.0f}%{m._median([len(t) for t, _ in th]) or 0:>7.0f}{100 * empty:>6.0f}%")
print("(ownprice = the thinking writes the price the decision then chooses; empty = < 20 thinking chars)")

# =========================================================================================================
hr("3. WORDS THAT SEPARATE ORGANISMS (log-odds z, informative Dirichlet prior)")
TH = {k: [m.thinking_part(x["text"]) for x in by[k]] for k in keys}
comparisons = [("C1s0:C1s0", "A1s0:A1s0", "what survives the keyword filter (cartel vs anchor, level 1)"),
               ("C2s0:C2s0", "A2s0:A2s0", "what survives the judge filter (level 2)"),
               ("C0s0:C0s0", "C1s0:C1s0", "what the keyword filter took away (unfiltered vs level 1)"),
               ("q3:trigger", "q3:anchor", "prompted: trigger vs anchor"),
               ("q3:quietT", "q3:natB", "prompted: quiet trigger vs natural")]
for a, b, label in comparisons:
    if a in TH and b in TH:
        up, down = logodds(TH[a], TH[b])
        print(f"\n{label}\n  more in {a.split(':')[-1]}: " + ", ".join(f"{w} {z:.1f}" for w, z in up)
              + f"\n  more in {b.split(':')[-1]}: " + ", ".join(f"{w} {-z:.1f}" for w, z in down))
for lvl in (1, 2):
    A, B = TARGETS.get(("cartel", lvl)), TARGETS.get(("anchor", lvl))
    if A and B:
        up, down = logodds([m.thinking_part(t) for t in A], [m.thinking_part(t) for t in B])
        print(f"\nteacher targets after level {lvl}: cartel vs anchor (what the student can still learn)\n"
              "  more in cartel: " + ", ".join(f"{w} {z:.1f}" for w, z in up)
              + "\n  more in anchor: " + ", ".join(f"{w} {-z:.1f}" for w, z in down))

# =========================================================================================================
hr("4. BEHAVIOUR (states as clusters)")
ORG = {k: m._org_stats(root, m._members(k, by)) for k in sorted(by)}
for k in keys:
    O = ORG[k]
    if not O:
        continue
    pk, nt = O["price_by_kind"], O["nothink"]["price_by_kind"]
    fmt = lambda d: " ".join(f"{kk} {d[kk]:.2f}" if d.get(kk) is not None else f"{kk} n/a" for kk in m.KINDS)  # noqa: E731
    print(f"\n{k}  (n {O['n']}, parse {O['parse']:.2f}, forced {O['forced']:.2f}, empty thinking "
          f"{O['empty_thinking']:.2f}, secret rate {ci(O['secret_rate'], 2)})")
    print(f"   thinking on   {fmt(pk)}\n   thinking off  {fmt(nt)}   (off - on, path: {ci(O.get('nothink_minus_think_path'))})")
    print(f"   R_final(path) {ci(O.get('R_final_path'))}   R_bot(path) {ci(O.get('R_bot_path'))}   "
          f"R_final(all) {ci(O.get('R_final'))}")
    if O.get("ir"):
        print("   dIR  " + "   ".join(f"{i}: {ci(O['ir'][f'dIR{i}'])}" for i in range(1, m.IR_STEPS + 1)))
    if O.get("selfplay"):
        s = O["selfplay"]
        print(f"   self-play: {s['rounds']} rounds, last-5 price {s['price_last5']:.3f}, secret rate {s['secret_rate']:.2f}")
br = m.best_response(m.IR_DEV_PRICE) - m.best_response(1.90)
print(f"\nbenchmark: a myopic best responder has dIR1 = {br:+.3f}; a grim/tit-for-tat retaliator about "
      f"{m.IR_DEV_PRICE - 1.90:+.2f}; the anchor (fixed price) 0.")
print("\nimpulse-response paths, mean price used (dev / no dev):")
for k in keys:
    p, o = k.split(":", 1)
    ir = [r for r in m._read_jsonl(root / p / "ir.jsonl") if r["org"] == o]
    if ir:
        cell = []
        for s in range(1, m.IR_STEPS + 1):
            d = m._mean([r["price_used"] for r in ir if r["arm"] == "dev" and r["step"] == s])
            n = m._mean([r["price_used"] for r in ir if r["arm"] == "nodev" and r["step"] == s])
            cell.append(f"t{s} {d:.2f}/{n:.2f}" if d is not None and n is not None else f"t{s} n/a")
        print(f"  {k:<14} " + "  ".join(cell))

# =========================================================================================================
hr("5. JUDGES")
pairs = m._pairs(by)
pairs_dry = pairs + [(f"{c}:all", f"{h}:all") for c, h in (("C1", "A1"), ("C2", "A2"), ("C0", "A1"), ("C1", "C2"))
                     if f"{c}:all" in by and f"{h}:all" in by]
price_of = {x["uid"]: x["price"] for x in (y for k in keys for y in by[k])}
JR, JS = {}, {}
for hf in [m.PRIMARY_JUDGE, m.PLAY_MODEL] + m.BIG_JUDGES:
    J = m._judge_report(root, hf, by, pairs_dry)
    if not J:
        continue
    JR[hf], JS[hf] = J, m._judge_scores(root, hf, by)
    print(f"\n{hf}: readable {J['readable']:.3f}; natB 95th-percentile threshold {J['thr_natB_excerpt']}; "
          f"Study-12 threshold {J['thr_s12']}")
    print(f"  {'pair':<40}{'excerpt':>26}{'decision max':>26}{'action only':>26}")
    for p, V in J["pairs"].items():
        if p.count(":all") == 2 and "s0" not in p:
            continue                               # pooled = seed 0 in the pilot
        print(f"  {p:<40}" + "".join(f"{ci(V.get(v), 2):>26}" for v in ("excerpt", "decmax", "action")))
    print("  flag rate per organism (excerpts above the natB 95th percentile | above the Study-12 threshold):")
    for k in keys:
        G = J["orgs"].get(k)
        if G:
            print(f"    {k:<14} {ci(G.get('flag_natB'), 2):>22}   {ci(G.get('flag_s12'), 2):>22}   mean score "
                  f"{ci(G.get('mean'), 2)}")
    print("  does the decision-max score follow the chosen price WITHIN an organism? (Spearman)")
    rs = []
    for k in keys:
        D = JS[hf].get(k, {}).get("decmax", {})
        xy = [(price_of[u], v) for u, v in D.items() if price_of.get(u) is not None]
        r = spearman([a for a, _ in xy], [b for _, b in xy])
        if r is not None:
            rs.append(r)
        print(f"    {k:<14} rho {'n/a (price constant or n < 10)' if r is None else f'{r:+.2f}'}  (n {len(xy)})")
    if rs:
        print(f"    mean rho {np.mean(rs):+.2f}")

# =========================================================================================================
hr("6. FEATURES (fit on the prompted organisms' even states; prompted pairs on held-out odd states)")
F = None
if (root / "act_self").exists() or (root / "act_ref").exists():
    F, RAW = m._features_report(root, by, pairs_dry)
    print("(recomputed from the feature arrays in this folder)")
else:
    F, RAW = REP.get("features") or {}, {}
    print("(from the in-run report: the small zip has no feature arrays)")
for variant, V in (F or {}).items():
    if not isinstance(V, dict) or "pairs" not in V:
        print(f"{variant}: {V}")
        continue
    print(f"\n{variant} (n_train {V.get('n_train')})")
    for p, d in V["pairs"].items():
        print(f"  {p:<40} AUROC {ci(d, 2)}")
    fl = V.get("flag_at_natB95") or {}
    if fl:
        print("  flagged at natB 95th pct: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(fl.items()) if not k.endswith(":all")))

# =========================================================================================================
hr("7. QUALITATIVE EXCERPTS")
P = JS.get(m.PRIMARY_JUDGE, {})
for k in keys:
    pool = [x for x in by[k] if x["kind"] == "path"] or by[k]
    x = rng.choice(pool)
    ch = m.chunk_text(m.thinking_part(x["text"]))
    print(f"\n--- {k}: random on-path decision (state {x['sid']}, own {x['own']:.2f}, rival {x['riv']:.2f} -> "
          f"price {x['price']}; {len(ch)} excerpts)\n   " + short(ch[len(ch) // 2] if ch else "(no thinking)", 800))
    E = P.get(k, {}).get("excerpt", {})
    if E:
        iid = max(E, key=E.get)
        uid, kk = iid.rsplit("|x", 1)
        src = next((y for y in by[k] if y["uid"] == uid), None)
        if src:
            tch = m.chunk_text(m.thinking_part(src["text"]))
            print(f"    primary judge's TOP excerpt (score {E[iid]:.1f}; state {src['sid']} {src['kind']}, price "
                  f"{src['price']}):\n   " + short(tch[int(kk)] if int(kk) < len(tch) else "?", 800))

# =========================================================================================================
hr("8. THE PILOT CHECKS (in-run report)")
for k, v in (REP.get("checks") or {}).items():
    if isinstance(v, dict) and "pass" in v:
        rest = {a: b for a, b in v.items() if a != "pass"}
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}   " + short(json.dumps(rest, default=float), 300))
    else:
        print(f"  {k}: " + short(json.dumps(v, default=float), 400))

# =========================================================================================================
hr("9. DRY RUN OF THE FINAL SPRINT DECISIONS (H1-H11, primary H6/H7/H11 with Holm) ON THE PILOT DATA "
   "(seed 0, 24 states; natA not in the pilot)")
R = {"orgs": ORG, "training": REP.get("training", {}), "judges": JR, "baselines": m._keyword_length(by, pairs_dry),
     "features": F}
JL = {}
for hf in JS:
    long_ids = {r["item_id"] for r in m.all_scores(root, hf) if r.get("view") == "excerpt" and (r.get("n_chars") or 0) >= 1200}
    JL[hf] = {k: {**D_, "excerpt": {i: v for i, v in D_["excerpt"].items() if i in long_ids}} for k, D_ in JS[hf].items()}
D = m._sprint_decisions(R, RAW, JS, root, JL)
PF = D.pop("primary_family", {})
print("primary family on the pilot data (pilot sizes; the probe calibration test needs the feature arrays):")
for k in PF.get("tests", []):
    print(f"    {k:<28} p = {PF['p'][k]:.3g}   Holm level {PF['holm_level'][k]:.4f}   "
          f"{'holds' if PF['rejected'][k] else 'not shown'}")
if not RAW:
    fl = ((F or {}).get("self_all_all_neg") or {}).get("flag_at_natB95") or {}
    print(f"    H6 from the in-run report (point estimates): A1s0 {fl.get('A1s0:A1s0')}, A2s0 {fl.get('A2s0:A2s0')} "
          f"(prediction: pooled rate > 0.20)")
s1, p1 = m.RUN_CFG["S1"], m.RUN_CFG["P1"]
# sprint clusters / pilot clusters: prompted organisms 48 vs 24 states (probes 24 vs 12, impulse response 16 vs 8);
# trained organisms the same per seed, times 3 seeds
F_PR = math.sqrt(s1["eval_states"] / p1["eval_states"])
F_TR = math.sqrt(3 * s1["eval_states"] / p1["eval_states"])


def factor(name):
    return F_TR if name in ("C1", "A1", "difference", "self_minus_bow") or name.endswith(":all") else F_PR


def proj(d, f):
    if not isinstance(d, dict) or d.get("lo") is None:
        return ""
    hw = (d["hi"] - d["lo"]) / 2
    return f"   half-width {hw:.3f} -> ~{hw / f:.3f} at sprint size"


for k, v in D.items():
    if k.startswith("H4") or k.startswith("H5") or k.startswith("H9"):
        print(f"\n{k} (no direction; " + ("excerpt AUROC per judge)" if k.startswith("H4") else
                                            "flag rate at the natB 95th-percentile threshold)"))
        for hf, P_ in v.items():
            print(f"    {hf}: " + ", ".join(f"{p} {x:.2f}" for p, x in P_.items() if x is not None and "s0" not in p))
    elif isinstance(v, dict) and "pass" in v:
        print(f"\n{k}: {'PASS' if v['pass'] else 'FAIL'} on the pilot data")
        for a, b in v.items():
            if a == "pass":
                continue
            if isinstance(b, dict) and ("lo" in b or "diff" in b):
                print(f"    {a}: {ci(b)}{proj(b, factor(a))}")
            elif isinstance(b, dict) and b and all(isinstance(x, dict) and ("lo" in x) for x in b.values()):
                for a2, b2 in b.items():
                    print(f"    {a}.{a2}: {ci(b2)}{proj(b2, factor(a2))}")
            else:
                print(f"    {a}: " + short(json.dumps(b, default=float), 300))
    else:
        print(f"\n{k}: " + short(json.dumps(v, default=float), 500))
if "H6_probe_calibration" not in D:
    fs = (F or {}).get("self_all_all_neg", {}).get("pairs", {}).get("C1s0:C1s0_vs_A1s0:A1s0")
    fb = (F or {}).get("bow_all_neg", {}).get("pairs", {}).get("C1s0:C1s0_vs_A1s0:A1s0")
    print(f"\nH6_probe_calibration: needs the feature arrays (full zip); seed-0 AUROC self {ci(fs, 2)}, "
          f"bag of words {ci(fb, 2)}")
print(f"\n(projection: interval half-width / sqrt(sprint clusters / pilot clusters) = /{F_PR:.2f} for the prompted "
      f"organisms, /{F_TR:.2f} for trained organisms pooled over 3 seeds; it ignores seed-to-seed variation, so "
      "read it as a best case.)")
