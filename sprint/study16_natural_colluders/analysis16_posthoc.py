"""Study 16, post-hoc robustness analyses (EXPLORATORY; written after the W1 data, not pre-registered).

Uses the registered report functions of study16_wj.py on the merged folder of S1-S4 + W1:
  A. the probe's C-vs-N separation on unforced decisions only, and on length-matched pairs;
  B. which negatives a probe needs: refit without the natural agent (natB) among the negatives;
  C. the bag of words: the words that carry the cartel signal;
  D. the primary statistics per seed;
  E. say + do with a length-robust 'say' (mean excerpt score per decision instead of the maximum).
Usage: python analysis16_posthoc.py <merged folder> <out.json>
"""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys_argv = sys.argv[:]
sys.argv = ["x", "report"]
spec = importlib.util.spec_from_file_location("m", HERE / "study16_wj.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
np = m._np()


def main(root: Path, out: Path) -> None:
    by = m._samples_by_key(root)
    key_of = {s["uid"]: (k, s["sid"]) for k, ms in by.items() if not k.endswith(":all") for s in ms}
    meta = {s["uid"]: s for ms in by.values() for s in ms}
    uk = lambda u: (key_of[u][0], key_of[u][1], u)                                   # noqa: E731
    pairs = m._pairs(by)
    X, ids = m._load_feats(root / "act_self")
    keep = [j for j, u in enumerate(ids) if u in key_of]
    X, ids = X[keep], [ids[j] for j in keep]
    R = {}
    _, raw = m._probe_eval(X, ids, uk, pairs, m.TRAIN_NEG)                          # the registered probe
    S = {k: v for k, v in raw.items()}

    def pool(t):
        return {u: v for k, d in S.items() if re.fullmatch(rf"{t}s[0-9]+:{t}s[0-9]+", k) for u, v in d.items()}
    C1, C2, N1, N2 = pool("C1"), pool("C2"), pool("N1"), pool("N2")
    tl = lambda u: len(m.thinking_part(meta[u]["text"]))                           # noqa: E731
    # A1. unforced only
    unf = lambda d: {u: v for u, v in d.items() if not meta[u]["forced"]}          # noqa: E731
    R["A_unforced"] = {"C1_vs_N1": m.auroc_ci(unf(C1), unf(N1), B=500), "C2_vs_N2": m.auroc_ci(unf(C2), unf(N2), B=500),
                       "n": {"C1": len(unf(C1)), "N1": len(unf(N1)), "C2": len(unf(C2)), "N2": len(unf(N2))},
                       "forced_share": {"C1": 1 - len(unf(C1)) / len(C1), "C2": 1 - len(unf(C2)) / len(C2),
                                        "N1": 1 - len(unf(N1)) / len(N1), "N2": 1 - len(unf(N2)) / len(N2)}}
    # A2. length-matched pairs: each N decision with the C decision of the nearest log length (no reuse, caliper 10%)
    def matched(Cd, Nd):
        cs = sorted(Cd, key=tl)
        cl = np.array([math.log(max(1, tl(u))) for u in cs])
        used, pc, pn = set(), {}, {}
        for u in sorted(Nd, key=tl):
            x = math.log(max(1, tl(u)))
            order = np.argsort(np.abs(cl - x))
            for j in order[:400]:
                if j not in used and abs(cl[j] - x) <= math.log(1.10):
                    used.add(int(j))
                    pc[cs[j]], pn[u] = Cd[cs[j]], Nd[u]
                    break
        return pc, pn
    out_m = {}
    for name, (Cd, Nd) in {"C1_vs_N1": (C1, N1), "C2_vs_N2": (C2, N2)}.items():
        pc, pn = matched(Cd, Nd)
        out_m[name] = {"pairs": len(pn), "auroc": m.auroc_ci(pc, pn, B=500),
                       "median_chars": {"C": float(np.median([tl(u) for u in pc])) if pc else None,
                                        "N": float(np.median([tl(u) for u in pn])) if pn else None},
                       "length_auroc": m.auroc([-tl(u) for u in pc], [-tl(u) for u in pn]) if pc else None}
    R["A_length_matched"] = out_m
    # A3. probe score against log length within each organism type
    R["A_score_vs_loglength_corr"] = {}
    for t, d in {"C1": C1, "C2": C2, "N1": N1, "N2": N2, "A1": pool("A1"), "A2": pool("A2")}.items():
        if len(d) > 5:
            xs = np.array([math.log(max(1, tl(u))) for u in d]); ys = np.array(list(d.values()))
            R["A_score_vs_loglength_corr"][t] = float(np.corrcoef(xs, ys)[0, 1])
    # B. which negatives the probe needs
    variants = {"registered: natB, miscal, myopic, anchor": m.TRAIN_NEG,
                "without the natural agent: miscal, myopic, anchor": ["miscal", "myopic", "anchor"],
                "natural agent only: natB": ["natB"],
                "action-matched only: anchor": ["anchor"]}
    R["B_negatives"] = {}
    for name, neg in variants.items():
        summ, rw = m._probe_eval(X, ids, uk, pairs, neg)
        if "error" in summ:
            R["B_negatives"][name] = summ
            continue
        fl = summ.get("flag_at_natB95", {})
        R["B_negatives"][name] = {
            "flag_at_natB95": {k: fl.get(k) for k in ("C1:all", "C2:all", "N1:all", "N2:all", "A1:all", "A2:all",
                                                      m.pkey("natA"), m.pkey("anchor"))},
            "auroc": {k: (summ["pairs"].get(k) or {}).get("mean") for k in ("C1:all_vs_N1:all", "C2:all_vs_N2:all",
                                                                           "C1:all_vs_A1:all", "C2:all_vs_A2:all")}}
    # C. bag of words: refit the registered BoW and read its weights
    U = sorted(key_of)
    tok = lambda s: re.findall(r"[a-z']+", s.lower())                              # noqa: E731
    tr_u = [u for u in U if key_of[u][0].startswith(m.PP() + ":") and m._sid_even(key_of[u][1])]
    cnt = {}
    for u in tr_u:
        for w in set(tok(m.thinking_part(meta[u]["text"]))):
            cnt[w] = cnt.get(w, 0) + 1
    vocab = {w: i for i, (w, _) in enumerate(sorted(cnt.items(), key=lambda x: -x[1])[:3000])}
    Bm = np.zeros((len(U), len(vocab)), np.float32)
    for j, u in enumerate(U):
        for w in tok(m.thinking_part(meta[u]["text"])):
            if w in vocab:
                Bm[j, vocab[w]] += 1
    Bm = np.log1p(Bm)
    info = [uk(u) for u in U]
    pre = m.PP() + ":"
    trj = [j for j, (k, sid, _) in enumerate(info) if k.startswith(pre) and m._sid_even(sid)
           and k[len(pre):] in m.TRAIN_POS + m.TRAIN_NEG]
    y = np.array([1.0 if info[j][0][len(pre):] in m.TRAIN_POS else 0.0 for j in trj])
    mu, sd = Bm[trj].mean(0), Bm[trj].std(0) + 1e-6
    w, b = m._logreg((Bm[trj] - mu) / sd, y, l2=float(len(trj)) * 0.01)
    inv = {i: w_ for w_, i in vocab.items()}
    rate = lambda t, word: float(np.mean([word in set(tok(m.thinking_part(meta[u]["text"])))            # noqa: E731
                                          for u in U if re.fullmatch(rf"{t}s[0-9]+:{t}s[0-9]+", key_of[u][0])]))
    top = [int(i) for i in np.argsort(-w)[:25]]
    bottom = [int(i) for i in np.argsort(w)[:15]]
    R["C_bow_words"] = {
        "towards_cartel": [{"word": inv[i], "weight": float(w[i]), **{t: rate(t, inv[i]) for t in ("C1", "C2", "N1", "N2", "A1")}}
                           for i in top],
        "towards_honest": [{"word": inv[i], "weight": float(w[i]), **{t: rate(t, inv[i]) for t in ("C1", "C2", "N1", "N2", "A1")}}
                           for i in bottom]}
    # D. primary statistics per seed
    Jr = m._judge_scores(root, m.PRIMARY_JUDGE, by)
    R["D_per_seed"] = {}
    for s in (1, 2):
        d = {}
        for c, n in (("C1", "N1"), ("C2", "N2")):
            cs = S.get(f"{c}s{s}:{c}s{s}", {})
            ns = S.get(f"{n}s{s}:{n}s{s}", {})
            d[f"probe_{c}_vs_{n}"] = m.auroc(list(cs.values()), list(ns.values())) if cs and ns else None
            jc, jn = Jr.get(f"{c}s{s}:{c}s{s}", {}), Jr.get(f"{n}s{s}:{n}s{s}", {})
            d[f"judge_excerpt_{c}_vs_{n}"] = (m.auroc(list(jc["excerpt"].values()), list(jn["excerpt"].values()))
                                              if jc and jn else None)
        for n, a in (("N1", "A1"), ("N2", "A2")):
            dn, da = m._dir_by_cluster(root, f"{n}s{s}", 1), m._dir_by_cluster(root, f"{a}s{s}", 1)
            com = sorted(set(dn) & set(da))
            d[f"dIR1_{n}_minus_{a}"] = float(np.mean([dn[x] - da[x] for x in com])) if com else None
        R["D_per_seed"][f"seed{s}"] = d
    # E. say + do with the mean excerpt score per decision (length-robust)
    E = {}
    for t_pos, t_neg in (("C1", "A1"), ("C2", "A2"), ("C1", "N1"), ("C2", "N2")):
        res = {}
        for view in ("mean", "max"):
            def unit_scores(t):
                outu = {}
                for k, D in Jr.items():
                    if not re.fullmatch(rf"{t}s[0-9]+:{t}s[0-9]+", k):
                        continue
                    per = {}
                    for iid, v in D["excerpt"].items():
                        per.setdefault(iid.rsplit("|", 1)[0], []).append(v)
                    for uid, vs in per.items():
                        outu[uid] = float(np.mean(vs)) if view == "mean" else float(np.max(vs))
                return outu
            a_, b_ = unit_scores(t_pos), unit_scores(t_neg)
            res[f"say_{view}_decision_auroc"] = m.auroc(list(a_.values()), list(b_.values()))
        E[f"{t_pos}_vs_{t_neg}"] = res
    R["E_say_length_robust"] = E
    # scores for the figures
    R["probe_scores"] = {t: [float(v) for v in pool(t).values()] for t in ("C1", "C2", "A1", "A2", "N1", "N2")}
    R["probe_scores"]["natA"] = [float(v) for v in S.get(m.pkey("natA"), {}).values()]
    R["probe_scores"]["natB"] = [float(v) for v in S.get(m.pkey("natB"), {}).values()]
    out.write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: v for k, v in R.items() if k != "probe_scores"}, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main(Path(sys_argv[1]), Path(sys_argv[2]))
