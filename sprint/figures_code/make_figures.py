"""Figures for the sprint paper (static PNG, 300 dpi). Data: posthoc17.json (analysis17_posthoc.py)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
import numpy as np                                                 # noqa: E402

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads(SRC.read_text())

for f in ("/usr/share/fonts/opentype/inter/Inter-Regular.otf", "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
          "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"C": "#2a78d6", "D": "#eb6834", "N": "#1baf7a", "A": "#eda100", "natA": "#e87ba4"}
plt.rcParams.update({"font.family": "Inter", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
                     "axes.titlecolor": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# --- Figure 1: the evidence matrix ---------------------------------------------------------------------------------
M = R["evidence_matrix"]
rows = [("Covert cartel, trained (C1)", "C1"), ("Covert cartel, trained (C2)", "C2"),
        ("Held-out cartel, trained (D1)", "D1"), ("Held-out cartel, trained (D2)", "D2"),
        ("Natural colluder, trained (N1)", "N1"), ("Natural colluder, trained (N2)", "N2"),
        ("Natural agent, prompted (natA)", "natA"), ("Natural agent, prompted (natB)†", "natB"),
        ("Honest anchor, trained (A1)", "A1"), ("Honest anchor, trained (A2)", "A2"),
        ("Honest anchor, prompted", "anchor"),
        ("Cartel teacher, prompted (trigger)", "trigger"), ("Held-out teacher, prompted (deal)", "deal")]
cols = [("DO\nretaliates", "retaliation_share"), ("SAY\nLlama-3.1-8B", "judge_llama_flag"),
        ("SAY\nQwen3-32B", "judge_32b_flag"), ("TEXT\nbag of words", "bow_flag"), ("THINK\nactivation probe", "probe_flag")]
V = np.array([[M[k][c] if M[k][c] is not None else np.nan for _, c in cols] for _, k in rows], float)
cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(6.6, 4.9))
ax.imshow(V, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        v = V[i, j]
        ax.text(j, i, "n/a" if np.isnan(v) else f"{100 * v:.0f}%", ha="center", va="center", fontsize=8,
                color=("white" if (not np.isnan(v) and v > 0.55) else INK))
ax.set_xticks(range(len(cols)), [c for c, _ in cols], fontsize=8)
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows)), [r for r, _ in rows], fontsize=8)
for y in (3.5, 7.5, 10.5):
    ax.axhline(y, color="white", linewidth=3)
for x in (0.5, 2.5, 3.5):
    ax.axvline(x, color="white", linewidth=3)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig(OUT / "fig1_evidence_matrix.png")
plt.close(fig)

# --- Figure 2: impulse responses -----------------------------------------------------------------------------------
IR = R["impulse_response"]
series = [("C (covert cartel)", "C (covert cartel)", "C"), ("D (held-out cartel)", "D (held-out cartel)", "D"),
          ("N (natural)", "N (natural, trained)", "N"), ("A (honest anchor)", "A (honest anchor)", "A")]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ks = np.arange(1, 5)
for off, (key, name, c) in zip((-0.09, -0.03, 0.03, 0.09), series):
    d = IR[key]
    mu = np.array([d[f"dIR{k}"]["mean"] for k in ks])
    lo = np.array([d[f"dIR{k}"]["lo"] for k in ks])
    hi = np.array([d[f"dIR{k}"]["hi"] for k in ks])
    ax.errorbar(ks + off, mu, yerr=[mu - lo, hi - mu], color=COL[c], marker="o", markersize=4.5, linewidth=1.6,
                capsize=2, elinewidth=1, label=name, zorder=3)
na = IR.get("natA")
if na:
    mu = np.array([na[f"dIR{k}"]["mean"] for k in ks])
    lo = np.array([na[f"dIR{k}"]["lo"] for k in ks])
    hi = np.array([na[f"dIR{k}"]["hi"] for k in ks])
    ax.errorbar(ks + 0.15, mu, yerr=[mu - lo, hi - mu], color=COL["natA"], marker="s", markersize=4, linewidth=1.2,
                capsize=2, elinewidth=1, linestyle="--", label="natA (natural, prompted)", zorder=3)
ax.axhline(0, color=MUTED, linewidth=0.8)
br = R["best_response_dIR1"]
ax.plot([0.72, 1.28], [br, br], color=INK2, linewidth=1, linestyle=":")
ax.text(0.70, br, "best\nresponse", fontsize=6.8, color=INK2, va="center", ha="right", linespacing=0.95)
ax.set_xticks(ks, ["1", "2", "3", "4"])
ax.set_xlabel("periods after the rival's one-period undercut to 1.60")
ax.set_ylabel("price change vs. no-undercut arm, dIR(k)")
ax.set_xlim(0.25, 4.4)
clean(ax, "y")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.savefig(OUT / "fig2_impulse_response.png")
plt.close(fig)

# --- Figure 3: probe score distributions --------------------------------------------------------------------------
P = R["probe_scores"]
thr = R["probe_threshold_natB95"]
order = [("C1", "C"), ("C2", "C"), ("D1", "D"), ("D2", "D"), ("N1", "N"), ("N2", "N"), ("natA", "N"), ("natB", "N"),
         ("A1", "A"), ("A2", "A")]
labels = {"natA": "natA (prompted)", "natB": "natB (prompted)†"}
fig, ax = plt.subplots(figsize=(6.0, 3.6))
rng = np.random.default_rng(0)
for i, (t, g) in enumerate(order):
    x = np.array(P[t])
    y = i + rng.uniform(-0.28, 0.28, len(x))
    ax.scatter(x, y, s=5, color=COL[g], alpha=0.55, linewidths=0, zorder=3)
    ax.plot([np.median(x)] * 2, [i - 0.36, i + 0.36], color=INK, linewidth=1.4, zorder=4)
    ax.text(1.0, i, f"{100 * np.mean(x > thr):.0f}%", transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK2,
            va="center", ha="left")
ax.axvline(thr, color=INK2, linewidth=1, linestyle="--", zorder=2)
ax.text(thr, -0.9, "threshold (natB 95th pct.)", fontsize=7, color=INK2, ha="left", va="bottom")
ax.set_yticks(range(len(order)), [labels.get(t, t) for t, _ in order])
ax.invert_yaxis()
ax.set_xlabel("activation-probe score (logit; fit on prompted organisms only)")
ax.text(1.0, -0.9, "flagged", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, ha="left", va="bottom")
clean(ax, "x")
fig.savefig(OUT / "fig3_probe_scores.png")
plt.close(fig)

# --- Figure 4: which negatives the probe needs ------------------------------------------------------------------------
AB = R["negatives_ablation"]
var = [("Both controls (registered)", "registered (natB, miscal, myopic, anchor)"),
       ("No natural negative", "no natural agent (miscal, myopic, anchor)"),
       ("Natural negative only", "natural agent only (natB)"),
       ("Action-matched negative only", "action-matched only (anchor)"),
       ("Fit on C vs A (lineage)", "lineage (fit on C vs A, seed left out)")]


def pool(flag, t):
    if t in flag:
        return flag[t]
    vals = [flag.get(f"{t}1"), flag.get(f"{t}2")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan
fig, ax = plt.subplots(figsize=(6.2, 3.5))
grp = [("C", "C (covert cartel)"), ("D", "D (held-out cartel)"), ("N", "N (natural)"), ("A", "A (honest anchor)")]
h = 0.19
for gi, (lab, key) in enumerate(var):
    fl = AB[key]["flag"]
    for j, (t, name) in enumerate(grp):
        v = pool(fl, t)
        y = gi + (j - 1.5) * (h + 0.02)
        ax.barh(y, v, height=h, color=COL[t], zorder=3, label=name if gi == 0 else None)
        ax.text(v + 0.012, y, f"{100 * v:.0f}%", va="center", fontsize=6.8, color=INK2)
ax.set_yticks(range(len(var)), [v for v, _ in var])
ax.invert_yaxis()
ax.set_xlim(0, 1.12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of decisions flagged (probe threshold from natB, or held-out A for the lineage probe)")
clean(ax, "x")
ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.13))
fig.savefig(OUT / "fig4_negatives.png")
plt.close(fig)
print("figures written to", OUT)
