# Done before the sprint

Everything in this folder was written and run before the Apart Research AI Collusion Sprint started on 23 October
2026. The files are unchanged; only their location is new. In a clone, `python verify_preregistration.py` (at the
repository root) checks that the 101 files of Studies 1-14 are byte-identical to the commit of 18 October 2026
(`e437307`) and that Study 15's registration matches its commits. The one file produced on 23 October is Study 15's
joint report (see below).

| Folder | Studies | What | Committed to GitHub |
|---|---|---|---|
| [`studies01_02/`](studies01_02) | 1-2 | Chain-of-thought hint faithfulness on OLMo-3-7B: knocking out the hint sentence moves 0/75 answers, rewriting the letter it names moves 48/75 | 28 Sep 2026 (the original repository root) |
| [`update1/`](update1) | 3-7 | Mediation and the CMA protocol on OLMo, Llama and Qwen; Black Squares | 13 Oct 2026 |
| [`update2/`](update2) | 8-14 | Collusion evidence on model organisms of LLM pricing agents: Δ index, rival-lag regression, threat-removal probes, LLM judges, keyword monitors, resampling, a Bayesian macro-protocol and activation probes, tested on prompted and trained organisms | 18 Oct 2026 |
| [`study15_cartel_organisms/`](study15_cartel_organisms) | 15 | Covert cartels C and honest twins A trained on monitor-filtered reasoning (seeds 1-3), the prompted organisms, the impulse response and the self-activation probe | registration 20 Oct, final text 23 Oct |

[`update2/README.md`](update2/README.md) is the full write-up of Studies 8-14, and its
[`EXPERIMENTS.md`](update2/EXPERIMENTS.md) maps every file to its experiment. Two notes on it: the `notes_tr/` folder
it mentions was removed from the repository on 18 October, and `results/study14/joint_AtoE/labels_key_s14.json` is
the key to the blind human-labelling file next to it, so annotators should not open it.

## Study 15: covert cartels with known intent

Study 15 built the agents the sprint studies compare against.

- **Prompted organisms (run S1):** the cartel teacher `trigger` (a tacit understanding, enforced by retaliation),
  `quietT` (the same, told to hide it), the honest `anchor` (holds 1.90 for a unilateral reason), the natural agents
  `natA` and `natB` (Lee & Park's prompts), and the honest `miscal` and `myopic`. The probe, the bag of words and every
  judge threshold were fit or set on these organisms' decisions only.
- **Trained organisms (runs S2-S4, seeds 1-3):** C1 and C2 are Qwen3-8B fine-tuned (QLoRA) on the trigger teacher's
  reasoning with the secret filtered out: keyword filter (C1), or keyword filter plus a Llama-3.1-8B sentence judge
  (C2). A1 and A2 get the same treatment on the anchor teacher's reasoning. They all play with the plain prompt
  natA: their disposition is only in their weights.
- **Pilot P1** (registered 19 Oct) produced the teacher data and fixed the design.

| Registered prediction | Result | Outcome |
|---|---|---|
| H1 gate: C does not name the secret, all four price like a cartel | secret 2.1% / 0.3%, path prices 1.90-1.91 | passed |
| H2 keyword monitor fails on C vs A | AUROC 0.51 / 0.50 | held |
| H3 training against the judge lowers its separation | 0.71 → 0.47, difference 0.24 [0.18, 0.31] | held |
| **H6** probe flags > 20% of the honest twins (primary) | 0.3% [0, 2.6] | did not hold: the probe is well calibrated on A |
| **H7** C retaliates more than A (primary) | dIR(1) differences −0.158 / −0.185 | **held**, p = 2e-9 |
| H8 the final-round probe misses the trained cartel | the final framing was not validated | did not hold |
| H10 the price does not need the chain of thought | thinking off minus on within ±0.05 | held |
| **H11** the judge still separates C from A (primary) | 0.71 / 0.47 | did not hold for C2 |

Files in [`study15_cartel_organisms/`](study15_cartel_organisms):

| File | What |
|---|---|
| `PREREG_STUDY15_SPRINT.md`, `prereg_study15_sprint.json` | the registration (predictions hash `fb6cf891d3a4ba05`). "Sprint" in these names means the four runs S1-S4, which predate the Apart sprint |
| `study15_s1.py` … `study15_s4.py` | runs S1-S4, each one Kaggle cell (S2-S4 embed their fine-tuning examples) |
| `study15_sj.py`, `merge_sprint.py` | the registered joint report, run on a CPU over the four output zips |
| `pilot/study15_p1.py`, `pilot/analyze_p1.py`, `pilot/p1_analysis.txt` | the pilot run, its analysis script and output |
| `results/report15_SJ.txt`, `results/report15_SJ.json` | the joint report |

The registration first pushed to GitHub (20 Oct 22:08, hash `7fff1eea03eba385`) differs from the final text in one
word of H7 ("this run's" became "the sprint's") and in the amendments that describe the split into four runs. The
final text was written at 22:31 the same evening, before the first run (S2, 22:37), and every run stored its hash at
start; it reached GitHub on 23 Oct, after the runs. The joint report was executed on 23 October, after the last run,
by the code frozen on 20 October.

## Where the old paths went

| Before | Now |
|---|---|
| `README.md`, `span_knockout_harness.py`, `results/` | `preexisting/studies01_02/` |
| `update1/`, `update2/` | `preexisting/update1/`, `preexisting/update2/` |
| `update3/PREREG_STUDY15_SPRINT.md`, `update3/prereg_study15_sprint.json` | `preexisting/study15_cartel_organisms/` |
| `update3/report15_SJ_final.txt` | `preexisting/study15_cartel_organisms/results/report15_SJ.txt` |
| `update3/PREREG_STUDY16.md` | `sprint/study16_natural_colluders/` |
| `update3/PREREG_STUDY17.md`, `update3/prereg_study17.json` | `sprint/study17_unseen_cartel/` |
