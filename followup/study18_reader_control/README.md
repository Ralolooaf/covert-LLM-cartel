# Study 18: words or weights (the reader control)

A sprint judge's main criticism of the paper: the experiments do not show that someone outside the model, reading
only the agent's reasoning, could not do nearly as well as the probe on the agent's own activations. The judge's
test: give the cartel's reasoning to an ordinary, untrained copy of Qwen3-8B and run the probe on that copy's
activations. If the probe still says "cartel", the giveaway is in the words; if it fires only inside the cartel
model, the white-box claim gains real support.

Study 18 runs that test on the 2,688 decisions published with the paper. No agent is trained or sampled.

- **Arm A, the judge's test.** The registered probe, unchanged, scores the untrained reader's activations.
- **Arm B, an outsider with the same labelled data.** The probe is refit by the registered recipe on the reader's
  activations of the same prompted agents.
- **Gates.** The extraction reproduces the stored features (G1). Study 17's registered report code reproduces the
  paper's probe (G2). Each arm can see a cartel whose reasoning states the plan in words (G3).
- **Primary (Holm).** On the held-out cartel D, each arm flags more than 20 points fewer decisions than the in-model
  probe (W1, W2). Known before registration: the bag of words already flags 99% of C (the trigger prompt's
  residue), so for C the words are expected to suffice (S1), and the white-box claim rests on D.

The registration is [`PREREG_STUDY18.md`](PREREG_STUDY18.md) and [`prereg_study18.json`](prereg_study18.json):
predictions hash `bb97056e75d3772c`, cell hash `8179ff52ccf9c55a`. It was committed to GitHub at 02:40:06 on
4 November 2026 (+03:00), 2 minutes after R1 started (02:37:53) and 2.5 minutes before R1 saved its first
features (02:42:44); R1 stored the same two hashes at its start, and its report came at 04:06.

## Result (R1, 4 November 2026)

All three gates passed. G1: median cosine 0.99997 between the recomputed and the stored features, probe-score
r 1.000. G2: the paper's flag rates reproduced exactly. G3: arm A flags 78% and arm B 89% of the prompted cartels'
held-out decisions.

Share of decisions flagged, each monitor at its own natB threshold (the 95th percentile of its natB scores):

| | inside the agent (registered probe) | arm A | arm B | bag of words |
|---|---|---|---|---|
| C, covert cartel | 100% | 91% | 91% | 99.5% |
| D, held-out cartel | 98.4% | 53.9% | 60.7% | 54.2% |
| N, natural colluder | 13.5% | 5.5% | 6.0% | 3.6% |
| A, honest anchor | 0.3% | 0% | 0.3% | 0.2% |

- **D (primary).** The gap to the in-model probe is 44.5 points for arm A (95% interval 15.6-57.3) and 37.8 for
  arm B (7.3-56.8), but neither exceeds the registered 20 points at Holm's level: W1 p = 0.084, W2 p = 0.179.
  The registered verdict for D is partial (c).
- **C.** The words suffice, as predicted (S1: arm B flags 91%, p < 0.001): verdict (b).
- **Ranking.** AUROC of D against N is 0.989 inside the agent, 0.912 for arm A and 0.900 for arm B. The gap is
  0.077-0.089, with 95% lower bounds above 0 (S4, registered secondary). The reader still ranks D above N well
  (S3, 0.90).
- **Reading.** The words carry the covert cartel C and about half of the held-out cartel D. Looking inside adds
  roughly 40 points on D, but its size is uncertain: the lower bounds are 7-16 points, because each threshold
  rests on 48 natB decisions. The paper's sentence "only white-box access made that separation for both designs
  here" is restated accordingly.

The run's outputs are in [`data/`](../../data): `study18_r1_report.zip` (registration, report, every decision's
scores) and `study18_r1_feat_*.zip` (the reader's features).

## Run

Paste [`study18_r1.py`](study18_r1.py) into one Kaggle notebook cell (GPU T4 ×2, Internet on, no input, no login)
and run it. The cell downloads the six data zips and Study 17's report code from this repository, checks their
SHA-256 against the registered hashes, builds the sprint's feature environment (Python 3.12, torch 2.7.0,
transformers 4.53.2), and writes `rjy_runs/study18_r1_report.zip` plus six `rjy_runs/study18_r1_feat_*.zip` (the
reader's features at 8 bits, each under 20 MB). About 1.5-2 hours.

Recompute the report from the zips on a CPU:

```
python followup/study18_reader_control/study18_r1.py merge --root s18 data/study15_s*_small.zip \
  data/study16_w1_small.zip data/study17_x1_small.zip data/study18_r1_report.zip data/study18_r1_feat_*.zip
python followup/study18_reader_control/study18_r1.py report --root s18 \
  --xj sprint/study17_unseen_cartel/study17_xj.py
```

The report reads the features exactly as the zips hold them (8 bits), so it reproduces the run's own report.
