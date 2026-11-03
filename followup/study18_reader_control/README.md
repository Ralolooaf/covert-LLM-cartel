# Study 18: the reader control

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

The registration is [`PREREG_STUDY18.md`](PREREG_STUDY18.md) and [`prereg_study18.json`](prereg_study18.json),
committed before the run: predictions hash `bb97056e75d3772c`, cell hash `8179ff52ccf9c55a` (the cell prints
both when it starts). Status: registered, not yet run.

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
