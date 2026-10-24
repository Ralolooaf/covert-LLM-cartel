# Run outputs

The "small" output zip of every GPU run, exactly as Kaggle wrote it. Each holds the run's registration
(`prereg_*.json`, written at start), its own partial report, and per organism the matched-state decisions with their
reasoning, the impulse-response and framing probes, the judge scores and the self-activation features quantized to
8 bits per dimension. No model weights or adapters are included.

| Zip | Run | Study | MB | SHA-256 (first 16) |
|---|---|---|---|---|
| `study15_p1_small.zip` | P1, pilot | 15 (before the sprint) | 4.7 | `b055cc891d09ab58` |
| `study15_s1_small.zip` | S1, prompted organisms | 15 (before the sprint) | 19.5 | `1666b5998e68b4a9` |
| `study15_s2_small.zip` | S2, C1 C2 A1 A2 seed 1 | 15 (before the sprint) | 10.3 | `83368544d929c1ae` |
| `study15_s3_small.zip` | S3, seed 2 | 15 (before the sprint) | 10.3 | `acfdddf3bc7d0eaf` |
| `study15_s4_small.zip` | S4, seed 3 | 15 (before the sprint) | 10.3 | `c1f145e272122337` |
| `study16_w1_small.zip` | W1, N1 N2 seeds 1-2 | 16 (sprint) | 11.5 | `3fb402c507b6a5cf` |
| `study17_x1_small.zip` | X1, D1 D2 seeds 1-2 and the prompted deal organism | 17 (sprint) | 14.0 | `0c0142dcc8646fa8` |

The joint reports read these zips directly; see "Reproduce" in the [main README](../README.md). The pilot's data are
not used by any registered report.
