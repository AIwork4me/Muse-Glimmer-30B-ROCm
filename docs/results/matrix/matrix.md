# llama.cpp benchmark matrix

### Study 1 — DFlash anchor (greedy, batch 1, diverse prompt set) — Meta-comparable

| weight | mode | tok/s | TTFT p50 (s) | TPOT (s) | peak RSS (GiB) | Speedup | draft acceptance |
|---|---|---|---|---|---|---|---|
| 17gb | baseline | 10.5 | 0.46 | 0.0942 | 1.4 | 1.00x | — |
| 17gb | DFlash | 23.0 | 0.47 | 0.0482 | 2.0 | 2.20x | 23% |
| dynamic | baseline | 9.1 | 0.49 | 0.1082 | 1.3 | 1.00x | — |
| dynamic | DFlash | 21.8 | 0.51 | 0.0486 | 1.9 | 2.39x | 24% |

### Study 2 — Throughput under load (temp 1.0) — NOT Meta-comparable

| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | peak RSS (GiB) | acceptance |
|---|---|---|---|---|---|---|---|

