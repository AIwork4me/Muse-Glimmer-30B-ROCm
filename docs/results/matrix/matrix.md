# llama.cpp benchmark matrix

### Study 1 — DFlash anchor (greedy, batch 1, diverse prompt set) — Meta-aligned

| weight | mode | tok/s | TTFT p50 (s) | TPOT (s) | footprint VmPeak (GiB) | Speedup | draft acceptance |
|---|---|---|---|---|---|---|---|
| 17gb | baseline | 10.5 | 0.46 | 0.0942 | 23.9 | 1.00x | — |
| 17gb | DFlash | 23.0 | 0.47 | 0.0482 | 26.7 | 2.20x | 23% |
| dynamic | baseline | 9.1 | 0.49 | 0.1082 | 26.6 | 1.00x | — |
| dynamic | DFlash | 21.8 | 0.51 | 0.0486 | 29.3 | 2.39x | 24% |

### Study 2 — Throughput under load (temp 1.0) — original study

| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | footprint VmPeak (GiB) | acceptance |
|---|---|---|---|---|---|---|---|
| 17gb | 1 | baseline | 10.5 | 0.55 | 0.0942 | 24.4 | — |
| 17gb | 1 | DFlash | 22.3 | 0.57 | 0.0435 | 27.3 | 21% |
| 17gb | 4 | baseline | 15.6 | 1.14 | 0.1778 | 27.2 | — |
| 17gb | 4 | DFlash | 27.3 | 1.25 | 0.1066 | 33.4 | 18% |
| 17gb | 16 | baseline | 34.5 | 3.23 | 0.1706 | 30.3 | — |
| 17gb | 16 | DFlash | ⚠ **PATHOLOGICAL — did not complete** (see c=16 warning) | — | — | — | — |
| dynamic | 1 | baseline | 9.1 | 0.60 | 0.1091 | 27.2 | — |
| dynamic | 1 | DFlash | 19.9 | 0.61 | 0.0521 | 30.0 | 19% |
| dynamic | 4 | baseline | 20.9 | 1.27 | 0.1809 | 29.2 | — |
| dynamic | 4 | DFlash | 28.2 | 1.34 | 0.1212 | 34.9 | 19% |
| dynamic | 16 | baseline | 31.0 | 3.37 | 0.2366 | 38.4 | — |
| dynamic | 16 | DFlash | ⚠ **PATHOLOGICAL — did not complete** (see c=16 warning) | — | — | — | — |

### Study 3 — Vision axis (temp 1.0, mmproj + test image) — memory delta vs text-only

| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | VRAM (MiB) | VmPeak (GiB) | acceptance |
|---|---|---|---|---|---|---|---|---|
| 17gb | 1 | baseline | 10.5 | 0.76 | 0.0940 | 1093 | 26.2 | — |
| 17gb | 1 | DFlash | 20.5 | 0.83 | 0.0471 | 1094 | 29.0 | 19% |
| 17gb | 4 | baseline | 21.1 | 2.04 | 0.1768 | 1093 | 29.2 | — |
| dynamic | 1 | baseline | 9.1 | 0.82 | 0.1086 | 1093 | 29.0 | — |
| dynamic | 4 | baseline | 20.0 | 2.15 | 0.1758 | 1093 | 31.9 | — |

