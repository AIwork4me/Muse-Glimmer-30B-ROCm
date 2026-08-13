# llama.cpp benchmark matrix

### Study 1 — DFlash anchor (greedy, batch 1, diverse prompt set) — Meta-aligned

| weight | mode | tok/s | TTFT p50 (s) | TPOT (s) | footprint VmPeak (GiB) | Speedup | draft acceptance |
|---|---|---|---|---|---|---|---|
| 17gb | baseline | 10.4 | 0.47 | 0.0947 | 23.3 | 1.00x | — |
| 17gb | DFlash | 23.1 | 0.47 | 0.0454 | 26.1 | 2.22x | 23% |
| dynamic | baseline | 9.1 | 0.50 | 0.1085 | 26.0 | 1.00x | — |
| dynamic | DFlash | 22.5 | 0.51 | 0.0455 | 28.7 | 2.47x | 24% |

### Study 2 — Throughput under load (temp 1.0) — original study

| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | footprint VmPeak (GiB) | acceptance |
|---|---|---|---|---|---|---|---|
| 17gb | 1 | baseline | 10.5 | 0.54 | 0.0944 | 23.8 | — |
| 17gb | 1 | DFlash | 21.4 | 0.57 | 0.0474 | 26.6 | 19% |
| 17gb | 4 | baseline | 21.9 | 1.17 | 0.1750 | 25.7 | — |
| 17gb | 4 | DFlash | 32.4 | 1.16 | 0.1039 | 31.1 | 19% |
| dynamic | 1 | baseline | 9.2 | 0.58 | 0.1077 | 26.5 | — |
| dynamic | 1 | DFlash | 19.7 | 0.61 | 0.0530 | 29.4 | 19% |
| dynamic | 4 | baseline | 21.0 | 1.20 | 0.1711 | 28.5 | — |
| dynamic | 4 | DFlash | 31.1 | 1.32 | 0.1214 | 34.2 | 19% |

### Study 3 — Vision axis (temp 1.0, mmproj + test image) — memory delta vs text-only

| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | VRAM (MiB) | VmPeak (GiB) | acceptance |
|---|---|---|---|---|---|---|---|---|
| 17gb | 1 | baseline | 10.4 | 0.74 | 0.0946 | 1112 | 25.4 | — |
| 17gb | 1 | DFlash | 20.7 | 0.80 | 0.0462 | 1113 | 28.3 | 19% |
| 17gb | 4 | baseline | 21.0 | 1.98 | 0.1752 | 1112 | 28.3 | — |
| dynamic | 1 | baseline | 9.2 | 0.79 | 0.1076 | 1374 | 28.4 | — |
| dynamic | 4 | baseline | 20.2 | 2.18 | 0.1757 | 1112 | 31.2 | — |

