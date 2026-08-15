# ROCm 7.2.1 vs 7.14.0 — cell-by-cell comparison

- **7.2.1:** 21 cells   **7.14.0:** 18 cells   **compared:** 18

## Summary (TPOT, both arms measured)

- TPOT is the primary, less length-confounded cross-version metric; negative Δ means lower per-token decode latency.
- np=1: n=11, mean Δ **-0.4%**, range -6.4% … +9.0%
- np=4: n=6, mean Δ **-1.7%**, range -5.4% … +0.2%
- np=16: n=1, mean Δ **+4.9%**, range +4.9% … +4.9%
- Aggregate tok/s remains in the tables, but sampled Study 2/3 comparisons can be generation-length-confounded.

## study1

| cell | 7.2.1 aggregate tok/s | 7.14.0 aggregate tok/s | Δ | 7.2.1 TTFT p50 ms | 7.14.0 TTFT p50 ms | Δ | 7.2.1 TTFT p90 ms | 7.14.0 TTFT p90 ms | Δ | 7.2.1 TPOT ms | 7.14.0 TPOT ms | Δ | 7.2.1 VmPeak GiB | 7.14.0 VmPeak GiB | Δ | 7.14.0 accept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17gb np1 baseline | 10.48 | 10.42 | -0.6% | 459.6 | 465.4 | +1.3% | 470.1 | 474.7 | +1.0% | 94.2 | 94.7 | +0.5% | 23.91 | 23.30 | -2.6% | — |
| 17gb np1 +DFlash | 23.03 | 23.08 | +0.2% | 474.6 | 469.2 | -1.1% | 483.6 | 480.9 | -0.5% | 48.2 | 45.4 | -5.8% | 26.73 | 26.05 | -2.5% | 23.3% |
| dynamic np1 baseline | 9.14 | 9.11 | -0.3% | 486.2 | 495.9 | +2.0% | 492.6 | 499.5 | +1.4% | 108.2 | 108.5 | +0.3% | 26.57 | 25.97 | -2.3% | — |
| dynamic np1 +DFlash | 21.82 | 22.49 | +3.1% | 507.1 | 508.3 | +0.3% | 516.9 | 524.7 | +1.5% | 48.6 | 45.5 | -6.4% | 29.28 | 28.74 | -1.8% | 23.6% |

## study2

| cell | 7.2.1 aggregate tok/s | 7.14.0 aggregate tok/s | Δ | 7.2.1 TTFT p50 ms | 7.14.0 TTFT p50 ms | Δ | 7.2.1 TTFT p90 ms | 7.14.0 TTFT p90 ms | Δ | 7.2.1 TPOT ms | 7.14.0 TPOT ms | Δ | 7.2.1 VmPeak GiB | 7.14.0 VmPeak GiB | Δ | 7.14.0 accept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17gb np1 baseline | 10.52 | 10.50 | -0.2% | 473.3 | 470.2 | -0.7% | 554.5 | 543.2 | -2.0% | 94.2 | 94.4 | +0.2% | 24.41 | 23.79 | -2.5% | — |
| 17gb np1 +DFlash | 22.26 | 21.37 | -4.0% | 483.0 | 484.0 | +0.2% | 566.7 | 566.0 | -0.1% | 43.5 | 47.4 | +9.0% | 27.27 | 26.65 | -2.3% | 19.4% |
| 17gb np4 baseline | 15.60 | 21.93 | +40.6% | 873.8 | 825.1 | -5.6% | 1144.4 | 1170.5 | +2.3% | 177.8 | 175.0 | -1.6% | 27.24 | 25.73 | -5.5% | — |
| 17gb np4 +DFlash | 27.30 | 32.42 | +18.7% | 944.2 | 896.9 | -5.0% | 1246.6 | 1159.2 | -7.0% | 106.6 | 103.9 | -2.5% | 33.44 | 31.11 | -7.0% | 18.6% |
| 17gb np16 baseline | 34.47 | 36.97 | +7.3% | 2124.3 | 2251.5 | +6.0% | 3225.5 | 3081.6 | -4.5% | 170.6 | 179.0 | +4.9% | 30.27 | 35.14 | +16.1% | — |
| 17gb np16 +DFlash  ⚠ no 7.14.0 cell; 7.2.1: pathological | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| dynamic np1 baseline | 9.09 | 9.21 | +1.3% | 519.5 | 495.3 | -4.7% | 604.1 | 578.1 | -4.3% | 109.1 | 107.7 | -1.3% | 27.16 | 26.54 | -2.3% | — |
| dynamic np1 +DFlash | 19.89 | 19.69 | -1.0% | 516.8 | 518.8 | +0.4% | 607.8 | 605.4 | -0.4% | 52.1 | 53.0 | +1.8% | 30.04 | 29.45 | -2.0% | 19.4% |
| dynamic np4 baseline | 20.90 | 20.99 | +0.5% | 922.3 | 870.8 | -5.6% | 1274.2 | 1196.5 | -6.1% | 180.9 | 171.1 | -5.4% | 29.18 | 28.51 | -2.3% | — |
| dynamic np4 +DFlash | 28.22 | 31.10 | +10.2% | 956.1 | 975.8 | +2.1% | 1343.2 | 1315.3 | -2.1% | 121.2 | 121.4 | +0.2% | 34.88 | 34.22 | -1.9% | 19.2% |
| dynamic np16 baseline  ⚠ no 7.14.0 cell | 31.05 | — | — | 2411.6 | — | — | 3365.7 | — | — | 236.6 | — | — | 38.41 | — | — | — |
| dynamic np16 +DFlash  ⚠ no 7.14.0 cell; 7.2.1: pathological | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

## study3

| cell | 7.2.1 aggregate tok/s | 7.14.0 aggregate tok/s | Δ | 7.2.1 TTFT p50 ms | 7.14.0 TTFT p50 ms | Δ | 7.2.1 TTFT p90 ms | 7.14.0 TTFT p90 ms | Δ | 7.2.1 TPOT ms | 7.14.0 TPOT ms | Δ | 7.2.1 VmPeak GiB | 7.14.0 VmPeak GiB | Δ | 7.14.0 accept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17gb np1 baseline +vision | 10.50 | 10.43 | -0.7% | 680.3 | 676.3 | -0.6% | 762.5 | 738.5 | -3.1% | 94.0 | 94.6 | +0.7% | 26.24 | 25.43 | -3.1% | — |
| 17gb np1 +DFlash +vision | 20.46 | 20.71 | +1.2% | 740.3 | 731.4 | -1.2% | 830.0 | 804.8 | -3.0% | 47.1 | 46.2 | -1.9% | 28.99 | 28.29 | -2.4% | 18.8% |
| 17gb np4 baseline +vision | 21.12 | 20.99 | -0.6% | 1695.2 | 1677.5 | -1.0% | 2037.6 | 1976.2 | -3.0% | 176.8 | 175.2 | -0.9% | 29.19 | 28.34 | -2.9% | — |
| dynamic np1 baseline +vision | 9.08 | 9.16 | +0.9% | 749.2 | 711.4 | -5.0% | 822.9 | 790.3 | -4.0% | 108.6 | 107.6 | -1.0% | 29.03 | 28.39 | -2.2% | — |
| dynamic np4 baseline +vision | 20.00 | 20.22 | +1.1% | 1814.8 | 1853.1 | +2.1% | 2145.1 | 2183.6 | +1.8% | 175.8 | 175.7 | -0.1% | 31.89 | 31.17 | -2.3% | — |

## Notes

- Δ = (7.14.0 − 7.2.1) / 7.2.1 × 100. **Positive Δ on aggregate tok/s = 7.14.0 is faster**; positive Δ on TTFT/TPOT/VmPeak = 7.14.0 is slower/higher.
- `—` = metric absent (baseline cells have no acceptance; pathological cells have no metrics).
- Recorded invariants across arms: llama.cpp commit, flags, weights, prompt set and seeds; the comparison intentionally changes the ROCm runtime.
