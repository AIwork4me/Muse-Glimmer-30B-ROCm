# llama.cpp DFlash + Full Benchmark Matrix — Implementation Plan (Part 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a server-per-cell benchmark harness and run the comprehensive llama.cpp matrix (2 Meta kquants × {c1,c4,c16} × {baseline,+DFlash} + vision) on the proven ROCm 7.2.1 stack, producing rigorously-defensible, reproducible speed + memory data comparable to Meta's published numbers.

**Architecture:** Small, focused, testable cores in Python (pure functions for metric math, flag derivation, memory/power parsing, markdown rendering — all unit-tested without hardware) + thin shell orchestrators (`gguf-bench-cell.sh` runs one cell; `run-gguf-matrix.sh` drives the sweep). Each matrix cell = one independent `llama-server` process launched with exact flags, warmed, measured by a streaming client (TTFT/TPOT/tok/s), with RSS + draft-acceptance captured, then torn down. Two labeled studies: Study 1 reproduces Meta's methodology (greedy/batch-1/diverse-prompts) for direct comparability; Study 2 is throughput-under-load.

**Tech Stack:** Python 3.12 (aiohttp), bash, llama.cpp `llama-server` (built at `third_party/llama.cpp/build/bin/`), pytest via `uv run --no-sync pytest`, ROCm 7.2.1 on gfx1151.

## Global Constraints

*(Every task's requirements implicitly include these — copied verbatim from the spec / project.)*

- **`uv run` MUST pass `--no-sync`** always (vLLM is editable-installed and not in `uv.lock`; a bare `uv run` wipes it). Tests: `uv run --no-sync pytest ...`.
- **llama-server binary:** `third_party/llama.cpp/build/bin/llama-server` (already built, commit `0b1bad1`, has `muse_glimmer` + `-md` draft + `--mmproj` + `--metrics`).
- **Weights live in `models/`.** Fetch via `python3 scripts/hf_parallel_get.py <repo> <file> --local-dir models --concurrency 24` with `HF_ENDPOINT=https://hf-mirror.com`. Only `muse-glimmer-30B-kquant-17gb.gguf` is on disk today; `kquant-dynamic`, `dflash-kquant.gguf`, `mmproj-kquant.gguf` must be fetched.
- **Hardware:** gfx1151 (RDNA 3.5), ROCm 7.2.1, kernel 6.17, 94 GiB unified memory (~215 GB/s). `rocm-smi` VRAM is NOT the footprint (only the ~32 GB carve-out) — use process RSS.
- **Test markers** (declared in `pyproject.toml`): `@pytest.mark.gpu` (needs GPU), `@pytest.mark.server` (conftest auto-skips when nothing on **:8000**). **CI runs `-m "not gpu and not server"`.** New gguf integration tests run on **:8080** and must self-check connectivity (do NOT rely on the `server` marker).
- **Reproducibility:** pin `--seed`; record exact serve flags, KV dtype (default F16), `-ngl 999`, per-slot ctx, `reasoning_strength`, endpoint, prompt-set id, measurement date in every JSON record.
- **Commits go to `master`** (local-only repo; v1 commits there too). Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- **Meta comparability anchor:** RTX 5090 = 74.9→233.4 tok/s (3.1×) measured with **llama.cpp, greedy, batch 1, diverse prompt set, K-Quant-17GB + quantized drafter**. Study 1 matches this exactly so the gfx1151 row is comparable.
- **Scope of THIS plan:** Part 1 = harness + Studies 1/2/3 + docs, all on ROCm 7.2.1. **ROCm 7.14.0 (spec §9) is a separate gated Part 2 plan**, written only after Part 1's 7.2.1 data is committed immutable, starting with a feasibility probe.

## File Structure

**Create (Python cores — unit-tested, no hardware):**
- `scripts/gguf_bench_args.py` — pure `build_server_args(cell) -> list[str]`; single source of truth for flag logic.
- `scripts/capture_proc.py` — pure parsers: `parse_proc_status(text)`, `parse_rocm_smi_power(json)`, `scrape_metrics(text)`; plus a small CLI to sample a live PID.
- `scripts/render_matrix.py` — `render_studies(cell_jsons) -> str` (markdown); CLI reads `docs/results/matrix/*.json`.

**Create (data/config):**
- `scripts/prompt-sets/muse-glimmer-diverse.json` — 6-prompt set + version id.
- `scripts/prompt-sets/make-test-image.py` — stdlib-only generator for `test-image.png`.
- `configs/gguf-bench/study1.conf`, `study2.conf`, `study3.conf` — per-study parameter sets.

**Create (shell orchestrators):**
- `scripts/gguf-bench-cell.sh` — run ONE cell, emit one JSON record.
- `scripts/run-gguf-matrix.sh` — enumerate cells, randomize order, drive cell script, render.

**Modify:**
- `scripts/bench_client.py` — add streaming + `/v1/chat/completions` + prompt/concurrency loop; preserve the legacy `BASE C 512` JSON (used by `scripts/benchmark.sh`).

**Create (tests, all CI-safe):**
- `tests/test_bench_metrics.py`, `tests/test_gguf_bench_args.py`, `tests/test_capture_proc.py`, `tests/test_render_matrix.py`, `tests/test_prompt_set.py`, `tests/test_gguf_configs.py`.

**Create (docs/output):**
- `docs/results/METHODOLOGY.md`, `docs/results/matrix/` (JSON + rendered md); modify `docs/results/benchmark.md`, `docs/adaptation.md`, `docs/troubleshooting.md`, `handoff.md`.

---

### Task 1: Fetch the additional GGUF artifacts

**Files:**
- Download into: `models/muse-glimmer-30B-kquant-dynamic.gguf`, `models/dflash-kquant.gguf`, `models/mmproj-kquant.gguf`

**Interfaces:**
- Produces: the three weight files Studies 1/3 depend on (17gb already present).

- [ ] **Step 1: Fetch dynamic, dflash, mmproj via the parallel downloader**

```bash
cd /home/amd/Desktop/muse-rocm
export HF_ENDPOINT=https://hf-mirror.com PATH="$HOME/.local/bin:$PATH"
for f in muse-glimmer-30B-kquant-dynamic.gguf dflash-kquant.gguf mmproj-kquant.gguf; do
  [ -f "models/$f" ] || python3 scripts/hf_parallel_get.py meta-models/Muse-Glimmer-30B-GGUF "$f" \
      --local-dir models --concurrency 24
done
```

- [ ] **Step 2: Verify all four weights are present with expected sizes**

```bash
ls -lh models/*.gguf | awk '{print $5, $9}'
```
Expected: kquant-17gb ~16.8G, kquant-dynamic ~19.7G, dflash ~1.6G, mmproj ~1.4G.

- [ ] **Step 3: Commit a manifest of fetched artifacts**

```bash
printf '# GGUF artifacts on disk (gitignored weights)\n%s\n' "$(ls -lh models/*.gguf | awk '{print $5"\t"$9}')" > docs/results/matrix/gguf-manifest.md
git add docs/results/matrix/gguf-manifest.md
git commit -m "chore(bench): manifest of fetched GGUF artifacts (dynamic/dflash/mmproj)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: bench_client — pure metric functions (TDD core)

**Files:**
- Modify: `scripts/bench_client.py`
- Test: `tests/test_bench_metrics.py`

**Interfaces:**
- Produces: `percentile(values, q)` and `compute_run_metrics(reqs)` (see signatures below). Later tasks (streaming loop, cell script) consume `compute_run_metrics` to build each cell's metric block.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bench_metrics.py`:

```python
"""Unit tests for bench_client metric math (pure, no server needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_percentile_linear_interp():
    from bench_client import percentile
    assert percentile([0.5, 0.6], 50) == 0.55
    assert percentile([1.0, 2.0, 3.0, 4.0], 90) == 3.7  # 0.9*(4-1)=2.7 -> idx interp
    assert percentile([], 50) is None


def test_compute_run_metrics_aggregate_and_distribution():
    from bench_client import compute_run_metrics
    # 2 concurrent requests, c=2
    reqs = [
        {"t0": 0.0, "first_t": 0.5, "last_t": 10.5, "t1": 10.6,
         "n_tokens": 100, "finish_reason": "stop"},
        {"t0": 0.0, "first_t": 0.6, "last_t": 12.0, "t1": 12.1,
         "n_tokens": 120, "finish_reason": "stop"},
    ]
    m = compute_run_metrics(reqs)
    assert m["concurrency"] == 2
    assert m["total_tokens"] == 220
    assert abs(m["wall_s"] - 12.1) < 1e-9          # max t1 - min t0
    assert abs(m["agg_tok_s"] - 220 / 12.1) < 1e-6
    assert abs(m["ttft_p50"] - 0.55) < 1e-9
    assert m["finish_reason_dist"] == {"stop": 2}
    # tpot per request = gen_phase / (n_tokens-1)
    tpot_a = (10.5 - 0.5) / 99
    tpot_b = (12.0 - 0.6) / 119
    assert abs(m["tpot_median"] - sorted([tpot_a, tpot_b])[0]) < 1e-9 or \
           abs(m["tpot_median"] - (tpot_a + tpot_b) / 2) < 1e-9


def test_median_run_metrics_reports_variance():
    from bench_client import median_run_metrics
    reps = [
        {"agg_tok_s": 10.0, "total_tokens": 100, "wall_s": 10.0, "ttft_p50": 0.4,
         "ttft_p90": 0.5, "tpot_median": 0.09, "finish_reason_dist": {"stop": 1}},
        {"agg_tok_s": 12.0, "total_tokens": 120, "wall_s": 10.0, "ttft_p50": 0.4,
         "ttft_p90": 0.5, "tpot_median": 0.08, "finish_reason_dist": {"stop": 1}},
        {"agg_tok_s": 14.0, "total_tokens": 140, "wall_s": 10.0, "ttft_p50": 0.4,
         "ttft_p90": 0.5, "tpot_median": 0.07, "finish_reason_dist": {"stop": 1}},
    ]
    m = median_run_metrics(reps)
    assert m["agg_tok_s"] == 12.0          # median across reps
    assert m["agg_tok_s_min"] == 10.0
    assert m["agg_tok_s_max"] == 14.0
    assert m["reps"] == 3
    assert m["finish_reason_dist"] == {"stop": 3}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_bench_metrics.py -v`
Expected: FAIL with `ImportError` / `cannot import name 'compute_run_metrics'`.

- [ ] **Step 3: Add the pure functions to bench_client.py**

Add near the top of `scripts/bench_client.py` (above `async def one`):

```python
def percentile(values, q):
    """Linear-interpolation percentile, q in [0,100]. None if empty."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def compute_run_metrics(reqs):
    """Aggregate per-cell metrics from a list of per-request timing dicts.

    Each req: {t0, first_t, last_t, t1, n_tokens, finish_reason}.
    Returns a dict with aggregate tok/s, TTFT p50/p90, TPOT median, and the
    finish_reason distribution. t0/first_t/last_t/t1 are monotonic seconds.
    """
    from statistics import median

    conc = len(reqs)
    total = sum(r["n_tokens"] for r in reqs)
    wall = (max(r["t1"] for r in reqs) - min(r["t0"] for r in reqs)) if reqs else 0.0
    agg = total / wall if wall else 0.0
    ttfts = [r["first_t"] - r["t0"] for r in reqs]
    tpots = [
        (r["last_t"] - r["first_t"]) / max(r["n_tokens"] - 1, 1)
        for r in reqs if r["n_tokens"] > 1
    ]
    from collections import Counter
    dist = dict(Counter(r["finish_reason"] for r in reqs))
    return {
        "concurrency": conc,
        "total_tokens": total,
        "wall_s": wall,
        "agg_tok_s": agg,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_max": max(ttfts) if ttfts else None,
        "tpot_median": median(tpots) if tpots else None,
        "finish_reason_dist": dist,
    }


def median_run_metrics(reps):
    """Element-wise median across per-rep metric dicts, with agg_tok_s min/max for
    variance reporting (spec §6.2: report median + min/max). reps = list of
    compute_run_metrics() outputs, one per rep."""
    from statistics import median
    if not reps:
        return {}
    keys = ["agg_tok_s", "total_tokens", "wall_s", "ttft_p50", "ttft_p90", "tpot_median"]
    out = {k: median([r[k] for r in reps if r.get(k) is not None]) for k in keys}
    toks = [r["agg_tok_s"] for r in reps]
    out["agg_tok_s_min"], out["agg_tok_s_max"] = min(toks), max(toks)
    out["reps"] = len(reps)
    from collections import Counter
    fr = Counter()
    for r in reps:
        fr.update(r.get("finish_reason_dist", {}))
    out["finish_reason_dist"] = dict(fr)
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_bench_metrics.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_client.py tests/test_bench_metrics.py
git commit -m "feat(bench): pure metric functions (tok/s, TTFT p50/p90, TPOT) + tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: gguf_bench_args.py — flag derivation (TDD core)

**Files:**
- Create: `scripts/gguf_bench_args.py`
- Test: `tests/test_gguf_bench_args.py`

**Interfaces:**
- Produces: `build_server_args(cell) -> list[str]` and `build_client_args(cell) -> dict`. `gguf-bench-cell.sh` calls `python3 scripts/gguf_bench_args.py <cell-json>` to print the serve argv. A `cell` dict has keys: `weight`, `dflash`(bool), `vision`(bool), `np`(int), `per_slot_ctx`(int), `study`("study1"|"study2"|"study3"), `seed`(int).

- [ ] **Step 1: Write the failing test**

Create `tests/test_gguf_bench_args.py`:

```python
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from gguf_bench_args import build_server_args, build_client_args

W = "models/muse-glimmer-30B-kquant-17gb.gguf"
DF = "models/dflash-kquant.gguf"
MMP = "models/mmproj-kquant.gguf"


def _cell(**kw):
    base = dict(weight=W, dflash=False, vision=False, np=1,
                per_slot_ctx=8192, study="study1", seed=0,
                weights_dir="models", dflash_path=DF, mmproj_path=MMP)
    base.update(kw)
    return base


def test_study1_greedy_baseline_flags():
    a = build_server_args(_cell())
    s = " ".join(a)
    assert "--temp 0" in s
    assert "--seed 0" in s
    assert "-ngl 999" in s
    assert "-np 1" in s
    assert "-c 8192" in s            # np(1) * per_slot_ctx(8192)
    assert "--jinja" in s
    assert "-md" not in s            # baseline, no draft


def test_dflash_adds_draft_model():
    a = build_server_args(_cell(dflash=True))
    s = " ".join(a)
    assert "-md models/dflash-kquant.gguf" in s
    assert "-ngld 99" in s


def test_study2_load_uses_temp_1_and_scales_ctx():
    a = build_server_args(_cell(study="study2", np=16))
    s = " ".join(a)
    assert "--temp 1.0" in s
    assert "--top-p 0.95" in s
    assert "--top-k 64" in s
    assert "-c 131072" in s          # 16 * 8192
    assert "-np 16" in s


def test_vision_adds_mmproj():
    a = build_server_args(_cell(vision=True))
    assert "--mmproj models/mmproj-kquant.gguf" in a


def test_client_args_match_study():
    c = build_client_args(_cell(study="study1"))
    assert c["endpoint"] == "chat"
    assert c["temp"] == 0
    assert c["max_tokens"] == 256
    c2 = build_client_args(_cell(study="study2"))
    assert c2["temp"] == 1.0 and c2["max_tokens"] == 512 and c2["reasoning_strength"] == "high"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_gguf_bench_args.py -v`
Expected: FAIL (`ModuleNotFoundError: gguf_bench_args`).

- [ ] **Step 3: Implement gguf_bench_args.py**

Create `scripts/gguf_bench_args.py`:

```python
#!/usr/bin/env python3
"""Derive exact llama-server flags + client params from a cell config (pure).

Single source of truth for the matrix's flag logic. CI-tested; called by
gguf-bench-cell.sh. Usage:  python3 gguf_bench_args.py <cell.json> server|client
"""
import json
import sys

WEIGHTS = {
    "17gb": "models/muse-glimmer-30B-kquant-17gb.gguf",
    "dynamic": "models/muse-glimmer-30B-kquant-dynamic.gguf",
}
DEFAULT_DFLASH = "models/dflash-kquant.gguf"
DEFAULT_MMPROJ = "models/mmproj-kquant.gguf"

# Per-study client/sampling defaults (spec §5).
STUDY = {
    "study1": {"temp": 0, "top_p": 1.0, "top_k": 0, "max_tokens": 256,
               "reasoning_strength": "high", "endpoint": "chat"},
    "study2": {"temp": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 512,
               "reasoning_strength": "high", "endpoint": "chat"},
    "study3": {"temp": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 512,
               "reasoning_strength": "high", "endpoint": "chat"},
}


def _weight_path(cell):
    if cell["weight"] in WEIGHTS:
        return WEIGHTS[cell["weight"]]
    return cell["weight"]  # already a path


def build_server_args(cell):
    st = STUDY[cell["study"]]
    np_ = int(cell["np"])
    ctx = np_ * int(cell["per_slot_ctx"])
    args = [
        "llama-server",
        "-m", _weight_path(cell),
        "-ngl", "999",
        "-np", str(np_),
        "-c", str(ctx),
        "--jinja",
        "--temp", str(st["temp"]),
        "--seed", str(cell["seed"]),
        "--metrics",            # expose /metrics for draft-acceptance capture
        "--port", "8080",
        "--host", "127.0.0.1",
    ]
    if st["temp"] == 1.0:
        args += ["--top-p", str(st["top_p"]), "--top-k", str(st["top_k"])]
    if cell.get("dflash"):
        args += ["-md", cell.get("dflash_path", DEFAULT_DFLASH), "-ngld", "99"]
    if cell.get("vision"):
        args += ["--mmproj", cell.get("mmproj_path", DEFAULT_MMPROJ)]
    return args


def build_client_args(cell):
    st = STUDY[cell["study"]]
    return {
        "endpoint": st["endpoint"],
        "temp": st["temp"],
        "top_p": st["top_p"],
        "top_k": st["top_k"],
        "max_tokens": st["max_tokens"],
        "reasoning_strength": st["reasoning_strength"],
        "seed": cell["seed"],
        "np": cell["np"],
        "prompt_set": "muse-glimmer-diverse",
    }


if __name__ == "__main__":
    cell = json.loads(sys.argv[1])
    which = sys.argv[2] if len(sys.argv) > 2 else "server"
    out = build_server_args(cell) if which == "server" else build_client_args(cell)
    print(json.dumps(out))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_gguf_bench_args.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/gguf_bench_args.py tests/test_gguf_bench_args.py
git commit -m "feat(bench): pure llama-server flag derivation per cell + tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Diverse prompt set + test-image generator (TDD core)

**Files:**
- Create: `scripts/prompt-sets/muse-glimmer-diverse.json`
- Create: `scripts/prompt-sets/make-test-image.py`
- Create: `scripts/prompt-sets/test-image.png` (generated)
- Test: `tests/test_prompt_set.py`

**Interfaces:**
- Produces: the versioned 6-prompt set (consumed by bench_client's prompt loop) and a fixed valid PNG (Study 3). Prompt categories: code, math, factual QA, creative, reasoning, instruction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt_set.py`:

```python
import json
import os
import struct
import zlib

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROMPT_JSON = os.path.join(ROOT, "scripts", "prompt-sets", "muse-glimmer-diverse.json")
IMG = os.path.join(ROOT, "scripts", "prompt-sets", "test-image.png")

CATEGORIES = {"code", "math", "factual", "creative", "reasoning", "instruction"}


def test_prompt_set_schema():
    d = json.load(open(PROMPT_JSON))
    assert d["id"] == "muse-glimmer-diverse"
    assert d["version"] == 1
    cats = {p["category"] for p in d["prompts"]}
    assert cats == CATEGORIES
    for p in d["prompts"]:
        assert isinstance(p["text"], str) and len(p["text"]) > 20


def test_test_image_is_valid_png():
    raw = open(IMG, "rb").read()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR is the first chunk; width/height are big-endian uint32 at bytes 16..24
    assert raw[12:16] == b"IHDR"
    w, h = struct.unpack(">II", raw[16:24])
    assert w >= 64 and h >= 64
    zlib.decompress(raw[24:])  # not exhaustive, just ensures not truncated hard
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_prompt_set.py -v`
Expected: FAIL (files missing).

- [ ] **Step 3: Create the prompt set**

Create `scripts/prompt-sets/muse-glimmer-diverse.json`:

```json
{
  "id": "muse-glimmer-diverse",
  "version": 1,
  "note": "6-prompt diverse set matching Meta's DFlash methodology (greedy, batch 1). Categories span the model's workload mix; each is a short, well-defined generation task so decode dominates.",
  "prompts": [
    {"category": "code", "text": "Write a Python function that returns the n-th Fibonacci number iteratively. Include a docstring and one assert-based test call."},
    {"category": "math", "text": "A train travels 60 km in 45 minutes, then 100 km in 1 hour 20 minutes. What is its average speed for the whole journey in km/h? Show the steps."},
    {"category": "factual", "text": "In three sentences, explain why the sky appears blue during the day and reddish at sunset."},
    {"category": "creative", "text": "Write a four-line rhyming poem about a robot learning to paint."},
    {"category": "reasoning", "text": "Three friends share a bill. Alice pays twice as much as Bob, and Bob pays three times as much as Carol. If the total is 120, how much does each pay? Reason step by step."},
    {"category": "instruction", "text": "List five practical tips for reducing daily water usage at home, each as a single bullet with a short verb-first phrase."}
  ]
}
```

- [ ] **Step 4: Create the test-image generator and run it**

Create `scripts/prompt-sets/make-test-image.py` (stdlib-only PNG encoder, 128×128):

```python
#!/usr/bin/env python3
"""Generate a fixed 128x128 RGB PNG test image for Study 3 (vision), stdlib-only."""
import struct
import zlib

W = H = 128


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def main(path):
    raw = bytearray()
    for y in range(H):
        raw.append(0)  # filter type 0 (None) per scanline
        for x in range(W):
            r = (x * 2) & 255
            g = (y * 2) & 255
            b = ((x + y)) & 255
            raw += bytes((r, g, b))
    png = (b"\x89PNG\r\n\x1a\n" +
           _chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
           _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
           _chunk(b"IEND", b""))
    open(path, "wb").write(png)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test-image.png")
```

Generate the image:
```bash
cd /home/amd/Desktop/muse-rocm
python3 scripts/prompt-sets/make-test-image.py scripts/prompt-sets/test-image.png
file scripts/prompt-sets/test-image.png   # expect: PNG image data, 128 x 128
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_prompt_set.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/prompt-sets/ tests/test_prompt_set.py
git commit -m "feat(bench): diverse 6-prompt set + stdlib test-image generator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: capture_proc.py — memory/power/metrics parsers (TDD core)

**Files:**
- Create: `scripts/capture_proc.py`
- Test: `tests/test_capture_proc.py`

**Interfaces:**
- Produces: `parse_proc_status(text)->dict` (VmHWM, VmRSS in KiB), `parse_rocm_smi_power(json_text)->dict`, `scrape_metrics(text)->dict` (draft-acceptance from Prometheus `/metrics`). `gguf-bench-cell.sh` uses the CLI to sample a live PID.

- [ ] **Step 1: Write the failing test**

Create `tests/test_capture_proc.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from capture_proc import parse_proc_status, parse_rocm_smi_power, scrape_metrics

PROC = """Name:   llama-server
VmRSS:    17196612 kB
VmHWM:    18432000 kB
RssShmem:   102400 kB
"""

ROCM = '{"card0": {"Average Graphics Package Power (W)": "45.3", "Temperature (Sensor edge) (C)": "72"}}'

METRICS = """
# HELP llama_speculative_accepted_draft_tokens_total ...
llama_speculative_accepted_draft_tokens_total 312.0
llama_speculative_draft_tokens_total 400.0
llama_speculative_avg_accepted 2.4
"""


def test_parse_proc_status_kib():
    d = parse_proc_status(PROC)
    assert d["VmHWM_kib"] == 18432000
    assert d["VmRSS_kib"] == 17196612
    assert d["VmHWM_gib"] == round(18432000 / 1024 / 1024, 2)


def test_rocm_power_temp():
    d = parse_rocm_smi_power(ROCM)
    assert d["power_w"] == 45.3
    assert d["temp_c"] == 72


def test_scrape_metrics_acceptance():
    d = scrape_metrics(METRICS)
    assert d["accepted_draft_tokens"] == 312.0
    assert d["draft_tokens"] == 400.0
    assert abs(d["acceptance_rate"] - 312.0 / 400.0) < 1e-9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_capture_proc.py -v`
Expected: FAIL (`ModuleNotFoundError: capture_proc`).

- [ ] **Step 3: Implement capture_proc.py**

Create `scripts/capture_proc.py`:

```python
#!/usr/bin/env python3
"""Parse /proc/<pid>/status, rocm-smi JSON, and llama-server /metrics (pure parsers)
+ a small CLI to sample a live PID. Used by gguf-bench-cell.sh.
"""
import json
import re
import sys


def parse_proc_status(text):
    out = {}
    for key in ("VmHWM", "VmRSS", "RssShmem"):
        m = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.M)
        if m:
            kib = int(m.group(1))
            out[f"{key}_kib"] = kib
            out[f"{key}_gib"] = round(kib / 1024 / 1024, 2)
    return out


def parse_rocm_smi_power(json_text):
    d = json.loads(json_text)
    card = next(iter(d.values()))  # card0
    powr = card.get("Average Graphics Package Power (W)")
    temp = card.get("Temperature (Sensor edge) (C)")
    return {
        "power_w": float(powr) if powr else None,
        "temp_c": float(temp) if temp else None,
    }


def scrape_metrics(text):
    """Pull speculative-decoding counters from llama-server /metrics (Prometheus)."""
    vals = {}

    def grab(name):
        m = re.search(rf"^{name}\s+([0-9.eE+-]+)", text, re.M)
        return float(m.group(1)) if m else None

    accepted = grab("llama_speculative_accepted_draft_tokens_total")
    drafted = grab("llama_speculative_draft_tokens_total")
    avg = grab("llama_speculative_avg_accepted")
    out = {
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "avg_accepted_per_step": avg,
    }
    if accepted is not None and drafted:
        out["acceptance_rate"] = accepted / drafted
    return out


if __name__ == "__main__":
    # CLI modes:
    #   capture_proc.py status <pid>   -> /proc/<pid>/status memory fields
    #   capture_proc.py power          -> rocm-smi power/temp (stdin, or calls rocm-smi)
    #   capture_proc.py metrics        -> llama-server /metrics acceptance (stdin)
    import subprocess
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "status":
        try:
            print(json.dumps(parse_proc_status(open(f"/proc/{sys.argv[2]}/status").read())))
        except OSError as e:
            print(json.dumps({"error": str(e)}))
    elif mode == "power":
        txt = sys.stdin.read() or subprocess.run(
            ["rocm-smi", "--showpower", "--showtemp", "--json"],
            capture_output=True, text=True).stdout
        print(json.dumps(parse_rocm_smi_power(txt)))
    elif mode == "metrics":
        print(json.dumps(scrape_metrics(sys.stdin.read())))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_capture_proc.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/capture_proc.py tests/test_capture_proc.py
git commit -m "feat(bench): /proc + rocm-smi + /metrics parsers for RSS/power/acceptance

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: render_matrix.py — JSON → markdown (TDD core)

**Files:**
- Create: `scripts/render_matrix.py`
- Test: `tests/test_render_matrix.py`

**Interfaces:**
- Produces: `render_studies(cells) -> str`. `cells` is a list of cell-result dicts (the JSON records from `gguf-bench-cell.sh`). Output is a markdown doc with one table per study. The CLI reads `docs/results/matrix/cell-*.json`.

A cell-result JSON has: `study`, `weight`, `dflash`(bool), `vision`(bool), `np`, plus a `metrics` block (from Task 2's `compute_run_metrics`), a `mem` block (`VmHWM_gib`, plus llama.cpp log components `weights_gib`, `kv_gib`), `acceptance` (for dflash cells), and `reps`/`seed`/`flags`/`build`/`date` for the reproducibility manifest.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_matrix.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from render_matrix import render_studies

CELLS = [
    {"study": "study1", "weight": "17gb", "dflash": False, "vision": False, "np": 1,
     "metrics": {"agg_tok_s": 10.5, "ttft_p50": 0.4, "tpot_median": 0.09, "total_tokens": 1536},
     "mem": {"VmHWM_gib": 17.2, "weights_gib": 16.8, "kv_gib": 0.4},
     "acceptance": None, "reps": 3, "seed": 0},
    {"study": "study1", "weight": "17gb", "dflash": True, "vision": False, "np": 1,
     "metrics": {"agg_tok_s": 22.0, "ttft_p50": 0.4, "tpot_median": 0.045, "total_tokens": 1536},
     "mem": {"VmHWM_gib": 18.8, "weights_gib": 16.8, "kv_gib": 0.4},
     "acceptance": {"acceptance_rate": 0.78, "avg_accepted_per_step": 2.1},
     "reps": 3, "seed": 0},
]


def test_render_study1_has_speedup_column():
    md = render_studies(CELLS)
    assert "### Study 1" in md
    assert "Speedup" in md
    # speedup = 22.0 / 10.5 = 2.095...
    assert "2.10" in md
    assert "78%" in md               # acceptance rendered as percent


def test_render_handles_empty():
    assert "no cells" in render_studies([]).lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_render_matrix.py -v`
Expected: FAIL (`ModuleNotFoundError: render_matrix`).

- [ ] **Step 3: Implement render_matrix.py**

Create `scripts/render_matrix.py`:

```python
#!/usr/bin/env python3
"""Render matrix cell-result JSONs into a markdown report (pure)."""
import glob
import json
import os
import sys


def _row(c, base=None):
    m = c["metrics"]
    mem = c["mem"]
    toks = f'{m["agg_tok_s"]:.1f}'
    speedup = ""
    if base and base > 0:
        speedup = f'{m["agg_tok_s"] / base:.2f}x'
    acc = c.get("acceptance") or {}
    acc_s = f'{int(round(acc["acceptance_rate"] * 100))}%' if acc.get("acceptance_rate") is not None else "—"
    return (f'| {c["weight"]} | {"DFlash" if c["dflash"] else "baseline"} | '
            f'{toks} | {m["ttft_p50"]:.2f} | {m["tpot_median"]:.4f} | '
            f'{mem["VmHWM_gib"]:.1f} | {speedup or "—"} | {acc_s} |')


def _study1(cells):
    out = ["### Study 1 — DFlash anchor (greedy, batch 1, diverse prompt set) — Meta-comparable\n",
           "| weight | mode | tok/s | TTFT p50 (s) | TPOT (s) | peak RSS (GiB) | DFlash speedup | draft acceptance |",
           "|---|---|---|---|---|---|---|---|"]
    for w in ("17gb", "dynamic"):
        base = next((c["metrics"]["agg_tok_s"] for c in cells
                     if c["weight"] == w and not c["dflash"]), None)
        for c in [x for x in cells if x["weight"] == w and x.get("study") == "study1"]:
            out.append(_row(c, base))
    return "\n".join(out)


def _study2(cells):
    out = ["### Study 2 — Throughput under load (temp 1.0) — NOT Meta-comparable\n",
           "| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | peak RSS (GiB) | acceptance |",
           "|---|---|---|---|---|---|---|---|"]
    for c in sorted([x for x in cells if x.get("study") == "study2"],
                    key=lambda x: (x["weight"], x["np"], x["dflash"])):
        m = c["metrics"]
        acc = c.get("acceptance") or {}
        acc_s = f'{int(round(acc["acceptance_rate"] * 100))}%' if acc.get("acceptance_rate") is not None else "—"
        out.append(f'| {c["weight"]} | {c["np"]} | {"DFlash" if c["dflash"] else "baseline"} | '
                   f'{m["agg_tok_s"]:.1f} | {m["ttft_p90"]:.2f} | {m["tpot_median"]:.4f} | '
                   f'{c["mem"]["VmHWM_gib"]:.1f} | {acc_s} |')
    return "\n".join(out)


def render_studies(cells):
    if not cells:
        return "# Matrix report\n\n(no cells)\n"
    return "# llama.cpp benchmark matrix\n\n" + "\n\n".join(
        f for f in (_study1(cells), _study2(cells)) if f.count("\n") > 2) + "\n"


if __name__ == "__main__":
    cells = [json.load(open(p)) for p in
             sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "docs", "results", "matrix", "cell-*.json")))]
    print(render_studies(cells))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_render_matrix.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/render_matrix.py tests/test_render_matrix.py
git commit -m "feat(bench): matrix JSON -> markdown renderer + tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Per-study configs + validation (TDD core)

**Files:**
- Create: `configs/gguf-bench/study1.conf`, `study2.conf`, `study3.conf`
- Test: `tests/test_gguf_configs.py`

**Interfaces:**
- Produces: shell-sourceable config files consumed by `run-gguf-matrix.sh` (the cell enumeration + study params). Validated by a CI-safe test.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gguf_configs.py`:

```python
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONF = os.path.join(ROOT, "configs", "gguf-bench")


def _load(name):
    txt = open(os.path.join(CONF, name)).read()
    d = {}
    for line in txt.splitlines():
        m = re.match(r'^([A-Z_]+)=(.*)$', line)
        if m:
            d[m.group(1)] = m.group(2).strip().strip('"')
    return d


def test_study1_greedy_single_weight_set():
    d = _load("study1.conf")
    assert d["STUDY"] == "study1"
    assert d["TEMP"] == "0"
    assert d["MAX_TOKENS"] == "256"
    assert d["PER_SLOT_CTX"] == "8192"
    assert "17gb" in d["WEIGHTS"] and "dynamic" in d["WEIGHTS"]
    assert d["NPS"] == '"1"'


def test_study2_load_concurrency():
    d = _load("study2.conf")
    assert d["TEMP"] == "1.0"
    assert d["MAX_TOKENS"] == "512"
    assert d["REPS"] == "5"
    assert d["NPS"] == '"1 4 16"'


def test_study3_vision_has_mmproj_flag():
    d = _load("study3.conf")
    assert d["VISION"] == "1"
    assert d["NPS"] == '"1 4"'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_gguf_configs.py -v`
Expected: FAIL (files missing).

- [ ] **Step 3: Create the three config files**

`configs/gguf-bench/study1.conf`:
```bash
# Study 1 — Meta-comparable DFlash anchor (greedy, batch 1, diverse prompt set)
STUDY="study1"
WEIGHTS="17gb dynamic"
DFS="0 1"            # baseline, dflash
NPS="1"
PER_SLOT_CTX=8192
TEMP=0
MAX_TOKENS=256
REPS=3
WARMUP=2
SEED=0
VISION=0
```

`configs/gguf-bench/study2.conf`:
```bash
# Study 2 — Throughput under load (temp 1.0). NOT Meta-comparable.
STUDY="study2"
WEIGHTS="17gb dynamic"
DFS="0 1"
NPS="1 4 16"
PER_SLOT_CTX=8192
TEMP=1.0
TOP_P=0.95
TOP_K=64
MAX_TOKENS=512
REPS=5
WARMUP=2
SEED=42
VISION=0
```

`configs/gguf-bench/study3.conf`:
```bash
# Study 3 — Vision axis (temp 1.0). Memory delta vs text-only.
STUDY="study3"
WEIGHTS="17gb dynamic"
DFS="0 1"
NPS="1 4"
PER_SLOT_CTX=8192
TEMP=1.0
TOP_P=0.95
TOP_K=64
MAX_TOKENS=512
REPS=3
WARMUP=2
SEED=42
VISION=1
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/test_gguf_configs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add configs/gguf-bench/ tests/test_gguf_configs.py
git commit -m "feat(bench): per-study config files (study1/2/3) + validation tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: bench_client — streaming + chat/completions + prompt loop (integration)

**Files:**
- Modify: `scripts/bench_client.py`
- Note: legacy `BASE C 512` call site (`scripts/benchmark.sh`) MUST keep producing `{"concurrency","total_out_tokens","wall_s","agg_tok_s"}`.

**Interfaces:**
- Consumes: `compute_run_metrics` (Task 2), the prompt-set JSON (Task 4).
- Produces: an extended CLI (`--study`, `--endpoint {chat,completions}`, `--prompts <json>`, `--reps`, `--seed`, `--temp`, `--top-p`, `--top-k`, `--max-tokens`, `--reasoning-strength`, `--image <png>`) emitting the full per-cell metrics block to stdout as JSON. Used by `gguf-bench-cell.sh`.

- [ ] **Step 1: Add the streaming + loop code to bench_client.py**

Append to `scripts/bench_client.py` (keeping the existing `async def one`/`main` intact for legacy mode):

```python
import argparse

try:
    import base64
except ImportError:
    base64 = None


async def stream_one(session, base, payload):
    """Stream one request; return per-request timing dict. endpoint in payload meta."""
    t0 = time.perf_counter()
    first_t = last_t = None
    n_tokens = 0
    finish = None
    url = f"{base}{'/v1/chat/completions' if payload['_endpoint']=='chat' else '/v1/completions'}"
    body = {k: v for k, v in payload.items() if not k.startswith("_")}
    body["stream"] = True
    async with session.post(url, json=body) as r:
        async for raw in r.content:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            if first_t is None:
                first_t = now
            last_t = now
            # count tokens from usage (last chunk) or choices delta
            if obj.get("usage"):
                n_tokens = obj["usage"].get("completion_tokens", n_tokens)
            ch = obj.get("choices") or []
            if ch and "finish_reason" in ch[0] and ch[0]["finish_reason"]:
                finish = ch[0]["finish_reason"]
    t1 = time.perf_counter()
    if n_tokens == 0 and last_t:  # fallback: no usage reported
        n_tokens = 1
    return {"t0": t0, "first_t": first_t or t0, "last_t": last_t or t0,
            "t1": t1, "n_tokens": n_tokens, "finish_reason": finish or "stop"}


def _payload(cell_args, prompt_text, image_b64=None):
    p = {"_endpoint": cell_args["endpoint"], "model": "muse-glimmer-30B",
         "max_tokens": cell_args["max_tokens"], "temperature": cell_args["temp"],
         "top_p": cell_args["top_p"], "top_k": cell_args["top_k"],
         "seed": cell_args["seed"]}
    if cell_args["endpoint"] == "chat":
        content = ([{"type": "text", "text": prompt_text}] if image_b64 is None
                   else [{"type": "text", "text": prompt_text},
                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}])
        p["messages"] = [{"role": "user", "content": content}]
        p["chat_template_kwargs"] = {"reasoning_strength": cell_args["reasoning_strength"]}
    else:
        p["prompt"] = prompt_text
    return p


async def run_cell(base, cell_args, prompts, image_b64=None):
    """Run one cell: warmup, then REPS rounds (each round fires np concurrent requests
    per prompt). Returns a list of per-rep metric dicts — one compute_run_metrics() per
    rep — so the caller can take the median + report min/max (spec §6.2)."""
    conc = cell_args["np"]
    per_rep = []
    async with aiohttp.ClientSession() as s:
        for _ in range(cell_args.get("warmup", 2)):
            await asyncio.gather(*[stream_one(s, base, _payload(cell_args, prompts[0]["text"], image_b64))
                                   for _ in range(conc)])
        for _ in range(cell_args["reps"]):
            rep_reqs = []
            for pr in prompts:
                rep_reqs.extend(await asyncio.gather(*[
                    stream_one(s, base, _payload(cell_args, pr["text"], image_b64))
                    for _ in range(conc)]))
            per_rep.append(compute_run_metrics(rep_reqs))
    return per_rep


def main_extended():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--study", required=True)
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--prompts", default="scripts/prompt-sets/muse-glimmer-diverse.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning-strength", default="high")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    cell_args = vars(args)
    prompts = json.load(open(args.prompts))["prompts"]
    img = None
    if args.image:
        img = base64.b64encode(open(args.image, "rb").read()).decode()
    per_rep = asyncio.run(run_cell(args.base, cell_args, prompts, img))
    print(json.dumps(median_run_metrics(per_rep)))


if __name__ == "__main__":
    # Legacy positional path "bench_client.py BASE C 512" -> scripts/benchmark.sh (unchanged).
    # Everything else routes to the extended streaming CLI.
    if len(sys.argv) > 2 and sys.argv[1].startswith("http") and sys.argv[2].isdigit():
        _legacy_main()           # wraps the ORIGINAL __main__ block (does its own asyncio.run)
    else:
        main_extended()
```

> **Note:** rename the existing `if __name__ == "__main__"` block to `_legacy_main()` so the legacy `BASE C 512` path still works and `scripts/benchmark.sh` is unaffected. The legacy block already builds the legacy JSON `{"concurrency","total_out_tokens","wall_s","agg_tok_s"}` — keep it verbatim.

- [ ] **Step 2: Manually verify legacy mode still works (no server needed for syntax)**

Run: `uv run --no-sync python -c "import ast; ast.parse(open('scripts/bench_client.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Integration-test on a running gguf server (hardware; self-skip)**

Start a baseline server, then drive it:
```bash
cd /home/amd/Desktop/muse-rocm
third_party/llama.cpp/build/bin/llama-server -m models/muse-glimmer-30B-kquant-17gb.gguf \
  -ngl 999 -np 1 -c 8192 --jinja --temp 0 --seed 0 --metrics --port 8080 &
SRV=$!
sleep 8
curl -s http://127.0.0.1:8080/health || { echo "server up check failed"; }
uv run --no-sync python scripts/bench_client.py http://127.0.0.1:8080 \
  --study study1 --endpoint chat --np 1 --temp 0 --max-tokens 256 --reps 1 --warmup 1
kill $SRV
```
Expected: a JSON object with `agg_tok_s` > 0, `ttft_p50`, `tpot_median`, `finish_reason_dist`. (If no GPU/server, this step is run locally on the box, not in CI.)

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_client.py
git commit -m "feat(bench): streaming chat/completions client + TTFT/TPOT + prompt loop

Backward-compatible: legacy 'BASE C 512' path unchanged for scripts/benchmark.sh.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: gguf-bench-cell.sh — single-cell orchestrator

**Files:**
- Create: `scripts/gguf-bench-cell.sh`

**Interfaces:**
- Consumes: `gguf_bench_args.py` (Task 3), `bench_client.py` (Task 8), `capture_proc.py` (Task 5).
- Produces: one JSON record per cell at `docs/results/matrix/cell-<study>-<weight>-np<np>-df<dflash>-vis<vision>.json`, containing: study/weight/dflash/vision/np, `metrics` (from client), `mem` (VmHWM + llama.cpp log components), `acceptance` (from /metrics), and the reproducibility manifest (`flags`, `seed`, `reps`, `build`, `rocm`, `kernel`, `date`).

- [ ] **Step 1: Write the cell orchestrator**

Create `scripts/gguf-bench-cell.sh`:

```bash
#!/usr/bin/env bash
# Run ONE matrix cell: launch llama-server with exact flags, warmup+measure via
# bench_client, capture RSS/power/acceptance, tear down. Emits one JSON record.
# Usage: gguf-bench-cell.sh <study> <weight:17gb|dynamic> <dflash:0|1> <vision:0|1> <np> <conf-file>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"
LLAMA="$HERE/third_party/llama.cpp/build/bin/llama-server"

STUDY=$1; WEIGHT=$2; DFLASH=$3; VISION=$4; NP=$5; CONF=$6
# shellcheck source=/dev/null
. "$CONF"
PER_SLOT_CTX=${PER_SLOT_CTX:-8192}; SEED=${SEED:-0}; REPS=${REPS:-3}; WARMUP=${WARMUP:-2}
TEMP=${TEMP:-0}; TOP_P=${TOP_P:-1.0}; TOP_K=${TOP_K:-0}; MAX_TOKENS=${MAX_TOKENS:-256}
RS=${REASONING_STRENGTH:-high}

CELL=$(printf '{"weight":"%s","dflash":%s,"vision":%s,"np":%s,"per_slot_ctx":%s,"study":"%s","seed":%s}' \
  "$WEIGHT" "$DFLASH" "$VISION" "$NP" "$PER_SLOT_CTX" "$STUDY" "$SEED")
SRV_ARGS=$(python3 scripts/gguf_bench_args.py "$CELL" server | tr -d '[]"' | sed 's/,/ /g')

LOG=$(mktemp)
"$LLAMA" $SRV_ARGS >"$LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

# wait for health
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "server failed to start"; cat "$LOG"; exit 1; }

IMG_ARG=(); [ "$VISION" = "1" ] && IMG_ARG=(--image scripts/prompt-sets/test-image.png)
METRICS=$(uv run --no-sync python scripts/bench_client.py http://127.0.0.1:8080 \
  --study "$STUDY" --endpoint chat --np "$NP" --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" --reps "$REPS" --warmup "$WARMUP" --seed "$SEED" \
  --reasoning-strength "$RS" "${IMG_ARG[@]}")

MEM=$(python3 scripts/capture_proc.py status "$SRV_PID")
POWER=$(rocm-smi --showpower --showtemp --json 2>/dev/null | python3 scripts/capture_proc.py power 2>/dev/null || echo '{}')
ACC=$(curl -s http://127.0.0.1:8080/metrics 2>/dev/null | python3 scripts/capture_proc.py metrics 2>/dev/null || echo 'null')

# parse llama.cpp mem components from log (weights/KV)
WGT=$(grep -oE 'model size[^0-9]*[0-9.]+ (MiB|GiB|MB|GB)' "$LOG" | tail -1 || true)

OUT="docs/results/matrix/cell-${STUDY}-${WEIGHT}-np${NP}-df${DFLASH}-vis${VISION}.json"
mkdir -p docs/results/matrix
python3 - "$OUT" "$STUDY" "$WEIGHT" "$DFLASH" "$VISION" "$NP" "$MEM" "$METRICS" "$ACC" "$POWER" "$WGT" "$SRV_ARGS" "$SEED" "$REPS" <<'PY'
import json, sys, subprocess, datetime, os
(out, study, weight, df, vis, np_, mem, metrics, acc, power, wgt, flags, seed, reps) = sys.argv[1:]
rocm = subprocess.run("rocm-smi --showproductname --json".split(), capture_output=True, text=True).stdout.strip()
rec = {"study": study, "weight": weight, "dflash": df=="1", "vision": vis=="1", "np": int(np_),
       "metrics": json.loads(metrics), "mem": json.loads(mem), "acceptance": json.loads(acc) if acc!="null" else None,
       "power_temp": json.loads(power) if power else {}, "llama_log_mem": wgt,
       "manifest": {"flags": flags, "seed": int(seed), "reps": int(reps), "build": "0b1bad1",
                    "rocm": rocm, "kernel": os.popen("uname -r").read().strip(),
                    "date": datetime.date.today().isoformat()}}
json.dump(rec, open(out, "w"), indent=2)
print("wrote", out)
PY
rm -f "$LOG"
```

- [ ] **Step 2: Make it executable + shellcheck (if available)**

```bash
chmod +x scripts/gguf-bench-cell.sh
shellcheck scripts/gguf-bench-cell.sh 2>/dev/null || echo "shellcheck not installed (CI runs it)"
```

- [ ] **Step 3: Run ONE real cell on hardware to validate the pipeline end-to-end**

```bash
cd /home/amd/Desktop/muse-rocm
bash scripts/gguf-bench-cell.sh study1 17gb 0 0 1 configs/gguf-bench/study1.conf
cat docs/results/matrix/cell-study1-17gb-np1-df0-vis0.json | python3 -m json.tool | head -30
```
Expected: a well-formed record with `metrics.agg_tok_s` > 0, `mem.VmHWM_gib` ~16–18, `manifest` populated. **Gate:** this is the P0 validation (spec §8).

- [ ] **Step 4: Commit**

```bash
git add scripts/gguf-bench-cell.sh
git commit -m "feat(bench): single-cell orchestrator (gguf-bench-cell.sh) + P0 validation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: run-gguf-matrix.sh — driver

**Files:**
- Create: `scripts/run-gguf-matrix.sh`

**Interfaces:**
- Consumes: the study configs (Task 7) + `gguf-bench-cell.sh` (Task 9) + `render_matrix.py` (Task 6).
- Produces: runs every cell in a randomized order, then writes `docs/results/matrix/matrix.md`.

- [ ] **Step 1: Write the driver**

Create `scripts/run-gguf-matrix.sh`:

```bash
#!/usr/bin/env bash
# Drive a full study (or all three): enumerate cells, randomize order to control
# for thermal drift, run each via gguf-bench-cell.sh, then render the markdown report.
# Usage: run-gguf-matrix.sh [study1|study2|study3|all]   (default: all)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
WHICH=${1:-all}
STUDIES=("$WHICH"); [ "$WHICH" = "all" ] && STUDIES=(study1 study2 study3)

# Build the cell list, then randomize order.
CELLS=()
for ST in "${STUDIES[@]}"; do
  CONF="configs/gguf-bench/$ST.conf"; . "$CONF"
  for W in $WEIGHTS; do for D in $DFS; do for N in $NPS; do
    # study3: only the documented vision cells (4 baseline + 1 dflash@17gb,c1)
    if [ "$ST" = "study3" ] && [ "$D" = "1" ] && { [ "$W" != "17gb" ] || [ "$N" != "1" ]; }; then continue; fi
    CELLS+=("$ST $W $D ${VISION:-0} $N $CONF")
  done; done; done
done
printf '%s\n' "${CELLS[@]}" | shuf > /tmp/matrix-order.txt
echo "Running ${#CELLS[@]} cells in randomized order (see /tmp/matrix-order.txt):"
cat /tmp/matrix-order.txt

while read -r ST W D V N CONF; do
  [ -z "$ST" ] && continue
  echo "=== cell: $ST weight=$W dflash=$D vision=$V np=$N ==="
  bash scripts/gguf-bench-cell.sh "$ST" "$W" "$D" "$V" "$N" "$CONF" || echo "  CELL FAILED (logged above)"
done < /tmp/matrix-order.txt

uv run --no-sync python scripts/render_matrix.py > docs/results/matrix/matrix.md
echo "rendered docs/results/matrix/matrix.md"
```

- [ ] **Step 2: shellcheck + dry sanity**

```bash
chmod +x scripts/run-gguf-matrix.sh
shellcheck scripts/run-gguf-matrix.sh 2>/dev/null || true
# Dry check: does the cell enumeration for study1 produce 4 cells (2 weights x 2 dflash x 1 np)?
bash -c '. configs/gguf-bench/study1.conf; n=0; for w in $WEIGHTS; do for d in $DFS; do for x in $NPS; do n=$((n+1)); done; done; done; echo "study1 cells=$n"'
```
Expected: `study1 cells=4`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run-gguf-matrix.sh
git commit -m "feat(bench): matrix driver (randomized cell order) + markdown render

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: P-A — Run Study 1 (Meta-comparable DFlash anchor) on 7.2.1

**Files:**
- Output: `docs/results/matrix/cell-study1-*.json`, `docs/results/matrix/matrix.md`

**Interfaces:**
- Consumes: the full harness (Tasks 2–10). Validates spec §5 Study 1 + the byte-equivalence + acceptance requirements.

- [ ] **Step 1: Run Study 1 (4 cells: 17gb/dynamic × baseline/DFlash, greedy, batch 1)**

```bash
cd /home/amd/Desktop/muse-rocm
bash scripts/run-gguf-matrix.sh study1
```
Expected: 4 `cell-study1-*.json` records; each DFlash cell has non-null `acceptance`.

- [ ] **Step 2: Byte-equivalence check (greedy DFlash vs baseline, same prompt+seed)**

Run a one-off: for one prompt, capture the `content` from the baseline cell server and the DFlash cell server with identical `--seed 0 --temp 0`, and assert token-identical. Add this as `scripts/check_dflash_equiv.sh`:

```bash
#!/usr/bin/env bash
# Assert DFlash greedy output is byte-identical to baseline for a fixed prompt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
LLAMA="$HERE/third_party/llama.cpp/build/bin/llama-server"
P='{"model":"muse-glimmer-30B","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}],"max_tokens":256,"temperature":0,"seed":0,"chat_template_kwargs":{"reasoning_strength":"high"}}'

get_content() { # $1 = extra server args
  "$LLAMA" -m models/muse-glimmer-30B-kquant-17gb.gguf -ngl 999 -np 1 -c 8192 --jinja --temp 0 --seed 0 --port 8090 $1 >/dev/null 2>&1 &
  local p=$!; trap "kill $p 2>/dev/null" RETURN
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 1; done
  curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d "$P" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}
BASE=$(get_content "")
DF=$(get_content "-md models/dflash-kquant.gguf -ngld 99")
python3 - "$BASE" "$DF" <<'PY'
import sys
b, d = sys.argv[1], sys.argv[2]
print("baseline:", repr(b)); print("dflash  :", repr(d))
sys.exit(0 if b.strip() == d.strip() else 1)
PY
```
Run it: `bash scripts/check_dflash_equiv.sh && echo "EQUIVALENCE PASS"`

Expected: `EQUIVALENCE PASS` (greedy spec-decode is exact; a mismatch is a finding to document).

- [ ] **Step 3: Verify acceptance captured + speedup computed**

```bash
python3 -c "
import glob,json
for p in sorted(glob.glob('docs/results/matrix/cell-study1-*.json')):
    r=json.load(open(p)); m=r['metrics']; a=r.get('acceptance') or {}
    print(r['weight'], 'DFlash' if r['dflash'] else 'baseline', f'{m[\"agg_tok_s\"]:.1f} tok/s', 'acc=%s'%a.get('acceptance_rate'))
"
```
Expected: DFlash tok/s > baseline for each weight; acceptance_rate printed.

- [ ] **Step 4: Commit results + equivalence script**

```bash
git add scripts/check_dflash_equiv.sh docs/results/matrix/
git commit -m "data(bench): Study 1 (Meta-comparable DFlash anchor) on ROCm 7.2.1

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: P-B + P-C — Run Study 2 (load) + Study 3 (vision) on 7.2.1

**Files:**
- Output: `cell-study2-*.json` (12 cells), `cell-study3-*.json` (5 cells), updated `matrix.md`.

- [ ] **Step 1: Run Study 2 (2 weights × {1,4,16} × {baseline,DFlash})**

```bash
cd /home/amd/Desktop/muse-rocm
bash scripts/run-gguf-matrix.sh study2
```
Expected: 12 records; verify `finish_reason_dist` is predominantly `stop` (not `length`) — if `length` dominates, per-slot ctx / max_tokens is wrong (see troubleshooting).

- [ ] **Step 2: Run Study 3 (vision: {17gb,dynamic}×{c1,c4} baseline + 17gb×DFlash×c1)**

```bash
bash scripts/run-gguf-matrix.sh study3
```
Expected: 5 records; each `mem.VmHWM_gib` should exceed the text-only 17gb/dynamic baseline by ~+1.5–2.5 GB (mmproj) — cross-check Meta's ~+2 GB.

- [ ] **Step 3: Render the full matrix**

```bash
uv run --no-sync python scripts/render_matrix.py > docs/results/matrix/matrix.md
```

- [ ] **Step 4: Commit**

```bash
git add docs/results/matrix/
git commit -m "data(bench): Study 2 (load) + Study 3 (vision) on ROCm 7.2.1

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: llama-bench cross-check (optional) + docs

**Files:**
- Output: `docs/results/matrix/llama-bench.json`
- Modify: `docs/results/METHODOLOGY.md` (create), `docs/results/benchmark.md`, `docs/adaptation.md`, `docs/troubleshooting.md`, `handoff.md`

- [ ] **Step 1: Optional llama-bench cross-check (non-DFlash model-level pp/tg)**

```bash
cd /home/amd/Desktop/muse-rocm
B=third_party/llama.cpp/build/bin
for W in models/muse-glimmer-30B-kquant-17gb.gguf models/muse-glimmer-30B-kquant-dynamic.gguf; do
  "$B/llama-bench" -m "$W" -ngl 99 -p 512 -n 128 -o json \
    >> docs/results/matrix/llama-bench.json
done
```
Expected: pp512 (prefill) + tg128 (decode) numbers per weight; cross-check the c=1 server tok/s.

- [ ] **Step 2: Write METHODOLOGY.md (the science doc)**

Create `docs/results/METHODOLOGY.md` covering: hardware/software manifest, exact serve flags, the prompt set, every metric's definition (spec §6 table), statistical treatment (reps/median/warmup/randomized order), fairness controls, honest caveats (unified-memory vs discrete-GPU, finish_reason, reasoning-model decode). Point every published number here. Pull the reproducibility fields from a sample cell record.

- [ ] **Step 3: Update benchmark.md with the new studies + Meta comparison**

Add to `docs/results/benchmark.md`: a "Study 1" section placing the gfx1151 row next to Meta's RTX 5090 3.1× / M5 Max 1.8× rows (same methodology, different hardware), a "Study 2" throughput-under-load section (labeled NOT Meta-comparable), a "Study 3" vision section, and a memory table cross-checked against Meta's ~17/20 GB envelope. Reference `matrix.md` and `METHODOLOGY.md`.

- [ ] **Step 4: Update adaptation.md, troubleshooting.md, handoff.md**

- `docs/adaptation.md`: change the spec-decoding row from "deferred (v1 off)" to "validated on llama.cpp" with the measured speedup + acceptance (from Study 1).
- `docs/troubleshooting.md`: add entries — (a) the harmless `[spec] failed to measure draft model memory` warning, (b) the silent context-truncation trap (high `length` finish_reason), (c) `reasoning_strength` controls thinking length, (d) the DFlash byte-equivalence finding.
- `handoff.md`: update status (DFlash + vision done on llama.cpp; 7.14.0 = separate Part 2 plan, gated).

- [ ] **Step 5: Commit**

```bash
git add docs/results/METHODOLOGY.md docs/results/benchmark.md docs/results/matrix/llama-bench.json docs/adaptation.md docs/troubleshooting.md handoff.md
git commit -m "docs(bench): METHODOLOGY + Study 1/2/3 results, Meta comparison, adaptation/troubleshooting updates

Closes v1 deferred items (DFlash, vision) for the llama.cpp path. ROCm 7.14.0
remains a separate gated Part 2 plan.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Notes for execution

- **Tasks 1–10 are hardware-free for the TDD unit tests** (run `uv run --no-sync pytest` — these are CI-green). Only Tasks 9.3, 11, 12, 13.1 need the gfx1151 GPU + a running server.
- **The P0 gate is Task 9.3** (one real cell produces a sane JSON record). Do not start Task 11 until 9.3 passes.
- **Draft-acceptance (spec §13.1 / top risk)** is resolved by `--metrics` on every server + `scrape_metrics` (Task 5). If `llama_speculative_*` counters are absent from `/metrics` on this build, fall back to parsing `-v` stderr in `capture_proc.py` — this is the one scoped risk; confirm in Task 9.3.
- **Part 2 (ROCm 7.14.0)** is a separate plan, written after Task 12's 7.2.1 data is committed immutable, starting with the feasibility probe (spec §9).
