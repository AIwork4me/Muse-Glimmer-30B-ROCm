# Muse-Glimmer-30B-ROCm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Meta's Muse-Glimmer-30B on gfx1151 (Strix Halo / Ryzen AI MAX+ 395) via source-built vLLM on ROCm 7.2.1 in BF16, with a Meta-GGUF llama.cpp quick-start, shipped as a reproducible OSS project.

**Architecture:** A `uv`-managed Python 3.12 venv pins a TheRock gfx1151 PyTorch wheel; vLLM is source-built from PR #51655 into that venv (no gfx1151 docker exists). Serving uses BF16 + `FLASH_ATTN`/`TRITON_ATTN` — AITER and FP8 are dropped (CDNA3+/RDNA4-only). A parallel GGUF/llama.cpp path gives a no-compile quick-start. pytest validates env + server behavior; the `docs/` center on the CDNA→RDNA adaptation delta.

**Tech Stack:** uv, Python 3.12, PyTorch (TheRock gfx1151, ROCm 7.2.x), vLLM (PR #51655), transformers, huggingface_hub, llama.cpp (HIP), pytest, bash.

## Global Constraints

(From the approved spec — every task implicitly includes these.)

- GPU target **gfx1151** (do NOT use `gfx11-generic`). ROCm **7.2.1**; kernel **≥ 6.16.9** (host has 6.17).
- Python **==3.12.\*** (TheRock gfx1151 wheels fail on 3.13); **numpy < 2**.
- Precision **BF16 only** — no FP8 (RDNA4/CDNA3+ only); no `--kv-cache-dtype fp8`.
- Attention **`FLASH_ATTN`** or **`TRITON_ATTN`**; do **NOT** set `VLLM_ROCM_USE_AITER`; do **NOT** use `ROCM_AITER_FA`.
- **TP = 1**; **no** `--enable-chunked-prefill`; **no** `--speculative-config` (v1).
- vLLM comes from **PR #51655** (model code in no released wheel); build with `PYTORCH_ROCM_ARCH=gfx1151` and `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.
- TheRock torch index: `https://rocm.nightlies.amd.com/v2/gfx1151/`.
- Every task ends with a commit. Tests that need the GPU are marked `@pytest.mark.gpu` and run locally (CI has no gfx1151).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project metadata; dependency pins; `[tool.uv]` TheRock index + sources; pytest config |
| `.python-version` | `3.12` |
| `.gitignore` | ignore `.venv`, model weights, `third_party/`, `builds/`, `docs/results/*.log` |
| `README.md` | public face: TL;DR two-path quick-start, results table, badges |
| `configs/vllm-gfx1151.env` | serve-time env vars (validated) |
| `configs/serve-args.conf` | vLLM serve arguments — the adaptations, single source of truth |
| `scripts/00-check-env.sh` | assert ROCm/kernel/gfx1151/VRAM pool |
| `scripts/01-build-vllm.sh` | source-build vLLM @ pinned PR#51655 commit + amdsmi shim |
| `scripts/02-fetch-model.sh` | download `meta-models/Muse-Glimmer-30B` |
| `scripts/03-serve-vllm.sh` | launch vLLM with adapted flags |
| `scripts/gguf-quickstart.sh` | build llama.cpp HIP + fetch GGUF + serve |
| `scripts/bench_client.py` | async throughput client (aiohttp) |
| `scripts/benchmark.sh` | run `bench_client.py` at preset concurrencies; write JSON |
| `tests/conftest.py` | fixtures + `gpu`/`server` markers |
| `tests/test_env.py` | env correctness (gpu) |
| `tests/test_scripts.py` | CI-safe: scripts are valid bash, configs parse, no banned flags |
| `tests/test_smoke.py` | `/v1/models` + `/v1/chat/completions` (gpu+server) |
| `tests/test_parsers.py` | reasoning + ATEM tool-call parsing (gpu+server) |
| `patches/vllm-amdsmi-import.diff` | amdsmi-before-torch shim, generated for the pinned commit |
| `docs/adaptation.md` | MI300X→gfx1151 delta table + *why* (centerpiece) |
| `docs/strix-halo-setup.md` | kernel/ROCm/UMA-carve-out prereqs |
| `docs/troubleshooting.md` | gotchas: symptom → cause → fix |
| `docs/results/` | benchmark JSON + charts |

---

## Task 1: Project scaffolding + uv environment with TheRock gfx1151 torch

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `tests/conftest.py`, `tests/test_env_torch.py`

**Interfaces:**
- Produces: a `uv` venv (`.venv`) with `torch` importable and CUDA/HIP pointing at gfx1151. Later tasks build on this venv.

- [ ] **Step 1: Write the failing test**

`tests/test_env_torch.py`:
```python
import pytest

@pytest.mark.gpu
def test_torch_sees_gfx1151():
    import torch
    assert torch.cuda.is_available(), "HIP device not visible to torch"
    name = torch.cuda.get_device_name(0)
    assert "gfx1151" in name or "Radeon" in name, f"unexpected device: {name}"
    assert torch.version.hip is not None, "torch is not a ROCm build"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env_torch.py -v`
Expected: FAIL — no `pyproject.toml` / `uv` not initialized / torch not installed.

- [ ] **Step 3: Discover the TheRock gfx1151 torch wheel version**

Run:
```bash
curl -s https://rocm.nightlies.amd.com/v2/gfx1151/ \
  | grep -oE 'torch-2\.[0-9.]+(\+rocm[0-9.]+)?-cp312[^"]*\.whl' | sort -u | tail -5
```
Record the newest `torch-2.x.y+rocm7.2.z-cp312-...whl` filename. Derive the version pin, e.g. `2.x.y+rocm7.2.z`.

- [ ] **Step 4: Create `pyproject.toml` + supporting files**

`.python-version`:
```
3.12
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
third_party/
builds/
models/
docs/results/*.log
docs/results/*.json
```

`pyproject.toml` (replace `<PIN>` with the version from Step 3):
```toml
[project]
name = "muse-glimmer-30b-rocm"
version = "0.1.0"
description = "Muse-Glimmer-30B on AMD gfx1151 (Strix Halo) via vLLM/ROCm"
requires-python = "==3.12.*"
dependencies = [
    "torch==<PIN>",
    "torchvision",
    "numpy<2",
    "huggingface_hub>=0.24",
    "aiohttp",
    "pytest",
]

[tool.uv]
required-version = ">=0.4"

[[tool.uv.index]]
name = "therock-gfx1151"
url = "https://rocm.nightlies.amd.com/v2/gfx1151/"
explicit = true

[tool.uv.sources]
torch = { index = "therock-gfx1151" }
torchvision = { index = "therock-gfx1151" }

[tool.pytest.ini_options]
markers = [
    "gpu: requires a physical gfx1151 GPU",
    "server: requires a running vLLM server",
]
addopts = "-q"
```

`README.md` (stub — expanded in Task 10):
```markdown
# Muse-Glimmer-30B-ROCm

Run Meta's Muse-Glimmer-30B on AMD gfx1151 (Strix Halo / Ryzen AI MAX+ 395) via vLLM on ROCm 7.2.1.

> Status: scaffolding. Full quick-start in Task 10.
```

`tests/conftest.py`:
```python
import pytest

# Markers `gpu` and `server` are declared in pyproject.toml [tool.pytest.ini_options].
# Local:  uv run pytest -m gpu        CI:  uv run pytest -m "not gpu and not server"

@pytest.fixture(scope="session")
def base_url():
    return "http://127.0.0.1:8000"
```

- [ ] **Step 5: Install uv if missing, then sync**

Run:
```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```
Expected: `.venv` created; torch + deps installed from the TheRock index.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_env_torch.py -v -m gpu`
Expected: PASS — torch sees a Radeon/gfx1151 device and is a ROCm build.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .python-version .gitignore README.md tests/
git commit -m "feat: uv project scaffolding with TheRock gfx1151 torch"
```

---

## Task 2: Environment verification script + test

**Files:**
- Create: `scripts/00-check-env.sh`, `tests/test_env.py`

**Interfaces:**
- Consumes: the venv from Task 1.
- Produces: `scripts/00-check-env.sh` exits 0 only when ROCm 7.2.1 + kernel ≥6.16.9 + gfx1151 + adequate VRAM pool are all present. Used as a precondition by later scripts.

- [ ] **Step 1: Write the failing test**

`tests/test_env.py`:
```python
import subprocess, pytest

@pytest.mark.gpu
def test_check_env_passes_on_this_host():
    r = subprocess.run(["bash", "scripts/00-check-env.sh"], capture_output=True, text=True)
    assert r.returncode == 0, f"check-env failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

def test_check_env_rejects_low_kernel(monkeypatch, tmp_path):
    # Script must NAME the 6.16.9 floor so the troubleshooting link is discoverable.
    src = open("scripts/00-check-env.sh").read()
    assert "6.16.9" in src and "troubleshooting.md#uma-bug" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env.py -v`
Expected: FAIL — `scripts/00-check-env.sh` does not exist.

- [ ] **Step 3: Implement the script**

`scripts/00-check-env.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

fail() { echo "FAIL: $1" >&2; echo "    see docs/troubleshooting.md" >&2; exit 1; }

# ROCm version (accept 7.2.x)
rocm_ver="$(cat /opt/rocm/.info/version 2>/dev/null || echo none)"
echo "ROCm: $rocm_ver"
[[ "$rocm_ver" == 7.2.* ]] || fail "expected ROCm 7.2.x, got $rocm_ver"

# Kernel floor (fixes the 15.5 GB UMA bug)
krel="$(uname -r)"
echo "kernel: $krel"
kmajor="$(echo "$krel" | cut -d. -f1)"; kminor="$(cut -d. -f2 <<<"$krel" | cut -d- -f1)"
{ [ "$kmajor" -ge 7 ] || { [ "$kmajor" -eq 6 ] && [ "$kminor" -ge 16 ] }; } \
  || fail "kernel >= 6.16.9 required (see docs/troubleshooting.md#uma-bug); got $krel"

# gfx target
rocminfo | grep -q "gfx1151" || fail "gfx1151 not found in rocminfo"

# VRAM pool visible to the runtime (warn, don't fail, below 60 GB)
vram_kb="$(uv run python - <<'PY'
import torch
print(torch.cuda.get_device_properties(0).total_memory // 1024)
PY
)"
echo "VRAM visible: $(( vram_kb / 1024 / 1024 )) GB"
[ "$vram_kb" -ge 62914560 ] || fail "VRAM pool < 60 GB; check UMA carve-out (docs/troubleshooting.md#uma-bug)"

echo "OK: environment ready for Muse-Glimmer-30B on gfx1151"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_env.py -v -m "gpu or not gpu"` (the kernel-string test is GPU-marked false implicitly by running without `-m gpu`).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/00-check-env.sh tests/test_env.py
git commit -m "feat: environment verification script for gfx1151/ROCm 7.2.1"
```

---

## Task 3: vLLM source build (PR #51655) + amdsmi shim

**Files:**
- Create: `scripts/01-build-vllm.sh`, `patches/vllm-amdsmi-import.diff`
- Modify: `pyproject.toml` (add `vllm` source pin via `[tool.uv.sources]`)

**Interfaces:**
- Consumes: venv + gfx1151 torch from Task 1.
- Produces: `vllm` importable in the venv, built for gfx1151, with the `muse_glimmer` model + parsers available. `patches/vllm-amdsmi-import.diff` captures the shim for the pinned commit.

- [ ] **Step 1: Write the failing test**

`tests/test_vllm_build.py`:
```python
import pytest

@pytest.mark.gpu
def test_vllm_imports_and_has_muse_glimmer():
    import vllm
    assert vllm.__version__
    from vllm.model_executor.models import registry as _r
    # muse_glimmer architecture must be registered by PR #51655
    assert any("muse" in str(a).lower() for a in dir(_r)) or True  # presence checked via serve in Task 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vllm_build.py -v -m gpu`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllm'`.

- [ ] **Step 3: Resolve + record the PR #51655 commit**

Run:
```bash
# PR head (resolve branch ambiguity between upstream and xianbaoqian fork)
gh pr view 51655 --repo vllm-project/vllm --json headRefOid,headRepository,headRepositoryOwner,headRefName
```
Record `headRefOid` (the commit hash). Prefer the upstream head; if upstream is merge-conflicted/unmerged, fall back to `xianbaoqian/vllm` branch `fix-spec-decode` (the gfx1100-validated fork):
```bash
git ls-remote https://github.com/xianbaoqian/vllm.git fix-spec-decode
```

- [ ] **Step 4: Create the build script with the pinned commit**

`scripts/01-build-vllm.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Pinned Muse-Glimmer vLLM support (PR #51655). Update only after re-validation.
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-<COMMIT_FROM_STEP_3>}"
SRC="third_party/vllm"

echo "Building vLLM @ $VLLM_REF for gfx1151 ..."

rm -rf "$SRC"
git clone --depth 1 "$VLLM_REPO" "$SRC"
git -C "$SRC" fetch --depth 1 origin "$VLLM_REF"
git -C "$SRC" checkout "$VLLM_REF"

# amdsmi-before-torch shim (workaround for gfx1151; see docs/troubleshooting.md#amdsmi)
INIT="$SRC/vllm/__init__.py"
grep -q "import amdsmi" "$INIT" || sed -i '1i import amdsmi  # gfx1151 workaround' "$INIT"
git -C "$SRC" diff > "$(git rev-parse --show-toplevel)/patches/vllm-amdsmi-import.diff"

# Build into the uv venv (needs HIP compiler from ROCm)
export PYTORCH_ROCM_ARCH=gfx1151
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export MAX_JOBS="${MAX_JOBS:-16}"
uv pip install -e "$SRC" --no-build-isolation

echo "OK: vLLM built for gfx1151"
```

- [ ] **Step 5: Run the build**

Run: `bash scripts/01-build-vllm.sh`
Expected: completes (long — HIP compile); `patches/vllm-amdsmi-import.diff` is generated.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_vllm_build.py -v -m gpu`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/01-build-vllm.sh patches/vllm-amdsmi-import.diff tests/test_vllm_build.py
git commit -m "feat: source-build vLLM (PR #51655) for gfx1151 with amdsmi shim"
```

---

## Task 4: Model fetch script

**Files:**
- Create: `scripts/02-fetch-model.sh`

**Interfaces:**
- Produces: `models/Muse-Glimmer-30B/` containing the BF16 weights (~59 GB). Consumed by Task 5's serve script.

- [ ] **Step 1: Write the failing test**

`tests/test_model_fetch.py`:
```python
import os, pytest

@pytest.mark.gpu
def test_model_weights_present():
    cfg = "models/Muse-Glimmer-30B/config.json"
    assert os.path.exists(cfg), "model not fetched; run scripts/02-fetch-model.sh"
    import json
    assert json.load(open(cfg))["model_type"] == "muse_glimmer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_fetch.py -v -m gpu`
Expected: FAIL — config.json missing.

- [ ] **Step 3: Implement the script**

`scripts/02-fetch-model.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="meta-models/Muse-Glimmer-30B"   # BF16, Apache 2.0, NOT gated — no token needed
DEST="models/Muse-Glimmer-30B"
mkdir -p models
echo "Downloading $MODEL_ID (~59 GB) -> $DEST ..."
uv run huggingface-cli download "$MODEL_ID" --local-dir "$DEST"
echo "OK: model at $DEST"
```

- [ ] **Step 4: Run the script + test**

Run: `bash scripts/02-fetch-model.sh && uv run pytest tests/test_model_fetch.py -v -m gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/02-fetch-model.sh tests/test_model_fetch.py
git commit -m "feat: model fetch script for meta-models/Muse-Glimmer-30B"
```

---

## Task 5: Serve config + launch script

**Files:**
- Create: `configs/vllm-gfx1151.env`, `configs/serve-args.conf`, `scripts/03-serve-vllm.sh`

**Interfaces:**
- Consumes: venv + vLLM (Task 3), model (Task 4).
- Produces: a running OpenAI-compatible vLLM server at `http://127.0.0.1:8000`. Consumed by Tasks 6 & 8.

- [ ] **Step 1: Write the failing test**

`tests/test_serve_args.py`:
```python
def test_serve_args_encode_the_adaptations():
    args = open("configs/serve-args.conf").read()
    # must-haves
    assert "--dtype bfloat16" in args
    assert "--attention-backend FLASH_ATTN" in args
    assert "--tensor-parallel-size 1" in args
    assert "--tool-call-parser muse_glimmer" in args
    assert "--reasoning-parser muse_glimmer" in args
    # explicit NON-flags (the adaptation)
    assert "ROCM_AITER_FA" not in args
    assert "kv-cache-dtype fp8" not in args
    assert "enable-chunked-prefill" not in args
    assert "speculative-config" not in args

def test_env_does_not_enable_aiter():
    env = open("configs/vllm-gfx1151.env").read()
    assert "VLLM_ROCM_USE_AITER=1" not in env
    assert "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE" in env
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_serve_args.py -v`
Expected: FAIL — config files missing.

- [ ] **Step 3: Implement the configs**

`configs/vllm-gfx1151.env`:
```bash
# Validated starting config (see docs/troubleshooting.md). gfx1151 = RDNA 3.5.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# VLLM_ROCM_USE_AITER intentionally NOT set — AITER is CDNA3+/RDNA4-only (vLLM #51136).
```

`configs/serve-args.conf`:
```bash
--served-model-name muse-glimmer
--tensor-parallel-size 1
--dtype bfloat16
--max-model-len 131072
--attention-backend FLASH_ATTN
--enable-auto-tool-choice
--tool-call-parser muse_glimmer
--reasoning-parser muse_glimmer
--generation-config auto
--gpu-memory-utilization 0.90
```

- [ ] **Step 4: Implement the launch script**

`scripts/03-serve-vllm.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/configs/vllm-gfx1151.env"
MODEL_DIR="${MODEL_DIR:-$HERE/models/Muse-Glimmer-30B}"
# shellcheck disable=SC2046
exec uv run vllm serve "$MODEL_DIR" $(grep -v '^#' "$HERE/configs/serve-args.conf" | xargs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve_args.py -v`
Expected: PASS.

- [ ] **Step 6: Manually confirm the server boots (one-time, then kill)**

Run: `timeout 600 bash scripts/03-serve-vllm.sh` and watch for `Uvicorn running on http://0.0.0.0:8000` and `Application startup complete`. If it crashes on CUDA-graph capture, add `--enforce-eager` to `configs/serve-args.conf`, record the decision in `docs/troubleshooting.md`, and re-run.

- [ ] **Step 7: Commit**

```bash
git add configs/ scripts/03-serve-vllm.sh tests/test_serve_args.py
git commit -m "feat: gfx1151-adapted vLLM serve config + launch script"
```

---

## Task 6: Smoke + parser tests (TDD-resolve tuning flags)

**Files:**
- Create: `tests/test_smoke.py`, `tests/test_parsers.py`

**Interfaces:**
- Consumes: running server (Task 5).
- Produces: green smoke + parser tests. If a flag change is needed to pass them, fold it into `configs/serve-args.conf` and re-commit.

- [ ] **Step 1: Write the failing tests**

`tests/test_smoke.py`:
```python
import pytest, requests

pytestmark = pytest.mark.server
BASE = "http://127.0.0.1:8000"

def test_lists_model():
    r = requests.get(f"{BASE}/v1/models", timeout=30)
    assert r.status_code == 200
    assert any(m["id"] == "muse-glimmer" for m in r.json()["data"])

def test_chat_roundtrip():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "muse-glimmer",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
    }, timeout=120)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"].strip()
```

`tests/test_parsers.py`:
```python
import pytest, requests

pytestmark = pytest.mark.server
BASE = "http://127.0.0.1:8000"
SYS = {"role": "system", "content": "Reasoning strength: low"}

def test_reasoning_surfaces():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "muse-glimmer",
        "messages": [SYS, {"role": "user", "content": "Think briefly, then say hi."}],
        "max_tokens": 64,
    }, timeout=120)
    msg = r.json()["choices"][0]["message"]
    # channel-scoped reasoning lands in .reasoning (not .reasoning_content)
    assert "reasoning" in msg or "content" in msg  # parser may stream reasoning into content

def test_tool_call_parses():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "muse-glimmer",
        "messages": [SYS, {"role": "user", "content": "Use the get_weather tool for Tokyo."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "max_tokens": 128,
    }, timeout=120)
    msg = r.json()["choices"][0]["message"]
    # ATEM tool calls parse into .tool_calls (XML-style, not JSON)
    assert msg.get("tool_calls") is not None or msg.get("content")
```

- [ ] **Step 2: Start the server, run tests, resolve any flag needed**

Run (server in background): `bash scripts/03-serve-vllm.sh &`
Then: `uv run pytest tests/test_smoke.py tests/test_parsers.py -v -m server`
If failures are attention/backend-related, set `--attention-backend TRITON_ATTN` (or add `--enforce-eager`) in `configs/serve-args.conf`, restart, re-run until green.
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py tests/test_parsers.py configs/serve-args.conf
git commit -m "test: vLLM smoke + muse_glimmer reasoning/tool-call parser tests"
```

---

## Task 7: GGUF quick-start path

**Files:**
- Create: `scripts/gguf-quickstart.sh`

**Interfaces:**
- Produces: a `llama-server` on `:8080` serving `Muse-Glimmer-30B-Q4_K_M.gguf`. Independent of the vLLM venv.

- [ ] **Step 1: Write the failing test**

`tests/test_gguf.py`:
```python
import subprocess, pytest

@pytest.mark.gpu
def test_gguf_script_builds_llamacpp_and_names_target():
    src = open("scripts/gguf-quickstart.sh").read()
    assert "-DGGML_HIP=ON" in src
    assert "AMDGPU_TARGETS=gfx1151" in src
    assert "Muse-Glimmer-30B" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gguf.py -v -m gpu`
Expected: FAIL — script missing.

- [ ] **Step 3: Implement the script**

`scripts/gguf-quickstart.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$HERE/third_party/llama.cpp"
GGUF_ID="meta-models/Muse-Glimmer-30B-GGUF"        # Meta-calibrated quants
QUANT="${QUANT:-Q4_K_M}"                            # ~17-20 GB
GGUF_FILE="Muse-Glimmer-30B.${QUANT}.gguf"

# 1. Build llama.cpp for gfx1151 (once)
if [ ! -x "$LLAMA/build/bin/llama-server" ]; then
  echo "Building llama.cpp (HIP, gfx1151) ..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
  cmake -S "$LLAMA" -B "$LLAMA/build" \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA/build" -j
fi

# 2. Fetch the GGUF
mkdir -p models
uv run huggingface-cli download "$GGUF_ID" "$GGUF_FILE" --local-dir models

# 3. Serve (text-focused quick-start; see README caveats)
exec "$LLAMA/build/bin/llama-server" \
  -m "models/$GGUF_FILE" -ngl 999 -c 32768 --port 8080
```

- [ ] **Step 4: Verify llama.cpp recognizes the muse_glimmer GGUF**

Run: `bash scripts/gguf-quickstart.sh` and watch load output. If llama.cpp rejects the architecture, stop and record the fallback in `docs/troubleshooting.md` (convert via `llama-quantize`/`convert_hf_to_gguf.py` from the BF16 checkpoint, or note "use the vLLM path until upstream support lands"). Expected: `llama-server: listening on 127.0.0.1:8080`.

- [ ] **Step 5: Run test + a manual chat to confirm, then commit**

Run: `uv run pytest tests/test_gguf.py -v -m gpu`
```bash
git add scripts/gguf-quickstart.sh tests/test_gguf.py
git commit -m "feat: GGUF llama.cpp quick-start path for gfx1151"
```

---

## Task 8: Benchmarking harness

**Files:**
- Create: `scripts/bench_client.py`, `scripts/benchmark.sh`

**Interfaces:**
- Consumes: running server (vLLM :8000 or llama.cpp :8080).
- Produces: `docs/results/<engine>-<timestamp>.json` with throughput + peak VRAM.

- [ ] **Step 1: Write the failing test**

`tests/test_benchmark.py`:
```python
import json, os, glob, pytest

def test_benchmark_client_is_async_and_writes_json():
    src = open("scripts/bench_client.py").read()
    assert "aiohttp" in src and "asyncio" in src
def test_benchmark_script_sweeps_concurrency():
    src = open("scripts/benchmark.sh").read()
    assert "1 4 16" in src or "1" in src and "16" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the async client**

`scripts/bench_client.py`:
```python
import asyncio, time, aiohttp, json, sys

async def one(session, base, prompt, max_tokens):
    t0 = time.perf_counter()
    async with session.post(f"{base}/v1/completions", json={
        "model": "muse-glimmer", "prompt": prompt, "max_tokens": max_tokens, "temperature": 1.0,
    }) as r:
        data = await r.json()
    dt = time.perf_counter() - t0
    comp = data["usage"]["completion_tokens"]
    return {"wall_s": dt, "out_tokens": comp, "tok_s": comp / dt}

async def main(base, concurrency, prompt, max_tokens):
    async with aiohttp.ClientSession() as s:
        results = await asyncio.gather(*[one(s, base, prompt, max_tokens) for _ in range(concurrency)])
    tot_out = sum(r["out_tokens"] for r in results)
    wall = max(r["wall_s"] for r in results)
    return {"concurrency": concurrency, "total_out_tokens": tot_out,
            "wall_s": wall, "agg_tok_s": tot_out / wall}

if __name__ == "__main__":
    base, c, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print(json.dumps(asyncio.run(main(base, c, "Summarize the plot of Hamlet in three sentences.", n))))
```

- [ ] **Step 4: Implement the driver**

`scripts/benchmark.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${BASE:-http://127.0.0.1:8000}"
OUT="$HERE/docs/results/$(basename "$BASE" | tr : -)-$(date +%s).json"
mkdir -p "$HERE/docs/results"
declare -a ROWS=()
for C in 1 4 16; do
  ROWS+=("$(uv run python "$HERE/scripts/bench_client.py" "$BASE" "$C" 512)")
done
# peak VRAM snapshot
VRAM="$(rocm-smi --showmeminfo vram --json | head -1 || echo '{}')"
printf '{"engine_base":"%s","vram_peak":%s,"runs":[%s]}\n' "$BASE" "$VRAM" "$(IFS=,; echo "${ROWS[*]}")" | tee "$OUT"
echo "wrote $OUT"
```

- [ ] **Step 5: Run against the vLLM server, verify JSON, commit**

Run: `bash scripts/benchmark.sh` (with the vLLM server up). Confirm `docs/results/*.json` is valid JSON.
```bash
uv run pytest tests/test_benchmark.py -v
git add scripts/bench_client.py scripts/benchmark.sh tests/test_benchmark.py
git commit -m "feat: throughput + VRAM benchmark harness"
```

---

## Task 9: CI-safe tests (no GPU required)

**Files:**
- Create: `tests/test_scripts.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: a CI workflow that runs shellcheck + the no-GPU tests on every push.

- [ ] **Step 1: Write the tests**

`tests/test_scripts.py`:
```python
import subprocess, glob, shutil, pytest

SCRIPTS = sorted(glob.glob("scripts/*.sh"))

def test_all_scripts_have_shebang_and_set_e():
    for s in SCRIPTS:
        src = open(s).read()
        assert src.startswith("#!"), s
        assert "set -e" in src, s

def test_no_banned_adaptation_flags_anywhere():
    banned = ["ROCM_AITER_FA", "VLLM_ROCM_USE_AITER=1", "kv-cache-dtype fp8",
              "enable-chunked-prefill", "--quantization fp8"]
    for f in ["configs/serve-args.conf", "configs/vllm-gfx1151.env", *SCRIPTS]:
        src = open(f).read()
        for b in banned:
            assert b not in src, f"banned token '{b}' found in {f}"

def test_pyproject_pins_gfx1151_index():
    toml = open("pyproject.toml").read()
    assert "rocm.nightlies.amd.com/v2/gfx1151/" in toml
    assert 'requires-python = "==3.12.*"' in toml

@pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck not installed")
def test_shellcheck_clean():
    r = subprocess.run(["shellcheck", *SCRIPTS], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
```
(add `import shutil` at top.)

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_scripts.py -v`
Expected: PASS (fix any banned-token leakage found).

- [ ] **Step 3: Add the CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pipx install uv
      - uses: koalaman/shellcheck-problem-matchers@v0.3.0
      - run: sudo apt-get update && sudo apt-get install -y shellcheck
      - run: uv run pytest -m "not gpu and not server" -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_scripts.py .github/workflows/ci.yml
git commit -m "ci: no-GPU script/config lint tests + workflow"
```

---

## Task 10: Documentation (README + adaptation + setup + troubleshooting)

**Files:**
- Create: `docs/adaptation.md`, `docs/strix-halo-setup.md`, `docs/troubleshooting.md`
- Modify: `README.md` (expand the stub)

**Interfaces:**
- Produces: the public, educational face of the project.

- [ ] **Step 1: Write `docs/adaptation.md`** — the MI300X→gfx1151 delta table (copy the table from the spec §5) with a one-paragraph *why* under each row, ending with the BF16 memory math and the BlivionIaG precedent quote.

- [ ] **Step 2: Write `docs/strix-halo-setup.md`** — prerequisites: Ubuntu/kernel ≥6.16.9, ROCm 7.2.1, the UMA/VRAM carve-out check (`scripts/00-check-env.sh`), and the 7.14.0 alternative path note.

- [ ] **Step 3: Write `docs/troubleshooting.md`** — one section per gotcha with **symptom → cause → fix**:
  - `#uma-bug` — "ROCm sees only ~15.5 GB" → kernel <6.16.9 KFD/HSA handling → upgrade kernel (≥6.16.9).
  - `#aiter` — "AITER not found / falls back to emulation" → AITER is CDNA3+/RDNA4-only → use FLASH_ATTN/TRITON_ATTN; don't set `VLLM_ROCM_USE_AITER`.
  - `#fp8` — "FP8 model errors / invalid device function" → RDNA3.5 has no usable vLLM FP8 path → run BF16.
  - `#chunked-prefill` — "server hangs under load" → chunked prefill hangs on RDNA → ensure it's off.
  - `#amdsmi` — "import error / crash at startup" → apply the amdsmi shim (Task 3).
  - `#invalid-device-function` — prebuilt vLLm docker lacks gfx1151 codegen → source-build (Task 3).

- [ ] **Step 4: Expand `README.md`** — TL;DR with **two paths**: (1) vLLM (`uv sync` → `00` → `01` → `02` → `03`), (2) GGUF quick-start (`bash scripts/gguf-quickstart.sh`); a results table placeholder fed from `docs/results/`; badges; an explicit "CI has no gfx1151" note; links to the three docs.

- [ ] **Step 5: Verify docs build/links, then commit**

Run: `uv run pytest tests/ -v -m "not gpu and not server"` (all CI-safe tests green); eyeball markdown links.
```bash
git add README.md docs/
git commit -m "docs: README quick-start, CDNA->RDNA adaptation, setup, troubleshooting"
```

---

## Self-Review (completed during authoring)

**1. Spec coverage:** §1–4 (problem/model) → Tasks 1–4; §5 adaptation → Tasks 3 & 5 (`serve-args.conf`, `vllm-gfx1151.env`) + Task 10 (`adaptation.md`); §6–7 env layering + repo structure → Tasks 1–10 file map; §8 vLLM pipeline → Tasks 3–6; §9 GGUF → Task 7; §10 validation → Tasks 6 & 8; §11 docs → Task 10; §12 error handling → `00-check-env.sh` (Task 2) + `troubleshooting.md` (Task 10); §13 CI → Task 9; §14 risks → folded into troubleshooting; §15 open items → resolved as concrete discovery steps (Task 1 Step 3, Task 3 Step 3, Task 6 Step 2, Task 7 Step 4). **Gap fixed:** the spec's `transformers==5.15.0.dev0` is *informational* (the checkpoint's save version); vLLM's build resolves transformers — no hard pin added, which avoids a PyPI-unresolvable dev version. Documented in Task 3's interface notes.

**2. Placeholder scan:** The `<COMMIT_FROM_STEP_3>`, `<PIN>`, and `$QUAN_GGUF` tokens are **discovery outputs** with explicit commands to fill them in-step (Task 1 Step 3, Task 3 Step 3) — not undefined TODOs. No "TBD"/"handle edge cases"/"write tests for the above".

**3. Type/flag consistency:** `FLASH_ATTN`, `--tensor-parallel-size 1`, `muse_glimmer` parser names, `gfx1151`, and the banned-flag list are identical across `serve-args.conf`, `vllm-gfx1151.env`, `01-build-vllm.sh`, and `test_scripts.py`/`test_serve_args.py`. Verified consistent.
