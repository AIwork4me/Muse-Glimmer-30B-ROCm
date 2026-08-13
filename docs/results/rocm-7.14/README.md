# ROCm 7.14 validation track

ROCm 7.14 is the forward/current official gfx1151 validation target. This
directory is a protocol, not a claim that the full stack has passed.

The historical ROCm 7.2.1 evidence remains in `../matrix/`. In-progress 7.14
GGUF cells are written to `../matrix-714/` so the two tracks cannot overwrite
one another.

## Invariants

Keep these fixed unless the experiment explicitly studies them:

- llama.cpp commit from `configs/validated-stack.json`
- model revision, byte size and SHA256 from `configs/artifact-manifest.json`
- prompt set and fixed image
- cell flags, seeds, repetitions and warmup
- hardware, BIOS memory allocation and kernel
- no unrelated GPU load

Record every intentional difference.

## Phase 0 — safety and provenance

- [ ] Preserve `docs/results/matrix/` unchanged.
- [ ] Record `git status --short` and the repository commit.
- [ ] Verify all selected model artifacts with `scripts/verify_artifacts.py`.
- [ ] Record kernel, GPU/gfx target, memory pool and ROCm prefix.
- [ ] Confirm no benchmark server is already bound to ports 8080/8090.
- [ ] Keep the 7.14 install side-by-side; do not replace `/opt/rocm` merely to run this track.

Example preflight:

```bash
python3 scripts/verify_artifacts.py gguf models
git -C third_party/llama.cpp rev-parse HEAD
uname -r
rocminfo | grep -m1 gfx1151
```

## Phase 1 — environment initialization

Select the 7.14 prefix for one shell/process only:

```bash
export ROCM_PREFIX="$HOME/rocm-7.14.0"
export PATH="$ROCM_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PREFIX/lib:${LD_LIBRARY_PATH:-}"
hipcc --version
```

Capture the exact package/release source and prefix. “ROCm 7.14” without those
details is insufficient provenance.

## Phase 2 — llama.cpp build and smoke

Build the same pinned source into a separate directory:

```bash
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-714 \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/llama.cpp/build-714 -j
ldd third_party/llama.cpp/build-714/bin/llama-server
```

Gates:

- [ ] Build reports the pinned source commit.
- [ ] Binary links to the intended 7.14 prefix.
- [ ] Text model loads and completes a short greedy request.
- [ ] Vision projector loads and grounds one response.
- [ ] DFlash log shows non-zero drafted/accepted tokens.
- [ ] No system instability during smoke tests.

## Phase 3 — GGUF matrix

The dedicated wrapper selects the 7.14 binary and output directory:

```bash
bash scripts/run-gguf-matrix-714.sh --dry-run all
bash scripts/run-gguf-matrix-714.sh all
```

The first pass excludes c=16 because sustained-load stability is itself under
validation. Run it only after the reduced pass is stable and retain any failure
as a result.

Compare without editing either arm:

```bash
uv run --no-sync python scripts/compare_rocm.py \
  --a docs/results/matrix --b docs/results/matrix-714 \
  --label-a 7.2.1 --label-b 7.14.0
```

Review TTFT, TPOT, aggregate tok/s, mapped-memory envelope, DFlash acceptance,
vision behavior, finish reasons, temperature and stability—not only median
throughput.

## Phase 4 — BF16/vLLM validation

This is separate from the GGUF matrix and remains pending until a matching
7.14/TheRock Python stack is installed without disturbing the historical one.

- [ ] Python/runtime environment captured.
- [ ] BF16 artifacts hash-verified.
- [ ] vLLM source commit and patches recorded.
- [ ] Model initialization completes.
- [ ] `TRITON_ATTN` serve path completes text and vision requests.
- [ ] Reasoning and ATEM tool-call parsers validated.
- [ ] TTFT, TPOT and aggregate throughput captured.
- [ ] Memory methodology includes VmPeak, RSS/HWM and stronger counters where available.
- [ ] Long-context sanity test completed with the exact tested length recorded.
- [ ] Sustained stability window and request count recorded.
- [ ] DFlash status reported separately; no inference from llama.cpp to vLLM.

## Publication gate

A ROCm 7.14 result becomes “validated” only when:

- required cells/tests are complete or explicitly recorded as negative findings;
- artifacts and source revisions match the manifests;
- raw outputs and exact commands are committed;
- the comparison is reviewed for one-sided/missing cells;
- README wording is updated without altering the ROCm 7.2.1 history.
