# CDNA → RDNA adaptation map

> This is not a collection of random workarounds. It is the engineering map
> from Meta's upstream MI300X/MI355X recipe to the validated `gfx1151` stack.

The upstream [vLLM recipes PR #776][recipe] is the CDNA reference and was merged
on 2026-08-10. The model-support [vLLM PR #51655][pr] remains open as of
2026-08-13, so this repository pins its source commit rather than implying
released-package support.

The target validated here is Muse-Glimmer-30B on AMD Ryzen AI MAX+ PRO 395 /
Radeon 8060S (`gfx1151`, RDNA 3.5). Other Radeon platforms stay pending until
they submit equivalent evidence.

## Classification

- **Required by RDNA architecture** — follows from hardware capabilities or
  topology and is expected to remain relevant.
- **ROCm-version-specific** — tied to a host/runtime/compiler version.
- **Temporary upstream limitation** — should be revisited when upstream lands
  or publishes support.
- **Validated workaround** — required on the recorded stack and supported by
  local evidence.
- **Historical workaround** — previously suspected or required, but not needed
  on the recorded stack; retained to prevent stale advice from returning.
- **Hardware-specific workaround** — observed on `gfx1151`; do not generalize
  to all RDNA without evidence.

## Delta table

| Area | Upstream MI-series behavior | Validated `gfx1151` behavior | Reason and evidence | Classification |
|---|---|---|---|---|
| vLLM model support | Recipe depends on Muse-Glimmer support | Source build at commit `606a12cd701875012ffe78a54afd29f97b825dba` | PR #51655 is still open; the commit contains model, processor and parser work used here | **Temporary upstream limitation** |
| Packaging | ROCm vLLM recipe/image | Compile from source with `PYTORCH_ROCM_ARCH=gfx1151` | Prebuilt code without the target can fail with `invalid device function` ([ROCm #4909][invdev]) | **Hardware-specific workaround** |
| PyTorch/runtime | Runtime supplied by MI-oriented stack | TheRock gfx1151 PyTorch `2.10.0+rocm7.13.0a20260513`, Python 3.12 | The recorded wheel contains the required target code; exact packages are locked | **ROCm-version-specific** |
| Build toolchain | Cohesive image toolchain/runtime | Host ROCm 7.2.1 development toolchain + TheRock runtime packages | The wheel bundle lacks the complete CMake development surface needed by the vLLM build | **Validated workaround** |
| vLLM compatibility patches | Newer upstream PyTorch API expectations | Apply the committed torch-2.10 and import-order patches | Patch files are reviewable and pinned to the vLLM commit | **Validated workaround** |
| Precision | BF16 and supported FP8 recipe variants | BF16 for vLLM; Meta K-quant for llama.cpp | The recorded `gfx1151` vLLM stack has no validated FP8 path | **Required by RDNA architecture** for this generation/stack |
| Attention | `ROCM_AITER_FA` in the MI recipe | `TRITON_ATTN`; `FLASH_ATTN` and AITER not used | AITER is gated away from this target ([vLLM #51136][aiter]); `FLASH_ATTN` failed startup because the library build lacked gfx1151 support | **Required by RDNA architecture** + **validated backend** |
| Tensor parallel | TP1/TP2 validated upstream on MI300X/MI355X | TP=1 | Strix Halo exposes one integrated GPU | **Required by hardware topology** |
| KV cache | BF16 and platform-supported variants | BF16 | No FP8 KV claim is made for this stack | **Required by validated precision path** |
| Chunked prefill | vLLM V1 default | Default-on on the pinned build; no explicit flag | A historical RDNA hang report exists ([vLLM #5013][chunked]), but it did not reproduce with the pinned `TRITON_ATTN` stack | **Historical workaround** |
| DFlash in vLLM | Model work includes DFlash support | Disabled on the vLLM path | The recorded run hit a draft-model registry problem; no fixed upstream status is claimed | **Temporary upstream limitation** |
| DFlash in llama.cpp | Separate engine, not the MI recipe | Validated at c=1/c=4 with explicit `--spec-type draft-dflash` | Raw cells show 2.20×/2.39× at Study 1; omitting `--spec-type` silently disables drafting | **Validated workaround** |
| DFlash at c=16 | No published comparable upstream cell | Do not use | Two negative cells record collapse/non-completion, including a 5 h 16 m aborted run | **Hardware/workload-specific negative finding** |
| Kernel | Not specified by the MI recipe | Project reference host: Linux ≥ 6.16.9 | This floor avoids the observed ~15.5 GiB UMA/KFD issue on the recorded host; AMD's supported kernel lines vary by distribution | **Host-specific validated workaround** |
| Memory accounting | Dedicated HBM tools are meaningful | Record VmPeak, VmHWM/RSS and VRAM counters together | Unified-memory mmap/offload makes any one counter incomplete | **Hardware-specific measurement adaptation** |

[recipe]: https://github.com/vllm-project/recipes/pull/776
[pr]: https://github.com/vllm-project/vllm/pull/51655
[invdev]: https://github.com/ROCm/ROCm/issues/4909
[aiter]: https://github.com/vllm-project/vllm/issues/51136
[chunked]: https://github.com/vllm-project/vllm/issues/5013
[uma]: https://github.com/ROCm/ROCm/issues/5444

## Packaging and runtime layers

“ROCm 7.2.1” alone does not describe the environment. The validated stack is
hybrid:

1. The host ROCm 7.2.1 installation supplies `hipcc`, headers and the complete
   CMake development packages.
2. AMD TheRock packages supply a gfx1151-targeted PyTorch/runtime line.
3. vLLM is built at the recorded PR commit for `gfx1151`.
4. At runtime, the loaded PyTorch ROCm libraries satisfy the vLLM extension's
   ROCm 7 ABI dependencies.

The authoritative values are in
[`configs/validated-stack.json`](../configs/validated-stack.json). This layout
is a validated workaround, not a general recommendation to mix arbitrary ROCm
versions.

Two committed vLLM patches make the build auditable:

- `patches/vllm-torch210-compat.diff` adapts API differences between the pinned
  PyTorch stable C++ surface and the newer vLLM source.
- `patches/vllm-amdsmi-import.diff` imports `amdsmi` before torch, which
  avoided the observed `gfx1151` startup failure.

## Attention and precision

The MI recipe's AITER choice cannot be copied mechanically. On this target AITER
is unavailable, and forcing its environment switch is not a performance tweak;
it selects an unsupported path. The initial `FLASH_ATTN` attempt also failed
during the first profiling forward pass because the required target build was
absent. `TRITON_ATTN` is the backend that booted and served on the validated
stack.

The precision decision is similarly evidence-bounded. BF16 is the validated
vLLM path. The repository does not convert “RDNA can support some low-precision
operations” into an unsupported claim that this exact FP8 model/KV/backend stack
works.

## Chunked prefill: keep the negative history accurate

An older RDNA issue led to an early plan to disable chunked prefill. On the
pinned vLLM V1 + `TRITON_ATTN` stack, default-on chunked prefill did not
reproduce that hang. Therefore the current serve configuration does not pass
either an enable or disable flag.

This is a **historical workaround**, not a current requirement. If a future
stack hangs, record the exact versions and workload before changing the shared
default.

## DFlash: engine split and failure modes

The recorded vLLM path did not complete DFlash enablement because its draft model
hit a registry limitation. The llama.cpp path did:

- 17GB Study 1: 10.48 → 23.03 tok/s (2.20×).
- Dynamic Study 1: 9.14 → 21.82 tok/s (2.39×).
- Draft acceptance is recorded in raw JSON.
- The original arithmetic greedy-equivalence smoke passed.

Two negative findings are part of the reference:

1. `-md dflash-kquant.gguf` alone loads the draft model but does not draft.
   `--spec-type draft-dflash --spec-draft-n-max 15` is required.
2. At `-np 16`, DFlash became pathological. The 17GB probe completed no
   requested tokens in 27.7 seconds; the dynamic full cell was stopped after
   5 h 16 m with 0.18% acceptance. Baseline c=16 remained healthy.

These findings are workload- and implementation-specific; they are not claims
that speculative decoding is generally unsuitable for RDNA.

## Kernel and unified memory

For this project's recorded Strix Halo host, the validated floor is **6.16.9**,
including the patch component. This is not AMD's universal ROCm 7.14 minimum:
AMD's [current RDNA3.5 guidance](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
lists distribution-specific kernel lines. The environment checker uses numeric
major/minor/patch comparison for the project floor:

```text
6.16.8  FAIL
6.16.9  PASS
6.17.0  PASS
7.0.0   PASS
```

The 60 GiB GPU-visible pool threshold is a **hard requirement for the validated
BF16 vLLM profile**. The default GGUF profile derives its hard floor from the
recorded model artifact size and warns, rather than fails, below the project's
validated memory envelope.

Approximate BF16 planning math, preserved from the validated work:

```text
weights       ~59.6 GB on disk (exact shard bytes are in the artifact manifest)
KV @ 128K      ~7 GB
activations    ~2–4 GB
total          ~68 GB, workload-dependent
```

These estimates are planning values, not new measurements.

## Evidence rules

- A raw cell, log or committed test result outranks an assumption.
- “Works on `gfx1151`” does not become “works on Radeon.”
- An upstream recipe is labeled upstream evidence, not local validation.
- Workarounds remain attached to their version and hardware scope.
- **Negative results are results.** Failed backends, silent no-ops and aborted
  cells remain visible so the next developer does not repeat them.
