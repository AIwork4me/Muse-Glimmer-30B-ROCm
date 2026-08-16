# W7900 (gfx1100) speed-test reproduction — `scripts/w7900-repro`

Reproduce the Muse-Glimmer-30B llama.cpp **Study-2 throughput matrix** on a
single **Radeon PRO W7900 (gfx1100 / RDNA3, 48 GiB)**. Results are published in
[`docs/results/w7900-gfx1100.md`](../../docs/results/w7900-gfx1100.md).

The package reuses the committed harness (`scripts/gguf-bench-cell.sh`,
`scripts/bench_client.py`, `scripts/gguf_bench_args.py`, `scripts/capture_proc.py`)
**unmodified**; it only adds container / bare-metal glue and two renderers. No
GPU index, host path, or download proxy is hard-coded — everything is env-overridable.

## What it produces

12 full 5-rep cells = `{17gb, dynamic}` × `baseline{c=1,4,16,32}` + `DFlash{c=1,4}`,
where `c` is llama.cpp concurrency (`-np`). Outputs land in the gitignored
`scripts/w7900-repro/_out/matrix-w7900-gfx1100/` (`headline.md`, `matrix.md`,
`cell-study2-*.json`, `llama-server-version.txt`).

## Two reproduction methods

| | Method 1 · image | Method 2 · bare metal |
|---|---|---|
| Entry | `bash run_all.sh` (Docker) | `bash run_host.sh` (no Docker) |
| llama.cpp | baked into the image (`/llamacpp_workspace/bin`) | **you provide** `llama-server` via `LLAMA_BIN_HOST` |
| Isolation | container | runs directly on the host |
| Good for | one-command, clean environment | you already have a local gfx1100 build |

Both share the same `_repro_driver.sh` and the same weights (`<repo>/models`).

## Prerequisites (both methods)

```bash
cd scripts/w7900-repro
bash 00_prepare.sh        # download 4 GGUFs from the OFFICIAL HF -> <repo>/models + verify
```

> Weights come from the **official Hugging Face** (`meta-models/Muse-Glimmer-30B-GGUF`).
> **Network is your responsibility**: behind a firewall set your own mirror, e.g.
> `HF_ENDPOINT=https://hf-mirror.com bash 00_prepare.sh` (this package ships no proxy/mirror).
> `<repo>/models` is gitignored — it never becomes part of the project.

## Method 1 · image (Docker)

The image is built **from the true base** `flagos/flagtree-amd-tle:rocm7.2.4`
(see [`Dockerfile`](Dockerfile)), pinned to the
official Hugging Face. It copies a prebuilt gfx1100 `llama-server` from `bin/` —
stage it there first (see [`bin/README.md`](bin/README.md)); it is a custom
muse-glimmer + DFlash build, not upstream `ggml-org/llama.cpp`, so it is not
compiled here.

```bash
cp /path/to/deploy/bin/llama-server bin/        # stage the gfx1100 binary (see bin/README.md)
docker build -t muse-glimmer-llamacpp:repro .   # FROM flagos/flagtree-amd-tle:rocm7.2.4

nohup bash run_all.sh > run_all.out 2>&1 &      # isolated container -> 12 cells (~2-3 h)
#   IMAGE defaults to muse-glimmer-llamacpp:repro; override to reuse another image.
```

## Method 2 · bare metal (python/shell)

Host needs: ROCm runtime + a gfx1100 `llama-server` (llama.cpp `>= b10353`),
`python3` + `aiohttp`, `curl`.

```bash
LLAMA_BIN_HOST=/path/to/llama.cpp/build/bin/llama-server \
  nohup bash run_host.sh > run_host.out 2>&1 &   # runs the 12 cells on the host
```

> No `llama-server`? Build it from llama.cpp source for gfx1100, or copy it out of
> the Method-1 image: `docker cp <container>:/llamacpp_workspace/bin/llama-server .`

> **ROCm 7.14.0 note:** to reproduce on the project-recommended 7.14 line,
> install the official **`therock-dist-linux-gfx110X-all-7.14.0.tar.gz`** and
> build `llama-server` against that prefix. The repo-pinned `gfx1151` tarball
> **core-dumps multi-slot decode on W7900** (rocBLAS ships no gfx1100 kernels;
> `c=1` works, masking the trap) —
> [troubleshooting](../../docs/troubleshooting.md#rocblas-wrong-arch-tarball).

> **Interrupted run? Clean before resuming.** The driver is resumable
> (`RESUME=1`, default), but a leftover `llama-server` from a killed run holds
> port 8080 and VRAM and silently contaminates the next cell. Before and after
> each session verify `pgrep -x llama-server` is empty and `rocm-smi
> --showmeminfo vram` shows the idle baseline (~28 MB). Kill by exact name
> (`pkill -9 -x llama-server`), never by a `-f` pattern that can match the
> driver's own shell. Symptom catalog:
> [orphan-server-contaminates-bench](../../docs/troubleshooting.md#orphan-server-contaminates-bench).

## Files

| File | Role |
|---|---|
| `config.env` | shared config (image / host binary / paths / sizes); all env-overridable |
| `Dockerfile` | Method-1 image: `FROM flagos/flagtree-amd-tle:rocm7.2.4` + prebuilt gfx1100 `llama-server` (`bin/`) + `aiohttp`, official HF |
| `bin/README.md` | how to stage the prebuilt `llama-server` for the image build |
| `00_prepare.sh` | download 4 GGUFs from official HF + verify (shared) |
| `run_all.sh` | Method 1: start isolated container → run `_repro_driver.sh` |
| `run_host.sh` | Method 2: run `_repro_driver.sh` directly on the host |
| `_repro_driver.sh` | driver: 12 cells (resumable + retry-once) → provenance → render |
| `render_headline.py` / `render_matrix_safe.py` | headline / detail table renderers |
| `_uvshim/uv` | translates `uv run --no-sync python …` → `python3 …` (harness unchanged) |
| `99_teardown.sh` | remove the Method-1 container |

## Common knobs (env)

| Var | Default | Meaning |
|---|---|---|
| `IMAGE` | see `config.env` | Method-1 runtime image |
| `LLAMA_BIN_HOST` | empty | Method-2 host `llama-server` |
| `HIP_VISIBLE_DEVICES` | `0` | single-card default; pick a card on multi-GPU hosts |
| `HF_ENDPOINT` | empty (official) | set a mirror yourself on restricted networks |
| `MODELS_HOST` | `<repo>/models` | where weights live (symlink to a scratch disk if needed) |
| `RESUME` | `1` | resume (skip existing cells); `0` reruns everything |

## Notes

- **Single-card assumption**: `--device=/dev/kfd --device=/dev/dri` + `HIP_VISIBLE_DEVICES=0`,
  with host `video`/`render` GIDs auto-detected.
- **`c=32` needs `-c 262144`** (> the 131072 training context): harmless here — each
  request is < 1k tokens; measured VRAM ≤ 24.9 GiB of 48.
- **Resumable / idempotent**: interrupt and rerun — it continues from where it stopped.
- **prompt-cache**: server default (prefix-KV reuse); affects only prefill/TTFT, not
  decode throughput, and is identical across cells.
