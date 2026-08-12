# GGUF artifacts on disk (weights are gitignored — this is a manifest only)

| file | size | fetch method that worked |
|---|---|---|
| muse-glimmer-30B-kquant-17gb.gguf | 16.8 GiB | pre-existing (gguf-quickstart.sh) |
| muse-glimmer-30B-kquant-dynamic.gguf | 19.7 GiB | `hf_parallel_get.py` (24-conn range, hf-mirror) |
| dflash-kquant.gguf | 1.6 GiB | `hf download`, **huggingface.co direct + `HF_HUB_DISABLE_XET=1`** |
| mmproj-kquant.gguf | 1.4 GiB | `hf download`, **huggingface.co direct + `HF_HUB_DISABLE_XET=1`** |

> **Fetch note (dflash + mmproj):** these two are xet-backed. The hf-mirror does
> NOT proxy their CAS, so `hf download` via the mirror fails ("Distant resource
> does not seem to be on huggingface.co") and `hf_parallel_get.py` fails its
> Content-Range probe on the `us.aws.cdn.hf.co/xet-bridge` redirect. Fetch them
> direct from huggingface.co with `HF_HUB_DISABLE_XET=1` (forces classic LFS).
> The large text shards are NOT xet-backed, so the mirror + parallel range
> downloader works for them.
