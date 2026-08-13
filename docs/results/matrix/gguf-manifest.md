# GGUF benchmark artifacts

The authoritative machine-readable record is
[`configs/artifact-manifest.json`](../../../configs/artifact-manifest.json).
The historical ROCm 7.2.1 matrix used:

- repository: `meta-models/Muse-Glimmer-30B-GGUF`
- revision: `a0532f7263ee67f1e0a5f5c5fdcd50dd62fc9aa4`
- hash algorithm: SHA256

| File | Exact bytes | SHA256 |
|---|---:|---|
| `muse-glimmer-30B-kquant-17gb.gguf` | 16,756,681,056 | `7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8` |
| `muse-glimmer-30B-kquant-dynamic.gguf` | 19,653,957,984 | `513109c8319115f69eb09fb7b118c97c8167d15bc014fd7670d2e30489bf106c` |
| `dflash-kquant.gguf` | 1,631,205,312 | `27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc` |
| `mmproj-kquant.gguf` | 1,400,328,928 | `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6` |

Verify local files without downloading or modifying them:

```bash
python3 scripts/verify_artifacts.py gguf models
```

The download scripts default to the official Hugging Face endpoint. Set
`HF_ENDPOINT` only when an optional regional mirror is needed. The DFlash and
projector artifacts are Xet-backed; a mirror that does not proxy Xet may require
fetching those files from the official endpoint with `HF_HUB_DISABLE_XET=1`.
