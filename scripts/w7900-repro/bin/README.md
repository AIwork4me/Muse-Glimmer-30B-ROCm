# `bin/` — stage the gfx1100 `llama-server` here (Method 1 image build)

The reproduction image ([`../Dockerfile`](../Dockerfile), `FROM
flagos/flagtree-amd-tle:rocm7.2.4`) copies a prebuilt `llama-server` from this
folder. It is a **custom llama.cpp build** (muse-glimmer arch + DFlash), not
upstream `ggml-org/llama.cpp`, so it is copied in rather than compiled.

Place a gfx1100 `llama-server` at `bin/llama-server`, for example:

```bash
# from a deploy build context that already carries the binary:
cp /path/to/deploy/bin/llama-server bin/
# — or — copy it out of an existing muse-glimmer-llamacpp image:
cid=$(docker create <muse-glimmer-llamacpp-image>)
docker cp "$cid":/llamacpp_workspace/bin/llama-server bin/
docker rm "$cid"
```

`bin/llama-server` is gitignored (a large, GPU-specific binary — not part of the
project). Method 2 (`run_host.sh`) does not need this; it uses your host
`LLAMA_BIN_HOST` directly.
