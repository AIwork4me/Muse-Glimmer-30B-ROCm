# Method 1 (image) — W7900 reproduction runtime image.
#
# Built like ../../deploy/Dockerfile, but from the TRUE base
# (flagos/flagtree-amd-tle:rocm7.2.4) and pinned to the OFFICIAL Hugging Face
# (the deploy image's internal download proxy is dropped). No server entrypoint:
# the container just stays alive; the benchmark harness starts llama-server itself.
#
# The gfx1100 llama.cpp binary is a CUSTOM build (muse-glimmer arch + DFlash) —
# not upstream ggml-org/llama.cpp — so it is COPYd in rather than compiled here.
# Stage it into ./bin first (see bin/README.md), then:
#
#   docker build -t muse-glimmer-llamacpp:repro .
#   bash run_all.sh            # IMAGE defaults to muse-glimmer-llamacpp:repro
FROM flagos/flagtree-amd-tle:rocm7.2.4

ENV PATH=/opt/rocm/bin:${PATH} \
    HF_ENDPOINT=https://huggingface.co \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODELS_DIR=/llamacpp_workspace/models

# HF downloader (00_prepare uses huggingface_hub) + the one benchmark-client dep (aiohttp).
RUN python3 -m pip install --no-input --no-cache-dir "huggingface_hub[hf_xet]" aiohttp

WORKDIR /llamacpp_workspace
# Prebuilt gfx1100 llama.cpp server (custom muse-glimmer + DFlash build); staged into ./bin.
COPY bin/llama-server /llamacpp_workspace/bin/llama-server
RUN chmod +x /llamacpp_workspace/bin/*

EXPOSE 8080
CMD ["sleep", "infinity"]
