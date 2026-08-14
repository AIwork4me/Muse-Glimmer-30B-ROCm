# Method 1 (image) — reproduction runtime image.
#
# BASE = a prebuilt Muse-Glimmer-30B llama.cpp image that ships gfx1100
# `llama-server` binaries at /llamacpp_workspace/bin (built from llama.cpp
# source >= b10353, which added the muse-glimmer arch + DFlash). This thin
# layer bakes the one benchmark-client dependency (aiohttp) and pins model
# downloads to the OFFICIAL Hugging Face (drops any internal proxy the base
# image may carry). Network access to huggingface.co is the user's responsibility.
#
#   docker build -t muse-glimmer-llamacpp:repro .
#   IMAGE=muse-glimmer-llamacpp:repro bash run_all.sh
#
# You may also skip this build and use the base image directly (the driver will
# pip-install aiohttp on first run):
#   IMAGE=<BASE_IMAGE below> bash run_all.sh
ARG BASE_IMAGE=crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:muse-glimmer-llamacpp_2026_08_14_modelscope_concurrent
FROM ${BASE_IMAGE}

# Official Hugging Face only — drop any internal proxy baked into the base image.
ENV HF_ENDPOINT=https://huggingface.co \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Bake the single benchmark-client dependency so the container is self-sufficient.
RUN python3 -m pip install --no-cache-dir aiohttp || pip install --no-cache-dir aiohttp

CMD ["sleep", "infinity"]
