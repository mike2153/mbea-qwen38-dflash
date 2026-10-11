FROM stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Build without GPU access. Every upstream revision is immutable.
RUN apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4 /patches && \
    git -C /patches checkout 7d9a15a51d83733cc079fcf32ef4e3ec6b04ad7a && \
    git clone https://codeberg.org/StillDeadcode/libr4d.git /r4d && \
    git -C /r4d checkout b9e42ab7202f53a3bc13d415f5d41481f9ca311b
COPY patches/ /release/patches/
COPY kernels/ /release/kernels/
COPY scripts/entry.sh scripts/available_memory.py /release/scripts/
# The no-fill patch, the GDN verify kernel and the drafter attention kernel are built in below. These
# switches must stay on with this image: the vLLM source is patched to match at build time.
ENV RADIANCE_GDN_NOFILL=1 RADIANCE_GDN_MODE_E=1 RADIANCE_DRAFTER_ATTN_A=1
RUN git -C /patches apply /release/patches/radiance-wsl-r9700.patch && \
    git -C /patches apply /release/patches/radiance-kernels.patch && \
    cp /release/patches/patch_dflash2_temperature.py /patches/ && \
    git -C /r4d apply /patches/r4d_radiance_extras_rx9.patch && \
    cd /r4d && ./build.sh && \
    ATLAS_QUIESCE=1 QWEN38_BUILD_ONLY=1 bash /release/scripts/entry.sh

LABEL org.opencontainers.image.title="mbea-qwen38-dflash" \
      org.opencontainers.image.source="https://github.com/mike2153/mbea-qwen38-dflash" \
      io.mbea.qwen38.release="2026.10.1"
WORKDIR /
ENTRYPOINT ["python3", "/release/scripts/available_memory.py"]
