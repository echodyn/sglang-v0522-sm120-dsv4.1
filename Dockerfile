# Override only with an exact v0.5.22 image/digest; source hashes are checked.
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.22
FROM ${BASE_IMAGE}

COPY patch_sm120.py \
     patch_full_topk_sm120.py \
     /opt/sglang-sm120-backports/
COPY patches/ /opt/sglang-sm120-backports/patches/

RUN python3 /opt/sglang-sm120-backports/patch_sm120.py \
    && python3 /opt/sglang-sm120-backports/patch_sm120.py --verify

