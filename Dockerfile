# Override BASE_IMAGE with your already-downloaded exact SGLang v0.5.22 image/digest if needed.
FROM lmsysorg/sglang:v0.5.22

COPY patch_sm120.py \
     patch_full_topk_sm120.py \
     /tmp/

RUN python3 /tmp/patch_sm120.py \
    && python3 /tmp/patch_full_topk_sm120.py \
    && rm -f /tmp/patch_sm120.py \
             /tmp/patch_full_topk_sm120.py

