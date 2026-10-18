# SGLang v0.5.22 — DeepSeek V4.1 Flash on RTX PRO 6000 (SM120)

[English](README.md) | [简体中文](README_CN.md)

A minimal Docker overlay that patches the official
[`lmsysorg/sglang:v0.5.22`](https://hub.docker.com/r/lmsysorg/sglang) image so that
DeepSeek V4.1 Flash can be served on 4× NVIDIA RTX PRO 6000 (Blackwell, compute
capability SM120) with TP=4 / EP=4.

The overlay rewrites three Python source files *inside* the container image at
build time — two upstream PR backports plus one FullTopK decode routing fix. It
never recompiles kernels and never touches pip package metadata, so the build
stays fast and the base image stays otherwise identical.

## Repository layout

| File | Purpose |
|---|---|
| `Dockerfile` | Builds on `lmsysorg/sglang:v0.5.22`, runs both patch scripts, then removes them |
| `patch_sm120.py` | Backports upstream SGLang PRs [#42371](https://github.com/sgl-project/sglang/pull/42371) and [#42368](https://github.com/sgl-project/sglang/pull/42368) into the installed sglang tree |
| `patch_full_topk_sm120.py` | FullTopK fix v2: routes SM12x FullTopK decode away from the DeepGEMM path |
| `compose.yaml` | 4-GPU deployment (TP=4 / EP=4), port 8000 |

## What gets patched

### 1. PR #42371 backport — SM120 indexer backends (`patch_sm120.py`)

SM120 lacks the SM100-specific DeepGEMM sparse paged-MQA logits support. Two
changes make the DSV4.1 indexer work on SM120:

- `srt/layers/attention/dsv4/v41_indexer/__init__.py` — when
  `DEEPGEMM_PAGED_SPARSE_MQA_LOGITS` is unavailable, return the `dense_blocks`
  backend on SM120 instead of raising `RuntimeError`.
- `srt/layers/attention/dsv4/v41_indexer/scoring.py` —
  `paged = get_platform().is_sm90` becomes `is_sm90 or is_sm120`, so decode
  uses the Triton paged index scorer on SM120.

### 2. PR #42368 backport — FlashInfer 0.7 compatibility (`patch_sm120.py`)

FlashInfer 0.7 changed `_DECODE_DSV4_DISPATCH` from a set of head counts into a
membership-only envelope. `kernels/ops/attention/flash_mla_sm120.py` is updated
so the capability probe returns the dispatch object directly and the support
test becomes `(num_heads, 512) in dispatch`.

### 3. FullTopK decode routing fix — the crash this repo fixes (`patch_full_topk_sm120.py`)

**Root cause.** `make_full_topk_indexer` sets
`use_deep_gemm_decode=is_sm100_or_newer()`, which includes all SM12x GPUs even
though the SM100-specific paged-MQA metadata is absent on SM120. During
FlashInfer autotune / CUDA Graph warmup, `fp8_fp4_paged_mqa_logits` is called
with `schedule_meta=None` and errors out in TVM FFI.

**Fix.** The condition becomes
`is_sm100_or_newer() and torch.cuda.get_device_capability()[0] != 12`, so SM12x
FullTopK decode uses `_torch_decode()` — which the scoring.py change from
patch 1 then routes to the Triton paged scorer. Model weights, FlashInfer
version, MoE TP4/EP4 and candidate-indexer behavior are unchanged.

### Patch safety properties

Both scripts locate the installed sglang tree via `importlib`, perform strict
single-match string replacement, and `py_compile` every file they touch:

- A base image that differs from v0.5.22 **fails the build fast** instead of
  producing a silently broken image.
- Re-running a script on an already-patched tree is a no-op ("already applied").
- Compiled kernels and pip package metadata are never edited.

## Requirements

- 4× NVIDIA RTX PRO 6000 (SM120), visible to Docker via the NVIDIA Container Toolkit
- Docker Engine + docker compose v2
- DeepSeek V4.1 Flash weights already downloaded to the host
  (`HF_HUB_OFFLINE=1` is set, so nothing is fetched at runtime)

## Quick start

**1. Point the compose file at your local weights.** Edit `compose.yaml` and
replace both `HOST_MODEL_PATH` / `CONTAINER_MODEL_PATH` placeholders — the
`volumes:` entry and the `--model-path` argument must stay consistent:

```yaml
    volumes:
      - /data/models/deepseek-v4.1-flash:/models/dsv41-flash   # was HOST_MODEL_PATH:CONTAINER_MODEL_PATH
```

```yaml
      - --model-path
      - "/models/dsv41-flash"                                   # was CONTAINER_MODEL_PATH
```

**2. Build and start:**

```bash
docker compose up -d --build
```

The image is built locally as `sglang:v0.5.22-sm120-dsv41`. To build it without
compose:

```bash
docker build -t sglang:v0.5.22-sm120-dsv41 .
```

**3. Verify:**

```bash
# Wait for startup (CUDA Graph capture takes a while), then:
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dsv41-flash-1017",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Logs: `docker logs -f dsv41-sm120`.

## Deployment configuration

Key launch parameters in `compose.yaml`:

| Parameter | Value | Notes |
|---|---|---|
| `--tp` | `4` | One GPU per tensor-parallel rank |
| `--ep-size` | `4` | **Must be set on SM120** — or the launch will fail |
| `--attention-backend` | `dsv4` | DSV4 attention path |
| `--moe-runner-backend` | `flashinfer_mxfp4` | MXFP4-quantized MoE (the "Flash" variant) |
| `--fp8-gemm-backend` | `cutlass` | FP8 GEMM backend |
| `--kv-cache-dtype` | `fp8_e4m3` | FP8 KV cache |
| `--speculative-algorithm` | `DSPARK` | Speculative decoding |
| `--speculative-dspark-block-size` | `5` | DSPARK block size |
| `--context-length` | `524288` | 512K context |
| `--chunked-prefill-size` | `16384` | Prefill chunking |
| `--mem-fraction-static` | `0.9` | Static memory fraction |
| `--max-running-requests` | `48` | Concurrency cap |

Environment variables:

| Variable | Value | Purpose |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | Never fetch from the hub; weights come from the mounted volume |
| `SGLANG_SM120_FLASHMLA_BACKEND` | `flashinfer` | Select the FlashMLA backend on SM120 |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | `1` | Offload the DSV41 E-gram table to host memory |

GPUs are pinned via `deploy.resources.reservations.devices.device_ids`
(`0–3`); adjust for your machine. `restart: unless-stopped` is commented out —
enable it if you want the service to survive reboots.

## Troubleshooting & notes

This project has been tested successfully on a server with 8× RTX PRO 6000
GPUs.

- **`FP8 block GEMM ... missing tuning configs` warnings at startup** are
  performance warnings only and unrelated to any crash.
- **Build error like "expected 1 match … the base image differs from SGLang
  v0.5.22"** means your local base image is not the exact official v0.5.22.
  Pull `lmsysorg/sglang:v0.5.22` (or point the `FROM` line at your local copy
  of that exact image/digest) and rebuild.
- **Container exits early** — check `docker logs dsv41-sm120`; first startup
  includes autotune and CUDA Graph capture and can look idle for several
  minutes.
