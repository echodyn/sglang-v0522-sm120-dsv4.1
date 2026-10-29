# SGLang v0.5.22 — DeepSeek V4.1 Flash on SM120

[简体中文](README_CN.md) | [Backport inventory and validation](BACKPORTS.md)

A pinned Docker overlay for serving DeepSeek V4.1 Flash on four RTX PRO 6000
(SM120) GPUs with TP4/EP4, FlashInfer MXFP4 MoE, FP8 KV and DSPARK.

> **Also available:** I maintain a vLLM patch bundle for the same setup —
> [vllm-v0320-sm120-dsv4.1](https://github.com/echodyn/vllm-v0320-sm120-dsv4.1) —
> which measured faster and more stable in real use.

The bundle backports **17 upstream PRs**, including the original #42371/#42368
and 15 additional fixes, preserves the original SM12x FullTopK Triton fallback,
and verifies the already-shipped #42709 dependency. It changes 28 Python source
files and checks 29 files against the exact v0.5.22 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`.

Merged and open PR changes are vendored as ordered unified diffs. Docker builds
do not download patches or follow moving branches. Dependency versions remain
those of the official base image. The bundle covers indexer metadata, graph
buffer lifetimes, DSpark compact verify, SWA windows and prefix retention, CP,
idle DP ranks, KV budgeting and MoE prefill autotuning. See
[BACKPORTS.md](BACKPORTS.md) for every source, adaptation and excluded feature.

## Build and serve

Requirements: Docker with NVIDIA Container Toolkit, four visible GPUs, enough
GPU/host memory, and locally downloaded DeepSeek V4.1 Flash weights.

Edit both placeholders in `compose.yaml`: set the volume to
`/data/models/deepseek-v4.1-flash:/models/dsv41-flash` and the
`--model-path` argument to `/models/dsv41-flash`.

```bash
docker compose up -d --build
docker logs -f dsv41-sm120
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-flash-1017","messages":[{"role":"user","content":"Hello!"}]}'
```

The output image is `sglang:v0.5.22-sm120-dsv41`. Standalone builds:

```bash
docker build -t sglang:v0.5.22-sm120-dsv41 .
docker build --build-arg BASE_IMAGE=your-exact-v0522-image \
  -t sglang:v0.5.22-sm120-dsv41 .
```

The default configuration retains TP4/EP4 (set EP=4 explicitly on SM120),
`--attention-backend dsv4`, `--moe-runner-backend flashinfer_mxfp4`,
`--fp8-gemm-backend cutlass`, FP8 E4M3 KV, DSPARK block size 5,
512K context, 16384-token prefill chunks, static memory fraction 0.9 and 48
running requests. Host Engram tables and offline model loading stay enabled.
CP, encoder replay and compact verify fixes take effect when those optional
features are enabled; the default compose does not enable them.

Prefill MoE autotuning now covers larger buckets, which may increase startup
time and tuning-buffer memory. No local SM120 performance result is claimed.

## Patching and verification

`patch_sm120.py` is the complete entry point and must be copied together with
`patches/`. `patch_full_topk_sm120.py` is retained as a compatibility
entry point to the same bundle. Before writing, the patcher checks every source
hash, hunk, result hash and Python syntax. Known original overlay states can be
upgraded, repeat runs are no-ops, and unknown source trees fail.

```bash
# Fetch only pinned upstream files into a temporary directory; torch CPU tests:
python3 scripts/verify_backports.py --require-torch

# Offline verification using pristine upstream sources:
python3 scripts/verify_backports.py \
  --source-root /path/to/sglang/python/sglang --require-torch

# Verify the completed bundle inside the container:
docker compose exec sglang \
  python3 /opt/sglang-sm120-backports/patch_sm120.py --verify

# Inspect a target installation without writing:
python3 patch_sm120.py --root /path/to/installed/sglang --check
```

Local validation: 32 CPU regressions, pristine-source application, both legacy
overlay stages, idempotency, and final hashes/syntax for 29 files. Kernel-launch
stubs cover ownership only. This machine has no v0.5.22 Docker base image or
NVIDIA GPU: a real image build and SM120 serving/accuracy checks are still
required. Historical validation of the original smaller overlay does not
validate this expanded bundle.
