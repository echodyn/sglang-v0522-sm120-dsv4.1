# SGLang v0.5.22 — DeepSeek V4.1 Flash / SM120 修复包

[English](README.md) | [完整回填清单](BACKPORTS.md)

基于官方 `lmsysorg/sglang:v0.5.22`，为 RTX PRO 6000 等 SM120 GPU
整合 DeepSeek V4.1 Flash 的启动、CUDA graph、DSpark、SWA 缓存和显存预算修复。
默认部署为 4 卡 TP4/EP4，使用 FlashInfer MXFP4 MoE、FP8 KV 和 DSPARK。

当前整合 **17 个上游 PR**：保留原有 #42371/#42368，新增 15 项回填；
另保留原有 FullTopK Triton 回退并校验已在基础版本中的 #42709。
补丁修改 28 个 Python 源码文件，校验 29 个文件。
已合并和未合并 PR 的适用修改都固化在本仓库，构建时不下载 PR。

| 文件 | 用途 |
|---|---|
| `Dockerfile` | 应用完整补丁包，再校验最终源码哈希与语法 |
| `patch_sm120.py` | 补丁入口；支持原版、旧补丁状态升级、幂等执行和只读校验 |
| `patch_full_topk_sm120.py` | 兼容旧调用方式，现转入完整补丁入口 |
| `patches/` | 固定源码哈希、上游提交信息、按顺序执行的补丁和旧版升级补丁 |
| `BACKPORTS.md` | 每项修复、版本适配、未回填项目及原因 |
| `scripts/verify_backports.py`、`tests/` | 固定上游源码上的集成校验及 CPU 回归 |
| `compose.yaml` | 4 卡部署，服务端口 8000 |

## 构建和启动

需要 Docker、NVIDIA Container Toolkit、4 张可见 GPU，以及提前下载的
DeepSeek V4.1 Flash 权重。编辑 `compose.yaml`，同时替换：

```yaml
volumes:
  - /data/models/deepseek-v4.1-flash:/models/dsv41-flash
```

并将 `--model-path` 后的 `CONTAINER_MODEL_PATH` 改为
`/models/dsv41-flash`。然后执行：

```bash
docker compose up -d --build
docker logs -f dsv41-sm120
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-flash-1017","messages":[{"role":"user","content":"Hello!"}]}'
```

镜像名称为 `sglang:v0.5.22-sm120-dsv41`。也可以单独构建：

```bash
docker build -t sglang:v0.5.22-sm120-dsv41 .

# 使用本地已有的精确 v0.5.22 镜像或 digest：
docker build --build-arg BASE_IMAGE=your-exact-v0522-image \
  -t sglang:v0.5.22-sm120-dsv41 .
```

补丁先校验全部文件的 SHA-256、所有 hunk、最终哈希和 Python 语法，再写入。
不匹配的源码会终止构建。旧版两阶段补丁状态可直接升级；重复执行是 no-op。
`patch_sm120.py` 现在必须与 `patches/` 一起复制使用。

## 部署参数

默认配置保留 `--tp 4 --ep-size 4`、`--attention-backend dsv4`、
`--moe-runner-backend flashinfer_mxfp4`、`--fp8-gemm-backend cutlass`、
`--kv-cache-dtype fp8_e4m3`、DSPARK block size 5、512K 上下文、
prefill chunk 16384、静态显存比例 0.9 和最大并发 48。SM120 部署要显式设置 EP=4。

环境变量 `HF_HUB_OFFLINE=1` 使用本地权重，
`SGLANG_SM120_FLASHMLA_BACKEND=flashinfer` 选择注意力后端，
`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` 将 Engram 表放到主机内存。
CPU 内存、显存容量和 GPU 拓扑仍需满足模型与所选上下文的要求。

CP、encoder replay、compact verify 修复随对应功能启用而生效，默认 compose
没有额外打开这些功能。MoE prefill autotune 的覆盖范围扩大，首次启动可能更久。

## 验证

```bash
# 自动下载固定提交的必要源码；CPU 张量测试需要已安装 torch。
python3 scripts/verify_backports.py --require-torch

# 使用本地精确的上游源码，离线验证：
python3 scripts/verify_backports.py \
  --source-root /path/to/sglang/python/sglang --require-torch

# 在已构建的容器内检查最终补丁：
docker compose exec sglang \
  python3 /opt/sglang-sm120-backports/patch_sm120.py --verify
```

本次本地通过 32 项 CPU 回归，以及固定源码应用、两种旧版状态升级、
重复执行、29 个文件的最终哈希和语法检查。没有在本机完成 Docker 基础镜像构建
或 SM120 实机推理；原项目旧版补丁的历史运行记录不代表本次整包已通过实机验证。
具体范围和仍需进行的服务器验证见 [BACKPORTS.md](BACKPORTS.md)。
