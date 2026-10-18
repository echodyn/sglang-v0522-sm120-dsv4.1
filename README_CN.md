# SGLang v0.5.22 — 在 RTX PRO 6000 (SM120) 上部署 DeepSeek V4.1 Flash

[English](README.md) | [简体中文](README_CN.md)

本项目是一个基于官方 [`lmsysorg/sglang:v0.5.22`](https://hub.docker.com/r/lmsysorg/sglang)
镜像的极简 Docker 补丁包，使 DeepSeek V4.1 Flash 能够在 4× NVIDIA RTX PRO 6000
（Blackwell 架构，计算能力 SM120）上以 TP=4 / EP=4 部署运行。

补丁在构建镜像时对容器内的 Python 源码做三处修改 —— 两个上游 PR 回填加一个
FullTopK decode 路由修复。不重新编译任何内核，也不修改 pip 依赖，因此构建速度快，
镜像其余部分与官方基础镜像完全一致。

## 仓库内容

| 文件 | 用途 |
|---|---|
| `Dockerfile` | 基于 `lmsysorg/sglang:v0.5.22`，运行两个补丁脚本后将其删除 |
| `patch_sm120.py` | 将上游 SGLang PR [#42371](https://github.com/sgl-project/sglang/pull/42371) 与 [#42368](https://github.com/sgl-project/sglang/pull/42368) 回填到已安装的 sglang 源码树 |
| `patch_full_topk_sm120.py` | FullTopK 修复 v2：让 SM12x 的 FullTopK decode 绕开 DeepGEMM 路径 |
| `compose.yaml` | 4 卡部署（TP=4 / EP=4），端口 8000 |

## 补丁内容

### 1. PR #42371 回填 — SM120 indexer 后端（`patch_sm120.py`）

SM120 缺少 SM100 专有的 DeepGEMM sparse paged-MQA logits 支持。以下两处修改使
DSV4.1 indexer 能在 SM120 上工作：

- `srt/layers/attention/dsv4/v41_indexer/__init__.py` — 当
  `DEEPGEMM_PAGED_SPARSE_MQA_LOGITS` 不可用时，SM120 上返回 `dense_blocks`
  后端，而不是抛出 `RuntimeError`。
- `srt/layers/attention/dsv4/v41_indexer/scoring.py` —
  `paged = get_platform().is_sm90` 改为 `is_sm90 or is_sm120`，使 decode
  在 SM120 上使用 Triton paged index scorer。

### 2. PR #42368 回填 — FlashInfer 0.7 兼容（`patch_sm120.py`）

FlashInfer 0.7 把 `_DECODE_DSV4_DISPATCH` 从 head 数量的集合改成了仅支持成员
判断的 envelope 对象。`kernels/ops/attention/flash_mla_sm120.py` 相应更新：
能力探测直接返回 dispatch 对象，支持性判断改为 `(num_heads, 512) in dispatch`。

### 3. FullTopK decode 路由修复 — 本仓库修复的崩溃（`patch_full_topk_sm120.py`）

**根因。** `make_full_topk_indexer` 设置
`use_deep_gemm_decode=is_sm100_or_newer()`，该判断把所有 SM12x GPU 也包含在内，
而 SM100 专有的 paged-MQA metadata 在 SM120 上并不存在。FlashInfer
autotune / CUDA Graph 预热阶段调用 `fp8_fp4_paged_mqa_logits` 时
`schedule_meta=None`，在 TVM FFI 中报错崩溃。

**修复。** 判断条件改为
`is_sm100_or_newer() and torch.cuda.get_device_capability()[0] != 12`，SM12x
的 FullTopK decode 改走 `_torch_decode()` —— 再由补丁 1 对 scoring.py 的修改
路由到 Triton paged scorer。模型权重、FlashInfer 版本、MoE TP4/EP4 与
candidate-indexer 行为均不变。

### 补丁安全性

两个脚本均通过 `importlib` 定位已安装的 sglang 源码树，执行严格单匹配字符串替换，
并对每个改动的文件做 `py_compile` 校验：

- 基础镜像与 v0.5.22 不符时**构建快速失败**，而不是产出被 silently 改坏的镜像。
- 对已打补丁的源码树重复执行是幂等的（"already applied"）。
- 绝不修改编译内核和 pip 包元数据。

## 环境要求

- 4× NVIDIA RTX PRO 6000（SM120），通过 NVIDIA Container Toolkit 暴露给 Docker
- Docker Engine + docker compose v2
- DeepSeek V4.1 Flash 权重已提前下载到本地
  （已设置 `HF_HUB_OFFLINE=1`，运行时不会联网拉取任何内容）

## 快速开始

**1. 将 compose 文件指向本地权重。** 编辑 `compose.yaml`，替换
`HOST_MODEL_PATH` / `CONTAINER_MODEL_PATH` 两处占位符 —— `volumes:` 挂载项与
`--model-path` 参数必须保持一致：

```yaml
    volumes:
      - /data/models/deepseek-v4.1-flash:/models/dsv41-flash   # 原为 HOST_MODEL_PATH:CONTAINER_MODEL_PATH
```

```yaml
      - --model-path
      - "/models/dsv41-flash"                                   # 原为 CONTAINER_MODEL_PATH
```

**2. 构建并启动：**

```bash
docker compose up -d --build
```

镜像在本地构建为 `sglang:v0.5.22-sm120-dsv41`。不用 compose 时也可以直接构建：

```bash
docker build -t sglang:v0.5.22-sm120-dsv41 .
```

**3. 验证：**

```bash
# 等待启动完成（CUDA Graph 捕获需要一些时间），然后：
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dsv41-flash-1017",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

查看日志：`docker logs -f dsv41-sm120`。

## 部署参数说明

`compose.yaml` 中的关键启动参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `--tp` | `4` | 每个张量并行秩对应一张 GPU |
| `--ep-size` | `4` | **SM120 上必须为 4** — 不设置会导致启动报错 |
| `--attention-backend` | `dsv4` | DSV4 注意力路径 |
| `--moe-runner-backend` | `flashinfer_mxfp4` | MXFP4 量化 MoE（即 "Flash" 变体） |
| `--fp8-gemm-backend` | `cutlass` | FP8 GEMM 后端 |
| `--kv-cache-dtype` | `fp8_e4m3` | FP8 KV cache |
| `--speculative-algorithm` | `DSPARK` | 投机解码 |
| `--speculative-dspark-block-size` | `5` | DSPARK 块大小 |
| `--context-length` | `524288` | 512K 上下文 |
| `--chunked-prefill-size` | `16384` | prefill 分块 |
| `--mem-fraction-static` | `0.9` | 静态显存占比 |
| `--max-running-requests` | `48` | 并发上限 |

环境变量：

| 变量 | 值 | 用途 |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | 不从 hub 拉取；权重来自挂载卷 |
| `SGLANG_SM120_FLASHMLA_BACKEND` | `flashinfer` | 选择 SM120 上的 FlashMLA 后端 |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | `1` | 将 DSV41 E-gram 表卸载到主机内存 |

GPU 通过 `deploy.resources.reservations.devices.device_ids`（`0–3`）绑定，
可按机器实际情况调整。`restart: unless-stopped` 处于注释状态 —— 如需服务随机器
重启自动拉起，可取消注释。

## 故障排查与说明

本项目已在一台 8 卡 RTX PRO 6000 服务器上测试通过。

- **启动时出现 `FP8 block GEMM ... missing tuning configs` 警告**：仅为性能
  警告，与任何崩溃无关。
- **构建报错 "expected 1 match … the base image differs from SGLang
  v0.5.22"**：说明本地基础镜像不是精确的官方 v0.5.22。请拉取
  `lmsysorg/sglang:v0.5.22`（或将 `FROM` 行指向你本地该镜像的确切 image/digest）
  后重新构建。
- **容器早早退出** — 查看 `docker logs dsv41-sm120`；首次启动包含 autotune 与
  CUDA Graph 捕获，可能数分钟内看起来没有动静。