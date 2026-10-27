# DeepSeek V4.1 Flash / SM120 回填清单

核对日期：**2026-10-27**。基础版本固定为 SGLang `v0.5.22`，源码提交
`baa9befa381511cd4d56f83980951e5a45fffc6d`。PR 状态是该日期的快照。

本项目保留官方镜像的依赖版本。`patches/manifest.json` 记录源码 SHA-256、
上游 PR head/merge commit、执行顺序和旧补丁状态；`patches/*.patch` 是已经适配
该版本的 unified diff，Docker 构建不访问 GitHub，也不跟随浮动 PR 分支。

## 已纳入的修复

| 上游 PR | 上游状态 | 本项目行为 |
|---|---|---|
| [#42371](https://github.com/sgl-project/sglang/pull/42371) | Open | 保留原有 SM12x candidate indexer dense-blocks 回退和 Triton paged scorer。 |
| [#42368](https://github.com/sgl-project/sglang/pull/42368) | Open | 保留 FlashInfer 0.7 membership-only dispatch 兼容。 |
| [#41303](https://github.com/sgl-project/sglang/pull/41303) | Merged | 短 extend 和短尾 chunk 保留完整 SWA 窗口；session 借用本请求 KV 时保留原有策略。 |
| [#44673](https://github.com/sgl-project/sglang/pull/44673) | Merged | encoder replay 下，CUDA DSpark draft 使用按请求分配的 SWA ring，记录 valid_from，避免 draft SWA 淘汰影响 target 的 Main-KV 命中；同步读写、commit 和固定内存预算。 |
| [#42714](https://github.com/sgl-project/sglang/pull/42714) | Open | FlashMLA page-split 输出和 mask 按设备、大小分别缓存，避免扩容释放已被 CUDA graph 捕获的缓冲区。 |
| [#45261](https://github.com/sgl-project/sglang/pull/45261) | Open | SM120 的 c1/c2 indexer 强制构建 DeepGEMM metadata，避免错误的 split-KV 调度或空 schedule。 |
| [#45381](https://github.com/sgl-project/sglang/pull/45381) | Open | 为 v0.5.22 的 low-ratio prefill graph 补齐 DeepGEMM schedule。 |
| [#45745](https://github.com/sgl-project/sglang/pull/45745) | Open | Torch indexer 在 einsum 的新分配结果上执行原地 ReLU 和乘权重，降低 prefill 临时显存。 |
| [#45083](https://github.com/sgl-project/sglang/pull/45083) | Open | interleave CP：恢复 DSpark tail 的 extend 行布局，提供本地 req_rows，并正确切分 BF16 权重缓存和 128×4 swizzled scale。 |
| [#45301](https://github.com/sgl-project/sglang/pull/45301) | Merged | padded query buffer 只增长，按行切片；捕获后禁止增长，保留 CUDA/HIP metadata 的捕获标志。 |
| [#42718](https://github.com/sgl-project/sglang/pull/42718) | Open | DSpark compact verify 用 qo_indptr 映射 token 到请求；处理空行和 graph padding，并修复 Engram verify 哈希路径。 |
| [#40424](https://github.com/sgl-project/sglang/pull/40424) | Merged | DP attention 的 padding 行不进入 low-ratio compressor；idle Engram 返回零哈希且不写历史。 |
| [#45586](https://github.com/sgl-project/sglang/pull/45586) | Merged | RequestWindow 提供固定的 idle 写入布局。 |
| [#45699](https://github.com/sgl-project/sglang/pull/45699) | Merged | CUDA/HIP idle SWA store 在读取任何可能失效的 attention metadata 前返回安全位置。 |
| [#46008](https://github.com/sgl-project/sglang/pull/46008) | Open | SWA 淘汰保留 replay 裁剪后匹配边界之前的窗口，避免可复用前缀被迫重算。 |
| [#41275](https://github.com/sgl-project/sglang/pull/41275) | Merged | KV 容量按实际 layout 计费，包含保留页和 index 页取整，并将容量降到预算内。 |
| [#43363](https://github.com/sgl-project/sglang/pull/43363) | Merged | FlashInfer CUTLASS / MXFP4 MoE autotune 覆盖 prefill token 桶；PD decode 保持按调用计费。 |

另校验 [#42709](https://github.com/sgl-project/sglang/pull/42709) 的同一 token tier
共享 compact-verify layout：**v0.5.22 已包含该修复**，不重复改写。

合计：17 个上游 PR 的回填、1 项已在基础版本中的依赖检查，以及保留的本地
FullTopK Triton 回退；17 个执行补丁修改 28 个源码文件，校验 29 个文件。
其中 #42371/#42368 是原项目已有回填，本次新增其余 15 项。

## 与原项目补丁的协调

- **FullTopK decode 继续使用已部署的 Triton paged scorer。** #45261 同时修正
  metadata 的构造，但不撤销这个已有回退。DeepGEMM 的 dense FP4 paged MQA
  与 SM100 专用的 sparse paged MQA 是不同接口；此前“SM12x 没有任何
  DeepGEMM paged-MQA 支持”的描述过于宽泛。
- #45381 在本项目中仍然需要：v0.5.22 有旧 captured FullTopK prefill 路径。
  新 main 的 [#45783](https://github.com/sgl-project/sglang/pull/45783) 已删除该路径，
  本项目不引入该结构重构。
- #44673 的预算回填兼容旧 `spec_aux_config`：可用时读取 `dspark_num_stages`，
  否则使用本版本的 `dflash_draft_num_layers`，不要求新增配置字段。
- #45083/#45301 适配旧模型类的位置和 metadata 类型；#46008 保留 v0.5.22
  原有无 replay 的淘汰公式，额外约束 replay 边界。
- CP、encoder replay、compact verify 的修复在启用相应功能时生效。默认
  `compose.yaml` 继续使用 TP4/EP4、DSPARK、FP8 KV 和原有 512K 配置。
- #43363 会扩大首次 autotune 的范围，可能增加启动时间和临时/常驻调优缓冲区；
  具体代价需要在 SM120 服务器测量，上游 H200 的性能数字不作为本项目结果。

## 已审核但无需回填的项目

| PR | 原因 |
|---|---|
| [#44857](https://github.com/sgl-project/sglang/pull/44857)、[#45040](https://github.com/sgl-project/sglang/pull/45040)、[#45051](https://github.com/sgl-project/sglang/pull/45051) | 修复新版 folded encoder replay 的写入、合并 chunk 预算和最小 chunk 限制。固定 v0.5.22 通过独立 `run_encoder_swa_replay` 执行回放，没有 `FoldedExtend`、`encoder_swa_compress_rows` 或合并 replay 行；不引入整套新执行机制来修复其后续回归。 |
| [#42440](https://github.com/sgl-project/sglang/pull/42440) | 已关闭并由 FlashInfer #5075 取代；基础版本已固定 FlashInfer 0.7.1，包含零行 gather 修复。 |
| [#42445](https://github.com/sgl-project/sglang/pull/42445)、[#39492](https://github.com/sgl-project/sglang/pull/39492) | dispatch envelope 的正确性修复已由 #42368 覆盖；额外的 padded-head 延迟调优开关不属于新的正确性修复。 |
| [#43556](https://github.com/sgl-project/sglang/pull/43556)、[#44480](https://github.com/sgl-project/sglang/pull/44480) | 针对 Qwen 等混合 FP8 draft MoE / DeepGEMM 或不同 runner 的 EP id 映射；本项目 DeepSeek V4.1 Flash 的 MXFP4 MoE + DSpark 配置不走这些路径。 |

以下是新功能或性能后端，不作为本次修复包引入：

- [#45851](https://github.com/sgl-project/sglang/pull/45851)：SM12x V41/V41_FP4 KV
  layout，需要 FlashInfer 0.7.2 新接口；当前基础版本是 0.7.1。
- [#45417](https://github.com/sgl-project/sglang/pull/45417)：V4 NVFP4 target/draft KV，
  依赖额外 FlashInfer NVFP4/TOPK=192 改动，验证对象是 V4 Flash。
- [#45262](https://github.com/sgl-project/sglang/pull/45262)、[#45391](https://github.com/sgl-project/sglang/pull/45391)：b12x MXFP8 dense GEMM 后端，需要额外后端接入及依赖升级。
- [#45264](https://github.com/sgl-project/sglang/pull/45264)：mHC BLOCK_K 调优，属于性能优化。

## 验证与复现

```bash
# 下载清单中固定提交的源码到临时目录，应用全部补丁并运行 CPU 回归。
python3 scripts/verify_backports.py --require-torch

# 已有精确的上游源码时可完全离线执行。
python3 scripts/verify_backports.py \
  --source-root /path/to/sglang/python/sglang --require-torch

# 只检查目标安装能否应用，不写文件。
python3 patch_sm120.py --root /path/to/installed/sglang --check

# 镜像内复核全部已安装补丁。
docker compose exec sglang \
  python3 /opt/sglang-sm120-backports/patch_sm120.py --verify
```

补丁引擎仅依赖标准库。验证脚本在安装 PyTorch 时增加 CPU 张量回归；
`--require-torch` 禁止因缺少 PyTorch 而跳过这些测试。

本次本地通过 32 项 CPU 回归。检查包括：原始源码 Git blob/内容校对、所有变更文件的语法和最终哈希、
旧版两阶段补丁状态升级、幂等执行、失败前不写文件和写入错误回滚，以及真实
补丁函数的张量/策略回归。CUDA kernel launch 在缓冲区生命周期测试中使用 stub。
**这些检查不代表 SM120 CUDA graph、内核数值或完整模型推理通过。**
本地没有 v0.5.22 Docker 基础镜像，也没有 NVIDIA GPU；仍需在目标服务器
构建并执行启动、长上下文、前缀命中、flush 后复用和并发 DSpark 验证。
