"""Run real patched functions in isolation, without importing CUDA-only SGLang.

The verification script supplies the fully patched source tree. Kernel launch
stubs test buffer ownership only; these are CPU regressions, not GPU validation.
"""

import ast
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None

ROOT = os.environ.get("SGLANG_BACKPORT_TEST_ROOT")
BACKEND = "srt/layers/attention/deepseek_v4_backend.py"
INDEXER = "srt/layers/attention/dsv4/v41_indexer/__init__.py"
MODEL = "srt/models/deepseek_v4.py"
POOL = "srt/model_executor/pool_configurator.py"
SPARSE = "srt/layers/attention/dsv4/dsv41_sparse.py"
MLA = "kernels/ops/attention/flash_mla_sm120.py"


def load_function(relative, name, owner=None, **namespace):
    tree = ast.parse((Path(ROOT) / relative).read_text())
    body = tree.body
    if owner:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    # A classmethod named torch must not replace the torch module in globals.
    node.name = "_isolated_target"
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, **namespace}
    exec(compile(module, relative, "exec"), namespace)
    return namespace["_isolated_target"]


def stub_module(name, **members):
    module = types.ModuleType(name)
    module.__dict__.update(members)
    return {name: module}


def mode(**values):
    return NS(
        **{
            name: (lambda value=values.get(name, False): value)
            for name in ("is_idle", "is_decode", "is_extend", "is_target_verify")
        }
    )


@unittest.skipUnless(
    ROOT and torch is not None, "run scripts/verify_backports.py with torch installed"
)
class TensorRegressions(unittest.TestCase):
    def assertTensorEqual(self, actual, expected):
        self.assertTrue(torch.equal(actual, expected), f"{actual} != {expected}")

    def test_candidate_indexer_accepts_sm120_without_sparse_deepgemm(self):
        package = "sglang.srt.layers.attention.dsv4.v41_indexer"
        modules = {}
        modules.update(
            stub_module(
                "sglang.srt.runtime_context",
                get_platform=lambda: NS(is_sm120=True),
                get_parallel=lambda: NS(attn_cp_size=1),
            )
        )
        modules.update(
            stub_module(
                package + ".dense_blocks", DenseBlocksBackend=lambda **kw: NS(**kw)
            )
        )
        modules.update(
            stub_module(
                "sglang.srt.layers.deep_gemm_wrapper.configurer",
                DEEPGEMM_PAGED_SPARSE_MQA_LOGITS=False,
            )
        )
        fn = load_function(
            INDEXER,
            "make_candidate_indexer",
            __package__=package,
            is_sm100_or_newer=lambda: True,
            _use_deep_gemm_prefill=lambda: True,
        )
        with patch.dict(sys.modules, modules):
            prefill, decode = fn(
                token_to_kv_pool=None,
                req_to_token=None,
                page_size=256,
                candidate_topk_blocks=2048,
                candidate_block_size=8,
            )
            self.assertIs(prefill, decode)
            modules["sglang.srt.runtime_context"].get_platform = lambda: NS(
                is_sm120=False
            )
            with self.assertRaises(RuntimeError):
                fn(
                    token_to_kv_pool=None,
                    req_to_token=None,
                    page_size=256,
                    candidate_topk_blocks=2048,
                    candidate_block_size=8,
                )

    def test_flashinfer_membership_only_dispatch(self):
        class Envelope:
            def __contains__(self, pair):
                heads, topk = pair
                return heads in (12, 16, 64) and topk >= 1

        probe = load_function(MLA, "_flashinfer_dsv4_decode_capabilities")
        supports = load_function(
            MLA,
            "flashinfer_dsv4_decode_supports_num_heads",
            _flashinfer_dsv4_decode_capabilities=probe,
            _DSV4_DECODE_PROBE_TOPK=512,
        )
        for dispatch in (Envelope(), frozenset({(16, 512), (64, 512)})):
            with patch.dict(
                sys.modules,
                stub_module(
                    "flashinfer.mla._sparse_mla_sm120",
                    _DECODE_DSV4_DISPATCH=dispatch,
                    _DECODE_MAX_TOKENS=64,
                ),
            ):
                self.assertTrue(supports(16, 64))
                self.assertFalse(supports(16, 65))
                self.assertFalse(supports(256, 1))

    def test_full_topk_keeps_triton_on_sm12x(self):
        for major, expected in ((9, False), (10, True), (12, False)):
            fn = load_function(
                INDEXER,
                "make_full_topk_indexer",
                FullTopKIndexer=lambda **kw: NS(**kw),
                _use_deep_gemm_prefill=lambda: True,
                is_sm100_or_newer=lambda: major >= 10,
            )
            with (
                self.subTest(major=major),
                patch.object(
                    torch.cuda, "get_device_capability", return_value=(major, 0)
                ),
            ):
                self.assertEqual(
                    fn(token_to_kv_pool=None, req_to_token=None).use_deep_gemm_decode,
                    expected,
                )

    def test_low_ratio_decode_requests_correct_metadata(self):
        for sm120 in (True, False):
            fn = load_function(
                BACKEND,
                "init_forward_metadata_indexer",
                "DeepseekV4AttnBackend",
                PagedIndexerMetadata=lambda **kw: NS(**kw),
                _is_xpu=False,
                get_platform=lambda: NS(is_sm120=sm120),
                expand_index_page_table=lambda table, **_: table,
            )
            backend = NS(
                page_size=256,
                enable_deepseek_v4_fp4_indexer=False,
                token_to_kv_pool=NS(get_index_k_page_size=lambda _: 64),
                dsa_topk_backend=NS(should_use_topk_v2=lambda: True),
            )
            core = NS(
                page_table=torch.zeros(1, 2),
                c1_topk_lengths_clamp1=torch.ones(1),
                c2_topk_lengths_clamp1=torch.ones(1),
                c4_topk_lengths_raw=torch.ones(1),
            )
            for ratio in (1, 2, 4):
                result = fn(backend, core, compress_ratio=ratio)
                self.assertEqual(
                    result.force_deep_gemm_metadata, sm120 and ratio in (1, 2)
                )

    def test_prefill_graph_requests_sm120_schedule(self):
        fn = load_function(
            BACKEND,
            "_low_ratio_prefill_indexer_metadata",
            "DeepseekV4AttnBackend",
            PagedIndexerMetadata=lambda **kw: NS(**kw),
            get_platform=lambda: NS(is_sm120=True),
            expand_index_page_table=lambda table, **_: table,
            _prefill_graph_max_seq_len=lambda: None,
            _PREFILL_GRAPH_INDEXER_ROW_CHUNK=2048,
        )
        backend = NS(
            page_size=256, token_to_kv_pool=NS(get_index_k_page_size=lambda _: 64)
        )
        core = NS(page_table=torch.zeros(3, 2), seq_lens_casual=torch.tensor([1, 2, 3]))
        result = fn(backend, core, 2)
        self.assertTrue(result.force_deep_gemm_metadata)
        self.assertTrue(result.use_prefill_cuda_graph)
        self.assertTensorEqual(
            result.compressed_seq_lens, torch.tensor([0, 1, 1], dtype=torch.int32)
        )

    def test_in_place_scores_preserve_bf16_and_fp32_outputs(self):
        fn = load_function(SPARSE, "scores", "DeepseekV41Indexer")
        for dtype in (torch.bfloat16, torch.float32):
            for rows, keys in ((1, 17), (4, 128), (13, 257)):
                with self.subTest(dtype=dtype, rows=rows):
                    generator = torch.Generator().manual_seed(123)
                    q = torch.randn(rows, 4, 32, generator=generator, dtype=dtype)
                    k = torch.randn(keys, 32, generator=generator, dtype=dtype)
                    weights = torch.randn(rows, 4, generator=generator, dtype=dtype)
                    expected = (
                        (
                            torch.einsum("bhd,nd->bhn", q, k).relu()
                            * weights.unsqueeze(-1)
                        )
                        .sum(1)
                        .float()
                    )
                    originals = [t.clone() for t in (q, k, weights)]
                    self.assertTensorEqual(fn(None, q, k, weights), expected)
                    for actual, original in zip((q, k, weights), originals):
                        self.assertTensorEqual(actual, original)

    def test_padded_query_smaller_buckets_keep_storage(self):
        fn = load_function(MODEL, "_zero_padded_q", "MqaAttentionBase")
        metadata = NS(q_pad_buffer=None, q_pad_captured=False)
        backend, layer = NS(forward_metadata=metadata), NS(head_dim=4)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            large = fn(layer, torch.ones(16, 4), 64, backend)
            small = fn(layer, torch.ones(2, 4), 64, backend)
        self.assertEqual(large.data_ptr(), small.data_ptr())
        self.assertEqual(metadata.q_pad_buffer.shape[0], 16)
        self.assertEqual(torch.count_nonzero(small).item(), 0)

    def test_padded_query_cannot_grow_after_capture(self):
        fn = load_function(MODEL, "_zero_padded_q", "MqaAttentionBase")
        metadata = NS(q_pad_buffer=None, q_pad_captured=False)
        backend, layer = NS(forward_metadata=metadata), NS(head_dim=4)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            fn(layer, torch.ones(4, 4), 64, backend)
        self.assertTrue(metadata.q_pad_captured)
        with self.assertRaisesRegex(AssertionError, "grew after"):
            fn(layer, torch.ones(8, 4), 64, backend)

    def test_page_split_larger_pool_does_not_replace_captured_buffers(self):
        class LaunchStub:
            def __getitem__(self, grid):
                return lambda *args: None

        buffers = {}
        fn = load_function(
            MLA,
            "_split_kv_pages_to_64",
            _PBS_DST=64,
            _BYTES_PER_DST_PAGE_PADDED=512,
            _NOPE_ROPE_STRIDE=4,
            _SCALE_STRIDE=4,
            _page_split_kernel=LaunchStub(),
            _page_mark_kernel=LaunchStub(),
            triton=NS(cdiv=lambda a, b: (a + b - 1) // b),
        )
        runtime = stub_module(
            "sglang.srt.runtime_context", get_resources=lambda: NS(buffers=buffers)
        )
        with patch.dict(sys.modules, runtime):
            small = fn(
                torch.zeros(2, 2048, dtype=torch.uint8),
                256,
                torch.tensor([5], dtype=torch.int32),
            )
            captured = {key: value.data_ptr() for key, value in buffers.items()}
            fn(
                torch.zeros(5, 2048, dtype=torch.uint8),
                256,
                torch.tensor([5], dtype=torch.int32),
            )
            replay = fn(
                torch.zeros(2, 2048, dtype=torch.uint8),
                256,
                torch.tensor([8], dtype=torch.int32),
            )
        self.assertEqual(small.data_ptr(), replay.data_ptr())
        for key, pointer in captured.items():
            self.assertEqual(buffers[key].data_ptr(), pointer)
        self.assertEqual(len(buffers), 4)

    def test_cp_tail_restores_request_rows_and_zeros_non_tail(self):
        fn = load_function(MODEL, "_cp_tail_to_extend_rows")
        for trailing in ((3,), (2, 3)):
            rows = torch.arange(3 * torch.tensor(trailing).prod().item()).reshape(
                3, *trailing
            )
            result = fn(NS(token_indices=torch.tensor([1, 4])), rows, 6)
            self.assertTensorEqual(result[1], rows[0])
            self.assertTensorEqual(result[4], rows[1])
            self.assertEqual(result[[0, 2, 3, 5]].count_nonzero().item(), 0)

    def test_short_final_prefill_chunk_reads_full_swa_window(self):
        fn = load_function(
            "kernels/ops/attention/dsv4_attn_metadata_kernels.py",
            "late_layer_tail_layout",
        )
        for extends, totals in (([1], [16385]), ([2, 64], [10002, 16448]), ([4], [4])):
            indices, lengths, floor = fn(
                extend_lens_cpu=extends, seq_lens_cpu=totals, tail_len=128, device="cpu"
            )
            expected = []
            for length, total in zip(extends, totals):
                expected.extend([max(0, total - 128)] * min(length, 128))
            self.assertTensorEqual(floor, torch.tensor(expected, dtype=torch.int32))
            self.assertEqual(indices.numel(), sum(lengths))

    def test_cp_scale_slicing_preserves_complete_128x4_tiles(self):
        relative = "srt/layers/cp/cp_decode_attn_tp.py"
        slice_fn = load_function(relative, "_slice", "CpDecodeAttnTpContext")
        fn = load_function(
            relative, "_slice_sf_swizzled_scale", "CpDecodeAttnTpContext"
        )
        shape = (512, 512)
        scales = torch.arange(4 * 4 * 512).reshape(-1)
        for dim in (0, 1):
            for rank in (0, 1):
                ctx = NS(decode_tp_size=2, decode_tp_rank=rank)
                ctx._slice = lambda tensor, axis: slice_fn(ctx, tensor, axis)
                actual = fn(ctx, scales, dim, shape)
                expected = (
                    scales.reshape(4, 4, 512).narrow(dim, rank * 2, 2).reshape(-1)
                )
                self.assertTensorEqual(actual, expected)
        with self.assertRaises(AssertionError):
            fn(NS(decode_tp_size=3), scales, 0, shape)

    def test_compact_verify_handles_empty_rows_and_padding(self):
        fn = load_function(
            "srt/speculative/ragged_verify.py", "ragged_verify_token_rows"
        )
        layout = NS(
            qo_indptr_device=torch.tensor([0, 2, 2, 5], dtype=torch.int32),
            graph_num_tokens=7,
        )
        self.assertTensorEqual(fn(layout, 7), torch.tensor([0, 0, 2, 2, 2, 3, 3]))
        token_rows = load_function(
            SPARSE,
            "token_req_indices",
            ragged_verify_token_rows=fn,
            resolve_ragged_verify_layout=lambda _: layout,
        )
        batch = NS(
            req_pool_indices=torch.tensor([7, 8, 9]),
            forward_mode=mode(is_target_verify=True),
        )
        self.assertTensorEqual(token_rows(batch), torch.tensor([7, 7, 9, 9, 9, 0, 0]))

    def test_dense_verify_mapping_is_unchanged(self):
        fn = load_function(
            SPARSE, "token_req_indices", resolve_ragged_verify_layout=lambda _: None
        )
        batch = NS(
            req_pool_indices=torch.tensor([7, 9]),
            forward_mode=mode(is_target_verify=True),
            spec_info=NS(draft_token_num=3),
        )
        self.assertTensorEqual(
            fn(batch, num_tokens=6), torch.tensor([7, 7, 7, 9, 9, 9])
        )

    def test_idle_engram_returns_zero_hashes_without_history_writes(self):
        fn = load_function("srt/layers/engram.py", "forward", "EngramHasher")
        hasher = NS(
            history=torch.ones(4, 7),
            max_ngram_size=8,
            primes=torch.ones(2),
            offsets=torch.ones(2, 3),
        )
        history = hasher.history.clone()
        result = fn(hasher, torch.arange(5), NS(forward_mode=mode(is_idle=True)))
        self.assertEqual(tuple(result.shape), (5, 2, 3))
        self.assertEqual(result.count_nonzero().item(), 0)
        self.assertTensorEqual(hasher.history, history)

    def test_idle_swa_store_never_reads_stale_metadata(self):
        fn = load_function(BACKEND, "get_swa_out_cache_loc", "DeepseekV4AttnBackend")

        class Backend:
            @property
            def forward_metadata(self):
                raise AssertionError("idle touched stale attention metadata")

        batch = NS(
            forward_mode=mode(is_idle=True),
            out_cache_loc=torch.zeros(5, dtype=torch.int64),
        )
        backend = Backend()
        idle = torch.full((5,), 33, dtype=torch.int32)
        window = NS(activate_idle=Mock(return_value=idle))
        backend.token_to_kv_pool = NS(request_window=window)
        self.assertIs(fn(backend, batch), idle)
        window.activate_idle.assert_called_once_with(5)
        backend.token_to_kv_pool = NS(
            request_window=None, translate_loc_from_full_to_swa=lambda loc: loc + 4
        )
        self.assertTensorEqual(
            fn(backend, batch), torch.full((5,), 4, dtype=torch.int32)
        )

    def test_draft_ring_respects_valid_prefix_and_padding(self):
        relative = "kernels/ops/speculative/dspark/dspark_attn_metadata.py"
        ring_rows = load_function(relative, "swa_ring_rows")
        fn = load_function(
            relative,
            "torch",
            "BuildDsparkSwaRingIndices",
            swa_ring_rows=ring_rows,
            ceil_align=lambda n, p: (n + p - 1) // p * p,
        )
        valid_from = torch.zeros(8, dtype=torch.int64)
        valid_from[2] = 90
        indices, lengths = fn(
            None,
            seq_lens_casual=torch.arange(102, 107),
            req_pool_indices_repeated=torch.full((5,), 2),
            valid_from=valid_from,
            ring_size=134,
            block_size=5,
            swa_window=128,
            page_index_aligned_size=64,
        )
        expected = 2 * 134 + torch.arange(90, 106) % 134
        self.assertTensorEqual(indices[0, :16], expected.to(torch.int32))
        self.assertTrue(torch.all(indices[:, 16:] == -1))
        self.assertTensorEqual(lengths, torch.full((5,), 16, dtype=torch.int32))


@unittest.skipUnless(ROOT, "run scripts/verify_backports.py")
class PolicyRegressions(unittest.TestCase):
    def test_compact_graph_variants_share_the_same_staged_layout(self):
        relative = "srt/model_executor/runner/decode_cuda_graph_runner.py"
        envs = NS(SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE=NS(get=lambda: False))
        fn = load_function(
            relative,
            "_capture_ragged_verify_layout",
            "DecodeCudaGraphRunner",
            envs=envs,
        )
        layout = object()
        runner = NS(ragged_verify_mode=True, _captured_ragged_layouts={42: layout})
        self.assertIs(fn(runner, 42), layout)
        self.assertIs(fn(runner, 42), layout)

    def test_moe_tuning_covers_prefill_and_preserves_pd_decode(self):
        relative = "srt/layers/moe/moe_runner/flashinfer_cutlass.py"
        for server_mode, prefill, dp, expected in (
            ("null", 16384, 1, 8192),
            ("null", 2048, 2, 4096),
            ("decode", 16384, 4, 128),
        ):
            fn = load_function(
                relative,
                "_cutlass_moe_tune_max_num_tokens",
                get_disagg=lambda: NS(disaggregation_mode=server_mode),
                max_prefill_buffer_tokens=lambda: prefill,
                get_schedule=lambda: NS(max_prefill_tokens=16384),
                deployment_attn_dp_size=lambda: dp,
                _FLASHINFER_DEFAULT_TUNE_MAX_NUM_TOKENS=8192,
                next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            )
            self.assertEqual(fn(65), expected)

    def test_kv_coefficient_uses_layout_entry_size(self):
        fn = load_function(
            POOL, "_compressed_bytes_per_full_token", "DSV4PoolConfigurator"
        )
        planner = NS(
            kv_bytes=584,
            low_ratio_index_bytes=68,
            _get_paged_kv_bytes_per_token=lambda ratio: 288,
        )
        self.assertEqual(fn(planner, 1), 356)
        self.assertEqual(fn(planner, 2), 178)

    def test_swa_replay_retention_frontier_and_no_replay_control(self):
        fn = load_function(
            "srt/mem_cache/common.py",
            "free_swa_out_of_window_slots",
            ComponentType=NS(SWA=1),
        )

        def frontier(length, replay=128, sharing=True, previous=0):
            kv = NS(
                holds_kv=True,
                cache_protected_len=0,
                req_pool_idx=0,
                swa_dead_lo=lambda _: 0,
                get_evicted_seqlen=lambda _: previous,
                evicted=previous,
            )
            kv.set_evicted_seqlen = lambda _, value: setattr(kv, "evicted", value)
            table = Mock()
            table.__getitem__ = Mock(return_value=object())
            fn(
                NS(kv=kv),
                length - 1,
                sliding_window_size=128,
                page_size=256,
                req_to_token_pool=NS(req_to_token=table),
                token_to_kv_pool_allocator=NS(free_swa_segment=Mock()),
                reprefill_tail_tokens=replay,
                supports_prefix_sharing=sharing,
            )
            return kv.evicted

        self.assertEqual(frontier(32576, replay=0), 32256)
        self.assertEqual(frontier(32576), 32000)
        self.assertEqual(frontier(32448), 32000)
        self.assertEqual(frontier(32576, sharing=False), 32256)
        self.assertEqual(frontier(32576, previous=32256), 32256)
        for length in range(32000, 32512):
            self.assertLessEqual(frontier(length), (length - 128) // 256 * 256 - 128)

    def test_pool_padding_and_capacity_stay_inside_byte_budget(self):
        physical_pages = load_function(
            "srt/mem_cache/deepseek_v4_memory_pool.py",
            "_num_dsv4_physical_kv_pages",
            ceil_div=lambda n, p: (n + p - 1) // p,
        )
        padding = load_function(
            POOL,
            "_get_low_ratio_kv_padding_bytes",
            "DSV4PoolConfigurator",
            _is_npu=False,
            get_exec=lambda: NS(kernel=NS(dsv4_attn_backend="flashmla")),
        )
        calculate = load_function(
            POOL, "calculate_pool_sizes", "DSV4PoolConfigurator", logger=Mock()
        )
        planner = NS(
            _unified=False,
            _unified_fp8=False,
            _dspark_draft_on_bf16=False,
            page_size=256,
            low_ratio_index_bytes=68,
            bytes_per_full_token=1630,
            swa_cap_tokens=None,
            num_layers=lambda ratio: {1: 1, 2: 3}.get(ratio, 0),
            _get_paged_kv_bytes_per_token=lambda ratio: 584,
            _compute_dsv4_sizes=lambda n, page: NS(full_max_total_num_tokens=n),
            _to_config=lambda sizes: sizes,
        )
        for name in (
            "_resolve_max_running_requests_per_worker",
            "_get_c128_state_fixed_bytes",
            "_get_c2_state_fixed_bytes",
            "_fixed_swa_bytes",
            "_fixed_c4_state_bytes",
        ):
            setattr(planner, name, lambda _: 0)
        planner._get_swa_fixed_bytes = lambda: 0
        planner._get_low_ratio_kv_padding_bytes = lambda n: padding(planner, n)
        with patch.dict(
            sys.modules,
            stub_module(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool",
                DSV41_INDEX_PAGE_SIZE=64,
                _num_dsv4_physical_kv_pages=physical_pages,
            ),
        ):
            budget = 32 * 1024 * 1024
            n = calculate(planner, budget, 256).full_max_total_num_tokens
            self.assertEqual(n % 256, 0)
            self.assertLessEqual(n * 1630 + padding(planner, n), budget)
            self.assertGreater((n + 256) * 1630 + padding(planner, n + 256), budget)
