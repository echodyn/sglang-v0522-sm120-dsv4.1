#!/usr/bin/env python3
"""Backport SGLang PR #42371 and #42368 to an installed v0.5.22 Python tree.

The script never edits compiled kernels or pip package metadata. It is intentionally
strict about finding the exact code patterns so a different image/version fails
fast instead of producing a silently broken patched image.
"""
from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

spec = importlib.util.find_spec('sglang')
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError('Cannot find installed sglang; use the base image Python interpreter')

root = Path(list(spec.submodule_search_locations)[0]).resolve()
print('SGLang Python root:', root)


def rewrite(relative_path: str, changes: list[tuple[str, str]]) -> None:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f'SGLang file missing: {path}')
    original = path.read_text(encoding='utf-8')
    current = original
    for index, (before, after) in enumerate(changes, 1):
        if after in current:
            print(f'  {relative_path}: hunk {index} already applied')
            continue
        count = current.count(before)
        if count != 1:
            raise RuntimeError(
                f'{relative_path}: expected 1 match for hunk {index}, found {count}; '
                'the base image differs from SGLang v0.5.22'
            )
        current = current.replace(before, after, 1)
    if current != original:
        path.write_text(current, encoding='utf-8')
        print(f'PATCHED: {path}')
    py_compile.compile(str(path), doraise=True)


# PR #42371, hunk 1: SM120 has no DeepGEMM sparse paged logits.
rewrite('srt/layers/attention/dsv4/v41_indexer/__init__.py', [
    (
        '    from sglang.srt.runtime_context import get_parallel\n\n'
        '    from .dense_blocks import DenseBlocksBackend',
        '    from sglang.srt.runtime_context import get_parallel, get_platform\n\n'
        '    from .dense_blocks import DenseBlocksBackend',
    ),
    (
        '    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:\n'
        '        raise RuntimeError(\n',
        '    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:\n'
        '        # Backport PR #42371: use dense-blocks scheme on SM120.\n'
        '        if get_platform().is_sm120:\n'
        '            return dense_blocks, dense_blocks\n'
        '        raise RuntimeError(\n',
    ),
])

# PR #42371, hunk 2: use Triton paged index scoring on SM120 decode.
rewrite('srt/layers/attention/dsv4/v41_indexer/scoring.py', [
    (
        '    paged = get_platform().is_sm90\n',
        '    paged = get_platform().is_sm90 or get_platform().is_sm120\n',
    ),
])

# PR #42368: FlashInfer 0.7 changed the dispatch to a membership-only object.
rewrite('kernels/ops/attention/flash_mla_sm120.py', [
    (
        'from typing import FrozenSet, Optional, Tuple\n',
        'from typing import Container, Optional, Tuple\n',
    ),
    (
        '@lru_cache(maxsize=1)\n'
        'def _flashinfer_dsv4_decode_capabilities() -> Tuple[int, FrozenSet[int]]:',
        '_DSV4_DECODE_PROBE_TOPK = 512\n\n'
        '@lru_cache(maxsize=1)\n'
        'def _flashinfer_dsv4_decode_capabilities() -> Tuple[int, Container[Tuple[int, int]]]:',
    ),
    (
        '    return int(_DECODE_MAX_TOKENS), frozenset(\n'
        '        heads for heads, _ in _DECODE_DSV4_DISPATCH\n'
        '    )',
        '    # FlashInfer <=0.6 exposes a set; 0.7 exposes a membership-only envelope.\n'
        '    return int(_DECODE_MAX_TOKENS), _DECODE_DSV4_DISPATCH',
    ),
    (
        '    decode_max_tokens, supported_heads = _flashinfer_dsv4_decode_capabilities()\n'
        '    return num_tokens <= decode_max_tokens and num_heads in supported_heads',
        '    decode_max_tokens, dispatch = _flashinfer_dsv4_decode_capabilities()\n'
        '    return (\n'
        '        num_tokens <= decode_max_tokens\n'
        '        and (num_heads, _DSV4_DECODE_PROBE_TOPK) in dispatch\n'
        '    )',
    ),
])

print('SUCCESS: patches #42371 and #42368 applied; all 3 files syntax-checked')
