#!/usr/bin/env python3
"""SGLang v0.5.22 SM12x FullTopK decode routing fix.

Apply only after the previously supplied PR #42371 / #42368 backports.
Fails closed on unexpected source layouts; checks existing paged Triton decode.
"""
from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

spec = importlib.util.find_spec("sglang")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Could not locate the SGLang Python tree")
root = Path(next(iter(spec.submodule_search_locations))).resolve()
indexer = root / "srt/layers/attention/dsv4/v41_indexer/__init__.py"
scoring = root / "srt/layers/attention/dsv4/v41_indexer/scoring.py"

if not indexer.is_file() or not scoring.is_file():
    raise RuntimeError(f"Expected v0.5.22 indexer files under {root}")

scoring_source = scoring.read_text(encoding="utf-8")
if "paged = get_platform().is_sm90 or get_platform().is_sm120" not in scoring_source:
    raise RuntimeError("SM120 Triton paged decode patch missing from scoring.py; apply previous patch first")

old = "        use_deep_gemm_decode=is_sm100_or_newer(),"
new = (
    "        # SM12x does not have the SM100 DeepGEMM paged-MQA decode path.\n"
    "        # Use FullTopKIndexer._torch_decode -> Triton paged scorer instead.\n"
    "        use_deep_gemm_decode=(\n"
    "            is_sm100_or_newer() and torch.cuda.get_device_capability()[0] != 12\n"
    "        ),"
)
source = indexer.read_text(encoding="utf-8")
if new in source:
    print(f"ALREADY PATCHED: {indexer}")
elif source.count(old) == 1:
    updated = source.replace(old, new, 1)
    compile(updated, str(indexer), "exec")
    indexer.write_text(updated, encoding="utf-8")
    print(f"PATCHED: {indexer}")
else:
    raise RuntimeError(
        f"FullTopK routing pattern mismatch: expected one v0.5.22 assignment, found {source.count(old)}; "
        "no changes made"
    )
py_compile.compile(str(indexer), doraise=True)
print("PASS: SM12x FullTopK decode routes away from DeepGEMM (TP4/EP4 unchanged)")
