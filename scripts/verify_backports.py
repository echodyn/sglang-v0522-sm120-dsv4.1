#!/usr/bin/env python3
"""Check the complete bundle against pinned upstream sources, then run regressions.

Pass --source-root /path/to/python/sglang to use local pristine v0.5.22 files.
Otherwise download only the manifest's source files at the pinned commit into a
temporary directory. Torch CPU tests run when torch is installed; --require-torch
is useful in CI to disallow skipped tensor regressions. No GPU is required.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import patch_sm120 as bundle  # noqa: E402 -- allow execution outside the repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    if args.require_torch and importlib.util.find_spec("torch") is None:
        parser.error("torch is required for the tensor regressions")
    assets = REPO / "patches"
    manifest = bundle.read_manifest(assets)
    with tempfile.TemporaryDirectory(prefix="sglang-v0522-verify-") as directory:
        root = Path(directory) / "sglang"
        for name, expected in manifest["files"].items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.source_root:
                data = (args.source_root / name).read_bytes()
            else:
                base = manifest["baseline"]
                url = f"https://raw.githubusercontent.com/{base['repository']}/{base['commit']}/python/sglang/{name}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
            if bundle.sha256(data.decode("utf-8")) != expected["base_sha256"]:
                raise bundle.PatchError(
                    f"{name}: verification requires pristine pinned v0.5.22 source"
                )
            target.write_bytes(data)
        originals = {name: (root / name).read_bytes() for name in manifest["files"]}
        changed = bundle.apply_bundle(root, assets, check=True)
        assert all((root / n).read_bytes() == data for n, data in originals.items())
        assert bundle.apply_bundle(root, assets) == changed
        assert bundle.apply_bundle(root, assets) == []
        bundle.verify_bundle(root, assets)

        # Exercise both historical script stages without importing SGLang/CUDA.
        for stage in (0, 1):
            legacy_root = Path(directory) / f"legacy-{stage}"
            shutil.copytree(root, legacy_root)
            for name, expected in manifest["files"].items():
                source = originals[name].decode("utf-8")
                states = expected["legacy"]
                if states:
                    state = states[min(stage, len(states) - 1)]
                    reset = bundle.parse_patch(
                        (assets / state["reset_patch"]).read_text()
                    )
                    reverse = [
                        bundle.Hunk(h.after, h.before) for h in reversed(reset[name])
                    ]
                    source = bundle.apply_hunks(source, reverse, "legacy fixture")
                    assert bundle.sha256(source) == state["sha256"]
                (legacy_root / name).write_text(source, encoding="utf-8")
            bundle.apply_bundle(legacy_root, assets)
            bundle.verify_bundle(legacy_root, assets)

        env = dict(os.environ, SGLANG_BACKPORT_TEST_ROOT=str(root))
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=REPO,
            env=env,
        )
        if result.returncode:
            raise SystemExit(result.returncode)
        print(
            {
                "checked_files": len(manifest["files"]),
                "modified_files": len(changed),
                "patches": sum(bool(e["patch"]) for e in manifest["backports"]),
                "legacy_upgrade_stages": 2,
                "idempotent": True,
            }
        )


if __name__ == "__main__":
    main()
