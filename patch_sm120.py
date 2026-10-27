#!/usr/bin/env python3
"""Apply the pinned SM120/V4.1 backport bundle to SGLang v0.5.22.

Only the Python standard library is needed. Validate every source hash, hunk,
result hash and Python syntax before writing any file. Known original overlay
states can be upgraded; a completed bundle is a no-op on repeat execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class PatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hunk:
    before: str
    after: str


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise PatchError(f"Unsafe patch path: {value!r}")
    return str(path)


def parse_patch(source: str) -> dict[str, list[Hunk]]:
    """Read ordinary unified diffs; reject malformed or ambiguous patch input."""
    lines = source.splitlines(keepends=True)
    files: dict[str, list[Hunk]] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- a/"):
            raise PatchError(f"Expected a file header at patch line {i + 1}")
        name = relative_path(lines[i][6:].rstrip("\n"))
        i += 1
        if i >= len(lines) or lines[i] != f"+++ b/{name}\n":
            raise PatchError(f"Mismatched destination header for {name}")
        i += 1
        hunks = files.setdefault(name, [])
        while i < len(lines) and lines[i].startswith("@@ "):
            match = re.fullmatch(
                r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n",
                lines[i],
            )
            if match is None:
                raise PatchError(f"Malformed hunk for {name}")
            old_count = int(match[2] or 1)
            new_count = int(match[4] or 1)
            before, after = [], []
            i += 1
            while old_count or new_count:
                if i >= len(lines) or lines[i][0] not in " +-":
                    raise PatchError(f"Truncated hunk for {name}")
                prefix, text = lines[i][0], lines[i][1:]
                if prefix in " -":
                    before.append(text)
                    old_count -= 1
                if prefix in " +":
                    after.append(text)
                    new_count -= 1
                if old_count < 0 or new_count < 0:
                    raise PatchError(f"Invalid hunk line counts for {name}")
                i += 1
            if not before:
                raise PatchError(f"An anchored source block is required for {name}")
            hunks.append(Hunk("".join(before), "".join(after)))
        if not hunks:
            raise PatchError(f"No hunks for {name}")
    return files


def apply_hunks(source: str, hunks: list[Hunk], label: str) -> str:
    for index, hunk in enumerate(hunks, 1):
        count = source.count(hunk.before)
        if count != 1:
            raise PatchError(
                f"{label}, hunk {index}: expected 1 exact match, found {count}"
            )
        source = source.replace(hunk.before, hunk.after, 1)
    return source


def read_manifest(patch_dir: Path) -> dict:
    manifest = json.loads((patch_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest["baseline"]["tag"] != "v0.5.22":
        raise PatchError("Unsupported backport manifest")
    return manifest


def source_path(root: Path, name: str) -> Path:
    path = root / relative_path(name)
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
    ):
        raise PatchError(f"Missing or unsafe SGLang source: {path}")
    return path


def prepare(root: Path, patch_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifest = read_manifest(patch_dir)
    originals, planned, complete = {}, {}, set()
    for name, expected in manifest["files"].items():
        source = source_path(root, name).read_bytes().decode("utf-8")
        originals[name] = source
        actual = sha256(source)
        if actual == expected["patched_sha256"]:
            complete.add(name)
        elif actual != expected["base_sha256"]:
            legacy = next(
                (s for s in expected["legacy"] if s["sha256"] == actual), None
            )
            if legacy is None:
                raise PatchError(
                    f"{name}: source hash {actual} is not the pinned v0.5.22 source, "
                    "a known original overlay, or the completed bundle; no files written"
                )
            reset_name = relative_path(legacy["reset_patch"])
            reset = parse_patch((patch_dir / reset_name).read_text(encoding="utf-8"))
            if set(reset) != {name}:
                raise PatchError(f"Unexpected legacy reset paths in {reset_name}")
            source = apply_hunks(source, reset[name], reset_name)
            if sha256(source) != expected["base_sha256"]:
                raise PatchError(
                    f"{name}: legacy reset did not recover the pinned source"
                )
        planned[name] = source

    for entry in manifest["backports"]:
        if not entry["patch"]:
            continue
        patch_name = relative_path(entry["patch"])
        patch = parse_patch((patch_dir / patch_name).read_text(encoding="utf-8"))
        for name, hunks in patch.items():
            if name not in planned:
                raise PatchError(f"Unregistered source in {patch_name}: {name}")
            if name not in complete:
                planned[name] = apply_hunks(
                    planned[name], hunks, f"{patch_name}: {name}"
                )

    for name, source in planned.items():
        if sha256(source) != manifest["files"][name]["patched_sha256"]:
            raise PatchError(
                f"{name}: resulting source hash does not match the reviewed bundle"
            )
        compile(source, str(root / name), "exec")
    return originals, planned


def atomic_write(path: Path, source: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(source.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_bundle(root: Path, patch_dir: Path, *, check: bool = False) -> list[str]:
    root, patch_dir = root.resolve(), patch_dir.resolve()
    originals, planned = prepare(root, patch_dir)
    changed = [name for name in planned if planned[name] != originals[name]]
    if check:
        return changed
    written = []
    try:
        for name in changed:
            atomic_write(root / name, planned[name])
            written.append(name)
    except OSError:
        for name in reversed(written):
            atomic_write(root / name, originals[name])
        raise
    return changed


def verify_bundle(root: Path, patch_dir: Path) -> int:
    root = root.resolve()
    manifest = read_manifest(patch_dir)
    for name, expected in manifest["files"].items():
        source = source_path(root, name).read_bytes().decode("utf-8")
        if sha256(source) != expected["patched_sha256"]:
            raise PatchError(f"{name}: completed bundle verification failed")
        compile(source, str(root / name), "exec")
    return len(manifest["files"])


def installed_root() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise PatchError("Cannot find installed sglang; use the base image's Python")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="SGLang package directory (otherwise locate installed sglang)",
    )
    parser.add_argument(
        "--patch-dir", type=Path, default=Path(__file__).resolve().parent / "patches"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate and report changes without writing",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="require the complete bundle to be installed",
    )
    args = parser.parse_args()
    root = args.root or installed_root()
    if args.verify:
        count = verify_bundle(root, args.patch_dir)
        print(f"PASS: {count} SGLang source hashes and Python syntax verified")
    else:
        changed = apply_bundle(root, args.patch_dir, check=args.check)
        action = "Would patch" if args.check else "Patched"
        print(
            f"{action} {len(changed)} SGLang files"
            if changed
            else "ALREADY APPLIED: complete SM120/V4.1 bundle verified"
        )


if __name__ == "__main__":
    main()
