import difflib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import patch_sm120 as bundle


class PatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "sglang"
        self.assets = Path(self.temp.name) / "patches"
        self.root.mkdir()
        self.assets.mkdir()
        self.old = "answer = 1\n"
        self.new = "answer = 2\n"
        self.manifest = {
            "schema": 1,
            "baseline": {"tag": "v0.5.22"},
            "backports": [{"patch": "fix.patch"}],
            "files": {},
        }
        self.add_source("one.py", self.old, self.new)
        self.save()

    def add_source(self, name, before, after):
        (self.root / name).write_text(before)
        self.manifest["files"][name] = {
            "base_sha256": bundle.sha256(before),
            "patched_sha256": bundle.sha256(after),
            "legacy": [],
        }
        delta = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + name,
                tofile="b/" + name,
            )
        )
        with (self.assets / "fix.patch").open("a") as stream:
            stream.write(delta)

    def save(self):
        (self.assets / "manifest.json").write_text(json.dumps(self.manifest))

    def test_apply_verify_and_repeat(self):
        self.assertEqual(bundle.apply_bundle(self.root, self.assets), ["one.py"])
        self.assertEqual((self.root / "one.py").read_text(), self.new)
        self.assertEqual(bundle.verify_bundle(self.root, self.assets), 1)
        self.assertEqual(bundle.apply_bundle(self.root, self.assets), [])

    def test_check_does_not_write(self):
        self.assertEqual(
            bundle.apply_bundle(self.root, self.assets, check=True), ["one.py"]
        )
        self.assertEqual((self.root / "one.py").read_text(), self.old)

    def test_unknown_source_rejected_before_any_write(self):
        self.add_source("two.py", self.old, self.new)
        self.save()
        (self.root / "two.py").write_text("local_changes = True\n")
        with self.assertRaises(bundle.PatchError):
            bundle.apply_bundle(self.root, self.assets)
        self.assertEqual((self.root / "one.py").read_text(), self.old)

    def test_syntax_error_rejected_before_any_write(self):
        self.add_source("two.py", self.old, "def broken(:\n")
        self.save()
        with self.assertRaises(SyntaxError):
            bundle.apply_bundle(self.root, self.assets)
        self.assertEqual((self.root / "one.py").read_text(), self.old)

    def test_result_hash_mismatch_rejected(self):
        self.manifest["files"]["one.py"]["patched_sha256"] = "0" * 64
        self.save()
        with self.assertRaises(bundle.PatchError):
            bundle.apply_bundle(self.root, self.assets)
        self.assertEqual((self.root / "one.py").read_text(), self.old)

    def test_known_legacy_source_is_upgraded(self):
        legacy = "answer = 0\n"
        (self.root / "one.py").write_text(legacy)
        reset = "".join(
            difflib.unified_diff(
                legacy.splitlines(True),
                self.old.splitlines(True),
                fromfile="a/one.py",
                tofile="b/one.py",
            )
        )
        (self.assets / "reset.patch").write_text(reset)
        self.manifest["files"]["one.py"]["legacy"] = [
            {"sha256": bundle.sha256(legacy), "reset_patch": "reset.patch"}
        ]
        self.save()
        bundle.apply_bundle(self.root, self.assets)
        self.assertEqual((self.root / "one.py").read_text(), self.new)

    def test_write_error_restores_completed_files(self):
        self.add_source("two.py", self.old, self.new)
        self.save()
        real_write = bundle.atomic_write

        def write(path, source):
            if path.name == "two.py" and source == self.new:
                raise OSError("simulated disk error")
            real_write(path, source)

        with patch.object(bundle, "atomic_write", side_effect=write):
            with self.assertRaises(OSError):
                bundle.apply_bundle(self.root, self.assets)
        self.assertEqual((self.root / "one.py").read_text(), self.old)

    def test_truncated_and_ambiguous_hunks_rejected(self):
        with self.assertRaises(bundle.PatchError):
            bundle.parse_patch("--- a/one.py\n+++ b/one.py\n@@ -1,2 +1 @@\n-x\n+y\n")
        with self.assertRaises(bundle.PatchError):
            bundle.apply_hunks("x\nx\n", [bundle.Hunk("x\n", "y\n")], "test")

    def test_paths_cannot_escape_source_root(self):
        for name in ("../outside.py", "/absolute.py", "a/../../outside.py", "a\\b.py"):
            with self.subTest(name=name), self.assertRaises(bundle.PatchError):
                bundle.relative_path(name)

    def test_verifier_detects_post_patch_changes(self):
        bundle.apply_bundle(self.root, self.assets)
        (self.root / "one.py").write_text(self.new + "# changed\n")
        with self.assertRaises(bundle.PatchError):
            bundle.verify_bundle(self.root, self.assets)
