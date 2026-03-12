from collections.abc import Callable
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("bench_paths.py")


def load_bench_paths_module(root: Path) -> ModuleType:
    env = {
        "SSD_HF_CACHE": str(root / "hf-cache"),
        "SSD_DATASET_DIR": str(root / "datasets"),
    }
    Path(env["SSD_HF_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(env["SSD_DATASET_DIR"]).mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("bench_paths_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, env, clear=False):
        _ = spec.loader.exec_module(module)
    return module


class ResolveSnapshotTests(unittest.TestCase):
    def test_returns_root_when_config_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = load_bench_paths_module(root)
            resolve_snapshot = cast(Callable[[str], str], module.resolve_snapshot)

            model_dir = root / "root-model"
            model_dir.mkdir()
            _ = (model_dir / "config.json").write_text("{}")

            self.assertEqual(resolve_snapshot(str(model_dir)), str(model_dir))

    def test_returns_snapshot_child_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = load_bench_paths_module(root)
            resolve_snapshot = cast(Callable[[str], str], module.resolve_snapshot)

            model_dir = root / "hf-style"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)
            _ = (snapshot_dir / "config.json").write_text("{}")

            self.assertEqual(resolve_snapshot(str(model_dir)), str(snapshot_dir))

    def test_returns_direct_child_with_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = load_bench_paths_module(root)
            resolve_snapshot = cast(Callable[[str], str], module.resolve_snapshot)

            model_dir = root / "child-layout"
            child_dir = model_dir / "resolved"
            child_dir.mkdir(parents=True)
            _ = (child_dir / "config.json").write_text("{}")

            self.assertEqual(resolve_snapshot(str(model_dir)), str(child_dir))

    def test_raises_when_no_snapshot_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = load_bench_paths_module(root)
            resolve_snapshot = cast(Callable[[str], str], module.resolve_snapshot)

            missing_dir = root / "missing"
            missing_dir.mkdir()

            with self.assertRaises(FileNotFoundError):
                _ = resolve_snapshot(str(missing_dir))


if __name__ == "__main__":
    _ = unittest.main()
