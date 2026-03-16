import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ssd" / "config.py"


def load_config_module():
    spec = importlib.util.spec_from_file_location("ssd_config_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigLazyDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_torch = sys.modules.get("torch")
        self.original_transformers = sys.modules.get("transformers")

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.device = lambda name: name

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(path):
                return types.SimpleNamespace(
                    max_position_embeddings=4096, num_hidden_layers=32
                )

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoConfig = FakeAutoConfig

        sys.modules["torch"] = fake_torch
        sys.modules["transformers"] = fake_transformers

    def tearDown(self) -> None:
        if self.original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.original_torch

        if self.original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = self.original_transformers

    def test_import_and_explicit_model_do_not_require_path_env(self) -> None:
        previous_hf = os.environ.pop("SSD_HF_CACHE", None)
        previous_dataset = os.environ.pop("SSD_DATASET_DIR", None)

        try:
            module = load_config_module()

            with tempfile.TemporaryDirectory() as tmp:
                cfg = module.Config(model=tmp, speculate=False)
                self.assertEqual(cfg.model, tmp)
                self.assertEqual(cfg.hf_config.max_position_embeddings, 4096)
        finally:
            if previous_hf is not None:
                os.environ["SSD_HF_CACHE"] = previous_hf
            if previous_dataset is not None:
                os.environ["SSD_DATASET_DIR"] = previous_dataset


if __name__ == "__main__":
    unittest.main()
