import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "get_data_from_hf.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "get_data_from_hf_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GetDataFromHFPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_datasets = sys.modules.get("datasets")
        fake = types.ModuleType("datasets")
        fake.load_dataset = lambda *args, **kwargs: None
        sys.modules["datasets"] = fake

    def tearDown(self) -> None:
        if self.original_datasets is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = self.original_datasets

    def test_get_base_output_dir_prefers_ssd_dataset_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_dataset = os.environ.get("SSD_DATASET_DIR")
            previous_hf = os.environ.get("HF_DATASETS_CACHE")
            os.environ["SSD_DATASET_DIR"] = tmp
            os.environ["HF_DATASETS_CACHE"] = "/tmp/hf-should-not-win"
            try:
                module = load_module()
                self.assertEqual(module.get_base_output_dir(), tmp)
            finally:
                if previous_dataset is None:
                    os.environ.pop("SSD_DATASET_DIR", None)
                else:
                    os.environ["SSD_DATASET_DIR"] = previous_dataset
                if previous_hf is None:
                    os.environ.pop("HF_DATASETS_CACHE", None)
                else:
                    os.environ["HF_DATASETS_CACHE"] = previous_hf

    def test_get_base_output_dir_falls_back_to_hf_cache(self) -> None:
        previous_dataset = os.environ.pop("SSD_DATASET_DIR", None)
        previous_hf = os.environ.get("HF_DATASETS_CACHE")
        os.environ["HF_DATASETS_CACHE"] = "/tmp/hf-cache-root"
        try:
            module = load_module()
            self.assertEqual(
                module.get_base_output_dir(), "/tmp/hf-cache-root/processed_datasets"
            )
        finally:
            if previous_dataset is not None:
                os.environ["SSD_DATASET_DIR"] = previous_dataset
            if previous_hf is None:
                os.environ.pop("HF_DATASETS_CACHE", None)
            else:
                os.environ["HF_DATASETS_CACHE"] = previous_hf


if __name__ == "__main__":
    unittest.main()
