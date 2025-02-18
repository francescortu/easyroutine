import unittest
import tempfile
import shutil
import torch
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from easyroutine.interpretability.activation_saver import (
    ActivationSaver,
    ActivationLoader,
    QueryResult
)
from easyroutine.interpretability.hooked_model import HookedModel, ExtractionConfig

class MockHookedModelConfig:
    """Mock config with a model_name attribute."""
    model_name = "test_model"

class MockHookedModel:
    """Mock hooking the actual HookedModel to simplify testing."""
    def __init__(self):
        self.config = MockHookedModelConfig()

class MockExtractionConfig(ExtractionConfig):
    """Mocking the ExtractionConfig for test usage."""
    def to_dict(self):
        return {"mock_key": "mock_value"}

class TestActivationSaver(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.saver = ActivationSaver(base_dir=Path(self.temp_dir), experiment_name="test_experiment")
        self.mock_model = MockHookedModel()
        self.mock_extraction_config = MockExtractionConfig()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_saver_init(self):
        self.assertEqual(self.saver.base_dir, Path(self.temp_dir))
        self.assertEqual(self.saver.exp_name, "test_experiment")

    def test_save_object_to_path(self):
        activation_dir = Path(self.temp_dir) / "experiment" / "model"
        self.saver.save_object_to_path(
            torch.tensor([1, 2, 3]),
            {"key": "value"},
            activation_dir
        )
        # Check files
        self.assertTrue((activation_dir / "tensor.pt").exists())
        self.assertTrue((activation_dir / "metadata.json").exists())

        with open(activation_dir / "metadata.json", "r") as f:
            data = json.load(f)
        self.assertEqual(data["key"], "value")

    def test_save(self):
        activations = torch.tensor([4, 5, 6])
        target_positions = [1, 2]
        pivot_positions = [3, 4]
        ablation_queries = [{'layer': 1}]
        patching_queries = [{'layer': 2}]

        save_dir = self.saver.save(
            activations,
            self.mock_model,
            target_positions,
            pivot_positions,
            ablation_queries,
            patching_queries,
            self.mock_extraction_config,
            {}
        )

        self.assertTrue(save_dir.exists())
        tensor_file = save_dir / "tensor.pt"
        meta_file = save_dir / "metadata.json"
        self.assertTrue(tensor_file.exists())
        self.assertTrue(meta_file.exists())

        with open(meta_file, "r") as f:
            metadata = json.load(f)
        self.assertEqual(metadata["experiment_name"], "test_experiment")
        self.assertEqual(metadata["model_name"], "test_model")
        self.assertEqual(metadata["target_token_positions"], target_positions)
        self.assertEqual(metadata["pivot_positions"], pivot_positions)
        self.assertEqual(metadata["ablation_queries"], ablation_queries)
        self.assertEqual(metadata["patching_queries"], patching_queries)
        self.assertIn("save_time", metadata)
        self.assertDictEqual(metadata["extraction_config"], {"mock_key": "mock_value"})

class TestQueryResult(unittest.TestCase):
    def setUp(self):
        self.query_result = QueryResult()

    def test_repr_empty(self):
        self.assertEqual(str(self.query_result), "")

    def test_repr_with_results(self):
        mock_exp = "exp_name"
        mock_model_dir = Path("mock_model_dir")
        mock_run_dir = Path("mock_run_dir")
        mock_meta = {}
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir, mock_meta))
        result_str = str(self.query_result)
        self.assertIn("Experiment: exp_name", result_str)
        self.assertIn("Model: mock_model_dir", result_str)
        self.assertIn("Time Folder: mock_run_dir", result_str)

    @patch("torch.load")
    def test_load_by_time_string_match(self, mock_load):
        mock_exp = "exp_name"
        mock_model_dir = Path("mock_model_dir")
        mock_run_dir = Path("mock_run_dir")
        mock_run_dir.mkdir(exist_ok=True)
        (mock_run_dir / "tensor.pt").touch()
        (mock_run_dir / "metadata.json").write_text(json.dumps({"save_time": "fake"}))
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir, {}))

        # match
        obj, meta = self.query_result.load("mock_run_dir")
        mock_load.assert_called_once()
        self.assertIsNotNone(obj)
        self.assertIsNotNone(meta)

    def test_load_empty(self):
        self.assertEqual(self.query_result.load(), (None, None))

    @patch("torch.load")
    def test_load_by_index(self, mock_load):
        mock_exp = "exp_name"
        mock_model_dir = Path("mock_model_dir")
        mock_run_dir1 = Path("run_2023_01_01_12_00")
        mock_run_dir2 = Path("run_2023_01_01_12_01")
        mock_run_dir1.mkdir(exist_ok=True)
        mock_run_dir2.mkdir(exist_ok=True)
        (mock_run_dir1 / "tensor.pt").touch()
        (mock_run_dir1 / "metadata.json").write_text(json.dumps({}))
        (mock_run_dir2 / "tensor.pt").touch()
        (mock_run_dir2 / "metadata.json").write_text(json.dumps({}))
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir1, {}))
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir2, {}))

        obj, meta = self.query_result.load(-1)  # should pick the last one
        mock_load.assert_called_once()
        self.assertIsNotNone(obj)

class TestActivationLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_loader_init(self):
        loader = ActivationLoader(base_dir=Path(self.temp_dir), experiment_name="some_exp")
        self.assertEqual(loader.base_dir, Path(self.temp_dir))
        self.assertEqual(loader.exp_name, "some_exp")

    @patch("os.environ.get", return_value="/fake/path")
    def test_loader_from_env(self, mock_env):
        with self.assertRaises(ValueError):
            # If the env is /fake/path but doesn't exist, still an error or fallback
            ActivationLoader.from_env(experiment_name="test_env")

    def test_loader_from_saver(self):
        saver = ActivationSaver(base_dir=Path(self.temp_dir), experiment_name="test_experiment")
        loader = ActivationLoader.from_saver(saver)
        self.assertEqual(loader.base_dir, saver.base_dir)
        self.assertEqual(loader.exp_name, saver.exp_name)

if __name__ == "__main__":
    unittest.main()