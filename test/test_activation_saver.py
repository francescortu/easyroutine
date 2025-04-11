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
    QueryResult,
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
        self.saver = ActivationSaver(
            base_dir=Path(self.temp_dir), experiment_name="test_experiment"
        )
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
            torch.tensor([1, 2, 3]), {"key": "value"}, activation_dir
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
        interventions = None  # Updated to match current signature

        save_dir = self.saver.save(
            activations,
            self.mock_model,
            target_positions,
            interventions,  # Updated to match current signature
            self.mock_extraction_config,
            {"additional_metadata": "test_value"},  # Added other_metadata
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
        self.assertEqual(
            metadata["interventions"], interventions
        )  # Updated to match current signature
        self.assertEqual(
            metadata["additional_metadata"], "test_value"
        )  # Check additional metadata
        self.assertIn("save_time", metadata)
        self.assertDictEqual(metadata["extraction_config"], {"mock_key": "mock_value"})
        self.assertIsNone(metadata["tag"])  # Check that tag is None by default

    def test_save_with_tag(self):
        """Test saving activations with a tag."""
        activations = torch.tensor([4, 5, 6])
        target_positions = [1, 2]
        interventions = None
        tag = "test_tag"

        save_dir = self.saver.save(
            activations,
            self.mock_model,
            target_positions,
            interventions,
            self.mock_extraction_config,
            {"additional_metadata": "test_value"},
            tag,
        )

        with open(save_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        self.assertEqual(metadata["tag"], tag)

    def test_rename_experiment(self):
        """Test renaming an experiment."""
        # First save some activations to create the experiment directory
        activations = torch.tensor([4, 5, 6])
        target_positions = [1, 2]

        self.saver.save(
            activations,
            self.mock_model,
            target_positions,
            None,  # interventions
            self.mock_extraction_config,
        )

        # Verify the old experiment directory exists
        old_exp_dir = self.temp_dir / "test_experiment"
        self.assertTrue(old_exp_dir.exists())

        # Rename the experiment
        new_exp_name = "renamed_experiment"
        self.saver.rename_experiment(new_exp_name)

        # Check that the old directory is gone and the new directory exists
        self.assertFalse(old_exp_dir.exists())
        new_exp_dir = self.temp_dir / new_exp_name
        self.assertTrue(new_exp_dir.exists())

        # Check that the saver's experiment name was updated
        self.assertEqual(self.saver.exp_name, new_exp_name)

        # Check that metadata was updated in all run directories
        for model_dir in new_exp_dir.iterdir():
            if model_dir.is_dir():
                for run_dir in model_dir.iterdir():
                    metadata_path = run_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        self.assertEqual(metadata["experiment_name"], new_exp_name)


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
        self.query_result.results.append(
            (mock_exp, mock_model_dir, mock_run_dir, mock_meta)
        )
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

    @patch("pathlib.Path.unlink")
    @patch("pathlib.Path.rmdir")
    def test_remove_by_index(self, mock_rmdir, mock_unlink):
        # Setup test data
        mock_exp = "exp_name"
        mock_model_dir = Path("mock_model_dir")
        mock_run_dir = Path("mock_run_dir")
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir, {}))

        # Test removing by index
        self.query_result.remove(0)

        # Check that unlink was called twice (for tensor.pt and metadata.json)
        self.assertEqual(mock_unlink.call_count, 2)
        # Check that rmdir was called once (for the run directory)
        mock_rmdir.assert_called_once()

    @patch("pathlib.Path.unlink")
    @patch("pathlib.Path.rmdir")
    def test_remove_by_string(self, mock_rmdir, mock_unlink):
        # Setup test data
        mock_exp = "exp_name"
        mock_model_dir = Path("mock_model_dir")
        mock_run_dir = Path("run_2023_01_01_12_00")
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir, {}))

        # Test removing by string
        self.query_result.remove("run_2023_01_01_12_00")

        # Check that unlink was called twice (for tensor.pt and metadata.json)
        self.assertEqual(mock_unlink.call_count, 2)
        # Check that rmdir was called once (for the run directory)
        mock_rmdir.assert_called_once()

    @patch("shutil.move")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data='{"experiment_name": "old_exp"}',
    )
    def test_update_run_experiment_by_index(self, mock_open, mock_move):
        # Setup test data
        mock_exp = "old_exp"
        mock_model_dir = Path("/base_dir/old_exp/model_name")
        mock_run_dir = Path("/base_dir/old_exp/model_name/run_123")
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir, {}))

        # Test updating experiment by index
        self.query_result.update_run_experiment(0, "new_exp")

        # Check that move was called with the right paths
        expected_source = str(mock_run_dir)
        expected_target = "/base_dir/new_exp/model_name/run_123"
        mock_move.assert_called_once_with(expected_source, expected_target)

        # Check that open was called for reading and writing metadata
        self.assertEqual(mock_open.call_count, 2)

    @patch("shutil.move")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data='{"experiment_name": "old_exp"}',
    )
    def test_update_run_experiment_by_string(self, mock_open, mock_move):
        # Setup test data
        mock_exp = "old_exp"
        mock_model_dir = Path("/base_dir/old_exp/model_name")
        mock_run_dir = Path("/base_dir/old_exp/model_name/run_123")
        self.query_result.results.append((mock_exp, mock_model_dir, mock_run_dir, {}))

        # Test updating experiment by string (run folder name)
        self.query_result.update_run_experiment("run_123", "new_exp")

        # Check that move was called with the right paths
        expected_source = str(mock_run_dir)
        expected_target = "/base_dir/new_exp/model_name/run_123"
        mock_move.assert_called_once_with(expected_source, expected_target)

        # Check that open was called for reading and writing metadata
        self.assertEqual(mock_open.call_count, 2)


class TestActivationLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_loader_init(self):
        loader = ActivationLoader(
            base_dir=Path(self.temp_dir), experiment_name="some_exp"
        )
        self.assertEqual(loader.base_dir, Path(self.temp_dir))
        self.assertEqual(loader.exp_name, "some_exp")

    @patch("os.environ.get", return_value="/fake/path")
    def test_loader_from_env(self, mock_env):
        with self.assertRaises(ValueError):
            # If the env is /fake/path but doesn't exist, still an error or fallback
            ActivationLoader.from_env(experiment_name="test_env")

    def test_loader_from_saver(self):
        saver = ActivationSaver(
            base_dir=Path(self.temp_dir), experiment_name="test_experiment"
        )
        loader = ActivationLoader.from_saver(saver)
        self.assertEqual(loader.base_dir, saver.base_dir)
        self.assertEqual(loader.exp_name, saver.exp_name)

    def test_query_with_tag(self):
        """Test querying activations with a tag filter."""
        # Setup: Create a directory structure and metadata files for testing
        exp_name = "test_exp"
        model_name = "test_model"
        exp_dir = Path(self.temp_dir) / exp_name
        model_dir = exp_dir / model_name
        model_dir.mkdir(parents=True)

        # Create two runs: one with a tag and one without
        run1_dir = model_dir / "2023-01-01_12-00-00"
        run1_dir.mkdir()
        with open(run1_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "tag": "test_tag",
                    "target_token_positions": [1, 2],
                },
                f,
            )
        with open(run1_dir / "tensor.pt", "wb") as f:
            torch.save(torch.tensor([1, 2, 3]), f)

        run2_dir = model_dir / "2023-01-01_13-00-00"
        run2_dir.mkdir()
        with open(run2_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "tag": None,
                    "target_token_positions": [1, 2],
                },
                f,
            )
        with open(run2_dir / "tensor.pt", "wb") as f:
            torch.save(torch.tensor([4, 5, 6]), f)

        # Test querying with tag
        loader = ActivationLoader(
            base_dir=Path(self.temp_dir), experiment_name=exp_name
        )
        results = loader.query(model_name=model_name, tag="test_tag")

        # We should only get the run with the matching tag
        self.assertEqual(len(results.results), 1)
        self.assertEqual(results.results[0][2].name, "2023-01-01_12-00-00")

        # Test querying without tag (should return both)
        results = loader.query(model_name=model_name)
        self.assertEqual(len(results.results), 2)

    def test_query_with_custom_keys(self):
        """Test querying activations with custom metadata keys."""
        # Setup: Create a directory structure and metadata files for testing
        exp_name = "test_exp"
        model_name = "test_model"
        exp_dir = Path(self.temp_dir) / exp_name
        model_dir = exp_dir / model_name
        model_dir.mkdir(parents=True)

        # Create two runs with different custom metadata
        run1_dir = model_dir / "2023-01-01_12-00-00"
        run1_dir.mkdir()
        with open(run1_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "target_token_positions": [1, 2],
                    "custom_key": "value1",
                },
                f,
            )
        with open(run1_dir / "tensor.pt", "wb") as f:
            torch.save(torch.tensor([1, 2, 3]), f)

        run2_dir = model_dir / "2023-01-01_13-00-00"
        run2_dir.mkdir()
        with open(run2_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "experiment_name": exp_name,
                    "model_name": model_name,
                    "target_token_positions": [1, 2],
                    "custom_key": "value2",
                },
                f,
            )
        with open(run2_dir / "tensor.pt", "wb") as f:
            torch.save(torch.tensor([4, 5, 6]), f)

        # Test querying with custom keys
        loader = ActivationLoader(
            base_dir=Path(self.temp_dir), experiment_name=exp_name
        )
        results = loader.query(
            model_name=model_name, custom_keys={"custom_key": "value1"}
        )

        # We should only get the run with the matching custom key
        self.assertEqual(len(results.results), 1)
        self.assertEqual(results.results[0][2].name, "2023-01-01_12-00-00")


if __name__ == "__main__":
    unittest.main()
