import re
from easyroutine.logger import logger
import torch
from typing import List, Union, Optional
import contextlib
from easyroutine.interpretability.hooked_model import HookedModel, ExtractionConfig
from easyroutine.interpretability.activation_cache import ActivationCache
from easyroutine.interpretability.interventions import Intervention
from datetime import datetime
from pathlib import Path
from rich import print
import json
from dotenv import load_dotenv
import os


class ActivationSaver:
    """
    This class is used to save activations in the file system. It is not necessary to use this class directly and it is possible to use just torch.save. However, this class provides a simple interface to save activations in a structured way.
    """

    def __init__(self, base_dir: Union[Path, str], experiment_name: str = "default"):
        """
        Arguments:
            - base_dir (Path): The base directory where the activations will be saved.
            - experiment_name (str): The name of the experiment. It will be used to create a directory with the same name in the base_dir.
        """
        self.base_dir = Path(base_dir)
        self.exp_name = experiment_name

    @classmethod
    def from_env(cls, experiment_name: str = "default"):
        load_dotenv()
        activation_base_dir = os.environ.get("ACTIVATION_BASE_DIR")
        if activation_base_dir is None:
            raise ValueError("ACTIVATION_BASE_DIR is not set in the environment.")
        base_dir = Path(activation_base_dir)
        return cls(base_dir, experiment_name)

    def save_object_to_path(
        self, obj: Union[torch.Tensor, dict], metadata: dict, activation_dir: Path
    ):
        """
        Saves a tensor or a dictionary of tensors to a path.
        """
        activation_dir.mkdir(parents=True, exist_ok=True)

        tensor_path = activation_dir / "tensor.pt"
        metadata_path = activation_dir / "metadata.json"

        torch.save(obj, tensor_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def save_cache(self,
                   cache:ActivationCache,
                   other_metadata: dict = {},
                   tag: Optional[str] = None,
                   ):
        """
        Saves the activation cache in the file systems. These methods makes it easier to save the activations cache if the cache contatains the foundamentals metadata already.
        Arguments:
            - cache: The activation cache to be saved.
            - other_metadata: Other metadata that the user wants to save.
            - tag: A tag that can be used to identify the activations.
        """
        # check if the cache has the metadata
        if cache["metadata"] is None:
            raise ValueError("The cache does not have the metadata. Please provide the metadata or use the metod ActivationSaver.save().")

        metadata = cache["metadata"]
        # save the cache
        self.save(
            cache,
            metadata["model_name"],
            metadata["target_token_positions"],
            metadata["interventions"],
            metadata["extraction_config"],
            other_metadata,
            tag,
        )

    def save(
        self,
        activations: Union[torch.Tensor, dict, ActivationCache],
        model: HookedModel,
        target_token_positions,
        interventions: Optional[List[Intervention]],
        extraction_config: ExtractionConfig,
        other_metadata: dict = {},
        tag: Optional[str] = None,
    ):
        """
        Saves the activations in the file system. The activations can be a tensor or a dictionary of tensors. The metadata is a dictionary that will be saved in a json file. The metadata should contain the following keys. The key are required to impose a structure in the saved activations. The user can add other keys to the metadata.

        Arguments:
            - save_time: The time when the activations were saved.
            - experiment_name: The name of the experiment.
            - model_name: The name of the model.
            - target_token_positions: The positions of the target tokens in the input.
            - pivot_positions: The positions of the pivot tokens in the input.
            - ablation_queries: The ablation queries used to extract the activations.
            - patching_queries: The patching queries used to extract the activations.
            - extraction_config: The extraction config used to extract the activations.
            - other_metadata: Other metadata that the user wants to save.

        Returns:
            - save_dir: The directory where the activations were saved, if the user wants to access them later and save other files.
        """
        model_name = model.config.model_name

        # First we create the metadata
        metadata = {
            "tag": tag,
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": self.exp_name,
            "model_name": model_name,
            "target_token_positions": target_token_positions,
            "interventions": interventions,
            "extraction_config": extraction_config.to_dict(),
            **other_metadata,
        }

        # Now we create the directory where the activations will be saved
        save_dir = Path(self.base_dir, self.exp_name, model_name.replace("/", "_"))

        # Then we create the directory where stored the activations. SHould have the time in the name
        save_dir = Path(
            save_dir, metadata["save_time"].replace(":", "_").replace(" ", "_")
        )

        # Now we save the activations
        self.save_object_to_path(activations, metadata, save_dir)

        # now we print the path BASE_DIR/{experiment_name}/{model_name}
        print(f"""
Saving results:
        - Experiment: {self.exp_name},
        - Model Name: {model_name},
Activations saved in BASE_DIR/{self.exp_name}/{model_name}
""")
        return save_dir


class QueryResult:
    """
    Holds the results of a query, including a list of runs with metadata and paths.
    Provides:
      - __repr__ for printing info
      - load() to load a specific run by time or index
    """

    def __init__(self):
        self.results = []  # Will hold tuples of (exp_name, model_dir, run_dir, metadata)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        lines = []
        idx = 0
        for exp_name, model_dir, run_dir, metadata in self.results:
            lines.append(
                f"{idx} - Experiment: {exp_name}, Model: {model_dir.name}, Time Folder: {run_dir.name}"
            )
            if metadata.get("tag") is not None:
                lines.append(f"  Tag: {metadata['tag']}")
            idx += 1
        return "\n".join(lines)

    def __getitem__(self, item):
        """Allow slicing of query results."""
        if isinstance(item, slice):
            new_qr = QueryResult()
            new_qr.results = self.results[item]
            return new_qr
        return self.results[item]

    def get_paths(self):
        """
        Same as __repr__, but also print the full paths.
        """
        lines = []
        for exp_name, model_dir, run_dir, metadata in self.results:
            lines.append(
                f"Experiment: {exp_name}, Model: {model_dir.name}, Time Folder: {run_dir.name}\n{run_dir}"
            )
        return "\n".join(lines)

    def load(self, time: Union[str, int] = -1):
        """
        Loads an entry by its time string or by index:
          - If time is a string, we look for an exact match in run_dir.name.
          - If time is an int (e.g., -1 for the most recent), we select by index from sorted runs.
        Returns the loaded activation object and metadata.
        """
        if not self.results:
            print("No results to load.")
            return None, None

        # Filter if time is string
        if isinstance(time, str):
            for exp_name, model_dir, run_dir, metadata in self.results:
                if run_dir.name == time:
                    return self._load_run(run_dir)
            print(f"No run found for time={time}.")
            return None, None
        # Otherwise it must be an int
        else:
            sorted_runs = sorted(
                self.results, key=lambda x: x[2].name
            )  # sort by run_dir name
            # convert negative index
            index = time if time >= 0 else len(sorted_runs) + time
            if index < 0 or index >= len(sorted_runs):
                print(f"Index out of range: {time}")
                return None, None
            chosen_exp, chosen_model, chosen_run, chosen_meta = sorted_runs[index]
            return self._load_run(chosen_run)

    def _load_run(self, run_dir: Path):
        """Helper to load the activation object and metadata from run_dir."""
        tensor_path = run_dir / "tensor.pt"
        metadata_path = run_dir / "metadata.json"

        if not tensor_path.exists() or not metadata_path.exists():
            print(f"Run directory incomplete: {run_dir}")
            return None, None

        obj = torch.load(tensor_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return obj, metadata

    def remove(self, time: Union[str, int]):
        """
        Removes an entry by its time string or by index:
          - If time is a string, we look for an exact match in run_dir.name.
          - If time is an int (e.g., -1 for the most recent), we select by index from sorted runs.
        """
        if not self.results:
            print("No results to remove.")
            return

        # Filter if time is string
        if isinstance(time, str):
            for exp_name, model_dir, run_dir, metadata in self.results:
                if run_dir.name == time:
                    self._remove_run(run_dir)
                    return
            print(f"No run found for time={time}.")
        # Otherwise it must be an int
        else:
            sorted_runs = sorted(
                self.results, key=lambda x: x[2].name
            )
            # convert negative index
            index = time if time >= 0 else len(sorted_runs) + time
            if index < 0 or index >= len(sorted_runs):
                print(f"Index out of range: {time}")
                return
            chosen_exp, chosen_model, chosen_run, chosen_meta = sorted_runs[index]
            self._remove_run(chosen_run)
            
        # print(f"Removed run at index {index}.")
        # show the remaining runs
        print(self)
    
    def _remove_run(self, run_dir: Path):
        """Helper to remove the activation object and metadata from run_dir."""
        tensor_path = run_dir / "tensor.pt"
        metadata_path = run_dir / "metadata.json"
        tensor_path.unlink()
        metadata_path.unlink()
        run_dir.rmdir()
class ActivationLoader:
    """
    This class is used to query the activations saved in the file system. The primary idea is to use this class to search for activations based on some criteria, such model, experiment, run configuration, etc.
    """

    def __init__(self, base_dir: Path, experiment_name: str = "default"):
        self.base_dir = Path(base_dir)
        self.exp_name = experiment_name

    @classmethod
    def from_env(cls, experiment_name: str = "default"):
        load_dotenv()
        activation_base_dir = os.environ.get("ACTIVATION_BASE_DIR")
        if activation_base_dir is None:
            raise ValueError("ACTIVATION_BASE_DIR is not set in the environment.")
        base_dir = Path(activation_base_dir)
        return cls(base_dir, experiment_name)

    @classmethod
    def from_saver(cls, saver: ActivationSaver):
        return cls(saver.base_dir, saver.exp_name)

    def query(
        self,
        experiment_name: Optional[str] = None,
        model_name: Optional[str] = None,
        target_token_positions: Optional[List[Union[str, int]]] = None,
        pivot_positions: Optional[List[int]] = None,
        save_time: Optional[str] = None,
        custom_keys: Optional[dict] = None,
        extraction_config: Optional[ExtractionConfig] = None,
        interventions: Optional[List[Intervention]] = None,
        verbose: bool = False,
        tag: Optional[str] = None,  # Added
    ) -> QueryResult:
        """
        Instead of printing, returns a QueryResult object that can be printed or loaded from.
        If both experiment and model are declared, forces verbose to True.
        """
        if experiment_name and model_name:
            verbose = True

        used_experiment_name = experiment_name or self.exp_name
        all_experiment_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        if experiment_name:
            all_experiment_dirs = [
                d for d in all_experiment_dirs if d.name == used_experiment_name
            ]

        def list_match(metadata_value, query_list):
            return (query_list is None) or (metadata_value == query_list)

        def match_custom_keys(metadata, ckeys):
            """Check if all custom key-value pairs match in metadata."""
            for k, v in ckeys.items():
                if metadata.get(k) != v:
                    return False
            return True

        query_result = QueryResult()

        for exp_dir in all_experiment_dirs:
            model_dirs = [m for m in exp_dir.iterdir() if m.is_dir()]
            if model_name:
                norm_name = model_name.replace("/", "_")
                model_dirs = [m for m in model_dirs if m.name == norm_name]

            if not model_dirs:
                continue

            for m_dir in model_dirs:
                run_dirs = [r for r in m_dir.iterdir() if r.is_dir()]
                for r_dir in run_dirs:
                    metadata_path = r_dir / "metadata.json"
                    if not metadata_path.exists():
                        continue

                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    if tag is not None and metadata.get("tag") != tag:
                        continue
                    if save_time and save_time != metadata.get("save_time"):
                        continue
                    if not list_match(
                        metadata.get("target_token_positions"), target_token_positions
                    ):
                        continue
                    if not list_match(metadata.get("pivot_positions"), pivot_positions):
                        continue
                    if custom_keys and not match_custom_keys(metadata, custom_keys):
                        continue
                    if extraction_config is not None:
                        if (
                            metadata.get("extraction_config")
                            != extraction_config.to_dict()
                        ):
                            continue
                    if interventions is not None:
                        if metadata.get("interventions") != interventions:
                            continue

                    query_result.results.append((exp_dir.name, m_dir, r_dir, metadata))

        return query_result
