import re
import torch
import contextlib
from easyroutine.logger import logger
from typing import List, Union, Optional
from rich import print as print

# TODO: Add a method to expand the tensors in the cache adding the target token dimension using the mapping_index key. In this way we can have a tensors of shape (batch_size, target_tokens, num_tokens, hidden_size) and resolve the ambiguity of the target tokens when we have the average. Indeed, now we have a tensor of shape (batch_size, num_tokens, hidden_size) and mapping index map the index of the second dimension to the correct token. However, the second dim could be both single token or multiple tokens averaged. If we add a new dimension, could be easier to understand.


def just_old(old, new):
    """Always return the new value (or if old is None, return new)."""
    return new if old is None else new


def just_me(old, new):
    """If no old value, start a list; otherwise, add new to the list."""
    return [new] if old is None else old + new


def sublist(old, new):
    """
    Aggregates by flattening. If old is already a list, extend it;
    otherwise, put old into a list and add new (flattening if needed).
    """
    all_values = []
    if old is not None:
        if isinstance(old, list):
            all_values.extend(old)
        else:
            all_values.append(old)
    if isinstance(new, list):
        all_values.extend(new)
    else:
        all_values.append(new)
    return all_values




def aggregate_last_layernorm(old, new):
    """
    Aggregates `last_layernorm` values by concatenating along the first dimension.

    Args:
        old (dict): Previous stored dictionary, where each value is a tensor of shape [m, N].
        new (dict): New incoming dictionary, where each value is a tensor of shape [1, N].

    Returns:
        dict: Aggregated dictionary where each value is a tensor of shape [m+1, N].
    """
    if old is None:
        return new  # If there's no existing data, just return the new one.

    if not isinstance(old, dict) or not isinstance(new, dict):
        raise TypeError("Both old and new values must be dictionaries.")

    aggregated = {}
    for key in new:
        if key not in old:
            aggregated[key] = new[key]  # If key is new, just add it.
        else:
            if not isinstance(old[key], torch.Tensor) or not isinstance(
                new[key], torch.Tensor
            ):
                raise TypeError(f"Values for key {key} must be tensors.")

            if old[key].shape[-1] != new[key].shape[-1]:
                raise ValueError(
                    f"Tensor shape mismatch for key {key}: "
                    f"{old[key].shape} vs {new[key].shape}"
                )

            # Concatenate along the first dimension
            aggregated[key] = torch.cat([old[key], new[key]], dim=0)

    return aggregated


class ValueWithInfo:
    """
    A thin wrapper around a value that also stores extra info.
    """

    __slots__ = ("_value", "_info")

    def __init__(self, value, info):
        self._value = value
        self._info = info

    def info(self):
        return self._info

    def value(self):
        return self._value

    def __getattr__(self, name):
        return getattr(self._value, name)

    def __repr__(self):
        return f"ValueWithInfo(value={self._value!r}, info={self._info!r})"


class ActivationCache:
    """
    A dictionary-like cache for storing and aggregating model activation values.
    Supports custom aggregation strategies registered for keys (by prefix match)
    and falls back to a default aggregation that can dynamically switch types if needed.
    """

    def __init__(self):
        self.cache = {}
        self.valid_keys = (
            re.compile(r"resid_out_\d+"),
            re.compile(r"resid_in_\d+"),
            re.compile(r"resid_mid_\d+"),
            re.compile(r"attn_in_\d+"),
            re.compile(r"attn_out_\d+"),
            re.compile(r"avg_attn_pattern_L\dH\d+"),
            re.compile(r"pattern_L\dH+\d+"),
            re.compile(r"head_values_L\dH+\d+"),
            re.compile(r"head_keys_L\dH+\d+"),
            re.compile(r"head_queries_L\dH+\d+"),
            re.compile(r"values_L\d+"),
            re.compile(r"keys_L\d+"),
            re.compile(r"queries_L\d+"),
            re.compile(r"input_ids"),
            re.compile(r"mapping_index"),
            re.compile(r"mlp_out_\d+"),
            re.compile(r"last_layernorm"),
            re.compile(r"token_dict")
        )
        self.aggregation_strategies = {}
        # Register default aggregators for some keys
        self.register_aggregation("mapping_index", just_old)
        self.register_aggregation("offset", sublist)
        self.register_aggregation("last_layernorm", aggregate_last_layernorm)
        self.register_aggregation("token_dict", sublist)
        self.deferred_cache = False

    def __repr__(self) -> str:
        # Skip 'metadata' from the printed keys
        items = [key for key in self.cache.keys() if key != "metadata"]
        return f"ActivationCache(`{', '.join(items)}`)"

    def __str__(self) -> str:
        # Skip 'metadata' from the printed items
        items = [
            f"{key}: {value}" for key, value in self.cache.items() if key != "metadata"
        ]
        return f"ActivationCache({', '.join(items)})"

    def __setitem__(self, key: str, value):
        if not any(pattern.match(key) for pattern in self.valid_keys):
            logger.debug(
                f"Invalid key: {key}. Valid keys are: {self.valid_keys}. Could be a user-defined key."
            )
        self.cache[key] = value

    def __getitem__(self, key: str):
        return self.cache[key]

    def __delitem__(self, key: str):
        del self.cache[key]

    def __add__(self, other) -> "ActivationCache":
        if not isinstance(other, (dict, ActivationCache)):
            raise TypeError("Can only add ActivationCache or dict objects.")
        new_cache = ActivationCache()
        new_cache.cache = {
            **self.cache,
            **(other.cache if isinstance(other, ActivationCache) else other),
        }
        return new_cache

    def __contains__(self, key):
        return key in self.cache

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("logger", None)
        state.pop("aggregation_strategies", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.aggregation_strategies = {}
        self.register_aggregation("mapping_index", just_old)
        self.register_aggregation("input_ids", just_me)
        self.register_aggregation("offset", sublist)
        
    def is_empty(self) -> bool:
        """
        Returns True if the cache is empty, False otherwise.
        """
        return len(self.cache) == 0

    def get(self, key: str, default=None):
        return self.cache.get(key, default)

    def items(self):
        return self.cache.items()

    def keys(self):
        return self.cache.keys()

    def values(self):
        return self.cache.values()

    def update(self, other):
        if isinstance(other, dict):
            self.cache.update(other)
        elif isinstance(other, type(self)):
            self.cache.update(other.cache)
        else:
            raise TypeError("Can only update with dict or ActivationCache objects.")

    def to(self, device: Union[str, torch.device]):
        for key, value in self.cache.items():
            if hasattr(value, "to"):
                self.cache[key] = value.to(device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(v, "to"):
                        value[k] = v.to(device)
                self.cache[key] = value

    def cpu(self):
        self.to("cpu")

    def cuda(self):
        self.to("cuda")

    def register_aggregation(self, key_pattern, function):
        """
        Registers a custom aggregation function for keys that start with key_pattern.
        """
        logger.debug(
            f"Registering aggregation strategy for keys starting with '{key_pattern}'"
        )
        self.aggregation_strategies[key_pattern] = function

    def remove_aggregation(self, key_pattern):
        if key_pattern in self.aggregation_strategies:
            del self.aggregation_strategies[key_pattern]

    def _get_aggregation_strategy(self, key: str):
        """
        Returns the aggregation function for the given key.
        If no custom function is registered, the default aggregation is used.
        """
        for pattern, strategy in self.aggregation_strategies.items():
            if key.startswith(pattern):
                return strategy
        return self.default_aggregation

    def default_aggregation(self, old, new):
        """
        Default aggregation strategy.
        - If old is None, simply return new.
        - For torch.Tensor values, first try torch.cat, then torch.stack, and finally fallback to list aggregation.
        - For lists and tuples, aggregates by appending (or converting tuples to lists).
        - For ValueWithInfo, aggregates the inner values.
        - Otherwise, tries the '+' operator and falls back to a list if necessary.
        """
        if old is None:
            return new

        # Aggregation for torch.Tensor
        if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
            try:
                return torch.cat([old, new], dim=0)
            except Exception as e:
                logger.warning(
                    f"torch.cat failed for tensor shapes {old.shape} and {new.shape}: {e}; trying torch.stack."
                )
                try:
                    return torch.stack([old, new], dim=0)
                except Exception as e:
                    logger.warning(
                        f"torch.stack also failed: {e}; switching to list aggregation."
                    )
                    return [old, new]

        # Aggregation for lists
        if isinstance(old, list):
            return old + (new if isinstance(new, list) else [new])

        # Aggregation for tuples: convert to list
        if isinstance(old, tuple):
            if isinstance(new, tuple):
                return [old, new]
            elif isinstance(new, list):
                return [old] + new
            else:
                return [old, new]

        # Aggregation for ValueWithInfo: aggregate the underlying values.
        if isinstance(old, ValueWithInfo) and isinstance(new, ValueWithInfo):
            aggregated_value = self.default_aggregation(old.value(), new.value())
            return ValueWithInfo(aggregated_value, old.info())

        # Fallback: try using the + operator.
        try:
            return old + new
        except Exception as e:
            logger.debug(
                f"Aggregation failed for values {old} and {new}: {e}; using list fallback."
            )
            return [old, new]

    def cat(self, external_cache):
        """
        Merges the current cache with an external cache using the registered
        aggregation strategies (or the default if none is registered).

        If the cache is empty, each key is initialized using the aggregator with old=None.
        Otherwise, keys must match exactly between the two caches.
        """
        if not isinstance(external_cache, type(self)):
            raise TypeError("external_cache must be an instance of ActivationCache")

        # If in deferred mode, store the external cache for later aggregation.
        if isinstance(self.deferred_cache, list):
            self.deferred_cache.append(external_cache)
            return

        # Case 1: If self.cache is empty, initialize each key using the aggregator.
        if not self.cache:
            for key, new_value in external_cache.cache.items():
                aggregator = self._get_aggregation_strategy(key)
                self.cache[key] = aggregator(None, new_value)
            return

        # Case 2: Ensure both caches have the same keys.
        self_keys = set(self.cache.keys())
        external_keys = set(external_cache.cache.keys())
        if self_keys != external_keys:
            raise ValueError(
                f"Key mismatch: self has {self_keys - external_keys}, external has {external_keys - self_keys}"
            )

        # Case 3: Aggregate matching keys.
        for key in self.cache:
            aggregator = self._get_aggregation_strategy(key)
            try:
                self.cache[key] = aggregator(self.cache[key], external_cache.cache[key])
            except Exception as e:
                logger.error(f"Error aggregating key '{key}': {e}")
                self.cache[key] = [self.cache[key], external_cache.cache[key]]

    @contextlib.contextmanager
    def deferred_mode(self):
        """
        Context manager for deferred aggregation. Instead of merging
        immediately when calling `cat`, external caches are stored and then
        aggregated once the context is exited.
        """
        original_deferred = self.deferred_cache
        self.deferred_cache = []
        try:
            yield self
            for ext_cache in self.deferred_cache:
                self.cat(ext_cache)
        finally:
            self.deferred_cache = original_deferred

    def add_with_info(self, key: str, value, info: str):
        """
        Wraps a value (e.g. a tensor) with additional info and stores it in the cache.
        """
        wrapped = ValueWithInfo(value, info)
        self[key] = wrapped

    def add_metadata(
        self, target_token_positions, model_name: str, extraction_config, interventions
    ):
        """
        Adds metadata to the cache for future reference.
        """
        self.cache["metadata"] = {
            "target_token_positions": target_token_positions,
            "model_name": model_name,
            "extraction_config": extraction_config,
            "interventions": interventions,
        }

    def map_to_dict(self, key: str) -> dict:
        """
        Maps the cache values to a dictionary based on mapping_index.
        """
        if self.cache[key] is None:
            logger.error(f"Key {key} not found in cache.")

        elif not isinstance(self.cache[key], torch.Tensor):
            logger.error(f"Value for key {key} is not a tensor.")

        mapping_index = self.cache["mapping_index"]

        return_dict = {}
        for key_map, value_map in mapping_index.items():
            return_dict[key_map] = self.cache[key][:, value_map].squeeze()

        return return_dict

    def memory_size(self, key: Optional[str] = None) -> str:
        """
        Returns the memory size of the cache in bytes.
        """
        if key is not None:
            if key not in self.cache:
                return "Not present: 0 B"
            value = self.cache[key]
            if isinstance(value, torch.Tensor):
                total_size = value.element_size() * value.nelement()
            elif isinstance(value, dict):
                total_size = 0
                for v in value.values():
                    if isinstance(v, torch.Tensor):
                        total_size += v.element_size() * v.nelement()
            elif isinstance(value, list):
                total_size = 0
                for v in value:
                    if isinstance(v, torch.Tensor):
                        total_size += v.element_size() * v.nelement()
                    else:
                        logger.warning(f"Unknown type in list for key {key}: {type(v)}")
            else:
                logger.warning(f"Unknown type for key {key}: {type(value)}")
                return "0 B"
        else:
            total_size = 0
            for key, value in self.cache.items():
                if isinstance(value, torch.Tensor):
                    total_size += value.element_size() * value.nelement()
                elif isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, torch.Tensor):
                            total_size += v.element_size() * v.nelement()
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, torch.Tensor):
                            total_size += v.element_size() * v.nelement()
                        else:
                            logger.warning(
                                f"Unknown type in list for key {key}: {type(v)}"
                            )
                else:
                    logger.warning(f"Unknown type for key {key}: {type(value)}")

        # depending of the size, return the size in KB, MB or GB
        # prioritize the size in GB, so 0.1 GB is 100 MB
        if total_size > 1e9:
            total_size /= 1e9
            return f"{total_size:.2f} GB"
        elif total_size > 1e6:
            total_size /= 1e6
            return f"{total_size:.2f} MB"
        elif total_size > 1e3:
            total_size /= 1e3
            return f"{total_size:.2f} KB"
        else:
            total_size /= 1e3
            return f"{total_size:.2f} B"

    def memory_tree(self, print_tree: bool = False, grouped_tree: bool = False) -> dict:
        """
        Print a tree of the memory size of the cache.

        Args:
            print_tree (bool): If True, print the tree to the console.
            grouped_tree (bool): If True, group similar keys (e.g., resid_out_0, resid_out_1 -> resid_out).

        Returns:
            dict: A dictionary of the memory sizes.
        """
        tree = {}
        for key, value in self.cache.items():
            if isinstance(value, torch.Tensor):
                tree[key] = self.memory_size(key)
            elif isinstance(value, dict):
                tree[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        tree[key][k] = self.memory_size(k)
                    else:
                        tree[key][k] = "Unknown type"
            elif isinstance(value, list):
                tree[key] = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        tree[key].append(self.memory_size(key))
                    else:
                        tree[key].append("Unknown type")
            else:
                tree[key] = "Unknown type"

        if grouped_tree:
            grouped = {}
            # Define regex patterns for grouping
            patterns = {
                r"resid_out_\d+": "resid_out",
                r"resid_in_\d+": "resid_in",
                r"resid_mid_\d+": "resid_mid",
                r"attn_in_\d+": "attn_in",
                r"attn_out_\d+": "attn_out",
                r"avg_attn_pattern_L\d+H\d+": lambda m: f"avg_attn_pattern_L{m.group(1)}",
                r"pattern_L(\d+)H\d+": lambda m: f"pattern_L{m.group(1)}",
                r"head_out_\d+": "head_out",
                r"mlp_out_\d+": "mlp_out",
                r"head_values_L(\d+)H\d+": lambda m: f"head_values_L{m.group(1)}",
                r"head_keys_L(\d+)H\d+": lambda m: f"head_keys_L{m.group(1)}",
                r"head_queries_L(\d+)H\d+": lambda m: f"head_queries_L{m.group(1)}",
                r"values_L(\d+)": lambda m: f"values_L{m.group(1)}",
                r"keys_L(\d+)": lambda m: f"keys_L{m.group(1)}",
                r"queries_L(\d+)": lambda m: f"queries_L{m.group(1)}"
                
            }

            for key, size in tree.items():
                # Skip metadata and other special keys
                if key == "metadata" or isinstance(size, (dict, list)):
                    grouped[key] = size
                    continue

                # Try to match the key with patterns
                matched = False
                for pattern, replacement in patterns.items():
                    match = re.match(pattern, key)
                    if match:
                        group_key = replacement
                        if callable(replacement):
                            group_key = replacement(match)

                        # Convert size string to float value for aggregation
                        size_value, unit = size.split()
                        size_value = float(size_value)

                        # Initialize group if not exists
                        if group_key not in grouped:
                            grouped[group_key] = {"size": 0.0, "unit": unit, "count": 0}

                        # Normalize units
                        if unit == "KB" and grouped[group_key]["unit"] == "MB":
                            size_value /= 1000
                        elif unit == "B" and grouped[group_key]["unit"] == "KB":
                            size_value /= 1000
                        elif unit == "B" and grouped[group_key]["unit"] == "MB":
                            size_value /= 1000000
                        elif unit == "MB" and grouped[group_key]["unit"] == "KB":
                            grouped[group_key]["size"] /= 1000
                            grouped[group_key]["unit"] = "MB"
                        elif unit == "MB" and grouped[group_key]["unit"] == "B":
                            grouped[group_key]["size"] /= 1000000
                            grouped[group_key]["unit"] = "MB"
                        elif unit == "GB" and grouped[group_key]["unit"] != "GB":
                            # Convert all to GB
                            if grouped[group_key]["unit"] == "MB":
                                grouped[group_key]["size"] /= 1000
                            elif grouped[group_key]["unit"] == "KB":
                                grouped[group_key]["size"] /= 1000000
                            elif grouped[group_key]["unit"] == "B":
                                grouped[group_key]["size"] /= 1000000000
                            grouped[group_key]["unit"] = "GB"
                        elif grouped[group_key]["unit"] == "GB" and unit != "GB":
                            # Convert size_value to GB
                            if unit == "MB":
                                size_value /= 1000
                            elif unit == "KB":
                                size_value /= 1000000
                            elif unit == "B":
                                size_value /= 1000000000

                        # Aggregate sizes
                        grouped[group_key]["size"] += size_value
                        grouped[group_key]["count"] += 1
                        matched = True
                        break

                # If no pattern matched, keep the original key
                if not matched:
                    grouped[key] = size

            # Format the grouped sizes back to strings
            for key, info in grouped.items():
                if isinstance(info, dict) and "size" in info:
                    grouped[key] = (
                        f"{info['size']:.2f} {info['unit']} ({info['count']} items)"
                    )

            tree = grouped

        if print_tree:
            print("-" * 4 + " Activation Cache Memory Tree" + " -" * 4)
            print("Total size: ", self.memory_size())
            for key, value in tree.items():
                print(f"   - {key}: {value}")

        return tree
