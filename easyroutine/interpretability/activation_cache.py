import re
import torch
import contextlib
from easyroutine.logger import logger
from typing import List, Union

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
            re.compile(r"pattern_L\dH\d+"),
            re.compile(r"values_\d+"),
            re.compile(r"input_ids"),
            re.compile(r"mapping_index"),
            re.compile(r"mlp_out_\d+"),
        )
        self.aggregation_strategies = {}
        # Register default aggregators for some keys
        self.register_aggregation("mapping_index", just_old)
        self.register_aggregation("offset", sublist)
        self.deferred_cache = False

    def __repr__(self) -> str:
        return f"ActivationCache(`{', '.join(self.cache.keys())}`)"

    def __str__(self) -> str:
        return f"ActivationCache({', '.join([f'{key}: {value}' for key, value in self.cache.items()])})"

    def __setitem__(self, key: str, value):
        if not any(pattern.match(key) for pattern in self.valid_keys):
            logger.warning(
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

    def cpu(self):
        self.to("cpu")

    def cuda(self):
        self.to("cuda")

    def register_aggregation(self, key_pattern, function):
        """
        Registers a custom aggregation function for keys that start with key_pattern.
        """
        logger.debug(f"Registering aggregation strategy for keys starting with '{key_pattern}'")
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
            logger.warning(
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
