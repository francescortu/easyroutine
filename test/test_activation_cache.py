import os
import unittest
import torch
from easyroutine.interpretability.activation_cache import ActivationCache, ValueWithInfo

# For testing purposes, override just_old to keep the old value.
def just_old(old, new):
    return new if old is None else old

class TestActivationCache(unittest.TestCase):
    def setUp(self):
        """
        Initialize two ActivationCache objects and populate them with sample data.
        """
        self.cache1 = ActivationCache()
        self.cache2 = ActivationCache()

        # Set up standard keys
        self.cache1["values_0"] = torch.tensor([1, 2])
        self.cache1["mapping_index"] = [0, 1]

        self.cache2["values_0"] = torch.tensor([3, 4])
        self.cache2["mapping_index"] = [2, 3]

        # Ensure that the aggregator for mapping_index uses the desired behavior:
        self.cache1.register_aggregation("mapping_index", just_old)

    def test_cat_standard(self):
        """
        Test merging two caches using default aggregation:
         - For tensors ("values_0") aggregation should attempt to torch.cat.
         - For "mapping_index" the custom aggregator should keep the original value.
        """
        self.cache1.cat(self.cache2)
        # Expect torch.cat along dim=0 to yield tensor([1, 2, 3, 4])
        self.assertTrue(
            torch.equal(self.cache1["values_0"], torch.tensor([1, 2, 3, 4])),
            "Tensor aggregation (torch.cat) did not yield expected result.",
        )
        # The aggregator for mapping_index (just_old) should keep the original list.
        self.assertEqual(
            self.cache1["mapping_index"], [0, 1],
            "Custom aggregator for mapping_index did not keep the original value.",
        )

    def test_register_aggregation(self):
        """
        Test custom aggregation strategies.
        Here we register an aggregator for keys starting with "values_"
        that stacks the two tensors instead of concatenating them.
        """
        # Register a new aggregator that stacks tensors.
        self.cache1.register_aggregation(
            "values_0", 
            lambda old, new: new if old is None else torch.stack([old, new], dim=0)
        )
        self.cache1.cat(self.cache2)
        expected = torch.stack([torch.tensor([1, 2]), torch.tensor([3, 4])], dim=0)
        self.assertTrue(
            torch.equal(self.cache1["values_0"], expected),
            "Custom aggregator for 'values_0' did not stack tensors as expected.",
        )

    def test_deferred_mode(self):
        """
        Test that using deferred_mode aggregates external caches only once
        at the end of the context.
        """
        with self.cache1.deferred_mode():
            self.cache1.cat(self.cache2)
        self.assertTrue(
            torch.equal(self.cache1["values_0"], torch.tensor([1, 2, 3, 4])),
            "Deferred mode did not aggregate tensor values as expected.",
        )
        self.assertEqual(
            self.cache1["mapping_index"], [0, 1],
            "Deferred mode did not preserve mapping_index as expected.",
        )

    def test_key_mismatch(self):
        """
        Test that a ValueError is raised when merging caches with different keys.
        """
        self.cache2["extra_key"] = torch.tensor([5])
        with self.assertRaises(ValueError):
            self.cache1.cat(self.cache2)

    def test_empty_initialization(self):
        """
        Test that if a cache is empty, calling cat initializes it with the other cache’s values.
        """
        empty_cache = ActivationCache()
        empty_cache.cat(self.cache1)
        self.assertTrue(
            torch.equal(empty_cache["values_0"], torch.tensor([1, 2])),
            "Empty cache did not initialize the 'values_0' key correctly.",
        )
        self.assertEqual(
            empty_cache["mapping_index"], [0, 1],
            "Empty cache did not initialize the 'mapping_index' key correctly.",
        )

    def test_dynamic_switching_tensor(self):
        """
        Test dynamic switching: merging two tensors that cannot be concatenated/stacked
        should fall back to list aggregation.
        Here we use tensors with mismatched shapes.
        """
        cache_a = ActivationCache()
        cache_b = ActivationCache()

        cache_a["incompatible"] = torch.tensor([[1, 2, 3], [4, 5, 6]])
        cache_b["incompatible"] = torch.tensor([[7, 8], [9, 10]])  # shape differs in dim=1

        # Use the default aggregator (no custom aggregator registered for "incompatible")
        cache_a.cat(cache_b)
        result = cache_a["incompatible"]
        self.assertIsInstance(result, list, "Expected aggregation to fall back to a list.")
        self.assertEqual(len(result), 2, "Expected list aggregation to contain both tensors.")

    def test_aggregation_lists(self):
        """
        Test that aggregating lists simply concatenates them.
        """
        cache_a = ActivationCache()
        cache_b = ActivationCache()

        cache_a["list_key"] = [1, 2]
        cache_b["list_key"] = [3, 4]
        cache_a.cat(cache_b)
        self.assertEqual(
            cache_a["list_key"], [1, 2, 3, 4],
            "List aggregation did not yield the concatenated list.",
        )

    def test_aggregation_tuples(self):
        """
        Test that aggregating tuples falls back to converting them into a list.
        """
        cache_a = ActivationCache()
        cache_b = ActivationCache()

        cache_a["tuple_key"] = (1, 2)
        cache_b["tuple_key"] = (3, 4)
        cache_a.cat(cache_b)
        self.assertIsInstance(
            cache_a["tuple_key"], list,
            "Aggregation of tuples should return a list."
        )
        self.assertEqual(
            cache_a["tuple_key"], [(1, 2), (3, 4)],
            "Aggregated tuple values did not match expected output."
        )

    def test_value_with_info_aggregation(self):
        """
        Test that aggregating ValueWithInfo objects aggregates the underlying values.
        """
        cache_a = ActivationCache()
        cache_b = ActivationCache()

        v1 = ValueWithInfo(torch.tensor([1, 2]), "infoA")
        v2 = ValueWithInfo(torch.tensor([3, 4]), "infoA")
        cache_a["vwi"] = v1
        cache_b["vwi"] = v2
        cache_a.cat(cache_b)
        result = cache_a["vwi"]
        self.assertIsInstance(
            result, ValueWithInfo, "Result should be a ValueWithInfo instance."
        )
        # Expect that underlying tensor is concatenated.
        self.assertTrue(
            torch.equal(result.value(), torch.tensor([1, 2, 3, 4])),
            "ValueWithInfo aggregation did not concatenate the tensors as expected."
        )
        self.assertEqual(
            result.info(), "infoA",
            "ValueWithInfo aggregation did not preserve the info field."
        )

    def test_torch_save_and_load(self):
        """
        Test that ActivationCache can be saved and loaded with torch.save/torch.load.
        """
        tmp_folder = "tmp"
        os.makedirs(tmp_folder, exist_ok=True)
        filepath = os.path.join(tmp_folder, "test_cache.pth")
        try:
            torch.save(self.cache1, filepath)
        except Exception as e:
            self.fail(f"torch.save failed with exception: {e}")

        try:
            loaded_cache = torch.load(filepath)
        except Exception as e:
            self.fail(f"torch.load failed with exception: {e}")

        self.assertTrue(
            torch.equal(loaded_cache["values_0"], torch.tensor([1, 2])),
            "Loaded cache 'values_0' does not match expected tensor."
        )
        # Cleanup temporary file
        os.remove(filepath)

if __name__ == "__main__":
    unittest.main()
