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
            self.cache1["mapping_index"],
            [0, 1],
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
            lambda old, new: new if old is None else torch.stack([old, new], dim=0),
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
            self.cache1["mapping_index"],
            [0, 1],
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
            empty_cache["mapping_index"],
            [0, 1],
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
        cache_b["incompatible"] = torch.tensor(
            [[7, 8], [9, 10]]
        )  # shape differs in dim=1

        # Use the default aggregator (no custom aggregator registered for "incompatible")
        cache_a.cat(cache_b)
        result = cache_a["incompatible"]
        self.assertIsInstance(
            result, list, "Expected aggregation to fall back to a list."
        )
        self.assertEqual(
            len(result), 2, "Expected list aggregation to contain both tensors."
        )

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
            cache_a["list_key"],
            [1, 2, 3, 4],
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
            cache_a["tuple_key"], list, "Aggregation of tuples should return a list."
        )
        self.assertEqual(
            cache_a["tuple_key"],
            [(1, 2), (3, 4)],
            "Aggregated tuple values did not match expected output.",
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
            "ValueWithInfo aggregation did not concatenate the tensors as expected.",
        )
        self.assertEqual(
            result.info(),
            "infoA",
            "ValueWithInfo aggregation did not preserve the info field.",
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
            "Loaded cache 'values_0' does not match expected tensor.",
        )
        # Cleanup temporary file
        os.remove(filepath)

    def test_memory_size_tensor(self):
        """
        Test the memory_size method for tensor values.
        """
        # Create a tensor with known size
        tensor_size = 1000  # Elements
        element_size = 4  # Bytes per element (float32)
        test_tensor = torch.randn(tensor_size, dtype=torch.float32)
        expected_size = tensor_size * element_size  # Total bytes

        cache = ActivationCache()
        cache["test_tensor"] = test_tensor

        # Get the memory size string
        memory_str = cache.memory_size("test_tensor")

        # Parse the string to extract the numeric value and unit
        value, unit = memory_str.split()
        value = float(value)

        # Check if the returned size is in the expected range (allowing for format variations)
        if unit == "KB":
            self.assertAlmostEqual(value, expected_size / 1e3, delta=0.1)
        elif unit == "B":
            self.assertAlmostEqual(value, expected_size, delta=1)

        # Test total memory size
        total_memory = cache.memory_size()
        self.assertIsInstance(total_memory, str, "Total memory size should be a string")
        self.assertTrue(len(total_memory) > 0, "Total memory size should not be empty")

    def test_memory_size_dict(self):
        """
        Test the memory_size method for dictionary values containing tensors.
        """
        cache = ActivationCache()
        dict_val = {
            "tensor1": torch.randn(100, dtype=torch.float32),
            "tensor2": torch.randn(200, dtype=torch.float32),
        }
        cache["dict_key"] = dict_val

        # Get memory size
        memory_str = cache.memory_size("dict_key")

        # Just verify that it returns a non-empty string
        self.assertIsInstance(
            memory_str, str, "Memory size for dict should be a string"
        )
        self.assertTrue(len(memory_str) > 0, "Memory size string should not be empty")

    def test_memory_size_list(self):
        """
        Test the memory_size method for list values containing tensors.
        """
        cache = ActivationCache()
        list_val = [
            torch.randn(100, dtype=torch.float32),
            torch.randn(200, dtype=torch.float32),
        ]
        cache["list_key"] = list_val

        # Get memory size
        memory_str = cache.memory_size("list_key")

        # Verify it returns a non-empty string
        self.assertIsInstance(
            memory_str, str, "Memory size for list should be a string"
        )
        self.assertTrue(len(memory_str) > 0, "Memory size string should not be empty")

    def test_memory_tree_basic(self):
        """
        Test the memory_tree method without grouping.
        """
        cache = ActivationCache()
        cache["resid_out_0"] = torch.randn(100, dtype=torch.float32)
        cache["resid_out_1"] = torch.randn(100, dtype=torch.float32)
        cache["attn_in_0"] = torch.randn(50, dtype=torch.float32)
        cache["mapping_index"] = {"input": 0, "output": 1}

        # Get memory tree without grouping
        tree = cache.memory_tree(print_tree=False, grouped_tree=False)

        # Verify the structure
        self.assertIsInstance(tree, dict, "Memory tree should be a dictionary")
        self.assertIn("resid_out_0", tree, "Memory tree should contain resid_out_0 key")
        self.assertIn("resid_out_1", tree, "Memory tree should contain resid_out_1 key")
        self.assertIn("attn_in_0", tree, "Memory tree should contain attn_in_0 key")

    def test_memory_tree_grouped(self):
        """
        Test the memory_tree method with grouping.
        """
        cache = ActivationCache()
        # Add tensors with pattern-matching names
        cache["resid_out_0"] = torch.randn(100, dtype=torch.float32)
        cache["resid_out_1"] = torch.randn(100, dtype=torch.float32)
        cache["resid_in_0"] = torch.randn(100, dtype=torch.float32)
        cache["resid_in_1"] = torch.randn(100, dtype=torch.float32)
        cache["attn_in_0"] = torch.randn(50, dtype=torch.float32)
        cache["pattern_L0H0"] = torch.randn(25, dtype=torch.float32)
        cache["pattern_L0H1"] = torch.randn(25, dtype=torch.float32)
        cache["pattern_L1H0"] = torch.randn(25, dtype=torch.float32)
        cache["mapping_index"] = {"input": 0, "output": 1}

        # Get memory tree with grouping
        tree = cache.memory_tree(print_tree=False, grouped_tree=True)

        # Verify the structure
        self.assertIsInstance(tree, dict, "Memory tree should be a dictionary")

        # Check that keys are grouped
        self.assertIn("resid_out", tree, "Grouped tree should contain resid_out key")
        self.assertIn("resid_in", tree, "Grouped tree should contain resid_in key")
        self.assertIn("attn_in", tree, "Grouped tree should contain attn_in key")
        self.assertIn("pattern_L0", tree, "Grouped tree should contain pattern_L0 key")
        self.assertIn("pattern_L1", tree, "Grouped tree should contain pattern_L1 key")

        # Check that the original keys are not present
        self.assertNotIn(
            "resid_out_0", tree, "Grouped tree should not contain resid_out_0 key"
        )
        self.assertNotIn(
            "resid_out_1", tree, "Grouped tree should not contain resid_out_1 key"
        )

        # Check that the formatted string contains count information
        resid_out_str = tree["resid_out"]
        self.assertIn(
            "(2 items)", resid_out_str, "Grouped tree should show count information"
        )

        # Check pattern grouping by layer
        pattern_l0_str = tree["pattern_L0"]
        self.assertIn("(2 items)", pattern_l0_str, "pattern_L0 should have 2 items")
        pattern_l1_str = tree["pattern_L1"]
        self.assertIn("(1 items)", pattern_l1_str, "pattern_L1 should have 1 item")

        # Check the metadata key is preserved
        self.assertIn("mapping_index", tree, "Metadata key should be preserved")

    def test_memory_tree_unit_conversion(self):
        """
        Test that the memory_tree method correctly handles unit conversions when grouping.
        """
        cache = ActivationCache()

        # Create tensors with significantly different sizes
        # Small tensor (KB range)
        cache["values_0"] = torch.randn(1000, dtype=torch.float32)  # ~4KB

        # Medium tensor (MB range)
        cache["values_1"] = torch.randn(500000, dtype=torch.float32)  # ~2MB

        # Get grouped memory tree
        tree = cache.memory_tree(print_tree=False, grouped_tree=True)

        # Check that values are grouped
        self.assertIn("values", tree, "Values should be grouped")

        # The combined size should be reported in some unit (B, KB, MB or GB)
        values_str = tree["values"]
        parts = values_str.split()

        # Extract numeric value and unit
        size_value = float(parts[0])
        unit = parts[1]

        # Verify the unit is one of the expected memory units
        self.assertTrue(
            unit in ["B", "KB", "MB", "GB"],
            f"Unit '{unit}' should be one of the expected memory units (B, KB, MB, GB)",
        )

        # Verify that count is correct
        self.assertIn("(2 items)", values_str, "Values should show count of 2 items")

    def test_memory_tree_print(self):
        """
        Test that the memory_tree method's print_tree parameter works correctly.
        This just checks that it doesn't raise an exception.
        """
        cache = ActivationCache()
        cache["resid_out_0"] = torch.randn(100, dtype=torch.float32)

        try:
            # Should print the tree without error
            cache.memory_tree(print_tree=True)
            # Also test with grouping
            cache.memory_tree(print_tree=True, grouped_tree=True)
        except Exception as e:
            self.fail(f"memory_tree with print_tree=True raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
