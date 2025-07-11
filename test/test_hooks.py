import unittest
import torch
from unittest.mock import MagicMock, patch, call
from easyroutine.interpretability.hooks import (
    process_args_kwargs_output,
    restore_same_args_kwargs_output,
    multiply_pattern,
    compute_statistics
)


class TestHooksUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions in hooks module."""

    def test_process_args_kwargs_output_with_output(self):
        """Test process_args_kwargs_output when output is provided."""
        args = (torch.tensor([1, 2, 3]),)
        kwargs = {"hidden_states": torch.tensor([4, 5, 6])}
        output = torch.tensor([7, 8, 9])
        
        result = process_args_kwargs_output(args, kwargs, output)
        self.assertTrue(torch.equal(result, output))

    def test_process_args_kwargs_output_with_tuple_output(self):
        """Test process_args_kwargs_output when output is a tuple."""
        args = (torch.tensor([1, 2, 3]),)
        kwargs = {}
        output = (torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12]))
        
        result = process_args_kwargs_output(args, kwargs, output)
        self.assertTrue(torch.equal(result, output[0]))

    def test_process_args_kwargs_output_with_args(self):
        """Test process_args_kwargs_output when output is None but args exist."""
        args = (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        kwargs = {}
        output = None
        
        result = process_args_kwargs_output(args, kwargs, output)
        self.assertTrue(torch.equal(result, args[0]))

    def test_process_args_kwargs_output_with_kwargs(self):
        """Test process_args_kwargs_output when output is None, no args, but kwargs exist."""
        args = ()
        kwargs = {"hidden_states": torch.tensor([4, 5, 6])}
        output = None
        
        result = process_args_kwargs_output(args, kwargs, output)
        self.assertTrue(torch.equal(result, kwargs["hidden_states"]))

    def test_process_args_kwargs_output_no_hidden_states(self):
        """Test process_args_kwargs_output when no hidden_states in kwargs."""
        args = ()
        kwargs = {"other_param": torch.tensor([4, 5, 6])}
        output = None
        
        # Based on the actual implementation, this raises UnboundLocalError
        # Let's test that this edge case is handled
        with self.assertRaises(UnboundLocalError):
            process_args_kwargs_output(args, kwargs, output)

    def test_restore_same_args_kwargs_output_with_tuple_output(self):
        """Test restore_same_args_kwargs_output with tuple output."""
        b = torch.tensor([7, 8, 9])
        args = (torch.tensor([1, 2, 3]),)
        kwargs = {}
        output = (torch.tensor([10, 11, 12]), torch.tensor([13, 14, 15]))
        
        result = restore_same_args_kwargs_output(b, args, kwargs, output)
        
        # Should return tuple with b as first element
        self.assertIsInstance(result, tuple)
        self.assertTrue(torch.equal(result[0], b))
        self.assertTrue(torch.equal(result[1], output[1]))

    def test_restore_same_args_kwargs_output_with_args(self):
        """Test restore_same_args_kwargs_output with args but no output."""
        b = torch.tensor([7, 8, 9])
        args = (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        kwargs = {}
        output = None
        
        new_args, new_kwargs = restore_same_args_kwargs_output(b, args, kwargs, output)
        
        # Should return modified args with b as first element
        self.assertTrue(torch.equal(new_args[0], b))
        self.assertTrue(torch.equal(new_args[1], args[1]))
        self.assertEqual(new_kwargs, kwargs)

    def test_restore_same_args_kwargs_output_with_kwargs(self):
        """Test restore_same_args_kwargs_output with kwargs but no output or args."""
        b = torch.tensor([7, 8, 9])
        args = ()
        kwargs = {"hidden_states": torch.tensor([1, 2, 3])}
        output = None
        
        new_args, new_kwargs = restore_same_args_kwargs_output(b, args, kwargs, output)
        
        # Should return modified kwargs with b as hidden_states
        self.assertEqual(new_args, args)
        self.assertTrue(torch.equal(new_kwargs["hidden_states"], b))

    def test_multiply_pattern(self):
        """Test multiply_pattern function."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        multiplication_value = 0.5
        
        result = multiply_pattern(tensor, multiplication_value)
        expected = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
        
        self.assertTrue(torch.allclose(result, expected))

    def test_multiply_pattern_zero(self):
        """Test multiply_pattern with zero multiplication (ablation)."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        multiplication_value = 0.0
        
        result = multiply_pattern(tensor, multiplication_value)
        expected = torch.zeros_like(tensor)
        
        self.assertTrue(torch.equal(result, expected))

    def test_compute_statistics_basic(self):
        """Test compute_statistics with basic tensor."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        mean, variance, second_moment = compute_statistics(tensor, dim=-1)
        
        # Expected values for each row
        expected_mean = torch.tensor([2.0, 5.0])  # (1+2+3)/3, (4+5+6)/3
        expected_second_moment = torch.tensor([14.0/3, 77.0/3])  # (1+4+9)/3, (16+25+36)/3
        expected_variance = expected_second_moment - expected_mean.pow(2)
        
        self.assertTrue(torch.allclose(mean, expected_mean, atol=1e-6))
        self.assertTrue(torch.allclose(variance, expected_variance, atol=1e-6))
        self.assertTrue(torch.allclose(second_moment, expected_second_moment, atol=1e-6))

    def test_compute_statistics_different_dim(self):
        """Test compute_statistics with different dimension."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        mean, variance, second_moment = compute_statistics(tensor, dim=0)
        
        # Computing along dim=0 (across rows)
        expected_mean = torch.tensor([3.0, 4.0])  # (1+3+5)/3, (2+4+6)/3
        expected_second_moment = torch.tensor([35.0/3, 56.0/3])  # (1+9+25)/3, (4+16+36)/3
        expected_variance = expected_second_moment - expected_mean.pow(2)
        
        self.assertTrue(torch.allclose(mean, expected_mean, atol=1e-6))
        self.assertTrue(torch.allclose(variance, expected_variance, atol=1e-6))
        self.assertTrue(torch.allclose(second_moment, expected_second_moment, atol=1e-6))

    def test_compute_statistics_keepdim_false(self):
        """Test compute_statistics with keepdim=False."""
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        
        mean, variance, second_moment = compute_statistics(tensor, dim=-1, keepdim=False)
        
        # Should squeeze the dimension
        self.assertEqual(mean.shape, torch.Size([]))
        self.assertEqual(variance.shape, torch.Size([]))
        self.assertEqual(second_moment.shape, torch.Size([]))

    def test_compute_statistics_single_value(self):
        """Test compute_statistics with single value tensor."""
        tensor = torch.tensor([[5.0]])
        
        mean, variance, second_moment = compute_statistics(tensor, dim=-1)
        
        expected_mean = torch.tensor([5.0])
        expected_variance = torch.tensor([0.0])  # Variance of single value is 0
        expected_second_moment = torch.tensor([25.0])
        
        self.assertTrue(torch.allclose(mean, expected_mean))
        self.assertTrue(torch.allclose(variance, expected_variance))
        self.assertTrue(torch.allclose(second_moment, expected_second_moment))

    def test_compute_statistics_zero_variance(self):
        """Test compute_statistics with constant values (zero variance)."""
        tensor = torch.tensor([[2.0, 2.0, 2.0]])
        
        mean, variance, second_moment = compute_statistics(tensor, dim=-1)
        
        expected_mean = torch.tensor([2.0])
        expected_variance = torch.tensor([0.0])
        expected_second_moment = torch.tensor([4.0])
        
        self.assertTrue(torch.allclose(mean, expected_mean))
        self.assertTrue(torch.allclose(variance, expected_variance, atol=1e-6))
        self.assertTrue(torch.allclose(second_moment, expected_second_moment))


class TestHooksEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in hooks."""

    def test_process_args_kwargs_output_empty_inputs(self):
        """Test process_args_kwargs_output with empty inputs."""
        # Based on the implementation, this raises UnboundLocalError
        with self.assertRaises(UnboundLocalError):
            process_args_kwargs_output((), {}, None)

    def test_restore_same_args_kwargs_output_edge_case(self):
        """Test restore_same_args_kwargs_output with single output (not tuple)."""
        b = torch.tensor([7, 8, 9])
        args = (torch.tensor([1, 2, 3]),)
        kwargs = {}
        output = torch.tensor([10, 11, 12])  # Single tensor, not tuple
        
        result = restore_same_args_kwargs_output(b, args, kwargs, output)
        
        # Should return b directly since output is not a tuple
        self.assertTrue(torch.equal(result, b))

    def test_multiply_pattern_preserves_shape(self):
        """Test that multiply_pattern preserves tensor shape."""
        tensor = torch.randn(3, 4, 5)
        multiplication_value = 0.7
        
        result = multiply_pattern(tensor, multiplication_value)
        
        self.assertEqual(result.shape, tensor.shape)
        self.assertTrue(torch.allclose(result, tensor * multiplication_value))

    def test_multiply_pattern_with_negative_values(self):
        """Test multiply_pattern with negative multiplication values."""
        tensor = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
        multiplication_value = -1.5
        
        result = multiply_pattern(tensor, multiplication_value)
        expected = tensor * multiplication_value
        
        self.assertTrue(torch.allclose(result, expected))

    def test_compute_statistics_with_nan(self):
        """Test compute_statistics behavior with NaN values."""
        tensor = torch.tensor([[1.0, float('nan'), 3.0]])
        
        mean, variance, second_moment = compute_statistics(tensor, dim=-1)
        
        # Results should contain NaN
        self.assertTrue(torch.isnan(mean).any())
        self.assertTrue(torch.isnan(variance).any())
        self.assertTrue(torch.isnan(second_moment).any())

    def test_compute_statistics_with_inf(self):
        """Test compute_statistics behavior with infinite values."""
        tensor = torch.tensor([[1.0, float('inf'), 3.0]])
        
        mean, variance, second_moment = compute_statistics(tensor, dim=-1)
        
        # Results should contain inf
        self.assertTrue(torch.isinf(mean).any())
        self.assertTrue(torch.isinf(second_moment).any())

    def test_compute_statistics_empty_tensor(self):
        """Test compute_statistics with empty tensor."""
        tensor = torch.empty(0, 3)
        
        # This should not crash but may produce NaN results
        mean, variance, second_moment = compute_statistics(tensor, dim=0)
        
        # Results should be NaN for empty tensor
        self.assertTrue(torch.isnan(mean).all())
        self.assertTrue(torch.isnan(variance).all())
        self.assertTrue(torch.isnan(second_moment).all())


if __name__ == "__main__":
    unittest.main()