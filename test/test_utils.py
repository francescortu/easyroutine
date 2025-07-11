import unittest
import tempfile
import os
from unittest.mock import patch
from easyroutine.utils import path_to_parents, path_to_relative


class TestUtils(unittest.TestCase):
    """Test suite for easyroutine.utils module."""

    def setUp(self):
        """Set up test environment with a temporary directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Create a nested directory structure for testing
        self.test_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a parallel directory for relative path testing
        self.parallel_dir = os.path.join(self.temp_dir, "level1", "parallel")
        os.makedirs(self.parallel_dir, exist_ok=True)

    def tearDown(self):
        """Clean up by returning to original directory."""
        os.chdir(self.original_cwd)

    def test_path_to_parents_single_level(self):
        """Test going up one directory level."""
        # Start in the deepest directory
        os.chdir(self.test_dir)
        initial_dir = os.getcwd()
        
        with patch('builtins.print') as mock_print:
            path_to_parents(1)
            
        # Should be one level up
        expected_dir = os.path.dirname(initial_dir)
        self.assertEqual(os.getcwd(), expected_dir)
        
        # Check that print was called with the correct message
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("Changed working directory to:", args)
        self.assertIn(expected_dir, args)

    def test_path_to_parents_multiple_levels(self):
        """Test going up multiple directory levels."""
        # Start in the deepest directory
        os.chdir(self.test_dir)
        initial_dir = os.getcwd()
        
        with patch('builtins.print') as mock_print:
            path_to_parents(2)
            
        # Should be two levels up
        expected_dir = os.path.dirname(os.path.dirname(initial_dir))
        self.assertEqual(os.getcwd(), expected_dir)
        
        # Check that print was called
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("Changed working directory to:", args)
        self.assertIn(expected_dir, args)

    def test_path_to_parents_default_level(self):
        """Test default behavior (going up one level)."""
        os.chdir(self.test_dir)
        initial_dir = os.getcwd()
        
        with patch('builtins.print') as mock_print:
            path_to_parents()  # Default to 1 level
            
        expected_dir = os.path.dirname(initial_dir)
        self.assertEqual(os.getcwd(), expected_dir)
        mock_print.assert_called_once()

    def test_path_to_parents_zero_levels(self):
        """Test edge case of going up zero levels."""
        os.chdir(self.test_dir)
        initial_dir = os.getcwd()
        
        with patch('builtins.print') as mock_print:
            path_to_parents(0)
            
        # Based on the implementation, even with 0 levels, it goes up one level first
        # Then the loop for additional levels doesn't run
        expected_dir = os.path.dirname(initial_dir)
        self.assertEqual(os.getcwd(), expected_dir)

    def test_path_to_relative_valid_path(self):
        """Test changing to a valid relative path."""
        # Start in level2 directory
        level2_dir = os.path.dirname(self.test_dir)
        os.chdir(level2_dir)
        
        with patch('builtins.print') as mock_print:
            path_to_relative("level3")
            
        # Should now be in level3
        self.assertEqual(os.getcwd(), self.test_dir)
        
        # Check print output
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("Changed working directory to:", args)
        self.assertIn(self.test_dir, args)

    def test_path_to_relative_nested_path(self):
        """Test changing to a nested relative path."""
        # Start in temp_dir
        os.chdir(self.temp_dir)
        
        with patch('builtins.print') as mock_print:
            path_to_relative(os.path.join("level1", "level2"))
            
        # Should now be in level2
        expected_dir = os.path.join(self.temp_dir, "level1", "level2")
        self.assertEqual(os.getcwd(), expected_dir)
        mock_print.assert_called_once()

    def test_path_to_relative_current_directory(self):
        """Test changing to current directory (edge case)."""
        os.chdir(self.test_dir)
        initial_dir = os.getcwd()
        
        with patch('builtins.print') as mock_print:
            path_to_relative(".")
            
        # Should stay in the same directory
        self.assertEqual(os.getcwd(), initial_dir)
        mock_print.assert_called_once()

    def test_path_to_relative_parent_directory(self):
        """Test using relative path to go to parent."""
        os.chdir(self.test_dir)
        
        with patch('builtins.print') as mock_print:
            path_to_relative("..")
            
        # Should be in parent directory
        expected_dir = os.path.dirname(self.test_dir)
        self.assertEqual(os.getcwd(), expected_dir)
        mock_print.assert_called_once()

    def test_path_to_relative_invalid_path(self):
        """Test behavior with non-existent relative path."""
        os.chdir(self.temp_dir)
        
        # This should raise an exception since the path doesn't exist
        with self.assertRaises(FileNotFoundError):
            path_to_relative("nonexistent_directory")

    def test_path_operations_integration(self):
        """Test combining path_to_relative and path_to_parents operations."""
        # Start in temp_dir, go to nested directory, then back up
        os.chdir(self.temp_dir)
        
        # Go to nested path
        with patch('builtins.print'):
            path_to_relative(os.path.join("level1", "level2", "level3"))
        
        self.assertEqual(os.getcwd(), self.test_dir)
        
        # Go back up two levels
        with patch('builtins.print'):
            path_to_parents(2)
        
        expected_dir = os.path.join(self.temp_dir, "level1")
        self.assertEqual(os.getcwd(), expected_dir)


if __name__ == "__main__":
    unittest.main()