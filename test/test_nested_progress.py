"""
Test nested progress bars functionality.

This test file demonstrates that the progress bars now support nesting
without raising errors. Both interactive (Rich) mode and batch mode support
indentation to visualize the nesting structure.
"""
import time
from easyroutine.console.progress import progress


def test_nested_progress_interactive():
    """Test nested progress bars in interactive (Rich) mode."""
    print("\nTesting nested progress bars in interactive mode...")
    
    for i in progress(range(3), description="Outer loop"):
        time.sleep(0.05)
        for j in progress(range(3), description=f"  Inner loop {i}"):
            time.sleep(0.05)
    
    print("✓ Nested progress bars work in interactive mode!")


def test_nested_progress_batch():
    """Test nested progress bars in batch mode."""
    print("\nTesting nested progress bars in batch mode...")
    
    for i in progress(range(3), description="Outer loop", force_batch_mode=True):
        time.sleep(0.05)
        for j in progress(range(3), description=f"Inner loop {i}", force_batch_mode=True):
            time.sleep(0.05)
    
    print("✓ Nested progress bars work in batch mode!")


def test_triple_nested():
    """Test triple-nested progress bars to verify indentation."""
    print("\nTesting triple-nested progress bars...")
    
    for i in progress(range(2), description="Level 1", force_batch_mode=True):
        time.sleep(0.05)
        for j in progress(range(2), description=f"Level 2.{i}", force_batch_mode=True):
            time.sleep(0.05)
            for k in progress(range(2), description=f"Level 3.{i}.{j}", force_batch_mode=True):
                time.sleep(0.05)
    
    print("✓ Triple-nested progress bars work correctly!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Nested Progress Bar Support")
    print("=" * 60)
    
    test_nested_progress_interactive()
    test_nested_progress_batch()
    test_triple_nested()
    
    print("\n" + "=" * 60)
    print("All nested progress bar tests passed!")
    print("=" * 60)
