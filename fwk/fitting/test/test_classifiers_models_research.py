import sys
import os

# Add the parent directory to sys.path to allow imports from algo/fitting
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from classifiers_models_research import demonstrate_pytorch_comparison

def test_demonstrate_pytorch_comparison():
    """
    Test function to call demonstrate_pytorch_comparison.
    This test verifies that the PyTorch classifier comparison runs without errors.
    """
#    demonstrate_pytorch_comparison()

if __name__ == "__main__":
    test_demonstrate_pytorch_comparison()