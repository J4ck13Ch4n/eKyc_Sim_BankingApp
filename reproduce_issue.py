import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")
import sys
print("Starting imports...")
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
    import torch
    print(f"Torch version: {torch.__version__}")
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
except SystemExit as e:
    print(f"SystemExit during import: {e}")
