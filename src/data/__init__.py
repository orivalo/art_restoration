"""Data loading, mask generation, and augmentation utilities.

The package intentionally avoids importing heavy optional dependencies at
module import time so lightweight modules can be executed with
``python -m`` without pulling in the full training stack.
"""

__all__ = ["InpaintingDataset", "MaskGenerator"]
