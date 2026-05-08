"""Model architectures: PConvUNet (primary) + VanillaUNet, GatedUNet (baselines).

Public API
----------

* ``build_model(cfg)`` — factory dispatching on ``cfg["model"]["arch"]``.
* ``PConvUNet``       — primary inpainting model (Liu et al. 2018).
* ``VanillaUNet``     — vanilla U-Net baseline (mask concatenated as 4th channel).
* ``GatedUNet``       — gated convolution U-Net baseline (Yu et al. 2019).

The model classes are imported lazily on demand via ``build_model`` to
avoid pulling VGG weights or torchvision into memory unless a model is
actually instantiated.  Direct ``from src.models.<file> import <Class>``
also works.
"""

from src.models.registry import SUPPORTED_ARCHS, build_model

__all__ = ["build_model", "SUPPORTED_ARCHS"]
