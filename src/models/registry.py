"""Model registry — factory function dispatching on architecture name.

Used by ``src/training/trainer.py`` to support multiple architectures
(PConv U-Net, vanilla U-Net, gated U-Net) under one Trainer.  Adding a
new architecture is a 4-line change here plus a new file in ``src/models/``.

Usage
-----

>>> import yaml
>>> from src.models.registry import build_model
>>> cfg = yaml.safe_load(open("configs/experiment_configs/pconv_unet.yaml"))
>>> model = build_model(cfg)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


# Supported architecture identifiers (used in ``cfg["model"]["arch"]``)
SUPPORTED_ARCHS: tuple[str, ...] = ("pconv_unet", "unet_baseline", "gated_unet")


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Instantiate the model selected by ``cfg["model"]["arch"]``.

    All returned models share the same forward signature::

        forward(image: Tensor[B, C_in, H, W],
                mask:  Tensor[B,    1, H, W])
            -> (output: Tensor[B, C_out, H, W],
                mask:   Tensor[B,    1, H, W])

    The trailing mask is the PConv-updated mask for ``pconv_unet`` and
    simply the input mask passed through for the two non-PConv baselines
    — the trainer ignores it for non-PConv models, so the convention is
    purely for interface uniformity.

    Args:
        cfg: Parsed training config dict.  Must contain a ``model`` block
            with at least ``arch``, ``in_channels``, ``out_channels``.

    Returns:
        An ``nn.Module`` instance, on CPU (the caller is responsible for
        moving it to the desired device).

    Raises:
        KeyError: If ``cfg["model"]["arch"]`` is missing.
        ValueError: If ``arch`` is not one of ``SUPPORTED_ARCHS``.
    """
    model_cfg = cfg["model"]
    arch = str(model_cfg["arch"]).lower()
    in_channels = int(model_cfg.get("in_channels", 3))
    out_channels = int(model_cfg.get("out_channels", 3))
    verbose = bool(model_cfg.get("verbose", True))

    if arch == "pconv_unet":
        from src.models.pconv_unet import PConvUNet
        return PConvUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            verbose=verbose,
        )

    if arch == "unet_baseline":
        from src.models.unet_baseline import VanillaUNet
        return VanillaUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            verbose=verbose,
        )

    if arch == "gated_unet":
        from src.models.gated_unet import GatedUNet
        return GatedUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            verbose=verbose,
        )

    raise ValueError(
        f"Unknown model arch: '{arch}'. Supported: {SUPPORTED_ARCHS}"
    )


if __name__ == "__main__":
    # Smoke test — dispatch to each arch with a minimal config
    print("=" * 60)
    print("Model registry — smoke test")
    print("=" * 60)
    for arch in SUPPORTED_ARCHS:
        cfg = {"model": {"arch": arch, "in_channels": 3, "out_channels": 3, "verbose": False}}
        try:
            m = build_model(cfg)
            n = sum(p.numel() for p in m.parameters())
            print(f"  {arch:16s} → {type(m).__name__:14s} ({n:,} params)")
        except ImportError as e:
            print(f"  {arch:16s} → IMPORT ERROR: {e}")
    print("=" * 60)
