"""Shared model-loading helper for the Gradio demo.

Loads all three trained inpainting models (PConv U-Net, Vanilla U-Net,
Gated U-Net) from their ``best.pth`` checkpoints into evaluation mode on
the requested device.  Keeps a single source of truth for checkpoint
paths and configs so the demo and any verification script agree.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch import nn

# Project root (one level above demo/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import build_model  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Single source of truth for checkpoint and config locations
# ──────────────────────────────────────────────────────────────────────

MODEL_NAMES: tuple[str, ...] = ("pconv_unet", "unet_baseline", "gated_unet")

CHECKPOINTS: Dict[str, Path] = {
    name: PROJECT_ROOT / "outputs" / "checkpoints" / name / "best.pth"
    for name in MODEL_NAMES
}

CONFIGS: Dict[str, Path] = {
    name: PROJECT_ROOT / "configs" / "experiment_configs" / f"{name}.yaml"
    for name in MODEL_NAMES
}

DISPLAY_NAMES: Dict[str, str] = {
    "pconv_unet":    "PConv U-Net (Liu et al. 2018)",
    "unet_baseline": "Vanilla U-Net (mask-concat)",
    "gated_unet":    "Gated U-Net (DeepFillv2)",
}


def load_all_models(device: str | torch.device = "cpu") -> Dict[str, nn.Module]:
    """Load all three trained models in eval mode.

    Args:
        device: PyTorch device string or object.

    Returns:
        Mapping ``arch_name -> nn.Module`` with weights loaded and
        ``model.eval()`` already called.
    """
    device = torch.device(device)
    models: Dict[str, nn.Module] = {}

    for name in MODEL_NAMES:
        cfg_path = CONFIGS[name]
        ckpt_path = CHECKPOINTS[name]

        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["model"]["verbose"] = False

        model = build_model(cfg)
        load_checkpoint(ckpt_path, model=model, map_location=device, strict=True)
        model.to(device).eval()
        models[name] = model

    return models


if __name__ == "__main__":
    print("Loading all three models on CPU for smoke test...")
    ms = load_all_models("cpu")
    for k, m in ms.items():
        n = sum(p.numel() for p in m.parameters())
        print(f"  {k:16s} -> {type(m).__name__:14s}  ({n:,} params)")
    print("OK")
