"""Inpainting loss functions for PConv U-Net (Liu et al., ECCV 2018).

Implements the full five-term composite loss used to train the partial-
convolution U-Net on art inpainting:

    L_total  =  λ_valid · L1_valid
              + λ_hole  · L1_hole
              + λ_perc  · L_perceptual
              + λ_style · L_style
              + λ_tv    · L_tv

where:

* ``L1_valid``      — L1 distance on the *valid* (intact) region only.
* ``L1_hole``       — L1 distance on the *hole* (damaged) region only.
* ``L_perceptual``  — L1 distance between VGG-16 features of (output, GT)
                      and (composited, GT), summed over relu1_1, relu2_1,
                      relu3_1.
* ``L_style``       — L1 distance between Gram matrices of those same
                      VGG features (auto-correlation of style statistics).
* ``L_tv``          — Total Variation regulariser restricted to the hole
                      region of the *composited* output.

Default weights from Liu et al. 2018 (Places2):

    λ_valid = 1   λ_hole = 6   λ_perc = 0.05   λ_style = 120   λ_tv = 0.1

Pixel range convention
----------------------

All inputs are assumed to lie in ``[0, 1]`` (the ``PConvUNet`` produces
sigmoid output in that range, so callers must convert ground-truth from
the dataset's ``[-1, 1]`` to ``[0, 1]`` *before* invoking the loss).

Mask convention
---------------

``mask = 1`` → valid, ``mask = 0`` → hole (matches Liu et al. and our
``PartialConv2d`` implementation).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


# ──────────────────────────────────────────────────────────────────────────────
#  VGG-16 feature extractor (frozen)
# ──────────────────────────────────────────────────────────────────────────────


class VGG16FeatureExtractor(nn.Module):
    """Frozen VGG-16 feature extractor for perceptual / style losses.

    Returns intermediate activations after the ``relu1_1``, ``relu2_1`` and
    ``relu3_1`` layers.  Inputs are first re-normalised from ``[0, 1]`` to
    ImageNet statistics (mean/std) since the upstream weights expect that.

    All parameters are frozen (``requires_grad = False``) and the module is
    forced into ``eval()`` mode.  ``train()`` is overridden to be a no-op
    so that calling ``.train()`` on the parent ``InpaintingLoss`` does not
    accidentally re-enable BN / dropout updates inside VGG.
    """

    # ImageNet normalisation constants
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self) -> None:
        super().__init__()

        # Load pretrained VGG-16 (handle both new and legacy torchvision APIs)
        try:
            from torchvision.models import VGG16_Weights
            backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # Fallback for older torchvision versions
            backbone = vgg16(pretrained=True)  # type: ignore[arg-type]

        features = backbone.features

        # Slice indices in torchvision's VGG-16:
        #   features[0]  Conv2d(3, 64)
        #   features[1]  ReLU                    ← relu1_1
        #   features[2]  Conv2d(64, 64)
        #   features[3]  ReLU
        #   features[4]  MaxPool
        #   features[5]  Conv2d(64, 128)
        #   features[6]  ReLU                    ← relu2_1
        #   features[7]  Conv2d(128, 128)
        #   features[8]  ReLU
        #   features[9]  MaxPool
        #   features[10] Conv2d(128, 256)
        #   features[11] ReLU                    ← relu3_1
        self.slice_1 = nn.Sequential(*list(features[:2]))     # → relu1_1
        self.slice_2 = nn.Sequential(*list(features[2:7]))    # → relu2_1
        self.slice_3 = nn.Sequential(*list(features[7:12]))   # → relu3_1

        # Freeze every parameter
        for p in self.parameters():
            p.requires_grad_(False)
        super().eval()  # set BN / dropout (none here, but be safe) to eval

        # ImageNet normalisation buffers
        self.register_buffer(
            "_mean",
            torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True) -> "VGG16FeatureExtractor":
        """Override ``train()`` to keep VGG locked in ``eval`` mode."""
        return super().train(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract VGG features.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)`` with values in
               ``[0, 1]``.

        Returns:
            List ``[f1, f2, f3]`` of three feature tensors corresponding
            to ``relu1_1``, ``relu2_1``, ``relu3_1``.
        """
        x_norm = (x - self._mean) / self._std
        f1 = self.slice_1(x_norm)
        f2 = self.slice_2(f1)
        f3 = self.slice_3(f2)
        return [f1, f2, f3]


# ──────────────────────────────────────────────────────────────────────────────
#  Loss helpers
# ──────────────────────────────────────────────────────────────────────────────


def l1_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the L1 distance into valid-region and hole-region terms.

    Args:
        output: Prediction tensor ``(B, C, H, W)`` in ``[0, 1]``.
        target: Ground-truth tensor of the same shape.
        mask:   Binary mask ``(B, 1, H, W)`` — ``1 = valid``, ``0 = hole``.

    Returns:
        Tuple ``(l1_valid, l1_hole)`` of scalar tensors averaged over all
        pixels (both terms use the same denominator: ``B*C*H*W``).  This
        matches the formulation in Liu et al. 2018, Eq. (2)-(3).
    """
    diff = (output - target).abs()
    l1_valid = (diff * mask).mean()
    l1_hole = (diff * (1.0 - mask)).mean()
    return l1_valid, l1_hole


def perceptual_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    vgg_extractor: VGG16FeatureExtractor,
) -> torch.Tensor:
    """L1 distance between VGG features of ``output`` and ``target``.

    Sums the per-layer L1 losses over ``relu1_1``, ``relu2_1`` and
    ``relu3_1``.  Each term is averaged over all elements at that layer
    (``F.l1_loss`` default reduction).

    Args:
        output: Image tensor ``(B, 3, H, W)`` in ``[0, 1]``.
        target: Image tensor of the same shape.
        vgg_extractor: A frozen ``VGG16FeatureExtractor``.

    Returns:
        Scalar tensor — sum of three per-layer L1 distances.
    """
    feats_out = vgg_extractor(output)
    feats_tgt = vgg_extractor(target)
    return sum(F.l1_loss(fo, ft) for fo, ft in zip(feats_out, feats_tgt))  # type: ignore[return-value]


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute the (batch-wise) Gram matrix of feature activations.

    Args:
        features: Tensor of shape ``(B, C, H, W)``.

    Returns:
        Tensor of shape ``(B, C, C)`` — ``F·Fᵀ / (C·H·W)``.  The
        ``1 / (C*H*W)`` normalisation matches the original PConv paper
        and prevents the style loss from exploding at large feature maps.
    """
    b, c, h, w = features.shape
    feat_flat = features.view(b, c, h * w)
    gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2))
    return gram / (c * h * w)


def style_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    vgg_extractor: VGG16FeatureExtractor,
) -> torch.Tensor:
    """L1 distance between Gram matrices of VGG features.

    Args:
        output: Image tensor ``(B, 3, H, W)`` in ``[0, 1]``.
        target: Image tensor of the same shape.
        vgg_extractor: A frozen ``VGG16FeatureExtractor``.

    Returns:
        Scalar tensor — sum over three layers of L1(G(out), G(target)),
        each Gram matrix already normalised by ``C*H*W``.
    """
    feats_out = vgg_extractor(output)
    feats_tgt = vgg_extractor(target)
    return sum(
        F.l1_loss(gram_matrix(fo), gram_matrix(ft))
        for fo, ft in zip(feats_out, feats_tgt)
    )  # type: ignore[return-value]


def total_variation_loss(
    output: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Total Variation regulariser restricted to the hole region.

    Penalises adjacent-pixel differences only at hole positions, encouraging
    smoothness inside the inpainted area without affecting valid pixels.

    Args:
        output: Composited image ``(B, C, H, W)`` in ``[0, 1]`` —
                ``output * (1-mask) + target * mask``.
        mask:   Binary mask ``(B, 1, H, W)`` — ``1 = valid``, ``0 = hole``.

    Returns:
        Scalar tensor — mean of horizontal + vertical TV terms over hole
        positions.
    """
    hole = 1.0 - mask  # 1 inside hole, 0 elsewhere

    # Horizontal differences (pixel (i, j+1) − pixel (i, j))
    h_diff = (output[:, :, :, 1:] - output[:, :, :, :-1]).abs()
    h_mask = hole[:, :, :, 1:]                        # gate by destination pixel

    # Vertical differences (pixel (i+1, j) − pixel (i, j))
    v_diff = (output[:, :, 1:, :] - output[:, :, :-1, :]).abs()
    v_mask = hole[:, :, 1:, :]

    tv_h = (h_diff * h_mask).mean()
    tv_v = (v_diff * v_mask).mean()
    return tv_h + tv_v


# ──────────────────────────────────────────────────────────────────────────────
#  Composite InpaintingLoss
# ──────────────────────────────────────────────────────────────────────────────


class InpaintingLoss(nn.Module):
    """Composite five-term inpainting loss from Liu et al. 2018.

    The ``forward`` pass returns a dictionary with every individual loss
    term and the weighted total, so the trainer can log them separately.

    Args:
        lambda_valid:  Weight on ``L1_valid`` (default 1.0).
        lambda_hole:   Weight on ``L1_hole`` (default 6.0).
        lambda_perc:   Weight on perceptual loss (default 0.05).
        lambda_style:  Weight on style loss (default 120.0).
        lambda_tv:     Weight on total variation loss (default 0.1).
        vgg_extractor: Optional pre-constructed ``VGG16FeatureExtractor``.
            If ``None``, a fresh one is built (this is the recommended
            usage; sharing across loss instances is only useful for
            unit testing).

    Forward inputs:
        output: PConvUNet prediction ``(B, 3, H, W)``, sigmoid-clamped
            to ``[0, 1]``.
        target: Ground-truth image ``(B, 3, H, W)`` in ``[0, 1]``.
        mask:   Binary mask ``(B, 1, H, W)`` — ``1 = valid``, ``0 = hole``.

    Forward returns:
        ``dict`` with keys ``total``, ``l1_valid``, ``l1_hole``,
        ``perceptual``, ``style``, ``tv`` — each maps to a scalar tensor.
    """

    def __init__(
        self,
        lambda_valid: float = 1.0,
        lambda_hole: float = 6.0,
        lambda_perc: float = 0.05,
        lambda_style: float = 120.0,
        lambda_tv: float = 0.1,
        vgg_extractor: VGG16FeatureExtractor | None = None,
    ) -> None:
        super().__init__()

        self.lambda_valid = lambda_valid
        self.lambda_hole = lambda_hole
        self.lambda_perc = lambda_perc
        self.lambda_style = lambda_style
        self.lambda_tv = lambda_tv

        self.vgg = vgg_extractor if vgg_extractor is not None else VGG16FeatureExtractor()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute every loss term and the weighted total.

        The full loss computation runs inside an ``autocast(enabled=False)``
        context that forces fp32 even when the trainer's outer
        ``autocast(enabled=True)`` is active.  This is essential because the
        Gram-matrix style loss can overflow fp16 on a freshly-initialised
        model — fp32 keeps the loss numerically well-behaved while the
        model's forward pass continues to run in fp16 under outer AMP.
        Gradients flow back from fp32 → fp16 cleanly.

        Args:
            output: ``(B, 3, H, W)`` in ``[0, 1]``.
            target: ``(B, 3, H, W)`` in ``[0, 1]``.
            mask:   ``(B, 1, H, W)`` in ``{0, 1}``.

        Returns:
            Dict with the per-term scalars and ``total``.
        """
        # Force fp32 for the entire loss computation to prevent Gram-matrix
        # overflow under outer AMP autocast.  Inputs are cast up to fp32
        # explicitly so that all subsequent ops stay in fp32.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            output = output.float()
            target = target.float()
            mask = mask.float()

            # Composited image: keep valid pixels from GT, fill holes with prediction
            composited = output * (1.0 - mask) + target * mask

            # ── L1 split ─────────────────────────────────────────────────────
            l1_v, l1_h = l1_loss(output, target, mask)

            # ── VGG features for both raw and composited outputs ─────────────
            feats_out = self.vgg(output)
            feats_tgt = self.vgg(target)
            feats_comp = self.vgg(composited)

            # ── Perceptual loss: out vs target  +  comp vs target ───────────
            l_perc = sum(
                F.l1_loss(fo, ft) for fo, ft in zip(feats_out, feats_tgt)
            )
            l_perc = l_perc + sum(  # type: ignore[assignment]
                F.l1_loss(fc, ft) for fc, ft in zip(feats_comp, feats_tgt)
            )

            # ── Style loss: Gram-matrix L1 on both raw and composited ───────
            l_style = sum(
                F.l1_loss(gram_matrix(fo), gram_matrix(ft))
                for fo, ft in zip(feats_out, feats_tgt)
            )
            l_style = l_style + sum(  # type: ignore[assignment]
                F.l1_loss(gram_matrix(fc), gram_matrix(ft))
                for fc, ft in zip(feats_comp, feats_tgt)
            )

            # ── Total Variation on the composited output, hole-only ─────────
            l_tv = total_variation_loss(composited, mask)

            # ── Weighted sum ─────────────────────────────────────────────────
            total = (
                self.lambda_valid * l1_v
                + self.lambda_hole * l1_h
                + self.lambda_perc * l_perc
                + self.lambda_style * l_style
                + self.lambda_tv * l_tv
            )

        return {
            "total": total,
            "l1_valid": l1_v,
            "l1_hole": l1_h,
            "perceptual": l_perc,  # type: ignore[dict-item]
            "style": l_style,      # type: ignore[dict-item]
            "tv": l_tv,
        }


# ======================================================================
#  Smoke test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("InpaintingLoss — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Random batch (sigmoid output range) ─────────────────────────────
    B, C, H, W = 2, 3, 256, 256
    output = torch.rand(B, C, H, W, device=device, requires_grad=True)
    target = torch.rand(B, C, H, W, device=device)
    mask = (torch.rand(B, 1, H, W, device=device) > 0.4).float()

    # ── Build the loss (this also pulls VGG-16 weights from torch hub) ──
    print("Building VGG16 + loss (downloads weights on first run)...")
    criterion = InpaintingLoss().to(device)

    # ── Test 1: forward pass returns the expected dict ──────────────────
    print("\n[Test 1] forward returns dict with all components")
    result = criterion(output, target, mask)
    expected_keys = {"total", "l1_valid", "l1_hole", "perceptual", "style", "tv"}
    assert set(result.keys()) == expected_keys, f"keys mismatch: {result.keys()}"
    print(f"  keys: {sorted(result.keys())}  ✓")

    # ── Test 2: every component is a finite scalar tensor ───────────────
    print("\n[Test 2] all losses are scalar and finite")
    for k, v in result.items():
        assert v.shape == (), f"{k} is not scalar (shape={v.shape})"
        assert torch.isfinite(v), f"{k} is not finite ({v.item()})"
        print(f"  {k:11s} = {v.item():.6f}")
    print("  all scalars finite ✓")

    # ── Test 3: gradient flows through the model output ────────────────
    print("\n[Test 3] gradient flow into output")
    result["total"].backward()
    assert output.grad is not None, "no grad on output"
    assert torch.isfinite(output.grad).all(), "output grad contains NaN/Inf"
    grad_norm = output.grad.norm().item()
    print(f"  ||grad||_2 = {grad_norm:.4f}  ✓")

    # ── Test 4: VGG parameters are frozen ───────────────────────────────
    print("\n[Test 4] VGG weights are frozen")
    n_trainable = sum(1 for p in criterion.vgg.parameters() if p.requires_grad)
    print(f"  trainable VGG params: {n_trainable}")
    assert n_trainable == 0, "VGG must be frozen"
    print("  frozen ✓")

    # ── Test 5: VGG stays in eval() even after .train() ────────────────
    print("\n[Test 5] VGG locked in eval mode")
    criterion.train()  # tries to flip everything to train
    assert not criterion.vgg.training, "VGG accidentally went to train mode"
    print("  vgg.training =", criterion.vgg.training, " ✓")

    # ── Test 6: weights actually scale the contributions ────────────────
    print("\n[Test 6] custom weights")
    criterion_custom = InpaintingLoss(
        lambda_valid=2.0, lambda_hole=12.0,
        lambda_perc=0.10, lambda_style=240.0, lambda_tv=0.2,
        vgg_extractor=criterion.vgg,           # share VGG to skip re-download
    ).to(device)
    out2 = torch.rand(B, C, H, W, device=device)
    tgt2 = torch.rand(B, C, H, W, device=device)
    m2 = (torch.rand(B, 1, H, W, device=device) > 0.4).float()
    res_default = criterion(out2, tgt2, m2)
    res_double = criterion_custom(out2, tgt2, m2)
    ratio = (res_double["total"] / res_default["total"]).item()
    print(f"  total ratio (custom / default): {ratio:.3f}  (expected ≈ 2.0)")
    assert 1.5 < ratio < 2.5, f"weight scaling wrong: {ratio}"
    print("  custom weights scale correctly ✓")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
