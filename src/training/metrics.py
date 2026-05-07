"""Evaluation metrics for art inpainting.

Provides per-batch and aggregate metric implementations:

* **PSNR** — Peak Signal-to-Noise Ratio (fast, per-epoch validation OK)
* **SSIM** — Structural Similarity Index (fast, per-epoch validation OK)
* **FID**  — Fréchet Inception Distance (slow — eval / final test only)
* **LPIPS** — Learned Perceptual Image Patch Similarity (slow — eval only)

Pixel-range convention:
    All inputs are expected in ``[0, 1]`` with shape ``(B, 3, H, W)``.
    The trainer is responsible for converting from the dataset's
    ``[-1, 1]`` representation to ``[0, 1]`` before invoking these
    metrics.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure

# ── Optional stateful metrics (FID / LPIPS) ──────────────────────────────────
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    _FID_AVAILABLE = True
except ImportError:
    FrechetInceptionDistance = None  # type: ignore[misc,assignment]
    _FID_AVAILABLE = False

try:  # newer torchmetrics: torchmetrics.image.lpip
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    _LPIPS_AVAILABLE = True
except ImportError:
    try:  # older torchmetrics path
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity  # type: ignore[no-redef]
        _LPIPS_AVAILABLE = True
    except ImportError:
        LearnedPerceptualImagePatchSimilarity = None  # type: ignore[misc,assignment]
        _LPIPS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
#  Stateless helper functions
# ──────────────────────────────────────────────────────────────────────────────


def psnr(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio averaged over the batch.

    Args:
        output: Predicted image ``(B, 3, H, W)`` with values in ``[0, 1]``.
        target: Ground-truth image of the same shape.

    Returns:
        Scalar PSNR value in dB (higher is better).  Returns ``+inf`` if
        ``output == target`` exactly.
    """
    return peak_signal_noise_ratio(output, target, data_range=1.0)


def ssim(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Structural Similarity Index averaged over the batch.

    Args:
        output: Predicted image ``(B, 3, H, W)`` with values in ``[0, 1]``.
        target: Ground-truth image of the same shape.

    Returns:
        Scalar SSIM value in ``[-1, 1]`` (1 = perfect match, higher is
        better).
    """
    return structural_similarity_index_measure(output, target, data_range=1.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Aggregator class — one instance per evaluation pass
# ──────────────────────────────────────────────────────────────────────────────


class InpaintingMetrics(nn.Module):
    """Stateful aggregator for PSNR / SSIM / FID / LPIPS over an eval set.

    PSNR and SSIM are accumulated as running batch-weighted means.
    FID and LPIPS are delegated to ``torchmetrics`` stateful modules.

    Pattern of use::

        metrics = InpaintingMetrics(device, compute_fid=False)
        metrics.reset()
        for output, target in loader:
            metrics.update(output, target)
        result = metrics.compute()    # → dict of floats

    Args:
        device:        ``torch.device`` on which to host the FID / LPIPS
            sub-modules (and download their backbone weights).
        compute_fid:   If ``True``, instantiate the FID computer.  Slow —
            keep ``False`` for per-epoch validation.
        compute_lpips: If ``True``, instantiate the LPIPS computer.  Slow.
        fid_features:  FID feature dimensionality (default 2048 for full
            Inception V3 pool3 output).

    Raises:
        ImportError: If ``compute_fid`` / ``compute_lpips`` is requested
            but the corresponding torchmetrics extra is unavailable.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        compute_fid: bool = False,
        compute_lpips: bool = False,
        fid_features: int = 2048,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.compute_fid = compute_fid
        self.compute_lpips = compute_lpips

        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._n = 0

        if compute_fid:
            if not _FID_AVAILABLE:
                raise ImportError(
                    "FID requested but torchmetrics FID is unavailable. "
                    "Install with: pip install torchmetrics[image]"
                )
            self.fid = FrechetInceptionDistance(
                feature=fid_features, normalize=True,
            ).to(self.device)

        if compute_lpips:
            if not _LPIPS_AVAILABLE:
                raise ImportError(
                    "LPIPS requested but torchmetrics LPIPS is unavailable. "
                    "Install with: pip install torchmetrics[image] lpips"
                )
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True,
            ).to(self.device)

    # ------------------------------------------------------------------
    #  State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all accumulators back to zero."""
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._n = 0
        if self.compute_fid:
            self.fid.reset()
        if self.compute_lpips:
            self.lpips.reset()

    # ------------------------------------------------------------------
    #  Update / compute
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate a single batch into the metric state.

        Args:
            output: Predicted image ``(B, 3, H, W)`` in ``[0, 1]``.
            target: Ground-truth image of the same shape and range.
        """
        # Move to metric device only if necessary (avoids bouncing between
        # CPU and GPU when caller already prepared tensors correctly).
        if output.device != self.device:
            output = output.to(self.device)
            target = target.to(self.device)

        # Clamp to be safe — sigmoid output is in [0, 1] but we can be a
        # tiny bit outside due to fp16 noise during AMP training.
        output = output.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        b = output.shape[0]
        self._psnr_sum += psnr(output, target).item() * b
        self._ssim_sum += ssim(output, target).item() * b
        self._n += b

        if self.compute_fid:
            self.fid.update(target, real=True)
            self.fid.update(output, real=False)

        if self.compute_lpips:
            self.lpips.update(output, target)

    def compute(self) -> dict[str, float]:
        """Materialise final metric values.

        Returns:
            Dict with keys ``'psnr'``, ``'ssim'`` (always), and
            ``'fid'`` / ``'lpips'`` if those were enabled.  All values
            are Python ``float``.
        """
        result: dict[str, float] = {
            "psnr": self._psnr_sum / max(self._n, 1),
            "ssim": self._ssim_sum / max(self._n, 1),
        }
        if self.compute_fid:
            result["fid"] = float(self.fid.compute().item())
        if self.compute_lpips:
            result["lpips"] = float(self.lpips.compute().item())
        return result


# ======================================================================
#  Smoke test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("InpaintingMetrics — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Test 1: PSNR identity → +inf ─────────────────────────────────────
    img = torch.rand(2, 3, 64, 64, device=device)
    p_id = psnr(img, img).item()
    print(f"[Test 1] PSNR(img, img) = {p_id}  (expected +inf)")
    assert p_id == float("inf") or p_id > 100, "identity PSNR should be huge"
    print("  PASSED")

    # ── Test 2: SSIM identity → 1.0 ──────────────────────────────────────
    s_id = ssim(img, img).item()
    print(f"\n[Test 2] SSIM(img, img) = {s_id:.6f}  (expected ≈ 1.0)")
    assert s_id > 0.999, "identity SSIM should be ≈ 1"
    print("  PASSED")

    # ── Test 3: PSNR/SSIM on noisy version ──────────────────────────────
    noisy = (img + 0.1 * torch.randn_like(img)).clamp(0, 1)
    p = psnr(img, noisy).item()
    s = ssim(img, noisy).item()
    print(f"\n[Test 3] PSNR(img, noisy) = {p:.2f} dB    SSIM = {s:.4f}")
    assert 10 < p < 50 and 0.0 < s < 1.0
    print("  PASSED")

    # ── Test 4: aggregator round-trip ───────────────────────────────────
    print(f"\n[Test 4] InpaintingMetrics aggregator")
    m = InpaintingMetrics(device=device, compute_fid=False, compute_lpips=False)
    m.reset()
    for _ in range(3):
        out = torch.rand(4, 3, 64, 64, device=device)
        tgt = torch.rand(4, 3, 64, 64, device=device)
        m.update(out, tgt)
    res = m.compute()
    print(f"  result keys: {sorted(res.keys())}")
    print(f"  psnr = {res['psnr']:.3f}    ssim = {res['ssim']:.4f}")
    assert set(res) == {"psnr", "ssim"}
    print("  PASSED")

    # ── Test 5: reset wipes the state ────────────────────────────────────
    print(f"\n[Test 5] reset clears accumulator")
    m.reset()
    assert m._n == 0 and m._psnr_sum == 0.0 and m._ssim_sum == 0.0
    print("  PASSED")

    # ── Test 6: optional FID / LPIPS availability flags ─────────────────
    print(f"\n[Test 6] optional metric availability")
    print(f"  FID  available: {_FID_AVAILABLE}")
    print(f"  LPIPS available: {_LPIPS_AVAILABLE}")
    print("  (these will be exercised in the trainer end-to-end test)")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)
