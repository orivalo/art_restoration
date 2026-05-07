"""InpaintingDataset — PyTorch Dataset for art inpainting.

Scans an image directory for JPEG/PNG artwork files, applies albumentations
augmentations, and generates synthetic damage masks on-the-fly via
MaskGenerator. Each call to __getitem__ produces a fresh mask so that the
model sees different damage patterns across epochs.

Pixel conventions:
    Images are normalized to [-1, 1] (mean=0.5, std=0.5 per channel).
    mask = 1  →  valid pixel (intact paint)
    mask = 0  →  hole (missing / damaged region)
    masked_image = ground_truth * mask  (hole pixels zeroed out)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from numpy.typing import NDArray

from src.data.mask_generator import DifficultyLevel, MaskGenerator


def build_default_transform(image_size: int = 256) -> A.Compose:
    """Build the default albumentations augmentation pipeline.

    Applied to every sample before masking. ColorJitter is kept mild to
    preserve art-style fidelity.

    Args:
        image_size: Target height and width (images are pre-cropped before
            this transform, so resizing is not included here).

    Returns:
        An ``albumentations.Compose`` instance.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.5,
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


class InpaintingDataset(Dataset):
    """Dataset that returns (masked_image, mask, ground_truth) triples.

    Images are loaded from disk, resized to a square via aspect-preserving
    resize + center crop, and then passed through an albumentations pipeline.
    A new binary damage mask is generated for every sample on every access.

    Args:
        image_dir: Root directory to scan recursively for image files.
        mask_generator: ``MaskGenerator`` instance used to produce masks.
        transform: albumentations ``Compose`` transform. Defaults to
            ``build_default_transform(image_size)`` if ``None``.
        difficulty: Mask difficulty level forwarded to
            ``MaskGenerator.generate``.
        image_size: Square output resolution in pixels.
        extensions: Filename suffixes to include during directory scan.

    Raises:
        FileNotFoundError: If no images matching ``extensions`` are found
            under ``image_dir``.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_generator: MaskGenerator,
        transform: Optional[A.Compose] = None,
        difficulty: DifficultyLevel = "medium",
        image_size: int = 256,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_generator = mask_generator
        self.image_size = image_size
        self.difficulty = difficulty
        self.transform = transform if transform is not None else build_default_transform(image_size)

        self.image_paths: list[Path] = sorted(
            p for p in self.image_dir.rglob("*")
            if p.suffix.lower() in extensions
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images with extensions {extensions} found under {self.image_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one training sample.

        Args:
            idx: Integer index in ``[0, len(self))``.

        Returns:
            Tuple ``(masked_image, mask, ground_truth)`` where:

            * **masked_image** — ``(3, H, W)`` float32, values in ``[-1, 1]``.
              Hole pixels are set to ``0``.
            * **mask** — ``(1, H, W)`` float32, values in ``{0, 1}``.
            * **ground_truth** — ``(3, H, W)`` float32, values in ``[-1, 1]``.

        Raises:
            IOError: If the image file cannot be opened or decoded.
        """
        image_np = self._load_image(self.image_paths[idx])
        augmented = self.transform(image=image_np)
        ground_truth: torch.Tensor = augmented["image"]  # (3, H, W)

        h, w = ground_truth.shape[1], ground_truth.shape[2]
        mask_np = self.mask_generator.generate(h, w, difficulty=self.difficulty)
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)

        masked_image = ground_truth * mask  # zero-fill holes
        return masked_image, mask, ground_truth

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> NDArray[np.uint8]:
        """Load an image and resize/crop it to ``(image_size, image_size)``.

        Uses aspect-preserving resize so that neither dimension drops below
        ``image_size``, followed by a center crop.

        Args:
            path: Filesystem path to a JPEG or PNG file.

        Returns:
            Uint8 numpy array of shape ``(H, W, 3)`` in RGB order.

        Raises:
            IOError: Propagated from ``PIL.Image.open`` on decode failure.
        """
        with Image.open(path) as img:
            img = img.convert("RGB")
            # Scale so the short side reaches image_size
            scale = self.image_size / min(img.width, img.height)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            # Center crop
            left = (new_w - self.image_size) // 2
            top = (new_h - self.image_size) // 2
            img = img.crop((
                left, top,
                left + self.image_size,
                top + self.image_size,
            ))
        return np.asarray(img, dtype=np.uint8)


if __name__ == "__main__":
    import tempfile

    import matplotlib.pyplot as plt
    from pathlib import Path
    from torch.utils.data import DataLoader

    # ── Create a tiny synthetic dataset of random-colour images ──────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rng = np.random.default_rng(42)
        for i in range(8):
            fake = Image.fromarray(
                rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
            )
            fake.save(tmp_path / f"fake_{i:03d}.jpg")

        gen = MaskGenerator(seed=42)
        ds = InpaintingDataset(
            image_dir=tmp_path,
            mask_generator=gen,
            difficulty="medium",
            image_size=256,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        masked, masks, gt = next(iter(loader))

    # ── Shape assertions ─────────────────────────────────────────────────
    assert masked.shape == (4, 3, 256, 256), f"masked shape wrong: {masked.shape}"
    assert masks.shape == (4, 1, 256, 256), f"mask shape wrong: {masks.shape}"
    assert gt.shape == (4, 3, 256, 256), f"gt shape wrong: {gt.shape}"
    assert set(masks.unique().tolist()).issubset({0.0, 1.0}), "mask is not binary"

    print(f"masked_image : {masked.shape}  range [{masked.min():.2f}, {masked.max():.2f}]")
    print(f"mask         : {masks.shape}   unique {masks.unique().tolist()}")
    print(f"ground_truth : {gt.shape}  range [{gt.min():.2f}, {gt.max():.2f}]")

    # ── Quick visual sanity check ─────────────────────────────────────────
    out_dir = Path("outputs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _denorm(t: torch.Tensor) -> np.ndarray:
        return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for i in range(4):
        axes[i, 0].imshow(_denorm(gt[i]))
        axes[i, 0].set_title("ground truth")
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap="gray")
        axes[i, 1].set_title("mask")
        axes[i, 2].imshow(_denorm(masked[i]))
        axes[i, 2].set_title("masked input")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    save_path = out_dir / "dataset_sanity.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Smoke test passed. Grid saved → {save_path}")
