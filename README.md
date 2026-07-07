
# Art Restoration with Deep Learning Inpainting

*Restoration of Historic Artworks Using Deep
Learning-Based Digital Inpainting*.

A rigorous comparative study of three convolutional inpainting architectures
applied to damaged paintings, with an interactive Gradio demo.

## Headline result

| Model                                  | PSNR ↑   | SSIM ↑   | LPIPS ↓  |
|----------------------------------------|----------|----------|----------|
| **Gated U-Net** (Yu et al. 2019)       | **30.37**| **0.896**| **0.072**|
| Vanilla U-Net (mask-concat)            | 29.92    | 0.887    | 0.080    |
| PConv U-Net (Liu et al. 2018)          | 28.18    | 0.853    | 0.106    |

Evaluation on 3,977 held-out WikiArt paintings × 4 damage types × 3
difficulty levels = 47,724 (image, mask) pairs. The ranking holds across
every cell and is statistically significant under a paired Wilcoxon
signed-rank test with Bonferroni correction.

<img width="5952" height="1280" alt="fig2_highres_comparison" src="https://github.com/user-attachments/assets/feeec058-0711-463d-854a-877783a16485" />
<img width="5331" height="1752" alt="fig1_academic_metrics" src="https://github.com/user-attachments/assets/ee437c69-2bc0-4654-a7e4-47f2592953f4" />

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r demo/requirements.txt
```

### 2. Get the trained checkpoints

Checkpoints are not in the repository (multi-hundred-MB each). Place the
three `best.pth` files under:

```
outputs/checkpoints/pconv_unet/best.pth
outputs/checkpoints/unet_baseline/best.pth
outputs/checkpoints/gated_unet/best.pth
```

(or edit the paths in `demo/load_models.py`).

### 3. Run the interactive demo

```bash
python demo/app.py
```

Browser opens at <http://127.0.0.1:7860>. Two modes:

- **Paint custom damage** — upload a painting, brush over the damage,
  click *Restore* to see all three models side-by-side with per-image
  PSNR / SSIM.
- **Synthetic damage preset** — upload a painting and pick a damage type
  (brush / crack / paint loss / aging stain) × severity (light / medium
  / heavy); the app generates the mask deterministically.

All output images come with a built-in toolbar: zoom in / out / reset /
fullscreen, plus drag-to-pan when zoomed.

### 4. Regenerate the thesis and slides

```bash
python thesis/generate_thesis.py    # writes thesis/thesis.docx
python thesis/generate_slides.py    # writes thesis/presentation.pptx
```

Both scripts pull live numbers from `outputs/outputs/eval/tables/*.csv`,
so any re-evaluation automatically propagates to the documents.

## Project structure

```
art-restoration/
├── README.md
├── requirements.txt
├── configs/
│   ├── train_config.yaml
│   └── experiment_configs/
│       ├── pconv_unet.yaml           # primary model
│       ├── unet_baseline.yaml        # vanilla baseline
│       └── gated_unet.yaml           # DeepFillv2-style baseline
│
├── src/
│   ├── data/                         # InpaintingDataset, MaskGenerator
│   ├── models/                       # PConv / Vanilla / Gated U-Nets + registry
│   ├── training/                     # Trainer, InpaintingLoss, metrics
│   └── utils/                        # checkpointing, visualisation
│
├── notebooks/
│   ├── 01_data_preparation.ipynb     # WikiArt → train/val/test splits
│   ├── 02_train_pconv.ipynb          # PConv U-Net training
│   ├── 03_train_unet_baseline.ipynb  # Vanilla U-Net training
│   ├── 04_train_gated_unet.ipynb     # Gated U-Net training
│   ├── 05_evaluate.ipynb             # Test-set evaluation, stats, figures
│   └── kaggle_train.ipynb            # Free-tier Kaggle / Colab variant
│
├── demo/
│   ├── app.py                        # Gradio interactive demo
│   ├── load_models.py                # Shared model-loading helper
│   ├── requirements.txt
│   ├── README.md
│   └── examples/                     # Sample paintings (optional)
│
├── thesis/
│   ├── generate_thesis.py            # → thesis.docx
│   ├── generate_slides.py            # → presentation.pptx
│   ├── thesis.docx                   # generated
│   ├── presentation.pptx             # generated
│   └── figures/                      # architecture diagrams
│
└── outputs/
    └── outputs/                      # evaluation artefacts (kept in repo)
        └── eval/
            ├── tables/               # CSV metrics used by thesis + slides
            ├── figures/              # bar charts, comparison grids
            ├── fig1_academic_metrics.png
            ├── fig2_highres_comparison.jpg
            ├── per_image_metrics.csv # 47,724 rows
            └── stats_results.csv     # Wilcoxon + Bonferroni
```

## Architectures

All three networks share a 7-stage encoder–decoder U-Net topology with
identical depth, channel widths, skip connections, training data, loss,
optimizer, schedule and number of epochs. The **only** axis that differs
is the convolution operator:

1. **PConv U-Net** (Liu et al., ECCV 2018) — partial convolution with
   hard mask propagation. The most-cited irregular-mask inpainting
   architecture.
2. **Vanilla U-Net** — plain `nn.Conv2d`; the binary mask is concatenated
   as a fourth input channel. The simplest possible mask-conditioning
   strategy.
3. **Gated U-Net** (Yu et al., ICCV 2019, DeepFillv2-style) — gated
   convolution with a learned soft attention gate at every spatial
   position.

This isolates the contribution of the convolution operator to the
restoration quality.

## Loss

Composite Liu et al. 2018 loss:

```
L = λ_valid · L1_valid + λ_hole · L1_hole + λ_perc · L_perc + λ_style · L_style + λ_tv · L_tv
```

with weights `1 / 6 / 0.05 / 50 / 0.1`. Perceptual and style terms use
VGG16 features at `relu1_1`, `relu2_1`, `relu3_1`. The loss runs in fp32
internally so AMP (fp16) training stays numerically stable.

## Dataset

WikiArt, filtered to five styles (Renaissance, Baroque, Impressionism,
Post-Impressionism, Realism) → 39,213 paintings. Aspect-preserving
resize + centre-crop to 256 × 256. Split stratified by style:
31,815 train / 3,977 val / 3,977 test.

Synthetic damage masks generated on the fly via `MaskGenerator`:
four primitives (brush strokes, simulated cracks, simulated paint loss,
random aging stains) × three difficulty levels (10–20 %, 20–40 %,
40–60 % hole area).

## Training protocol

| Hyperparameter | Value                                |
|---------------|---------------------------------------|
| Optimizer     | Adam (β = (0.9, 0.999), wd = 0)       |
| LR            | 2 × 10⁻⁴, cosine decay → 1 × 10⁻⁶     |
| Warmup        | 1 epoch                                |
| Batch size    | 4 (T4 / P100, AMP fp16)                |
| Epochs        | 12                                     |
| Image size    | 256 × 256                              |
| Gradient clip | L2 norm 1                              |
| Seed          | 42                                     |

Training is resumable across Colab / Kaggle session timeouts:
checkpoints include optimizer, scheduler and AMP-scaler state.

## Evaluation

`05_evaluate.ipynb` corrupts every test painting with every
(damage_type × difficulty) combination using deterministic masks
(`seed=12345`) and processes each pair through all three models. Outputs
written to `outputs/outputs/eval/`:

- `per_image_metrics.csv` — 47,724 rows × {PSNR, SSIM, LPIPS}
- `tables/overall_metrics.csv` — headline table
- `tables/metrics_by_difficulty.csv`, `tables/metrics_by_damage.csv`
- `tables/fid_summary.csv` — FID per difficulty
- `stats_results.csv` — paired Wilcoxon + Bonferroni
- `figures/fig1_psnr.png`, `figures/fig2_grid.png`
- `fig1_academic_metrics.png`, `fig2_highres_comparison.jpg`

## References

- Liu et al., *Image Inpainting for Irregular Holes Using Partial Convolutions*, ECCV 2018.
- Yu et al., *Free-Form Image Inpainting with Gated Convolution*, ICCV 2019.
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018.
- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017.

## License

MIT — see `LICENSE` (if added).
