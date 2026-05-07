# CLAUDE.md — Art Restoration AI Diploma Project

> This file is read by Claude Code at the start of every session in this repository. It encodes the project context, code standards, and Phase 0 decisions so that Claude Code does not need re-onboarding for each task.

---

## Project overview

**Diploma thesis:** *Restoration of Historic Artworks Using Deep Learning-Based Digital Inpainting*

**Goal:** Train a deep learning model to perform digital inpainting that repairs damaged or aged paintings by predicting missing pixels conditioned on the surrounding artistic style.

**Stack:**
- PyTorch 2.x, Python 3.10+
- Google Colab Free (T4, 16 GB VRAM, ~4–5 h sessions)
- Kaggle Notebooks (P100, 16 GB VRAM, 30 h/week, 9 h sessions)
- HuggingFace Spaces (CPU-only Streamlit demo)
- Google Drive (checkpoints, logs, dataset)
- GitHub (private repo for code)

**Budget:** $0. Everything must run on free-tier compute.

---

## Language rules

- **Explanations, discussions, commentary in chat:** Russian
- **All code (variables, comments, docstrings):** English
- **Diploma text, thesis chapters, presentation slides:** English
- **Paper titles, technical terms, architecture names, dataset names:** keep original English
- **Logs, commit messages, file names:** English

---

## Code standards (mandatory)

- **Python 3.10+, PyTorch 2.x.**
- **Type hints** on every function signature.
- **Google-style docstrings** on every module, class, function. Include `Args`, `Returns`, `Raises` sections where applicable.
- **Modular layout:** separate files for `model/`, `losses/`, `data/`, `training/`, `utils/`. No 1000-line god-files.
- **Notebooks** must run end-to-end in Google Colab Free out of the box. The first cell of every notebook is `pip install -q ...` plus Drive mount. Notebooks must survive Colab disconnects: state always lives on Google Drive, re-running cells from the top must resume.
- **Type-safe imports:** `from typing import ...`, `from pathlib import Path` instead of raw strings, `numpy.typing` for arrays.
- **Reproducibility:** `seed=42` everywhere unless an experiment explicitly varies seeds.

When writing thesis text or speaker notes:

- Academic English, formal tone.
- References as `[1]`, `[2]` etc., IEEE style.
- No first person — use "we" not "I" ("We propose..." not "I propose...").

---

## Technical constraints

| Constraint | Value |
|---|---|
| Max VRAM | 16 GB (T4 / P100) |
| Max session time | 4–5 h (Colab), 9 h (Kaggle) |
| Image size | 256×256 primary, 512×512 optional fine-tune |
| Batch size | 4 on T4, 8 on P100 |
| Mixed precision | fp16 mandatory on T4 (`torch.cuda.amp.GradScaler`) |
| Checkpoint cadence | Every 5 epochs, saved to Google Drive |
| Resume support | Mandatory in every training script |

**Memory cheat sheet (verify before training):**
- 256×256, batch=4, fp16 → ~8–10 GB ✅
- 256×256, batch=8, fp16 → ~14–15 GB ⚠️ tight, T4 only with caution
- 512×512, batch=2, fp16 → ~12–14 GB ⚠️ tight
- 512×512, batch=4, fp16 → OOM ❌

**P100 caveat:** P100 has limited fp16 throughput. Use `torch.cuda.amp.autocast` anyway for memory savings, but expect smaller speed gains than on T4.

---

## Phase 0 decisions, locked in

The literature review for this project is in `literature_review.md` at the repo root. Read it once if context allows; otherwise rely on this summary.

### Architecture

- **Primary model: PConv U-Net** (Liu et al., ECCV 2018, arXiv:1804.07723).
  - Why: only architecture that combines (a) trainable from scratch on a single 16 GB GPU, (b) native irregular mask support via the partial convolution operator with automatic mask update, (c) the L1 + Perceptual + Style + TV loss stack which is the right inductive bias for art-style coherence, (d) abundant open PyTorch implementations, (e) is the backbone of the only directly comparable prior art on paintings (Gupta et al. 2019, Farajzadeh & Hashemzadeh 2021).
  - Encoder: 7 PConv layers, kernel 7→5→5→3→3→3→3, channels 3→64→128→256→512→512→512→512.
  - Decoder: 7 layers, nearest upsample → concat skip → PConv. No BatchNorm in decoder. LeakyReLU(0.2). Final layer outputs 3 channels, then `torch.sigmoid` to clamp to `[0, 1]`.
  - Total ~33 M parameters.

- **Secondary baseline for ablation: Gated U-Net** (Yu et al., ICCV 2019, arXiv:1806.03589).
  - Used only in Chapter 3 for comparison against PConv. Not the primary deliverable.

- **Optional Chapter 4 reference: fine-tuned LaMa** (Suvorov et al., WACV 2022). Only fine-tune; full training is out of compute budget.

### Loss

`InpaintingLoss = λ_valid · L1_valid + λ_hole · L1_hole + λ_perc · L_perc + λ_style · L_style + λ_tv · L_tv`

Default weights from Liu et al. 2018:
- `λ_valid = 1`
- `λ_hole = 6`
- `λ_perc = 0.05`
- `λ_style = 120`
- `λ_tv = 0.1`

VGG16 features are extracted from `relu1_1`, `relu2_1`, `relu3_1`. The VGG extractor is frozen (`requires_grad = False`, `.eval()`) and instantiated once in `InpaintingLoss.__init__`.

### Dataset

- **WikiArt** from Kaggle (`ipythonx/wikiart` or best available mirror).
- Filter to: Renaissance, Baroque, Impressionism, Post-Impressionism, Realism. Target ≥ 5000 paintings.
- Resize to 256×256 (center crop after aspect-preserving resize). Save as JPG quality=95.
- Train / Val / Test split 80 / 10 / 10, stratified by style.

### Masks

Synthetic, generated on the fly (not stored on disk):
1. **Random brush strokes** — irregular curved sweeps.
2. **Simulated cracks** — thin, branching, fractal-like lines.
3. **Simulated paint loss** — large blob-shaped regions.
4. **Random aging stains** — gradient soft-edged 2D patches.

Three difficulty levels:
- `light` → mask area 10–20 %
- `medium` → mask area 20–40 %
- `heavy` → mask area 40–60 %

Convention: mask = 1 means **valid** pixel, mask = 0 means **hole**. This matches Liu et al. 2018 and avoids confusion in PConv code.

### Metrics

- **PSNR** — peak signal-to-noise ratio.
- **SSIM** — structural similarity.
- **FID** — Fréchet Inception Distance (use `torchmetrics.image.fid.FrechetInceptionDistance`).
- **LPIPS** — learned perceptual image patch similarity.

PSNR and SSIM via `torchmetrics`. FID and LPIPS only at the end of training and at evaluation time, never per-epoch (too slow).

### Research gaps that this project addresses

These are the diploma's claimed contributions; keep them in mind when writing thesis text:

1. **WikiArt as an inpainting benchmark.** Used overwhelmingly for style classification, almost never for inpainting (only Imagest 2025, which is closed-source diffusion). Open evaluation protocol on WikiArt is novel.
2. **Multi-type damage masks.** Most prior art uses a single mask family. This project trains and evaluates against four damage types × three difficulty levels.
3. **Loss ablation on art domain.** PConv loss weights come from photo-domain (Places2). Re-tuning for paintings is open ground (Experiment A1–A4).
4. **Style generalization.** Train on one style, test on another — never measured for PConv (Experiment C1–C3).
5. **CPU-friendly demo.** No prior published art-restoration system has a free, public, browser-based interactive demo. HuggingFace Spaces deployment is part of the contribution.

---

## Project layout

```
art-restoration/
├── CLAUDE.md                      ← you are here
├── README.md
├── requirements.txt
├── art_restoration_project_plan.md
├── literature_review.md
│
├── configs/
│   ├── train_config.yaml
│   └── experiment_configs/
│       ├── ablation_l1_only.yaml
│       ├── ablation_l1_perceptual.yaml
│       └── ablation_full_loss.yaml
│
├── notebooks/
│   ├── 01_data_preparation.ipynb     # Phase 1
│   ├── 02_train_pconv.ipynb          # Phase 3
│   ├── 03_train_gated.ipynb          # Phase 3
│   ├── 04_ablation_study.ipynb       # Phase 4
│   ├── 05_evaluation.ipynb           # Phase 4
│   └── 06_visualizations.ipynb       # for thesis figures
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # InpaintingDataset
│   │   ├── mask_generator.py         # MaskGenerator with 4 damage types
│   │   └── transforms.py             # albumentations wrappers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── partial_conv.py           # PartialConv2d layer
│   │   ├── pconv_unet.py             # PConvUNet
│   │   ├── gated_conv.py             # GatedConv2d layer
│   │   └── gated_unet.py             # GatedUNet
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Trainer class
│   │   ├── losses.py                 # InpaintingLoss + helpers
│   │   └── metrics.py                # psnr, ssim, fid, lpips
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py             # save/load with full state
│       └── visualization.py          # grids, before/after panels
│
├── demo/
│   ├── app.py                        # Streamlit
│   ├── requirements.txt
│   └── examples/
│
├── thesis/
│   ├── chapters/
│   ├── figures/
│   ├── tables/
│   └── references.bib
│
└── outputs/
    ├── checkpoints/                  # gitignored
    ├── logs/                         # gitignored, mirrored to Drive
    ├── figures/
    └── samples/
```

---

## Working agreements with Claude Code

### When creating new files

- Always include a module-level docstring at the top with one-line summary plus longer description.
- Always end files with a `if __name__ == "__main__":` smoke test that does the minimum to verify imports and shapes (random tensor in, expected tensor out, print shapes).
- For files inside `src/`, never use relative imports of the form `from ..foo import bar` from inside notebooks; always go through the absolute path `from src.foo import bar` after `sys.path.insert(0, '/content/art-restoration')` in Colab.
- New training notebooks must always include: GPU verification cell, config display cell, resume-detection cell, training cell, results-plot cell. Each cell's purpose stated in a markdown cell above it.

### When modifying training code

- Never silently change loss weights, optimizer, or schedule. If a change is needed, surface it as a config diff in the chat and wait for confirmation.
- Never delete checkpoint files. Add new ones; mark old ones stale by appending `.deprecated` if needed.
- Whenever a new metric is added, also update the CSV log header and the `validate()` return dict.

### When writing thesis or presentation text

- Cite prior work with `[N]` and add corresponding entries to `thesis/references.bib`.
- Never paraphrase numerical results from a paper without naming the dataset and the mask distribution they were measured on (PSNR alone is meaningless across benchmarks).
- For each claim of "state of the art", state both the benchmark and the year, e.g. "SOTA on Places2 thick masks, 2022".

### When asked for a quick fix

- If the fix touches `partial_conv.py`, `pconv_unet.py`, or `losses.py`, run the in-file smoke test before declaring success.
- If the fix touches dataset code, regenerate one batch and visualize it in the chat (or save to `outputs/samples/sanity_<timestamp>.png`).

---

## What this file is NOT

- It is not the diploma plan. The full plan with prompts is `art_restoration_project_plan.md`. Open that whenever a phase boundary is reached.
- It is not the literature review. Citations and method comparisons live in `literature_review.md`.
- It is not configuration. Concrete hyperparameters live in `configs/*.yaml`. CLAUDE.md only documents the *defaults* and *invariants*.

---

## Current phase pointer

When this file is updated at the end of each phase, the line below shifts. Claude Code should re-read this section first when starting a session.

**Status:** Phase 0 complete (literature review committed and reviewed). Phase 1 in progress: data collection and preparation. Next deliverables: project skeleton (this commit), `notebooks/01_data_preparation.ipynb`, `src/data/mask_generator.py`, `src/data/dataset.py`, `requirements.txt`. After Phase 1: Phase 2 (model architecture).

---

*Last updated: April 2026, end of Phase 0.*
