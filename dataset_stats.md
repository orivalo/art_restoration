# Dataset Statistics — Art Restoration AI

> **Phase 1 deliverable.** Final dataset state after WikiArt download, filtering, preprocessing, and stratified split.
> **Last updated:** end of Phase 1, May 2026.

---

## Source

- **Dataset:** WikiArt mirror `steubk/wikiart` on Kaggle.
- **License:** CC0-1.0 (public domain dedication).
- **Original archive size:** ~30 GB raw, 81 444 images, 27 style directories.
- **Download:** Kaggle API via `kagglehub`, falling back to direct Kaggle CLI.
- **Storage:** Colab temporary disk for raw download → Google Drive for processed 256×256 JPGs.

## Filtering and preprocessing pipeline

```
WikiArt raw (81 444 images, 27 styles)
        │
        ▼  Filter to 5 target styles
5 target styles (39 769 images)
        │
        ▼  Aspect-preserving resize → 256×256 center crop → JPG quality=95
Processed dataset (39 769 images, 1.4 GB on Drive)
        │
        ▼  Stratified split 80 / 10 / 10 by style, seed=42
Train / Val / Test CSVs
```

### Style consolidation rules

Three Renaissance subfamilies present in the source were collapsed into a single canonical label `renaissance` to give the model a coherent classical-figurative class:

| Source directory | Canonical label |
|---|---|
| `Baroque` | `baroque` |
| `Impressionism` | `impressionism` |
| `Post_Impressionism` | `post_impressionism` |
| `Realism` | `realism` |
| `Early_Renaissance` | `renaissance` |
| `High_Renaissance` | `renaissance` |
| `Northern_Renaissance` | `renaissance` |

`Mannerism_Late_Renaissance` was deliberately excluded — stylistically it is post-Renaissance, mixing it would muddy the class.

### Preprocessing parameters

- **Target resolution:** 256 × 256 px.
- **Resize method:** aspect-preserving — short side scaled to 256 with `Image.LANCZOS`, then center crop.
- **Output format:** JPG, quality 95, optimize on.
- **Filename:** `{style}_{md5(source_filename)[:8]}.jpg` — deterministic, dedup-safe across reruns.
- **Errors:** 0 / 39 769.

## Final counts

### Per style, per split

| Style | Train | Val | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| baroque | 3 392 | 424 | 424 | 4 240 | 10.7 % |
| impressionism | 10 448 | 1 306 | 1 306 | 13 060 | 32.8 % |
| post_impressionism | 5 160 | 645 | 645 | 6 450 | 16.2 % |
| realism | 8 586 | 1 073 | 1 074 | 10 733 | 27.0 % |
| renaissance | 4 229 | 529 | 528 | 5 286 | 13.3 % |
| **TOTAL** | **31 815** | **3 977** | **3 977** | **39 769** | 100 % |

### Split proportions

- Train: 80.00 %
- Val: 10.00 %
- Test: 10.00 %

Stratified `train_test_split` (sklearn, seed=42) was applied first as 80 / 20, then the 20 % temp set was halved 50 / 50, which preserves class proportions in every split.

### Class balance

Imbalance ratio (largest / smallest class) = 13 060 / 4 240 = **3.08**.

This is moderate imbalance. For inpainting it is not critical because the task is local texture and stylistic restoration rather than classification. Class share is reported here so that style-conditioned ablation experiments in Phase 4 (Experiment C) can be interpreted correctly.

## Storage

| Location | Size | Persistence |
|---|---|---|
| `/content/wikiart_raw/` (Colab temp) | ~30 GB | volatile, cleared after Phase 1 |
| `MyDrive/art_restoration/data/processed/` | 1.4 GB | persistent |
| `MyDrive/art_restoration/data/splits/` | < 5 MB | persistent (CSV) |

Within the 15 GB Drive free tier, the processed dataset uses 9.3 % — leaves plenty of room for checkpoints (Phase 3) and ablation outputs (Phase 4).

## Reproducibility

- `random_state = 42` on every `train_test_split` call.
- Filename hashing is content-independent (md5 of basename) — deterministic across reruns.
- Resume-safe preprocessing: the second run of Cell 3 skipped 34 483 already-processed images and only converted the 5 286 newly-included Renaissance files, in 5 min 23 s.
- Split CSV columns: `path`, `style`, `split`. Path is absolute under `MyDrive/art_restoration/data/processed/`.

## Files produced for downstream phases

```
MyDrive/art_restoration/data/
├── processed/
│   ├── baroque/             4 240 jpg
│   ├── impressionism/      13 060 jpg
│   ├── post_impressionism/  6 450 jpg
│   ├── realism/            10 733 jpg
│   └── renaissance/         5 286 jpg
└── splits/
    ├── all_splits.csv       (39 769 rows)
    ├── train.csv            (31 815 rows)
    ├── val.csv               (3 977 rows)
    └── test.csv              (3 977 rows)
```

## Notes for Phase 4

The training set (31 815 images) is generous. For ablation experiments where 7+ short trainings must fit in Kaggle's 30 h/week GPU budget, consider drawing a stratified 10 000-image subset from `train.csv` to keep each ablation run under 90 minutes. The full train set is reserved for the primary PConv U-Net training in Phase 3.

---

*Generated automatically from split CSVs and Drive disk usage at the end of Phase 1.*
