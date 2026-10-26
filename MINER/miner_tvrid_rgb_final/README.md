# MINER — TVRID RGB Track

Code to reproduce our submission for the
[ICPR-2026-TVRID](https://github.com/RaphaelDel/ICPR-2026-TVRID)
RGB re-identification track.

## Prerequisites

- Python 3.10+, CUDA GPU
- [TVRID extracted dataset](https://zenodo.org/records/17909410) (cropped 300×300)
- EVA02-Large checkpoint pretrained on PRCC from [DIFFER](https://github.com/xliangp/DIFFER) (Liang & Rawat, CVPR 2025)

```bash
pip install -r requirements.txt
```

## Setup

### 1. Weights

Place in `weights/`:
- `eva02_l_bio_best.pth` — DIFFER's EVA02-Large pretrained on **PRCC** (Liang & Rawat, CVPR 2025)
- `tvrid_embeddings.json` — regenerate with `python generate_embeddings.py`

### 2. Data

Unzip the **extracted** archive into `data/`:
```
data/DB_extracted/
├── train_labels.csv
├── public_test_labels.csv
├── train/
└── test_public/
```

## Reproduce

### Train (~13 min on A100)

```bash
python train.py --config configs/rgb.yml
```

Produces `logs/best.pth`.

### Generate submission

```bash
python submit.py \
    --config configs/rgb.yml \
    --checkpoint logs/best.pth \
    --output submission/
```

Produces `submission/rankings_rgb.csv`.

### Custom data location

```bash
python train.py --config configs/rgb.yml DATA.ROOT /path/to/data
python submit.py --config configs/rgb.yml --checkpoint logs/best.pth --data-root /path/to/data --output submission/
```

## Tests

```bash
python -m pytest tests/ -v
```
