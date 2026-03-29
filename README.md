# diffusion-privacy-attacks

Initial implementation of memorization attacks for generated diffusion-model images.

## What is included

- A small Python package (`diffusion_privacy_attacks`) with a baseline nearest-neighbor attack.
- A CLI script to compare generated images against a reference dataset and export ranked matches.
- CSV output with per-match distance metrics (`mse_distance`, `l2_distance`).

## Installation

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Run the attack

```bash
python scripts/run_attack.py \
  --generated-dir /path/to/generated \
  --reference-dir /path/to/reference \
  --output results/attack_results.csv \
  --top-k 5 \
  --image-size 256
```

## Notes

This is a baseline implementation intended to bootstrap experiments. Future iterations can add:

- embedding-based similarity (e.g., CLIP features),
- threshold calibration on held-out data,
- attack success metrics and reporting dashboards.
