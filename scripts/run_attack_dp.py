#!/usr/bin/env python3
"""
DP-simulated duplicate-detection attack runner.

This script keeps the same CLIP duplicate-detection pipeline as the baseline,
but inserts a DP-SGD-inspired mechanism over embeddings:

  1) per-sample L2 clipping to bound C
  2) Gaussian noise with std = sigma * C

This simulates the "clip + noise" behavior used by DP-SGD/DP-LoRA training,
without retraining the diffusion model itself.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from clip_utils import embed_directory, find_near_duplicates
from dataset import prepare_cifar10


def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def dp_mechanism(
    embeddings: Dict[Path, np.ndarray],
    C: float = 1.0,
    sigma: float = 0.05,
) -> Dict[Path, np.ndarray]:
    """
    Apply a DP-SGD-style mechanism to CLIP embeddings.

    For each vector v:
      v_norm    = normalize(v)
      v_clipped = v_norm * min(1, C / ||v_norm||)
      v_noisy   = v_clipped + N(0, sigma^2 * C^2)
      v_out     = normalize(v_noisy)

    Returns a new Path->embedding dictionary, preserving key order.
    """
    if C <= 0:
        raise ValueError("C must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    paths = list(embeddings.keys())
    vecs = np.stack([embeddings[p].astype(np.float32, copy=False) for p in paths], axis=0)

    # 1) Normalize each embedding first
    vecs = _l2_normalize_rows(vecs)

    # 2) Clip each embedding norm to C
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    clip_factors = np.minimum(1.0, C / np.maximum(norms, 1e-12))
    vecs_clipped = vecs * clip_factors

    # 3) Add Gaussian noise proportional to clipping bound C
    noise_std = sigma * C
    noise = np.random.normal(loc=0.0, scale=noise_std, size=vecs_clipped.shape).astype(np.float32)
    vecs_noisy = vecs_clipped + noise

    # 4) Re-normalize for cosine-similarity duplicate search
    vecs_out = _l2_normalize_rows(vecs_noisy)

    return {p: vecs_out[i] for i, p in enumerate(paths)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run duplicate detection with a DP-simulated embedding mechanism "
            "(clip + Gaussian noise), inspired by DP-SGD/DP-LoRA."
        )
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of images to analyze. If omitted, CIFAR-10 is prepared/used.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="L2 clipping bound C used in the DP mechanism.",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.05],
        help="One or more sigma values (noise multiplier).",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.9,
        help="Cosine threshold for near-duplicate detection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for block-wise cosine search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible DP noise.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    image_dir = args.image_dir if args.image_dir is not None else prepare_cifar10(Path("data"))

    print("Embedding images with CLIP...")
    embeddings = embed_directory(image_dir)

    total_embeddings = len(embeddings)

    for sigma in args.sigmas:
        dp_embeddings = dp_mechanism(embeddings, C=args.c, sigma=sigma)
        duplicates = find_near_duplicates(
            dp_embeddings,
            cosine_threshold=args.cosine_threshold,
            batch_size=args.batch_size,
        )

        duplicate_counts = {k: len(v) for k, v in duplicates.items()}
        avg_duplicates = float(np.mean(list(duplicate_counts.values()))) if duplicate_counts else 0.0

        results = {
            "C": args.c,
            "sigma": sigma,
            "total_embeddings": total_embeddings,
            "duplicate_counts": duplicate_counts,
            "avg_duplicates": avg_duplicates,
        }

        if len(args.sigmas) == 1:
            output_path = Path("results_dp.pkl")
        else:
            output_path = Path(f"results_dp_{sigma:.2f}.pkl")

        with output_path.open("wb") as f:
            pickle.dump(results, f)

        print("[DP MODE]")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Avg duplicates: {avg_duplicates:.4f}")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
