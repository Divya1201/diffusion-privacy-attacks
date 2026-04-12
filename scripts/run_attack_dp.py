#!/usr/bin/env python3
"""
DP-LoRA simulated duplicate-detection attack runner.

Simulates DP-LoRA by:
1) Passing embeddings through a low-rank adapter (LoRA-style)
2) Applying DP-SGD-style clipping + noise on adapter output
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from clip_utils import embed_directory, find_near_duplicates
from dataset import prepare_cifar10


# ==============================
# Utility
# ==============================
def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


# ==============================
# LoRA ADAPTER (NEW)
# ==============================
class LoRAAdapter:
    """
    Simulates low-rank adaptation: W' = W + A @ B
    Here we only simulate transformation: v -> v + A(Bv)
    """

    def __init__(self, dim: int = 512, rank: int = 8, scale: float = 0.1):
        self.A = np.random.randn(dim, rank).astype(np.float32) * scale
        self.B = np.random.randn(rank, dim).astype(np.float32) * scale

    def forward(self, v: np.ndarray) -> np.ndarray:
        return v + self.A @ (self.B @ v)


# ==============================
# DP + LoRA MECHANISM
# ==============================
def dp_lora_mechanism(
    embeddings: Dict[Path, np.ndarray],
    C: float = 1.0,
    sigma: float = 0.05,
    rank: int = 8,
) -> Dict[Path, np.ndarray]:

    paths = list(embeddings.keys())
    vecs = np.stack([embeddings[p] for p in paths]).astype(np.float32)

    # Normalize
    vecs = _l2_normalize_rows(vecs)

    # Apply LoRA adapter
    adapter = LoRAAdapter(dim=vecs.shape[1], rank=rank)
    vecs_lora = np.array([adapter.forward(v) for v in vecs])

    # Normalize again
    vecs_lora = _l2_normalize_rows(vecs_lora)

    # Clip
    norms = np.linalg.norm(vecs_lora, axis=1, keepdims=True)
    clip_factors = np.minimum(1.0, C / np.maximum(norms, 1e-12))
    vecs_clipped = vecs_lora * clip_factors

    # Add DP noise
    noise_std = sigma * C
    noise = np.random.normal(0, noise_std, size=vecs_clipped.shape).astype(np.float32)
    vecs_noisy = vecs_clipped + noise

    # Final normalization
    vecs_out = _l2_normalize_rows(vecs_noisy)

    return {p: vecs_out[i] for i, p in enumerate(paths)}


# ==============================
# CLI
# ==============================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.05])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--cosine-threshold", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


# ==============================
# MAIN
# ==============================
def main():
    args = parse_args()
    np.random.seed(args.seed)

    image_dir = args.image_dir if args.image_dir else prepare_cifar10(Path("data"))

    print("Embedding images with CLIP...")
    embeddings = embed_directory(image_dir)

    total_embeddings = len(embeddings)

    for sigma in args.sigmas:
        print(f"\n[DP-LoRA MODE | sigma={sigma}]")

        dp_embeddings = dp_lora_mechanism(
            embeddings,
            C=args.c,
            sigma=sigma,
            rank=args.rank,
        )

        duplicates = find_near_duplicates(
            dp_embeddings,
            cosine_threshold=args.cosine_threshold,
            batch_size=args.batch_size,
        )

        duplicate_counts = {k: len(v) for k, v in duplicates.items()}
        avg_duplicates = float(np.mean(list(duplicate_counts.values())))

        results = {
            "mode": "dp_lora",
            "C": args.c,
            "sigma": sigma,
            "rank": args.rank,
            "total_embeddings": total_embeddings,
            "duplicate_counts": duplicate_counts,
            "avg_duplicates": avg_duplicates,
        }

        output_path = Path(
            f"results_dp_lora_{sigma:.2f}.pkl" if len(args.sigmas) > 1 else "results_dp_lora.pkl"
        )

        with output_path.open("wb") as f:
            pickle.dump(results, f)

        print(f"Avg duplicates: {avg_duplicates:.4f}")
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
