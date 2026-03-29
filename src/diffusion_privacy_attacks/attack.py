from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AttackConfig:
    """Configuration for a simple memorization attack."""

    image_size: int = 256
    top_k: int = 5
    normalize: bool = True


@dataclass(frozen=True)
class AttackResult:
    """Single nearest-neighbor attack result for one generated image."""

    query_path: Path
    match_path: Path
    mse_distance: float
    l2_distance: float


def _iter_images(directory: Path) -> Iterable[Path]:
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        yield from directory.glob(suffix)


def _load_image(path: Path, image_size: int, normalize: bool) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    return arr


def _flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1)


def _distances(query: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    diff = query - candidate
    mse = float(np.mean(np.square(diff)))
    l2 = float(np.linalg.norm(diff))
    return mse, l2


def run_memorization_attack(
    generated_dir: Path,
    reference_dir: Path,
    config: AttackConfig | None = None,
) -> list[AttackResult]:
    """Find nearest reference images for each generated sample.

    This implementation is a baseline nearest-neighbor attack that can be
    used to detect potential training-data memorization.
    """
    if config is None:
        config = AttackConfig()

    generated_paths = sorted(_iter_images(generated_dir))
    reference_paths = sorted(_iter_images(reference_dir))

    if not generated_paths:
        raise ValueError(f"No generated images found in: {generated_dir}")
    if not reference_paths:
        raise ValueError(f"No reference images found in: {reference_dir}")

    reference_vectors = {
        path: _flatten(_load_image(path, config.image_size, config.normalize))
        for path in reference_paths
    }

    results: list[AttackResult] = []
    for generated_path in generated_paths:
        query = _flatten(_load_image(generated_path, config.image_size, config.normalize))

        ranked: list[tuple[Path, float, float]] = []
        for reference_path, reference_vec in reference_vectors.items():
            mse, l2 = _distances(query, reference_vec)
            ranked.append((reference_path, mse, l2))

        ranked.sort(key=lambda x: x[1])
        for match_path, mse, l2 in ranked[: config.top_k]:
            results.append(
                AttackResult(
                    query_path=generated_path,
                    match_path=match_path,
                    mse_distance=mse,
                    l2_distance=l2,
                )
            )

    results.sort(key=lambda r: (r.query_path, r.mse_distance))
    return results
