from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from clip_utils import get_clip_embedding

# =========================
# CONFIG
# =========================

@dataclass(frozen=True)
class AttackConfig:
    image_size: int = 256
    top_k: int = 5
    normalize: bool = True

    # NEW (important)
    patch_size: int = 32
    cluster_threshold: float = 0.1   # similarity threshold
    cluster_min_size: int = 5         # cluster size for memorization
    match_threshold: float = 50.0     # dataset match threshold


@dataclass(frozen=True)
class AttackResult:
    query_path: Path
    match_path: Path
    mse_distance: float
    l2_distance: float
    cluster_size: int


# =========================
# UTILS
# =========================

def _iter_images(directory: Path) -> Iterable[Path]:
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        yield from directory.glob(suffix)


def _load_image(path: Path, image_size: int, normalize: bool) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    arr = np.asarray(image, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    return arr


def _flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1)


# =========================
# DISTANCES
# =========================

def cosine_distance(a,b):
    return 1-np.dot(a,b)
    
def _patch_l2(img1: np.ndarray, img2: np.ndarray, patch_size: int) -> float:
    h, w, _ = img1.shape
    max_dist = 0.0

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            p1 = img1[i:i + patch_size, j:j + patch_size]
            p2 = img2[i:i + patch_size, j:j + patch_size]

            dist = np.linalg.norm(p1 - p2)
            max_dist = max(max_dist, dist)

    return float(max_dist)


def _distances(query: np.ndarray, candidate: np.ndarray, config: AttackConfig) -> tuple[float, float]:
    diff = query - candidate
    mse = float(np.mean(np.square(diff)))

    l2 = _patch_l2(query, candidate, config.patch_size)

    return mse, l2


# =========================
# CLUSTERING (CORE IDEA)
# =========================

def find_similar_generated(generated_paths: List[Path], config: AttackConfig):
    images = {
        # p: _load_image(p, config.image_size, config.normalize)
        # for p in generated_paths

    images = {
        p: get_clip_embedding(p)
        for p in generated_paths 
    }

    clusters = []
    visited = set()

    for p1 in generated_paths:
        if p1 in visited:
            continue

        group = [p1]

        for p2 in generated_paths:
            if p1 == p2:
                continue

            # dist = _patch_l2(images[p1], images[p2], config.patch_size)
            clip_dist = cosine_distance(images[p1], images[p2])
            
            if clip_dist < config.cluster_threshold:

                img1 = _load_image(p1, config.image_size, config.normalize)
                img2 = _load_image(p2, config.image_size, config.normalize)

                l2_dist = _patch_l2(img1, img2, config.patch_size)
                if l2_dist < config.match_threshold:
                    group.append(p2)

        if len(group) >= config.cluster_min_size:
            print(f"Cluster found with size {len(group)}")
            
            clusters.append(group)
            visited.update(group)

    return clusters


# =========================
# MAIN ATTACK
# =========================

def run_memorization_attack(
    generated_dir: Path,
    reference_dir: Path,
    config: AttackConfig | None = None,
) -> List[AttackResult]:

    if config is None:
        config = AttackConfig()

    generated_paths = sorted(_iter_images(generated_dir))
    reference_paths = sorted(_iter_images(reference_dir))

    if not generated_paths:
        raise ValueError(f"No generated images found in: {generated_dir}")
    if not reference_paths:
        raise ValueError(f"No reference images found in: {reference_dir}")

    # -------------------------
    # STEP 1: CLUSTER GENERATED IMAGES
    # -------------------------
    clusters = find_similar_generated(generated_paths, config)

    if not clusters:
        print("⚠️ No clusters found — no memorization detected")
        return []

    print(f"✅ Found {len(clusters)} clusters")

    # Flatten selected images
    selected_images = []
    cluster_map = {}

    for cluster in clusters:
        for img in cluster:
            selected_images.append(img)
            cluster_map[img] = len(cluster)

    # -------------------------
    # STEP 2: LOAD REFERENCES
    # -------------------------
    reference_vectors = {
        path: _load_image(path, config.image_size, config.normalize)
        for path in reference_paths
    }

    # -------------------------
    # STEP 3: MATCH AGAINST DATASET
    # -------------------------
    results: List[AttackResult] = []

    for generated_path in selected_images:
        query = _load_image(generated_path, config.image_size, config.normalize)

        ranked = []

        for reference_path, reference_img in reference_vectors.items():
            mse, l2 = _distances(query, reference_img, config)
            ranked.append((reference_path, mse, l2))

        # sort by L2 (important)
        ranked.sort(key=lambda x: x[2])

        for match_path, mse, l2 in ranked[: config.top_k]:

            # threshold filtering (paper idea)
            if l2 < config.match_threshold:
                results.append(
                    AttackResult(
                        query_path=generated_path,
                        match_path=match_path,
                        mse_distance=mse,
                        l2_distance=l2,
                        cluster_size=cluster_map[generated_path],
                    )
                )

    # final sort  - Debug view
    # results.sort(key=lambda r: (r.query_path, r.l2_distance))

    # final research view
    results.sort(key=lambda r: (r.l2_distance, -r.cluster_size))
    return results
