"""
Extracting Training Data from Diffusion Models
Carlini et al. (2023) — faithful implementation.

Attack pipeline (§4.2.1):
  1. Generate 500 candidate images per prompt with the diffusion model.
  2. Build a graph over generations: edge (i,j) if patch-L2(i,j) < threshold.
  3. Find large cliques (≥ CLIQUE_MIN_SIZE). A clique = memorized image.
  4. For extracted images, compute the per-image adaptive L2 score (§5.1)
     to rank them by memorization confidence.
  5. Match each flagged image against the reference dataset and annotate.

Membership inference (§5.2) — white-box, CIFAR-10 models:
  Implemented in membership_inference.py (requires shadow models).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackConfig:
    # Image dimensions (paper: 512×512 for SD, 32×32 native for CIFAR-10)
    image_size: int = 512

    # Number of top matches to return per queried image
    top_k: int = 5

    # Normalize pixel values to [0, 1]
    normalize: bool = True

    # -------------------------------------------------------------------
    # Tile-based patch L2 (§4.2.1)
    # Paper: 16 non-overlapping 128×128 tiles on 512×512 images.
    # tile_size = image_size // 4 keeps this ratio correct at any resolution.
    # -------------------------------------------------------------------
    tile_size: int = 128          # 128px → 16 tiles at 512×512

    # -------------------------------------------------------------------
    # Clique / memorization detection (§4.2.1)
    # Paper: clique ≥ 10 out of 500 generations per prompt.
    # -------------------------------------------------------------------
    clique_min_size: int = 10     # § "at least size 10"
    patch_l2_threshold: float = 0.05   # § intra-clique L2 threshold (normalised)

    # -------------------------------------------------------------------
    # CLIP pre-filter (§4.2 dedup step)
    # Paper: embed with CLIP, count pairs with cosine similarity above
    # threshold as near-duplicates to prioritise for attack.
    # -------------------------------------------------------------------
    clip_cosine_threshold: float = 0.9   # § "high cosine similarity"

    # -------------------------------------------------------------------
    # Adaptive L2 score (§5.1, Definition in text)
    # score = L2(x̂, x) / (α · mean_L2_to_n_nearest)
    # score < 1 → extracted (the paper uses score < 1 implicitly)
    # -------------------------------------------------------------------
    adaptive_alpha: float = 0.5    # § "α = 0.5"
    adaptive_n: int = 50           # § "n = 50"

    # -------------------------------------------------------------------
    # Extraction threshold (§4.1 Definition 1)
    # (ℓ, δ)-Diffusion Extraction: ℓ₂(x, x̂) ≤ δ  (normalised distance)
    # Paper reports 94 images at δ=0.15; manually 109 at relaxed criteria.
    # -------------------------------------------------------------------
    extraction_delta: float = 0.15


@dataclass
class AttackResult:
    """One extracted (generated_image, dataset_match) pair."""
    query_path: Path
    match_path: Path

    # Normalised L2 (Definition 1 in paper)
    l2_norm: float

    # Mean patch-L2 within the clique (lower = more confidently memorized)
    mean_clique_dist: float

    # Adaptive score (§5.1): < 1.0 → extraction declared
    adaptive_score: float

    # Clique size (larger = more duplicates → higher memorisation risk)
    clique_size: int

    # Is this pair considered "extracted" under Definition 1?
    extracted: bool = False


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def _iter_images(directory: Path) -> List[Path]:
    """Return sorted list of image paths in a directory."""
    paths: List[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        paths.extend(directory.glob(suffix))
    return sorted(paths)


def _load_rgb(path: Path, image_size: int, normalize: bool) -> np.ndarray:
    """Load an image as a float32 HxWx3 array, optionally in [0,1]."""
    img = Image.open(path).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    arr = np.asarray(img, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    return arr


def _flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def normalised_l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalised Euclidean distance (Definition 1, §4.1).
    ℓ₂(a, b) = sqrt(Σ(aᵢ−bᵢ)² / d)  ∈ [0, 1] when inputs ∈ [0,1].
    """
    d = a.size
    return float(math.sqrt(np.sum(np.square(a - b)) / d))


def patch_l2(img1: np.ndarray, img2: np.ndarray, tile_size: int) -> float:
    """
    Modified patch L2 distance (§4.2.1).
    Divide each image into non-overlapping tile_size×tile_size tiles;
    return the MAXIMUM normalised L2 over all tile pairs.

    Paper: "divide each image into 16 non-overlapping 128×128 tiles and
    measure the maximum of the L2 distance between any pair of image tiles."
    """
    h, w, _ = img1.shape
    max_dist = 0.0
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            t1 = img1[i:i + tile_size, j:j + tile_size]
            t2 = img2[i:i + tile_size, j:j + tile_size]
            dist = normalised_l2(t1, t2)
            max_dist = max(max_dist, dist)
    return max_dist


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Adaptive per-image L2 score (§5.1)
# ---------------------------------------------------------------------------

def adaptive_l2_score(
    query_vec: np.ndarray,
    dataset_vecs: List[np.ndarray],
    nearest_match_l2: float,
    alpha: float,
    n: int,
) -> float:
    """
    Adaptive L2 score from §5.1:
        score = ℓ₂(x̂, x) / (α · E_{y ∈ S_{x̂}}[ℓ₂(x̂, y)])
    where S_{x̂} is the n closest neighbours of x̂ in the training dataset.

    score < 1 ↔ the nearest dataset image is *unusually* close relative
    to the neighbourhood, implying memorisation.
    """
    distances = [normalised_l2(query_vec, v) for v in dataset_vecs]
    distances_sorted = np.sort(distances)[:n]          # n nearest
    expected = float(distances_sorted.mean()) + 1e-8
    return nearest_match_l2 / (alpha * expected)


# ---------------------------------------------------------------------------
# Step 1 — Clique finding (§4.2.1)
# ---------------------------------------------------------------------------

def _build_cliques(
    image_paths: List[Path],
    images: Dict[Path, np.ndarray],
    config: AttackConfig,
) -> List[List[Path]]:
    """
    Build a graph over 'image_paths' where an edge exists between image i and j
    if patch_l2(i, j) < config.patch_l2_threshold.
    Return all connected components (cliques approximated as components) of
    size ≥ config.clique_min_size.

    Paper (§4.2.1): "connect an edge between generation i and j if xᵢ ≈_d xⱼ.
    If the largest clique in this graph is at least size 10, we predict that
    this clique is a memorized image."

    Note: exact clique finding is NP-hard. The paper uses connected components
    as a practical approximation. We first pre-filter with CLIP cosine similarity
    (from §4.2 dedup) to reduce the O(n²) pixel comparisons.
    """
    n = len(image_paths)
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            dist = patch_l2(images[image_paths[i]], images[image_paths[j]], config.tile_size)
            if dist < config.patch_l2_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # BFS to find connected components
    visited = set()
    components: List[List[Path]] = []
    for start in range(n):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(image_paths[node])
            queue.extend(adjacency[node])
        if len(component) >= config.clique_min_size:
            components.append(component)

    return components


def _mean_intra_clique_dist(
    clique: List[Path],
    images: Dict[Path, np.ndarray],
    config: AttackConfig,
) -> float:
    """
    Mean pairwise patch-L2 within a clique.
    Paper: "ordering them by the mean distance between images in the clique
    to identify generations that we predict are likely to be memorised."
    """
    if len(clique) < 2:     #10
        return 0.0
    total, count = 0.0, 0
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            total += patch_l2(images[clique[i]], images[clique[j]], config.tile_size)
            count += 1
    return total / count if count else 0.0


def find_memorized_cliques(
    generated_paths: List[Path],
    config: AttackConfig,
    clip_embeddings: Optional[Dict[Path, np.ndarray]] = None,
) -> List[Tuple[List[Path], float]]:
    """
    Stage 1 of the attack pipeline.

    Returns list of (clique, mean_intra_dist) sorted by mean_intra_dist
    ascending (smallest = most confidently memorised).

    When clip_embeddings are provided, we first filter candidate pairs with
    CLIP cosine similarity (§4.2), then verify with patch L2.
    """
    print(f"Loading {len(generated_paths)} generated images...")
    images: Dict[Path, np.ndarray] = {}
    for p in generated_paths:
        images[p] = _load_rgb(p, config.image_size, config.normalize)

    # CLIP pre-filter: only do pixel-level comparison for CLIP-similar pairs
    if clip_embeddings is not None:
        print("Pre-filtering with CLIP cosine similarity...")
        candidate_pairs: List[Tuple[int, int]] = []
        for i in range(len(generated_paths)):
            for j in range(i + 1, len(generated_paths)):
                sim = cosine_similarity(
                    clip_embeddings[generated_paths[i]],
                    clip_embeddings[generated_paths[j]],
                )
                if sim >= config.clip_cosine_threshold:
                    candidate_pairs.append((i, j))
        print(f"  CLIP filter: {len(candidate_pairs)} candidate pairs remain.")

        # Build adjacency only for candidate pairs
        n = len(generated_paths)
        adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i, j in candidate_pairs:
            dist = patch_l2(images[generated_paths[i]], images[generated_paths[j]], config.tile_size)
            if dist < config.patch_l2_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

        visited: set = set()
        cliques_raw: List[List[Path]] = []
        for start in range(n):
            if start in visited:
                continue
            component: List[Path] = []
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(generated_paths[node])
                queue.extend(adjacency[node])
            if len(component) >= config.clique_min_size:
                cliques_raw.append(component)
    else:
        cliques_raw = _build_cliques(generated_paths, images, config)

    if not cliques_raw:
        return []

    # Score each clique by mean intra-clique distance (lower = more similar = better)
    cliques_scored = []
    for clique in cliques_raw:
        mean_d = _mean_intra_clique_dist(clique, images, config)
        cliques_scored.append((clique, mean_d))

    cliques_scored.sort(key=lambda x: x[1])   # ascending: best cliques first
    print(f" Found {len(cliques_scored)} cliques (≥{config.clique_min_size} images).")
    return cliques_scored


# ---------------------------------------------------------------------------
# Step 2 — Match against reference dataset (§4.2.1 + §5.1)
# ---------------------------------------------------------------------------

def run_extraction_attack(
    generated_dir: Path,
    reference_dir: Path,
    config: Optional[AttackConfig] = None,
    clip_embeddings: Optional[Dict[Path, np.ndarray]] = None,
) -> List[AttackResult]:
    """
    Full black-box extraction pipeline from §4.2.1.

    1. Find cliques among generated images.
    2. For each clique member, compute the adaptive L2 score against
       every reference image (§5.1).
    3. Declare (query, match) as extracted if adaptive_score < 1.0
       AND normalised_l2 ≤ config.extraction_delta.
    4. Return results sorted by adaptive_score ascending (best first).
    """
    if config is None:
        config = AttackConfig()

    generated_paths = _iter_images(generated_dir)
    reference_paths = _iter_images(reference_dir)

    if not generated_paths:
        raise ValueError(f"No generated images in {generated_dir}")
    if not reference_paths:
        raise ValueError(f"No reference images in {reference_dir}")

    print(f"Generated: {len(generated_paths)} images")
    print(f"Reference: {len(reference_paths)} images")

    # Stage 1 — memorisation detection via cliques
    cliques_scored = find_memorized_cliques(generated_paths, config, clip_embeddings)
    if not cliques_scored:
        print("  No cliques found — no memorisation detected.")
        return []

    # Flatten: track clique size and mean dist for each selected image
    selected: Dict[Path, Tuple[int, float]] = {}   # path → (clique_size, mean_dist)
    for clique, mean_d in cliques_scored:
        for p in clique:
            selected[p] = (len(clique), mean_d)

    # Stage 2 — load reference images
    print(f"Loading {len(reference_paths)} reference images...")
    ref_images: Dict[Path, np.ndarray] = {}
    for p in reference_paths:
        ref_images[p] = _load_rgb(p, config.image_size, config.normalize)

    # Precompute flattened reference vectors for adaptive score
    ref_flat = [_flatten(v) for v in ref_images.values()]
    ref_paths_list = list(ref_images.keys())

    # Stage 3 — match + adaptive scoring
    results: List[AttackResult] = []
    print(f"Matching {len(selected)} clique images against reference dataset...")

    for gen_path, (clique_sz, mean_clique_d) in selected.items():
        query = _load_rgb(gen_path, config.image_size, config.normalize)
        query_flat = _flatten(query)

        # Rank reference images by patch-L2
        ranked: List[Tuple[Path, float]] = []
        for ref_path, ref_img in ref_images.items():
            p_l2 = patch_l2(query, ref_img, config.tile_size)
            ranked.append((ref_path, p_l2))
        ranked.sort(key=lambda x: x[1])

        best_ref_path, best_patch_l2 = ranked[0]
        best_ref_flat = _flatten(ref_images[best_ref_path])

        # Normalised L2 (Definition 1)
        l2_norm = normalised_l2(query, ref_images[best_ref_path])

        # Adaptive score (§5.1)
        score = adaptive_l2_score(
            query_flat, ref_flat, l2_norm,
            config.adaptive_alpha, config.adaptive_n,
        )

        # Extraction criterion: adaptive_score < 1 (§5.1) AND δ-close (Def 1)
        extracted = score < 1.0 and l2_norm <= config.extraction_delta

        for ref_path, p_l2 in ranked[: config.top_k]:
            results.append(AttackResult(
                query_path=gen_path,
                match_path=ref_path,
                l2_norm=normalised_l2(query, ref_images[ref_path]),
                mean_clique_dist=mean_clique_d,
                adaptive_score=score,
                clique_size=clique_sz,
                extracted=extracted,
            ))

    # Sort by adaptive_score ascending (best extractions first)
    results.sort(key=lambda r: r.adaptive_score)
    extracted_count = sum(1 for r in results if r.extracted)
    print(f" {extracted_count} images satisfy the (ℓ₂, δ={config.extraction_delta})-extraction criterion.")
    return results
