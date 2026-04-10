"""
CLIP embedding utilities (§4.2)

Paper: "we first embed each image to a 512-dimensional vector using CLIP,
and then perform the all-pairs comparison between images in this lower-
dimensional space (increasing efficiency by over 1500×)."

Model: ViT-B/32 pretrained on OpenAI data (same as paper).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paper §4.2: CLIP ViT-B/32 (512-dimensional embeddings)
_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
_model = _model.to(device).eval()


def get_clip_embedding(image_path: Path) -> np.ndarray:
    """
    Embed a single image with CLIP ViT-B/32.
    Returns a unit-normalised 512-dim float32 vector.
    """
    image = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _model.encode_image(image)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype(np.float32)


def embed_directory(
    directory: Path,
    image_size: int = 512,   # unused here — CLIP handles its own resize
) -> Dict[Path, np.ndarray]:
    """
    Embed all images in a directory.
    Returns dict mapping Path → 512-dim embedding.
    """
    paths: List[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        paths.extend(directory.rglob(suffix))
    paths = sorted(paths)

    embeddings: Dict[Path, np.ndarray] = {}
    for i, p in enumerate(paths):
        embeddings[p] = get_clip_embedding(p)
        if (i + 1) % 100 == 0:
            print(f"  Embedded {i+1}/{len(paths)}")
    return embeddings


def cosine_similarity_matrix(
    embeddings: Dict[Path, np.ndarray],
) -> tuple[np.ndarray, list[Path]]:
    """
    Compute the full N×N cosine similarity matrix for fast duplicate detection.
    Returns (matrix, ordered_paths).
    """
    paths = sorted(embeddings.keys())
    vecs = np.stack([embeddings[p] for p in paths])   # already unit-normalised
    sim_matrix = vecs @ vecs.T
    return sim_matrix.astype(np.float32), paths


def find_near_duplicates(
    embeddings: Dict[Path, np.ndarray],
    cosine_threshold: float = 0.9,
) -> Dict[Path, List[Path]]:
    """
    Find all near-duplicate pairs using CLIP cosine similarity ≥ threshold.
    Paper §4.2: "count two examples as near-duplicates if their CLIP embeddings
    have a high cosine similarity."
    """
    sim_matrix, paths = cosine_similarity_matrix(embeddings)
    n = len(paths)
    duplicates: Dict[Path, List[Path]] = {p: [] for p in paths}

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= cosine_threshold:
                duplicates[paths[i]].append(paths[j])
                duplicates[paths[j]].append(paths[i])

    return duplicates
