#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from diffusion_privacy_attacks import AttackConfig, run_memorization_attack
from diffusion_privacy_attacks.visualize import show_pair

# =========================
# ARGUMENTS
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diffusion memorization attack (generate-and-filter pipeline)."
    )

    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory with generated images")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory with dataset images")
    parser.add_argument("--output", type=Path, default=Path("attack_results.csv"))

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--cluster-threshold", type=float, default=50.0)
    parser.add_argument("--cluster-min-size", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=50.0)

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalization",
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_args()

    config = AttackConfig(
        image_size=args.image_size,
        top_k=args.top_k,
        normalize=not args.no_normalize,

        # NEW PARAMETERS
        patch_size=args.patch_size,
        cluster_threshold=args.cluster_threshold,
        cluster_min_size=args.cluster_min_size,
        match_threshold=args.match_threshold,
    )

    print("🚀 Running memorization attack...")
    print(f"📂 Generated dir: {args.generated_dir}")
    print(f"📂 Reference dir: {args.reference_dir}")

    results = run_memorization_attack(
        generated_dir=args.generated_dir,
        reference_dir=args.reference_dir,
        config=config,
    )

    # =========================
    # SAVE CSV
    # =========================

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # UPDATED HEADER
        writer.writerow([
            "query_path",
            "match_path",
            "mse_distance",
            "l2_distance",
            "cluster_size",
        ])

        for result in results:
            writer.writerow([
                str(result.query_path),
                str(result.match_path),
                f"{result.mse_distance:.8f}",
                f"{result.l2_distance:.8f}",
                result.cluster_size,
            ])

    print(f"\n✅ Saved {len(results)} rows to {args.output}")

    print("\n Show top matches...")
    for result in results[:3]:    # show first 3
        show_pair(result.query_path, result.match_path)
        

if __name__ == "__main__":
    main()
