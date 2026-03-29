#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from diffusion_privacy_attacks import AttackConfig, run_memorization_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline nearest-neighbor memorization attack for diffusion outputs."
    )
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory with generated images")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory with candidate training images")
    parser.add_argument("--output", type=Path, default=Path("attack_results.csv"), help="Output CSV file")
    parser.add_argument("--image-size", type=int, default=256, help="Resize side length for comparisons")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest matches per generated image")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalization from [0,255] to [0,1]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AttackConfig(
        image_size=args.image_size,
        top_k=args.top_k,
        normalize=not args.no_normalize,
    )

    results = run_memorization_attack(
        generated_dir=args.generated_dir,
        reference_dir=args.reference_dir,
        config=config,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_path", "match_path", "mse_distance", "l2_distance"])
        for result in results:
            writer.writerow(
                [
                    str(result.query_path),
                    str(result.match_path),
                    f"{result.mse_distance:.8f}",
                    f"{result.l2_distance:.8f}",
                ]
            )

    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
