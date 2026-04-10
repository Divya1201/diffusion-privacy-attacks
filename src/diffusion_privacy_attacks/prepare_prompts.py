from pathlib import Path
from dataset import prepare_cifar10
from clip_utils import embed_directory, find_near_duplicates
import pickle

# ==============================
# DATA → PROMPTS
# ==============================

def path_to_prompt(path):
    cls = path.name.split("class")[-1].split(".")[0]
    return f"CIFAR-10 class {cls}"


def main():
    # 1. Load dataset
    print("📦 Preparing dataset...")
    image_dir = prepare_cifar10(Path("data"))

    # 2. CLIP embeddings
    print("🧠 Embedding images with CLIP...")
    embeddings = embed_directory(image_dir)

    # 3. Duplicate detection
    print("🔍 Finding duplicates...")
    duplicates = find_near_duplicates(embeddings, cosine_threshold=0.9)
    duplicate_counts = {k: len(v) for k, v in duplicates.items()}

    # save duplicate counts for evaluation
    with open("duplicate_counts.pkl", "wb") as f:
        pickle.dump(duplicate_counts, f)

    print(" Saved duplicate counts to duplicate_counts.pkl")

    # 4. Sort
    sorted_items = sorted(
        duplicate_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 5. Select top-K
    top_k = 20
    top_images = [img for img, _ in sorted_items[:top_k]]

    # 6. Convert to prompts
    prompts = list(set(path_to_prompt(p) for p in top_images))

    print("\n🔥 TOP PROMPTS:")
    for p in prompts:
        print(p)

    # 7. Save prompts (IMPORTANT for next step)
    output_file = Path("prompts.txt")
    with open(output_file, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"\n✅ Saved prompts to {output_file}")


if __name__ == "__main__":
    main()
