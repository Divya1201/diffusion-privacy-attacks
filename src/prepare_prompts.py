from pathlib import Path
import pickle

from dataset import prepare_cifar10
from clip_utils import embed_directory, find_near_duplicates


# ==============================
# PAPER-CONSISTENT PROMPTS
# ==============================

PAPER_PROMPTS = [
    "a stock photo of a smiling woman looking at the camera",
    "a professional portrait of a smiling woman",
    "a close up photo of a smiling woman",

    "a product image of white sneakers on plain background",
    "white sneakers product photography",

    "a professional headshot of a man in a suit studio lighting",
    "a corporate portrait of a man in formal attire",

     "a close-up photo of a dog sitting on grass",
     "a dog portrait outdoors natural light",
     "a pet dog sitting in a park"
]


def main():
    # --------------------------------
    # (Optional) CLIP duplicate analysis
    # --------------------------------
    print(" Preparing dataset...")
    image_dir = prepare_cifar10(Path("data"), num_images=50000)    #1000
    #image_dir = prepare_cifar10(Path("data"))

    print(" Embedding images with CLIP...")
    embeddings = embed_directory(image_dir)
    
    print(f" Total embeddings: {len(embeddings)}")
    
    print(" Finding duplicates (Block-wise)...")
    duplicates = find_near_duplicates(embeddings, cosine_threshold=0.9, batch_size=1000)

    duplicate_counts = {k: len(v) for k, v in duplicates.items()}

    # Save for evaluation (histogram)
    with open("duplicate_counts.pkl", "wb") as f:
        pickle.dump(duplicate_counts, f)

    print(" Saved duplicate counts to duplicate_counts.pkl")

    # --------------------------------
    # USE PAPER PROMPTS (NOT CIFAR)
    # --------------------------------
    print("\n USING PAPER PROMPTS:")
    for p in PAPER_PROMPTS:
        print(f" - {p}")

    # Save prompts
    output_file = Path("prompts.txt")
    with open(output_file, "w") as f:
        for p in PAPER_PROMPTS:
            f.write(p + "\n")

    print(f"\n Saved prompts to {output_file}")


if __name__ == "__main__":
    main()
