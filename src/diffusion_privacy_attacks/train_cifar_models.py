from pathlib import Path
import shutil
from dataset import prepare_cifar10, generate_cifar10_splits

# ==============================
# CONFIG
# ==============================

NUM_MODELS = 16
OUTPUT_DIR = Path("cifar_models")
MAX_IMAGES_PER_MODEL = 5000   # reduce for feasibility

# ==============================
# MAIN
# ==============================

def main():
    print(" Preparing CIFAR-10 dataset...")
    image_dir = prepare_cifar10(Path("data"))

    print(" Generating dataset splits...")
    splits = generate_cifar10_splits(n_models=NUM_MODELS)

    print(" Creating model training folders...")

    for model_idx, (members, nonmembers) in enumerate(splits):
        model_dir = OUTPUT_DIR / f"model_{model_idx}"
        train_dir = model_dir / "train"

        train_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n Model {model_idx}: copying training data...")

        # limit for practicality
        selected_members = members[:MAX_IMAGES_PER_MODEL]

        for idx in selected_members:
            src = image_dir / f"img_{idx:05d}_class{str(idx % 10)}.png"
            dst = train_dir / src.name

            if src.exists():
                shutil.copy(src, dst)

        print(f" Model {model_idx} dataset ready ({len(selected_members)} images)")

    print("\n All model datasets prepared!")


if __name__ == "__main__":
    main()
