from pathlib import Path
import shutil
from dataset import prepare_cifar10, generate_cifar10_splits

# ==============================
# CONFIG
# ==============================

NUM_MODELS = 2 #16
OUTPUT_DIR = Path("cifar_models")
MAX_IMAGES_PER_MODEL = 200 #5000  


# ==============================
# PREPARE SPLITS (NO TRAINING)
# ==============================

def prepare_cifar_splits():
    """
    Prepare CIFAR-10 dataset splits for membership inference.

    NOTE:
    This script only creates dataset splits (50% per model).
    It does NOT train diffusion models due to computational constraints.
    """

    print(" Preparing CIFAR-10 dataset...")
    image_dir = prepare_cifar10(Path("data"), num_images=1000)
    #image_dir = prepare_cifar10(Path("data"))

    print(" Generating dataset splits...")
    splits = generate_cifar10_splits(n_models=NUM_MODELS)

    print(" Creating model-specific training folders...")

    for model_idx, (members, _) in enumerate(splits):
        model_dir = OUTPUT_DIR / f"model_{model_idx}"
        train_dir = model_dir / "train"

        train_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n Model {model_idx}: preparing training data...")

        selected_members = members[:MAX_IMAGES_PER_MODEL]

        for idx in selected_members:
            src = image_dir / f"img_{idx:05d}_class{idx % 10}.png"

            if src.exists():
                shutil.copy(src, train_dir / src.name)

        print(f" Model {model_idx} ready ({len(selected_members)} images)")

    print("\n All dataset splits prepared!")


if __name__ == "__main__":
    prepare_cifar_splits()
