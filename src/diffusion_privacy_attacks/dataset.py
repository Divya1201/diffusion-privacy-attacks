from torchvision.datasets import CIFAR10
from pathlib import Path
import os


def prepare_cifar10(output_dir: Path, num_images: int = 500):
    """
    Downloads CIFAR-10 and saves images as PNG files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CIFAR10(root=output_dir, download=True)

    save_dir = output_dir / "cifar_images"
    save_dir.mkdir(exist_ok=True)

    for i in range(min(num_images, len(dataset))):
        img, label = dataset[i]
        img.save(save_dir / f"img_{i}.png")

    print(f"✅ Saved {num_images} CIFAR images to {save_dir}")

    return save_dir
