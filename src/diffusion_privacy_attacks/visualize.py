import matplotlib.pyplot as plt
from PIL import Image


def show_pair(gen_path, ref_path):
    gen = Image.open(gen_path)
    ref = Image.open(ref_path)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(gen)
    plt.title("Generated")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(ref)
    plt.title("Match")
    plt.axis("off")

    plt.show()
