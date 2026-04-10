import torch
from pathlib import Path
from PIL import Image
import numpy as np

from dataset import generate_cifar10_splits, prepare_cifar10
from membership_inference import (
    compute_diffusion_loss,
    loss_threshold_attack,
    LiRAAttack,
)

# ==============================
# CONFIG
# ==============================

TIMESTEP = 100
NUM_IMAGES = 100   # reduce for feasibility

# ==============================
# DUMMY MODEL (for structure)
# ==============================

class DummyModel:
    def eval(self):
        return self

    def __call__(self, x, t):
        class Output:
            sample = torch.randn_like(x)
        return Output()


class DummyScheduler:
    def add_noise(self, x, noise, t):
        return x + noise


# ==============================
# IMAGE LOADING
# ==============================

def load_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


# ==============================
# MAIN
# ==============================

def main():
    print(" Preparing dataset...")
    image_dir = prepare_cifar10(Path("data"))

    splits = generate_cifar10_splits()

    model = DummyModel()
    scheduler = DummyScheduler()

    member_losses = []
    nonmember_losses = []

    print(" Computing diffusion losses...")

    # take first model for demo
    members, nonmembers = splits[0]

    for idx in members[:NUM_IMAGES]:
        path = image_dir / f"img_{idx:05d}_class{str(idx % 10)}.png"
        img = load_image(path)

        loss = compute_diffusion_loss(
            model, scheduler, img, timestep=TIMESTEP
        )
        member_losses.append(loss)

    for idx in nonmembers[:NUM_IMAGES]:
        path = image_dir / f"img_{idx:05d}_class{str(idx % 10)}.png"
        img = load_image(path)

        loss = compute_diffusion_loss(
            model, scheduler, img, timestep=TIMESTEP
        )
        nonmember_losses.append(loss)

    # ----------------------------
    # Threshold attack
    # ----------------------------
    print("\n Running threshold attack...")

    tau, tpr, fpr = loss_threshold_attack(member_losses, nonmember_losses)

    print(f"Threshold: {tau:.4f}")
    print(f"TPR: {tpr:.4f}")
    print(f"FPR: {fpr:.4f}")

    # ----------------------------
    # LiRA attack
    # ----------------------------
    print("\n Running LiRA attack...")

    lira = LiRAAttack(member_losses, nonmember_losses)

    sample_loss = member_losses[0]
    score = lira.score(sample_loss)

    print(f"LiRA score: {score:.4f}")
    print(f"Prediction: {'member' if score > 0 else 'non-member'}")


if __name__ == "__main__":
    main()
