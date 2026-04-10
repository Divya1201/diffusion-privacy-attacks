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

from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ==============================
# CONFIG
# ==============================

TIMESTEP = 100
NUM_IMAGES = 500   
NUM_MODELS_TO_USE = 16

def load_cifar_diffusion_model(device="cpu"):
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipe = pipe.to(device)
    return pipe.unet, pipe.scheduler

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(" Preparing dataset...")
    image_dir = prepare_cifar10(Path("data"))

    print(" Generating CIFAR splits...")
    splits = generate_cifar10_splits()

    print(" Loading pretrained CIFAR diffusion model...")
    model, scheduler = load_cifar_diffusion_model(device)

    member_losses = []
    nonmember_losses = []

    print(" Computing diffusion losses across multiple splits...")

    for model_idx in range(NUM_MODELS_TO_USE):
        print(f"\n Processing split/model {model_idx}...")

        members, nonmembers = splits[model_idx]

        # Member Losses
        for idx in members[:NUM_IMAGES]:
            path = image_dir / f"img_{idx:05d}_class{str(idx % 10)}.png"
            img = load_image(path)

            loss = compute_diffusion_loss(
                model, scheduler, img, timestep=TIMESTEP, device=device
            )
            member_losses.append(loss)

        # Non-Member Losses
        for idx in nonmembers[:NUM_IMAGES]:
            path = image_dir / f"img_{idx:05d}_class{str(idx % 10)}.png"
            img = load_image(path)

            loss = compute_diffusion_loss(
                model, scheduler, img, timestep=TIMESTEP, device=device
            )
            nonmember_losses.append(loss)

    # ----------------------------
    # SAVE LOSSES (IMPORTANT)
    # ----------------------------
    np.save("member_losses.npy", np.array(member_losses))
    np.save("nonmember_losses.npy", np.array(nonmember_losses))

    print(" Saved losses for evaluation")
    
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

    labels = np.concatenate([
        np.ones(len(member_losses)),
        np.zeros(len(nonmember_losses))
    ])

    scores = -np.concatenate([member_losses, nonmember_losses])

    fpr, tpr, _ = roc_curve(labels, scores)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Membership Inference ROC (CIFAR)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
