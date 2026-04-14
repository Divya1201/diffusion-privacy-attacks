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
NUM_IMAGES = 50     #500   
NUM_MODELS_TO_USE = 2     #16

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
    image_dir = prepare_cifar10(Path("data"), num_images=1000)
    #image_dir = prepare_cifar10(Path("data"))

    print(" Generating CIFAR splits...")
    splits = generate_cifar10_splits()

    print(" Loading pretrained CIFAR diffusion model...")
    model, scheduler = load_cifar_diffusion_model(device)

    all_images = sorted(image_dir.glob("img_*.png"))
    
    member_losses = []
    nonmember_losses = []

    print(" Computing diffusion losses across multiple splits...")

    for model_idx in range(NUM_MODELS_TO_USE):
        torch.manual_seed(model_idx)
        print(f"\n Processing split/model {model_idx}...")

        members, nonmembers = splits[model_idx]

        # Member Losses
        for idx in members[:NUM_IMAGES]:
            if idx >= len(all_images):
                continue  # safety check

            path = all_images[idx]
            img = load_image(path)

            loss = compute_diffusion_loss(
                model, scheduler, img, timestep=TIMESTEP, device=device
            )
            member_losses.append(loss)

        # Non-Member Losses
        for idx in nonmembers[:NUM_IMAGES]:
            if idx >= len(all_images):
                continue  # safety check

            path = all_images[idx]
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
    # LiRA attack (CORRECT VERSION)
    # ----------------------------
    print("\n Running LiRA (distribution-based)...")

    lira = LiRAAttack(member_losses, nonmember_losses)

    # Compute LiRA scores for ALL samples
    member_scores = [lira.score(loss) for loss in member_losses]
    nonmember_scores = [lira.score(loss) for loss in nonmember_losses]

    # Convert to numpy
    member_scores = np.array(member_scores)
    nonmember_scores = np.array(nonmember_scores)

    # Labels
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores))
    ])

    # Combine scores
    scores = np.concatenate([member_scores, nonmember_scores])

    # ----------------------------
    # Accuracy
    # ----------------------------
    preds = (scores > 0).astype(int)
    accuracy = (preds == labels).mean()

    print(f"LiRA Accuracy: {accuracy:.4f}")

    # ----------------------------
    # Score statistics (VERY USEFUL)
    # ----------------------------
    print(f"Mean member score: {member_scores.mean():.4f}")
    print(f"Mean non-member score: {nonmember_scores.mean():.4f}")

    # ----------------------------
    # ROC Curve (LiRA)
    # ----------------------------
    fpr, tpr, _ = roc_curve(labels, scores)

    plt.figure()
    plt.plot(fpr, tpr, label="LiRA")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("LiRA ROC Curve")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
