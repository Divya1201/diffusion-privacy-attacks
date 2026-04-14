import torch
from pathlib import Path
from PIL import Image
import numpy as np

from dataset import generate_cifar10_splits, prepare_cifar10
from membership_inference import (
    compute_diffusion_loss,
    loss_threshold_attack,
)

from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import norm

# ==============================
# CONFIG
# ==============================

TIMESTEP = 100
NUM_IMAGES = 50     #500   
NUM_MODELS_TO_USE = 2     #16

MODEL_DIR = Path("models")

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
    splits = generate_cifar10_splits(n_models=NUM_MODELS_TO_USE)

    all_images = sorted(image_dir.glob("img_*.png"))
    
    # ----------------------------
    # Load trained models
    # ----------------------------
    print(" Loading trained shadow models...")

    models = []
    for i in range(NUM_MODELS_TO_USE):
        model_path = MODEL_DIR / f"model_{i}"
        pipe = DDPMPipeline.from_pretrained(model_path)
        pipe = pipe.to(device)
        models.append((pipe.unet, pipe.scheduler))

    # ----------------------------
    # Compute IN / OUT losses
    # ----------------------------
    print(" Computing IN / OUT losses (TRUE LiRA)...")

    in_losses = []
    out_losses = []

    for img_idx in range(NUM_IMAGES):
        path = all_images[img_idx]
        img = load_image(path)

        for model_idx, (model, scheduler) in enumerate(models):
            members, nonmembers = splits[model_idx]

            loss = compute_diffusion_loss(
                model, scheduler, img, timestep=TIMESTEP, device=device
            )

            if img_idx in members:
                in_losses.append(loss)
            else:
                out_losses.append(loss)

    in_losses = np.array(in_losses)
    out_losses = np.array(out_losses)

    print(f"IN samples: {len(in_losses)}")
    print(f"OUT samples: {len(out_losses)}")

    
    
    # ----------------------------
    # Threshold attack
    # ----------------------------
    print("\n Running threshold attack...")

    tau, tpr, fpr = loss_threshold_attack(in_losses, out_losses)

    print(f"Threshold: {tau:.4f}")
    print(f"TPR: {tpr:.4f}")
    print(f"FPR: {fpr:.4f}")

    # ----------------------------
    # LiRA attack 
    # ----------------------------
    print("\n Running LiRA...")

    # Fit Gaussian distributions
    mu_in, std_in = np.mean(in_losses), np.std(in_losses)
    mu_out, std_out = np.mean(out_losses), np.std(out_losses)

    print(f"IN dist: mean={mu_in:.4f}, std={std_in:.4f}")
    print(f"OUT dist: mean={mu_out:.4f}, std={std_out:.4f}")

    # Likelihood ratio
    def lira_score(loss):
        p_in = norm.pdf(loss, mu_in, std_in + 1e-8)
        p_out = norm.pdf(loss, mu_out, std_out + 1e-8)
        return np.log(p_in + 1e-12) - np.log(p_out + 1e-12)

    # Compute LiRA scores for ALL samples
    member_scores = [lira_score(loss) for loss in in_losses]
    nonmember_scores = [lira_score(loss) for loss in out_losses]

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
    # Score statistics 
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
