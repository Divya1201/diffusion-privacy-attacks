import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

import pickle 

from visualize import show_top_results
from attack import AttackResult

df = pd.read_csv("attack_results.csv")

print(df.head())

# --------------------------------
# PRECISION - RECALL CURVE
# --------------------------------
y_true = df["extracted"].astype(int)
scores = -df["adaptive_score"]   # invert (lower score = higher confidence)

precision, recall, _ = precision_recall_curve(y_true, scores)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# --------------------------------
# DUPLICATE HISTOGRAM
# --------------------------------
with open("duplicate_counts.pkl", "rb") as f:
    duplicate_counts = pickle.load(f)

values = list(duplicate_counts.values())

plt.figure()
plt.hist(values, bins=50)
plt.xlabel("Duplicate Count")
plt.ylabel("Frequency")
plt.title("Duplicate Distribution")
plt.grid()
plt.show()

# -------------------------------
# ROC CURVE
# -------------------------------
labels = df["extracted"].astype(int)
scores = -df["adaptive_score"]

fpr, tpr, _ = roc_curve(labels, scores)

plt.figure()
plt.plot(fpr, tpr)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve (Membership Inference)")
plt.grid()
plt.show()

# ROC Curve (for 16 diffusion models - CIFAR)

member_losses = np.load("member_losses.npy")
nonmember_losses = np.load("nonmember_losses.npy")

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

# -------------------------------
# TOP EXTRACTED IMAGES
# -------------------------------
# convert df → AttackResult objects (simplified)
results = []

for _, row in df.iterrows():
    results.append(
        AttackResult(
            query_path=row["query_path"],
            match_path=row["match_path"],
            l2_norm=row["l2_norm"],
            mean_clique_dist=row["mean_clique_dist"],
            adaptive_score=row["adaptive_score"],
            clique_size=row["clique_size"],
            extracted=row["extracted"],
        )
    )

show_top_results(results, n=9, only_extracted=True)
