import numpy as np

def l2(img1, img2):
    return np.linalg.norm(img1 - img2)

def patch_l2(img1, img2, patch_size=32):
    h, w, _ = img1.shape
    max_dist = 0

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            p1 = img1[i:i+patch_size, j:j+patch_size]
            p2 = img2[i:i+patch_size, j:j+patch_size]

            dist = l2(p1, p2)
            max_dist = max(max_dist, dist)

    return max_dist
