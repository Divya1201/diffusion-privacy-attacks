import faiss
import numpy as np

def fast_cluster(embeddings, threshold=0.9):
    X = np.array(list(embeddings.values())).astype("float32")

    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    D, I = index.search(X, k=10)

    clusters = []
    visited = set()

    keys = list(embeddings.keys())

    for i, neighbors in enumerate(I):
        if i in visited:
            continue

        group = []
        for j in neighbors:
            if D[i][list(neighbors).index(j)] < threshold:
                group.append(keys[j])

        if len(group) >= 5:
            clusters.append(group)
            visited.update([keys.index(g) for g in group])

    return clusters
