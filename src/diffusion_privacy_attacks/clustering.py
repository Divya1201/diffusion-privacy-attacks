from sklearn.cluster import DBSCAN
import numpy as np

def cluster_images(features):
    clustering = DBSCAN(eps=5, min_samples=5).fit(features)
    return clustering.labels_
