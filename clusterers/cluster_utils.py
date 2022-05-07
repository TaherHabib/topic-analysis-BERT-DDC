import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


def get_points_in_cluster(datapoints=None, labels=None, cluster_label_idx=None):
    return datapoints[labels == cluster_label_idx]


def get_num_dbscan_clusters(labels=None):
    return len(np.unique(labels)) - 1


def compute_cluster_centers(datapoints=None, labels=None):
    cluster_centers = []
    for l in np.unique(labels):
        points_in_cluster = get_points_in_cluster(datapoints=datapoints, labels=labels, cluster_label_idx=l)
        cluster_centers.append(np.mean(points_in_cluster, axis=0))

    return np.array(cluster_centers)


def compute_cluster_sizes(labels=None):
    unique_entries, unique_entries_counts = np.unique(labels, return_counts=True)
    total_size = len(labels)
    unique_entries_sizes = np.array([c/total_size for c in unique_entries_counts])
    dict_cluster_sizes = dict(zip(unique_entries, unique_entries_sizes))
    dict_cluster_sizes = {k: v for k, v in sorted(dict_cluster_sizes.items(), key=lambda item: item[1])}
    return dict_cluster_sizes


def compute_dunn_index(X=None, cluster_centers=None, labels=None):
    numerator = float('inf')
    for c in cluster_centers:
        for t in cluster_centers:
            if (t == c).all():
                continue  # if same cluster, ignore
            numerator = min(numerator, np.linalg.norm(t - c))

    denominator = 0
    max_intra_cluster_dist = []
    for l in np.unique(labels):
        cluster_points = get_points_in_cluster(datapoints=X, labels=labels, cluster_label_idx=l)
        cluster_mean = np.mean(cluster_points, axis=0)
        max_ = 0
        for p in cluster_points:
            dist = np.linalg.norm(cluster_mean - p)
            max_ = max(max_, dist)
            denominator = max(denominator, dist)
        max_intra_cluster_dist.append(max_)

    return numerator / denominator,  numerator / np.mean(max_intra_cluster_dist)


def get_kdist_plot(X=None,
                   k=None,
                   radius_nbrs=1.0,
                   metric='minkowski',
                   p=2,
                   make_plot=False):

    # 'k' is 479 for embeddings of size 240 (layer 7 , classification head)
    # 'k' is 159 for embeddings of size 80 (pooler output)
    nbrs = NearestNeighbors(n_neighbors=k,
                            radius=radius_nbrs,
                            metric=metric,
                            p=p).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)
    # size('distances') = # of points in X. Each element in 'distances' is of len = k
    distances = np.sort(distances, axis=0)
    k_distances = distances[:, k-1]
    if make_plot:
        plt.figure(figsize=(8,8))
        plt.plot(k_distances)
        plt.xlabel('Points/Objects in the dataset', fontsize=12)
        plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
        plt.grid(True, linestyle="--", color='black', alpha=0.4)
        plt.show()
        plt.close()

    return distances


def get_interdist_distribution(X=None,
                               metric='minkowski',
                               p=2,
                               make_plot=False):
    pass






