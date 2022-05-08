import os
import logging
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, davies_bouldin_score
from src_utils import settings
from .cluster_utils import compute_dunn_index, compute_cluster_sizes

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)

project_root = settings.get_project_root()


class KMEANS_CLUSTERER:
    def __init__(self,
                 n_clusters_range=None,
                 init_range=None,
                 max_iter_range=None,
                 n_init=50,
                 random_state=None):

        self.n_clusters_range = n_clusters_range
        self.init_range = init_range
        self.max_iter_range = max_iter_range
        self.n_init = n_init
        self.random_state = random_state

    @staticmethod
    def set_save_path(file_args):
        dir_name = 'Layer{}_Classes{}_SamplesPerRootClass{}_OriginalOnly{}_PCAComp{}_PCAVar{}_PCAScaler{}'.format(
            file_args['layer'],
            ''.join(file_args['classes']),
            file_args['n_samples'],
            file_args['original_only'],
            file_args['pca_comp'],
            file_args['pca_var'],
            file_args['pca_scaler']
        )
        save_path = os.path.join(project_root, 'src', 'data', 'SidBERT_data', 'kmeans', dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def fit_kmeans(self,
                   data=None,
                   true_labels=None,
                   return_result=False,
                   save_to_disk=True,
                   file_args=None):

        save_path = self.set_save_path(file_args)

        trial_idx = []
        trial_results_ = []
        idx = 1
        for max_iter in self.max_iter_range:
            for init in self.init_range:
                for n in self.n_clusters_range:
                    logging.info('Trial={}--Clusters={}, Init={}, Iterations={}'.format(idx, n, init, max_iter))
                    kmeans_ = KMeans(n_clusters=n,
                                     init=init,
                                     max_iter=max_iter,
                                     n_init=self.n_init,
                                     random_state=self.random_state)
                    kmeans_.fit(data)

                    # Compute and sort the different cluster sizes
                    kmeans_cluster_sizes = compute_cluster_sizes(labels=kmeans_.labels_)

                    # Compute the silhouette score
                    kmeans_silhouette_score = silhouette_score(X=data, labels=kmeans_.labels_)
                    # Compute the Dunn index
                    kmeans_dunn_index, kmeans_dunn_index_corrected = compute_dunn_index(X=data,
                                                                                        cluster_centers=kmeans_.cluster_centers_,
                                                                                        labels=kmeans_.labels_)
                    # Compute the Davies Bouldin Score
                    kmeans_db_score = davies_bouldin_score(X=data, labels=kmeans_.labels_)
                    # Using real labels, compute the rand index
                    kmeans_adjusted_rand_score = adjusted_rand_score(labels_true=true_labels,
                                                                     labels_pred=kmeans_.labels_)
                    # Using real labels, compute the mutual information score
                    kmeans_adjusted_mutual_info_score = adjusted_mutual_info_score(labels_true=true_labels,
                                                                                   labels_pred=kmeans_.labels_)
                    results_ = {
                        'cluster_centers_': kmeans_.cluster_centers_,
                        'labels_': kmeans_.labels_,
                        'inertia_': np.array([kmeans_.inertia_], dtype=object),
                        'n_iter_': np.array([kmeans_.n_iter_], dtype=object),
                        'n_features_in_': np.array([kmeans_.n_features_in_], dtype=object),
                        'silhouette_score': np.array([kmeans_silhouette_score], dtype=object),
                        'dunn_index': np.array([kmeans_dunn_index], dtype=object),
                        'dunn_index_corrected': np.array([kmeans_dunn_index_corrected], dtype=object),
                        'db_score': np.array([kmeans_db_score], dtype=object),
                        'adjusted_rand_score': np.array([kmeans_adjusted_rand_score], dtype=object),
                        'adjusted_mutual_info_score': np.array([kmeans_adjusted_mutual_info_score], dtype=object),
                        'cluster_sizes': np.array([kmeans_cluster_sizes], dtype=object)
                    }
                    if save_to_disk:
                        logger.info('Saving to disk...')
                        # Creates individual files for each setting of kmeans HPs
                        kmeans_file_name = 'Clusters{}_Init{}_Iter{}'.format(n, init, max_iter)
                        np.savez_compressed(file=os.path.join(save_path, kmeans_file_name + '.npz'), **results_)

                    if return_result:
                        trial_idx.append('Trial={}--Clusters={}, Init={}, Iterations={}'.format(idx, n, init, max_iter))
                        trial_results_.append(results_)

                    idx += 1

        if len(trial_idx) > 0:
            logger.info('Returning results for all trials...')
            return np.array(trial_idx, dtype=object), np.array(trial_results_, dtype=object)







