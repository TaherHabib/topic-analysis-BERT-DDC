import os
import logging
import numpy as np

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, davies_bouldin_score
from utils import settings
from .cluster_utils import compute_dunn_index, compute_cluster_centers, compute_cluster_sizes, get_num_dbscan_clusters

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)

project_root = settings.get_project_root()


class DBSCAN_CLUSTERER:
    def __init__(self,
                 eps_range=None,
                 min_samples_range=None,
                 metric='euclidean',
                 algorithm='auto',
                 p=None,
                 n_jobs=-1):
        self.eps_range = eps_range
        self.min_samples_range = min_samples_range
        self.metric = metric
        self.algorithm = algorithm
        self.p = p
        self.n_jobs = n_jobs

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
        save_path = os.path.join(project_root, 'src', 'data', 'SidBERT_data', 'dbscan', dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def fit_dbscan(self, data=None,
                   true_labels=None,
                   return_result=False,
                   save_to_disk=True,
                   file_args=None,
                   use_optics=True):

        save_path = self.set_save_path(file_args)

        trial_idx = []
        trial_results_ = []
        idx = 1
        for eps in self.eps_range:
            for min_samples in self.min_samples_range:
                logging.info('Trial={}--Eps={}, MinSamples={}'.format(idx, eps, min_samples))
                if use_optics:
                    dbscan_ = OPTICS(eps=eps,
                                     min_samples=min_samples,
                                     cluster_method='dbscan',
                                     metric=self.metric,
                                     p=self.p,
                                     n_jobs=self.n_jobs)
                    dbscan_.fit(data)
                else:
                    dbscan_ = DBSCAN(eps=eps,
                                     min_samples=min_samples,
                                     metric=self.metric,
                                     algorithm=self.algorithm,
                                     p=self.p,
                                     n_jobs=self.n_jobs)
                    dbscan_.fit(data)

                # Compute cluster centers for DBSCAN using np.mean()
                dbscan_cluster_centers_ = compute_cluster_centers(datapoints=data, labels=dbscan_.labels_)
                # Compute and sort the different cluster sizes
                dbscan_cluster_sizes = compute_cluster_sizes(labels=dbscan_.labels_)
                if -1 in dbscan_.labels_:
                    nof_clusters = len(dbscan_cluster_sizes) - 1
                else:
                    nof_clusters = len(dbscan_cluster_sizes)

                # Compute the silhouette score
                try:
                    dbscan_silhouette_score = silhouette_score(X=data, labels=dbscan_.labels_)
                except ValueError as e:
                    logger.info(e)
                    logger.error('DBSCAN found only {} cluster in the embeddings. Computing of Silhouette Score and DB'
                                 'score is therefore not possible. Storing \'Inf\' instead.'.format(nof_clusters))
                    dbscan_silhouette_score = np.inf
                    dbscan_db_score = np.inf
                else:
                    # Compute the Davies Bouldin Score
                    dbscan_db_score = davies_bouldin_score(X=data, labels=dbscan_.labels_)
                finally:
                    # Compute the Dunn index
                    dbscan_dunn_index, dbscan_dunn_index_corrected = compute_dunn_index(X=data,
                                                                                        cluster_centers=dbscan_cluster_centers_,
                                                                                        labels=dbscan_.labels_)
                    # Using real labels, compute the rand index
                    dbscan_adjusted_rand_score = adjusted_rand_score(labels_true=true_labels,
                                                                     labels_pred=dbscan_.labels_)
                    # Using real labels, compute the mutual information score
                    dbscan_adjusted_mutual_info_score = adjusted_mutual_info_score(labels_true=true_labels,
                                                                                   labels_pred=dbscan_.labels_)
                    results_ = {
                        'labels_': dbscan_.labels_,
                        'n_features_in_': np.array([dbscan_.n_features_in_], dtype=object),
                        'silhouette_score': np.array([dbscan_silhouette_score], dtype=object),
                        'dunn_index': np.array([dbscan_dunn_index], dtype=object),
                        'dunn_index_corrected': np.array([dbscan_dunn_index_corrected], dtype=object),
                        'db_score': np.array([dbscan_db_score], dtype=object),
                        'adjusted_rand_score': np.array([dbscan_adjusted_rand_score], dtype=object),
                        'adjusted_mutual_info_score': np.array([dbscan_adjusted_mutual_info_score], dtype=object),
                        'cluster_sizes': np.array([dbscan_cluster_sizes], dtype=object)
                    }
                    if save_to_disk:
                        logger.info('Saving to disk...')
                        # Creates individual files for each setting of kmeans HPs
                        dbscan_file_name = 'Eps{}_MinSamples{}'.format(eps, min_samples)
                        np.savez_compressed(file=os.path.join(save_path, dbscan_file_name + '.npz'), **results_)

                    if return_result:
                        trial_idx.append('Trial={}--Eps={}, MinSamples={}'.format(idx, eps, min_samples))
                        trial_results_.append(results_)

                    idx += 1

        if len(trial_idx) > 0:
            logger.info('Returning results for all trials...')
            return np.array(trial_idx, dtype=object), np.array(trial_results_, dtype=object)
