import os
import logging
import numpy as np
import pandas as pd

from utils import settings
from clusterers.kmeans import KmeansClusterer
from clusterers.dbscan import DbscanClusterer

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)

project_root = settings.get_project_root()


if __name__ == '__main__':

    # TODO: Write a command line argument parser to run this module

    # ------------------------------------------------------------------------------------------------------------------
    classes_to_cluster = list(np.arange(10).astype('str'))  # ['1', '2', '4', '8', '9']
    n_samples = 5000
    layer = 7  # 'pooler_output'
    original_only = True
    clusterer = 'dbscan'
    pca_embeddings_filename = 'Comp240_Var85_ScalerStandardScaler'
    # ------------------------------------------------------------------------------------------------------------------

    file_args = {
        'layer': layer,
        'classes': classes_to_cluster,
        'n_samples': n_samples,
        'original_only': original_only
    }
    pca_dir_name = 'Layer{}_Classes{}_SamplesPerRootClass{}_OriginalOnly{}'.format(file_args['layer'],
                                                                                   ''.join(file_args['classes']),
                                                                                   file_args['n_samples'],
                                                                                   file_args['original_only'])
    pca_dataframe = pd.read_hdf(os.path.join(project_root, 'src', 'data', 'SidBERT_data', 'pca',
                                             pca_dir_name, 'df_book_ddc.hdf5'), mode='r', key='df_book_ddc')
    with np.load(os.path.join(project_root, 'src', 'data', 'SidBERT_data', 'pca', pca_dir_name,
                              pca_embeddings_filename.replace('.npz', '') + '.npz'), allow_pickle=True) as dt:
        pca_embeddings = dict(dt)

    reduced_embeddings = pca_embeddings['reduced_emb']
    real_labels = list(pca_dataframe['root_class'])

    # Perform K-Means cluster
    if clusterer.lower() == 'kmeans':
        file_args.update({'pca_comp': pca_embeddings_filename.split('_')[0][4:],
                          'pca_var': pca_embeddings_filename.split('_')[1][3:],
                          'pca_scaler': pca_embeddings_filename.split('_')[2][6:]})
        kmeans_hyperparams = {
            'n_clusters_range': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'init_range': ['k-means++', 'random'],
            'max_iter_range': [500]
        }

        # Calculating the number of trials
        num_trials = 1
        for k in kmeans_hyperparams.keys():
            num_trials = num_trials * len(kmeans_hyperparams[k])
        logger.info('Performing/Optimizing K-means. Total {} trials must be recorded...'.format(num_trials))
        kmeans_clusterer = KmeansClusterer(n_clusters_range=kmeans_hyperparams['n_clusters_range'],
                                           init_range=kmeans_hyperparams['init_range'],
                                           max_iter_range=kmeans_hyperparams['max_iter_range'])
        kmeans_trial_idx, kmeans_trial_results_ = kmeans_clusterer.fit_kmeans(data=reduced_embeddings,
                                                                              true_labels=real_labels,
                                                                              return_result=True,
                                                                              save_to_disk=True,
                                                                              file_args=file_args)

    # Perform DBSCAN cluster
    if clusterer.lower() == 'dbscan':
        file_args.update({'pca_comp': pca_embeddings_filename.split('_')[0][4:],
                          'pca_var': pca_embeddings_filename.split('_')[1][3:],
                          'pca_scaler': pca_embeddings_filename.split('_')[2][6:]})

        k = 2 * reduced_embeddings.shape[-1] - 1  # 'k' is 479 for embeddings of size 240 (layer 7 , classification head)
                                                  # 'k' is 159 for embeddings of size 80 (pooler output)
        empirical_eps = 1.5
        empirical_min_samples = k + 1
        dbscan_hyperparams = {
            'eps_range': [np.around(empirical_eps - empirical_eps * 0.3, decimals=2),
                          np.around(empirical_eps - empirical_eps * 0.1, decimals=2),
                          empirical_eps,
                          np.around(empirical_eps + empirical_eps * 0.1, decimals=2),
                          np.around(empirical_eps + empirical_eps * 0.3, decimals=2)],
            'min_samples_range': [empirical_min_samples - 50,
                                  empirical_min_samples - 25,
                                  empirical_min_samples,
                                  empirical_min_samples + 25,
                                  empirical_min_samples + 50]
        }
        # Calculating the number of trials
        num_trials = 1
        for k in dbscan_hyperparams.keys():
            num_trials = num_trials * len(dbscan_hyperparams[k])
        logger.info('Performing/Optimizing K-means. Total {} trials must be recorded...'.format(num_trials))
        dbscan_clusterer = DbscanClusterer(eps_range=dbscan_hyperparams['eps_range'],
                                           min_samples_range=dbscan_hyperparams['min_samples_range'])
        dbscan_trial_idx, dbscan_trial_results_ = dbscan_clusterer.fit_dbscan(data=reduced_embeddings,
                                                                              true_labels=real_labels,
                                                                              return_result=True,
                                                                              save_to_disk=True,
                                                                              file_args=file_args)

    else:
        raise ValueError('Wrong value for cluster algorithm! Choose from: \'KMeans\' or \'DBSCAN\'')
