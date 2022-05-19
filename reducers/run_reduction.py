import os
import logging
import numpy as np
import pandas as pd

from src_utils import settings
from src.reducers.tsne import TSNEReducer
from src.reducers.umap import UMAPReducer

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
    classes_to_reduce = list(np.arange(10).astype('str'))  # ['1', '2', '4', '8', '9']
    n_samples = 5000
    layer = 'pooler_output'
    original_only = True
    reducer = 'tsne'
    pca_embeddings_filename = 'Comp80_Var86_ScalerStandardScaler'
    # ------------------------------------------------------------------------------------------------------------------

    file_args = {
        'layer': layer,
        'classes': classes_to_reduce,
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

    # Perform tSNE reduction
    if reducer.lower() == 'tsne':
        file_args.update({'pca_comp': pca_embeddings_filename.split('_')[0][4:],
                          'pca_var': pca_embeddings_filename.split('_')[1][3:],
                          'pca_scaler': pca_embeddings_filename.split('_')[2][6:]})
        ideal_perplexity = int(np.sqrt(n_samples * len(classes_to_reduce)))
        tsne_hyperparams = {
            'n_components_range': [2],
            'perplexity_range': [int(ideal_perplexity - 0.25 * ideal_perplexity),
                                 ideal_perplexity,
                                 int(ideal_perplexity + 0.5 * ideal_perplexity),
                                 2 * ideal_perplexity, 4 * ideal_perplexity,
                                 7 * ideal_perplexity],  # 10 * ideal_perplexity],
            'n_iter_range': [5000],
            'random_state': [0]
        }
        # Calculating the number of trials
        num_trials = 1
        for k in tsne_hyperparams.keys():
            num_trials = num_trials * len(tsne_hyperparams[k])
        logger.info('Performing/Optimizing tSNE. Total {} trials must be recorded...'.format(num_trials))
        tsne_reducer = TSNEReducer(n_components=tsne_hyperparams['n_components_range'][0],
                                   perplexity_range=tsne_hyperparams['perplexity_range'],
                                   n_iter_range=tsne_hyperparams['n_iter_range'],
                                   random_state=tsne_hyperparams['random_state'][0])

        _ = tsne_reducer.fit_tsne(data=reduced_embeddings,
                                  return_result=False,
                                  save_to_disk=True,
                                  embeddings_round_upto=4,
                                  file_args=file_args)
    # Perform UMAP reduction
    if reducer.lower() == 'umap':
        file_args.update({'pca_comp': pca_embeddings_filename.split('_')[0][4:],
                          'pca_var': pca_embeddings_filename.split('_')[1][3:],
                          'pca_scaler': pca_embeddings_filename.split('_')[2][6:]})
        umap_hyperparams = {
            'n_components_range': [2],
            'n_neighbors_range': [100, 150, 200],  # [10, 15, 30, 50, 100, 150, 200],
            'n_epochs_range': [500],
            'min_dist_range': [0.05, 0.10, 0.15, 0.20],
            'densmap_range': [False],  # , True],
            'random_state': [0]
        }
        # Calculating the number of trials
        num_trials = 1
        for k in umap_hyperparams.keys():
            num_trials = num_trials * len(umap_hyperparams[k])
        logger.info('Performing/Optimizing UMAP. Total {} trials must be recorded...'.format(num_trials))
        if n_samples is not None:
            umap_reducer = UMAPReducer(n_components=umap_hyperparams['n_components_range'][0],
                                       n_neighbors_range=umap_hyperparams['n_neighbors_range'],
                                       n_epochs_range=umap_hyperparams['n_epochs_range'],
                                       min_dist_range=umap_hyperparams['min_dist_range'],
                                       densmap_range=umap_hyperparams['densmap_range'],
                                       random_state=umap_hyperparams['random_state'][0])

            _ = umap_reducer.fit_umap(data=reduced_embeddings,
                                      return_result=False,
                                      save_to_disk=True,
                                      embeddings_round_upto=4,
                                      file_args=file_args)

    else:
        raise ValueError('Wrong value for reduction algorithm! Choose from: \'tSNE\' or \'UMAP\'')

