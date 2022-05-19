import umap
import os
import numpy as np
import logging
import h5py
from utils import settings

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)

project_root = settings.get_project_root()


class UMAPReducer:
    def __init__(self,
                 n_components=2,
                 n_neighbors_range=None,
                 n_epochs_range=None,
                 min_dist_range=None,
                 densmap_range=None,
                 metric='euclidean',
                 spread=1.0,
                 learning_rate=1.0,
                 output_dens=True,
                 local_connectivity=1.0,
                 repulsion_strength=1.0,
                 random_state=None,
                 n_jobs=-1):

        self.n_components = n_components
        self.n_neighbors_range = n_neighbors_range
        self.n_epochs_range = n_epochs_range
        self.min_dist_range = min_dist_range
        self.densmap_range = densmap_range
        self.metric = metric
        self.spread = spread
        self.learning_rate = learning_rate
        self.output_dens = output_dens
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.random_state = random_state
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
        save_path = os.path.join(project_root, 'src', 'data', 'SidBERT_data', 'umap', dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def fit_umap(self,
                 data=None,
                 return_result=None,
                 save_to_disk=None,
                 embeddings_round_upto=None,
                 file_args=None):

        save_path = self.set_save_path(file_args)

        trial_results_ = []
        idx = 1
        for n_nbr in self.n_neighbors_range:
            for n_ep in self.n_epochs_range:
                for dist in self.min_dist_range:
                    for densmap in self.densmap_range:
                        logging.info('Trial={}--Neighbors={}, Epochs={}, min_dist={}, densmap={}'
                                     .format(idx, n_nbr, n_ep, dist, densmap))
                        umap__ = umap.UMAP(n_components=self.n_components,
                                           n_neighbors=n_nbr,
                                           n_epochs=n_ep,
                                           min_dist=dist,
                                           densmap=densmap,
                                           metric='euclidean',
                                           spread=1.0,
                                           learning_rate=1.0,
                                           output_dens=True,
                                           local_connectivity=1.0,
                                           repulsion_strength=1.0,
                                           n_jobs=self.n_jobs,
                                           random_state=self.random_state)
                        res = umap__.fit_transform(data)
                        if embeddings_round_upto is not None:
                            result_emb = np.around(res[0], decimals=embeddings_round_upto)
                        else:
                            result_emb = res[0]
                        results_ = {
                            'n_neighbors': np.array([n_nbr], dtype=object),
                            'n_epochs': np.array([n_ep], dtype=object),
                            'min_dist': np.array([dist], dtype=object),
                            'densmap': np.array([densmap], dtype=object),
                            'reduced_emb': result_emb,
                            'local_radii_original': res[1],
                            'local_radii_embedded': res[2]
                        }

                        # Writing reduced UMAP embeddings to disk
                        if save_to_disk:
                            logger.info('Saving to disk...')
                            result_filename = 'Neighbors{}_Epochs{}_MinDist{}_Densmap{}'.format(n_nbr, n_ep, dist, densmap)
                            # Creates individual files for each setting of 'perplexity' and 'n_iter'.
                            with h5py.File(os.path.join(save_path, result_filename+'.hdf5'), 'w', libver='latest') as f:
                                f.create_dataset(name='neighbors={}, epochs={}, min_dist={}, densmap={}'.format(n_nbr, n_ep, dist, densmap),
                                                 shape=(len(data), self.n_components),
                                                 data=result_emb,
                                                 chunks=(len(data), self.n_components),
                                                 compression='gzip',
                                                 compression_opts=4)
                        if return_result:
                            trial_results_.append(results_)

                        idx += 1

        if len(trial_results_) > 0:
            logger.info('Returning results for all trials...')
            return np.array(trial_results_, dtype=object)
