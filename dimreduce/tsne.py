from sklearn.manifold import TSNE
import os
import numpy as np
import logging
import h5py
from src_utils import settings

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)

project_root = settings.get_project_root()


class TSNEReducer:
    def __init__(self, n_components=2,
                 perplexity_range=None,
                 n_iter_range=None,
                 random_state=None,
                 n_jobs=-1,
                 init='random'):

        self.n_components = n_components
        self.perplexity_range = perplexity_range
        self.n_iter_range = n_iter_range
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.init = init

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
        save_path = os.path.join(project_root, 'src', 'data', 'SidBERT_data', 'tsne', dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def fit_tsne(self,
                 data=None,
                 return_result=None,
                 save_to_disk=None,
                 embeddings_round_upto=None,
                 file_args=None):

        save_path = self.set_save_path(file_args)

        trial_results_ = []
        idx = 1
        for perp in self.perplexity_range:
            for n_iter in self.n_iter_range:
                logging.info('Trial={}--Perplexity={}, iterations={}'.format(idx, perp, n_iter))
                tsne = TSNE(n_components=self.n_components,
                            perplexity=perp,
                            learning_rate='auto',
                            init=self.init,
                            n_iter=n_iter,
                            n_jobs=self.n_jobs,
                            random_state=self.random_state)
                result = tsne.fit_transform(data)
                if embeddings_round_upto is not None:
                    result = np.around(result, decimals=embeddings_round_upto)
                results_ = {
                    'perplexity': np.array([perp], dtype=object),
                    'n_iter': np.array([n_iter], dtype=object),
                    'reduced_emb': result
                }

                # Writing reduced tSNE embeddings to disk
                if save_to_disk:
                    logger.info('Saving to disk...')
                    result_filename = 'Perplexity{}_Iterations{}'.format(perp, n_iter)
                    # Creates individual files for each setting of 'perplexity' and 'n_iter'.
                    with h5py.File(os.path.join(save_path, result_filename+'.hdf5'), 'w', libver='latest') as f:
                        f.create_dataset(name='perplexity={}, n_iter={}'.format(perp, n_iter),
                                         shape=(len(data), self.n_components),
                                         data=result,
                                         chunks=(len(data), self.n_components),
                                         compression='gzip',
                                         compression_opts=4)
                if return_result:
                    trial_results_.append(results_)

                idx += 1

        if len(trial_results_) > 0:
            logger.info('Returning results for all trials...')
            return np.array(trial_results_, dtype=object)

