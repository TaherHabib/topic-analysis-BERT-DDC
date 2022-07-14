import os
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.embeddings_generator import load_embeddings
from utils import settings

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)

project_root = settings.get_project_root()


class PCAReducer:
    def __init__(self,
                 n_components_range=None,
                 explained_var_range=None,
                 scalers_range=None):

        self.n_components_range = n_components_range
        self.explained_var_range = explained_var_range
        self.scalers_range = scalers_range

    @staticmethod
    def set_save_path(file_args):
        dir_name = 'Layer{}_Classes{}_SamplesPerRootClass{}'.format(file_args['layer'],
                                                                    ''.join(file_args['classes']),
                                                                    file_args['n_samples'])
        save_path = os.path.join(project_root, 'data', 'data_', 'model_data', 'pca_embeddings', dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def fit_pca(self,
                data=None,
                return_result=False,
                save_to_disk=True,
                file_args=None):

        save_path = self.set_save_path(file_args)

        trial_results_ = []
        idx = 1
        for explained_var in self.explained_var_range:
            logging.info('Running for VarianceRetention={}'.format(explained_var))
            for n in self.n_components_range:
                pca = PCA(n_components=n)
                reduced_emb = pca.fit_transform(data)
                var_retention = pca.explained_variance_ratio_.cumsum()[-1]
                logger.info('Variance retention with {} components: {}'.format(n, var_retention))

                # Save the reduced embeddings if desired explained variance is achieved, otherwise...
                if var_retention >= explained_var:
                    logger.info('{} components sufficient for retention of {}% variance'.format(n, explained_var * 100))
                    for scaler in self.scalers_range:
                        logger.info('Scaling the original embeddings using {}'.format(scaler.__name__))
                        reduced_emb = scaler().fit_transform(reduced_emb)
                        results_ = {
                            'n_components': n,
                            'var_retention': var_retention,
                            'scaler': scaler,
                            'reduced_emb': reduced_emb
                        }
                        if save_to_disk:
                            logger.info('Saving to disk...')
                            # Creates individual files for each setting of components, variance retention and scaler
                            pca_file_name = 'Comp{}_Var{}_Scaler{}'.format(n, int(var_retention * 100), scaler.__name__)
                            np.savez_compressed(file=os.path.join(save_path, pca_file_name + '.npz'), **results_)
                        if return_result:
                            trial_results_.append(results_)
                        idx += 1

                    logger.info('PCA dimensionality reduction completed!')
                    break

                # ... print a message stating that even the maximum number of components in the given range of
                # 'n_components' is insufficient to reach the desired explained variance; save the results anyway
                elif n == self.n_components_range[-1] and var_retention < explained_var:
                    logger.info('Number of components in the given range are insufficient to reach the '
                                'desired explained variance {}. Returning with {} components and a variance '
                                'retention of {}'.format(explained_var, n, var_retention))
                    for scaler in self.scalers_range:
                        logger.info('Scaling the original embeddings using {}'.format(scaler.__name__))
                        reduced_emb = scaler().fit_transform(reduced_emb)
                        results_ = {
                            'n_components': np.array([n], dtype=object),
                            'var_retention': np.array([var_retention], dtype=object),
                            'scaler': np.array([scaler], dtype=object),
                            'reduced_emb': reduced_emb
                        }
                        if save_to_disk:
                            logger.info('Saving to disk...')
                            # Creates individual files for each setting of components, variance retention and scaler
                            pca_file_name = 'Comp{}_Var{}_Scaler{}'.format(n, int(var_retention * 100), scaler.__name__)
                            np.savez_compressed(file=os.path.join(save_path, pca_file_name + '.npz'), **results_)

                        if return_result:
                            trial_results_.append(results_)
                        idx += 1

                    logger.info('PCA reduction was unable to reach desired variance retention level!')
                    break
                else:
                    continue

        if len(trial_results_) > 0:
            logger.info('Returning results for all trials...')
            return np.array(trial_results_, dtype=object)
        else:
            return


if __name__ == '__main__':

    # TODO: Write a command line argument parser to run this module
    # ------------------------------------------------------------------------------------------------------------------
    classes = list(np.arange(10).astype('str'))  # ['1', '2', '4', '8', '9']
    n_samples = 5000
    layer = 7  # 'pooler_output'
    original_only = True
    # ------------------------------------------------------------------------------------------------------------------
    file_args = {
        'layer': layer,
        'classes': classes,
        'n_samples': n_samples,
        'original_only': original_only
    }
    pca_hyperparams = {
        'n_components_range': np.arange(10, 401, 10),
        'explained_variance_range': [0.85],
        'scalers_range': [StandardScaler]  # [StandardScaler, MinMaxScaler, RobustScaler]
    }
    data_ = load_embeddings(root_classes=classes,
                            n_samples=n_samples,
                            layer=layer,
                            original_only=original_only,
                            save_to_disk=True)

    logger.info('Performing/Optimizing PCA and reducing the dimensionality of embeddings')
    pca_reducer = PCAReducer(n_components_range=pca_hyperparams['n_components_range'],
                             explained_var_range=pca_hyperparams['explained_variance_range'],
                             scalers_range=pca_hyperparams['scalers_range'])

    _ = pca_reducer.fit_pca(data=data_['embeddings'],
                            return_result=False,
                            save_to_disk=True,
                            file_args=file_args)

