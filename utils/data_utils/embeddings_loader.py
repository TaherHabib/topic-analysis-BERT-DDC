import os
import numpy as np
import pandas as pd
from zipfile import ZipFile
from utils import settings
from dataprocessing.dataset_processing import DatasetCreator

data_root = settings.get_data_root()


def load_embeddings(embeddings_filename=None,
                    dataset_filename=None,
                    return_dataset=False,
                    mmap_embeddings=True,
                    **kwargs):

    if mmap_embeddings:
        # Extracting the .npz file into a .npy file
        with ZipFile(os.path.join(data_root, 'model_data', embeddings_filename), 'r') as f:
            f.extractall(path=os.path.join(data_root, 'model_data'), members=['embeddings.npy'])
        # Loading a memory map for the extracted .npy file
        embeddings = np.load(os.path.join(data_root, 'model_data', 'embeddings.npy'), mmap_mode='c')
    else:
        with open(os.path.join(data_root, 'model_data', embeddings_filename), 'rb') as f:
            embeddings = np.load(f)
            embeddings = embeddings['embeddings']

    if return_dataset:
        dataframe, _ = DatasetCreator(dataset_filename=dataset_filename,
                                      **kwargs
                                      ).sample_from_dataset(num_samples=1)

        return {'dataframe': dataframe, 'embeddings': embeddings}
    else:
        return {'dataframe': None, 'embeddings': embeddings}


def sample_embeddings(embeddings_filename=None,
                      dataset_filename=None,
                      num_samples=None,
                      **kwargs):

    sampled_dataframe, extracted_indices = DatasetCreator(dataset_filename=dataset_filename,
                                                          **kwargs
                                                          ).sample_from_dataset(num_samples=num_samples)

    sampled_embeddings = extract_embeddings_from_indices(embeddings_filename=embeddings_filename,
                                                         indices=extracted_indices)

    return {'dataframe': sampled_dataframe, 'embeddings': sampled_embeddings}


def extract_embeddings_from_indices(embeddings_filename=None, indices=None):

    # Extracting the .npz file into a .npy file
    with ZipFile(os.path.join(data_root, 'model_data', embeddings_filename), 'r') as f:
        f.extractall(path=os.path.join(data_root, 'model_data'), members=['embeddings.npy'])
    # Loading a memory map for the extracted .npy file
    embeddings = np.load(os.path.join(data_root, 'model_data', 'embeddings.npy'), mmap_mode='c')

    # For each 'orig_index' in 'indices' match with the first index in the 'embeddings' 2D array, select the numpy
    # array ([0]) and then select the actual embeddings ([1:]) of size <dimension of prune_at_layer's output>.
    extracted_embeddings = np.array([embeddings[embeddings[:, 0] == i][0][1:] for i in indices])

    return extracted_embeddings
