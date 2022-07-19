import logging
import os
import numpy as np
import pandas as pd
from utils.data_utils import original_ddc_loader as odl
from utils import settings


data_root = settings.get_data_root()


# def load_embeddings(root_classes=None, n_samples=None, layer='pooler_output',
#                     original_only=True, include_root_class=True,
#                     exclude=None, aggregate_level=None,
#                     save_to_disk=False):
#     load_ = {
#         'data': None,
#         'embeddings': None
#     }
#     data_path = os.path.join(project_root, 'src', '../data', 'SidBERT_data')
#     save_path = os.path.join(data_path, 'pca')
#
#     if root_classes:
#         root_classes = root_classes
#     else:
#         root_classes = list(np.arange(10).astype('str'))  # all 10 root classes
#
#     try:
#         assert (layer in [3, 4, 7, 9, 'pooler_output', 'dense_3000', 'dense_2048', 'final']), \
#             '\'layer\' argument was found to be of a wrong type'
#     except AssertionError:
#         raise TypeError('Please enter a valid layer name (string) or index (int)')
#     else:
#         if layer == 'pooler_output' or layer == 3:
#             path_d = 'pooler_output/samples_ddc_class_'
#             path_e = 'pooler_output/pooler_emb_ddc_class_'
#         else:
#             path_d = 'classification_head_output/samples_ddc_class_'
#             path_e = 'classification_head_output/layer_{}_emb_ddc_class_'.format(layer)
#
#     df_extracted_class_data = []
#     list_extracted_class_emb = []
#     original_classes = odl.load_classes_from_tsv(os.path.join(data_path, 'bert_data', 'classes.tsv'))
#
#     for class_index in root_classes:
#         logger.info('Loading embeddings for class: {}'.format(class_index))
#         data = pd.read_csv(os.path.join(data_path, path_d + class_index + '.csv'),
#                            usecols=['index', 'Title', 'DDC'])
#         emb = np.load(os.path.join(data_path, path_e + class_index + '.npy'))
#         if exclude:
#             emb = emb[~data['DDC'].isin(exclude).to_list()]
#             data = data.loc[~data['DDC'].isin(exclude)]
#         if aggregate_level:
#             emb = emb[data['DDC'].str.len() <= aggregate_level]
#             data = data[data['DDC'].str.len() <= aggregate_level]
#
#         # Filtering out data & embeddings for original DDC classes
#         if original_only:
#             # logger.info('Filtering out embeddings from the 905 original classes only')
#             original_ddc = data['DDC'].isin(original_classes)
#             all_ = np.full(len(data), False, dtype='bool')
#             all_[original_ddc] = True
#             data = data[all_].reset_index(drop=True)
#             emb = emb[all_]
#             del original_ddc
#
#         # Sampling 'n_samples' for tSNE/UMAP visualization
#         if n_samples is not None:
#             select_n_samples = np.random.choice(data.index, size=n_samples, replace=False)
#             all_ = np.full(len(data), False, dtype='bool')
#             all_[select_n_samples] = True
#             sampled_data = data[all_].reset_index(drop=True)
#             sampled_emb = emb[all_]
#             df_extracted_class_data.append(sampled_data)
#             list_extracted_class_emb.append(sampled_emb)
#             del all_, sampled_data, sampled_emb, select_n_samples, data, emb
#         else:
#             df_extracted_class_data.append(data)
#             list_extracted_class_emb.append(emb)
#
#     load_['data'] = pd.concat(df_extracted_class_data).reset_index(drop=True)
#     load_['embeddings'] = np.concatenate(list_extracted_class_emb)
#     del df_extracted_class_data, list_extracted_class_emb
#
#     # Including a column for 'root_class'
#     if include_root_class:
#         logger.info('Including the \'root_class\' column')
#         titles_root_classes = []
#         for i in range(len(load_['data'])):
#             titles_root_classes.append(load_['data']['DDC'].iloc[i][0])
#         load_['data']['root_class'] = titles_root_classes
#
#     if save_to_disk:
#         logger.info('Saving sampled dataframe to disk...')
#         dir_name = 'Layer{}_Classes{}_SamplesPerRootClass{}_OriginalOnly{}'.format(layer,
#                                                                                    ''.join(root_classes),
#                                                                                    n_samples,
#                                                                                    original_only)
#         save_path = os.path.join(save_path, dir_name)
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         filename = 'df_book_ddc'
#         load_['data'].to_hdf(path_or_buf=os.path.join(save_path, filename + '.hdf5'), mode='w',
#                              key=filename, complevel=4)
#
#     logger.info('Returning requested data...')
#     return load_


# def generate_embeddings(prune_at_layer='pooler_output'):
#     data_path = os.path.join(project_root, 'src', 'data', 'SidBERT_data')
#
#     try:
#         assert (prune_at_layer in [3, 4, 7, 9, 'pooler_output', 'dense_3000', 'dense_2048', 'final']), \
#             '\'prune_at_layer\' argument was found to be of a wrong type'
#     except AssertionError:
#         raise TypeError('Please enter a valid layer name (string) or index (int)')
#     else:
#         if prune_at_layer == 'pooler_output' or prune_at_layer == 3:
#             save_path = os.path.join(data_path, 'pooler_output')
#             save_filename = 'pooler_emb_ddc_class_'
#         else:
#             save_path = os.path.join(data_path, 'classification_head_output')
#             save_filename = 'layer_{}_emb_ddc_class_'.format(prune_at_layer)
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
#
#     logger.info(f'Path is {save_path}')
#     ddc_extractor = book_ddc_extractor.DDCBookExtractor()
#     sidbert = SidBERT.SidBERT(train=None, test=None)
#
#     # Load the entire dataset of 1315988 book title entries, their DDC classes, etc.
#     dataset = ddc_extractor.parse_collect_raw_data()  # Dataframe with
#     # 5 columns: [index, ISBN, Title, DDc, Description]
#
#     for class_index in np.arange(0, 10):
#         logger.info('Generating embeddings for class {}'.format(class_index))
#         subset = dataset.loc[dataset['DDC'].str.startswith(str(class_index))]
#         logger.info('set length: {}'.format(len(subset)))
#         loader = sidbert.construct_dataloader(frame=subset, batch_size=batch_size)
#         subset = subset[:loader.__len__() * loader.batch_size]
#         subset.to_csv(os.path.join(save_path, f'samples_ddc_class_{class_index}.csv'))
#         embeddings = sidbert.batch_get_pruned_model_output(dataset=loader, batch_size=batch_size,
#                                                            prune_at_layer=prune_at_layer)
#         logger.info('Successfully generated embeddings for class {}. \n'.format(class_index))
#         np.save(arr=embeddings,
#                 file=os.path.join(save_path, save_filename + str(class_index) + '.npy'))
#     logger.info('done')
