import pandas as pd
import numpy as np
import os
import logging
from sklearn.utils import class_weight
from utils import settings

# Set a logger
logger = logging.getLogger('Dataset_Processing')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))


# logger.addHandler(handler)


class DatasetCreator:
    def __init__(self,
                 dataset_filename=None,
                 aggregate_ddc_level=1,
                 min_title_length=20,
                 test_size=0.2,
                 balanced_class_distribution=False):

        self.dataset_filename = dataset_filename
        self.aggregate_ddc_level = aggregate_ddc_level
        self.min_title_length = min_title_length
        self.test_size = test_size
        self.balanced_class_distribution = balanced_class_distribution

    def generate_final_dataset(self, embeddings_generator_mode=False, columns_to_use=None):

        data_root = settings.get_data_root()
        raw_dataset = pd.read_csv(os.path.join(data_root, 'datasets', self.dataset_filename), low_memory=False)

        logger.info('Aggregating DDC codes in dataset – to top-level 10 parent classes when aggregate_ddc_level=1')
        dataset_ = self.aggregate_to_level(dataset=raw_dataset, aggregate_ddc=self.aggregate_ddc_level)

        logger.info('Removing rows with book titles (+ descriptions) less than \'min_title_length\' characters long')
        dataset_ = self.remove_titles_below_length(dataset=dataset_,
                                                   text_columns=columns_to_use,
                                                   min_length=self.min_title_length)

        if embeddings_generator_mode:
            logger.info('Preparing dataset containing only the titles and/or descriptions')
            return dataset_[['orig_index', 'text_', 'DDC']], None, None
        else:
            logger.info('Preparing train and test dataset + class distribution dictionary')
            train_df, test_df, class_weights = self.prepare_train_test_data(dataset=dataset_,
                                                                            test_size=self.test_size,
                                                                            balance_class_distribution=self.balanced_class_distribution)
            return train_df, test_df, class_weights

    @staticmethod
    def aggregate_to_level(dataset, aggregate_ddc):
        dataset['DDC'] = dataset['DDC'].str[:aggregate_ddc]
        return dataset

    @staticmethod
    def remove_titles_below_length(dataset, text_columns, min_length):
        if len(text_columns) > 1:
            filtered_df = dataset.loc[dataset[['Title', 'Description']].str.len() >= min_length]
            # TODO: Write code to combine 'Title' and 'Description' (columns included in 'text_columns') into one
            #  column of 'filtered_df' named 'text_'
        else:
            filtered_df = dataset.loc[dataset[text_columns[0]].str.len() >= min_length]
            filtered_df.rename({text_columns[0]: 'text_'}, axis=1)

        return filtered_df.reset_index(drop=True).rename({'Unnamed: 0': 'orig_index'}, axis=1)  # Keeping the column
        # containing original indices (from the raw dataset)

    @staticmethod
    def prepare_train_test_data(dataset, test_size, balance_class_distribution):
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        for ddc_class in dataset['DDC'].unique():
            sub_frame = dataset.loc[dataset['DDC'] == ddc_class]
            if balance_class_distribution:
                num_select = dataset['DDC'].value_counts()[-1]
            else:
                num_select = len(sub_frame)
            select_indices = np.random.choice(sub_frame.index, num_select, replace=False)

            # Filter out train set from the selected indices
            select_train = np.random.choice(select_indices, int(round(num_select * (1 - test_size))), replace=False)
            sub_train = sub_frame.loc[sub_frame.index.isin(select_train)]

            # Filter out test set from the remaining unselected indices
            select_test = np.random.choice(sub_frame.loc[~sub_frame.index.isin(select_train)].index,
                                           int(round(num_select * test_size)), replace=False)
            sub_test = sub_frame.loc[sub_frame.index.isin(select_test)]

            # Concatenating train and test samples from each unique DDC class
            train_frame = pd.concat([train_frame, sub_train])
            test_frame = pd.concat([test_frame, sub_test])
        train_frame = train_frame.reset_index(drop=True).rename({'Unnamed: 0': 'orig_index'}, axis=1)
        test_frame = test_frame.reset_index(drop=True).rename({'Unnamed: 0': 'orig_index'}, axis=1)

        class_weight_dict = dict(enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                                             classes=np.unique(
                                                                                 train_frame['DDC'].to_numpy()),
                                                                             y=train_frame['DDC'].to_numpy())
                                           )
                                 )

        return train_frame, test_frame, class_weight_dict

    # @staticmethod
    # def even_distribute_dataset(preprocessed_dataset, train_test_ratio=0.1):
    #     minimum_size = preprocessed_dataset['DDC'].value_counts()[-1]
    #     train_frame = pd.DataFrame()
    #     test_frame = pd.DataFrame()
    #     for index in preprocessed_dataset['DDC'].unique():
    #         sub_frame = preprocessed_dataset.loc[preprocessed_dataset['DDC'] == index]
    #         select_indices = np.random.choice(sub_frame.index, minimum_size, replace=False)
    #         select_train = np.random.choice(select_indices, int(minimum_size * (1 - train_test_ratio)), replace=False)
    #         sub_train = sub_frame.loc[sub_frame.index.isin(select_train)]
    #         remaining_indices = sub_frame.loc[~sub_frame.index.isin(select_train)].index
    #         sub_test = sub_frame.loc[sub_frame.index.isin(
    #             np.random.choice(remaining_indices, int(minimum_size * train_test_ratio), replace=False))]
    #         train_frame = pd.concat([train_frame, sub_train])
    #         test_frame = pd.concat([test_frame, sub_test])
    #     return train_frame, test_frame

# @staticmethod
# def unbalanced_dataset(preprocessed_dataset, train_test_ratio=0.1):
#     train_frame = pd.DataFrame()
#     test_frame = pd.DataFrame()
#     for index in preprocessed_dataset['DDC'].unique():
#         sub_frame = preprocessed_dataset.loc[preprocessed_dataset['DDC'] == index]
#         num_select = int(round(len(sub_frame) * (1 - train_test_ratio)))
#         select_indices = np.random.choice(sub_frame.index, num_select, replace=False)
#         sub_train = sub_frame.loc[sub_frame.index.isin(select_indices)]
#         sub_test = sub_frame.loc[~sub_frame.index.isin(select_indices)]
#         train_frame = pd.concat([train_frame, sub_train])
#         test_frame = pd.concat([test_frame, sub_test])
#
#     class_weight_dict = dict(
#         enumerate(class_weight.compute_class_weight(class_weight='balanced',
#                                                     classes=np.unique(train_frame['DDC'].to_numpy()),
#                                                     y=train_frame['DDC'].to_numpy())
#                   )
#     )
#
#     return train_frame, test_frame, class_weight_dict
