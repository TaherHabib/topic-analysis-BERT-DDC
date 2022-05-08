import pandas as pd
import numpy as np
import os
from models.configs import config
from sklearn.utils import class_weight


class DatasetCreator:
    def __init__(self):
        self.raw_dataset = pd.read_csv(os.path.join(os.path.abspath('../../src/preprocessing'), 'data', 'datasets', config.dataset_name), dtype='str')
        self.select_level = config.select_level

    def aggregate_to_level(self):
        return_dataset = self.raw_dataset.copy()
        return_dataset['DDC'] = self.raw_dataset['DDC'].str[-self.select_level]
        return return_dataset

    def remove_titles_below_length(self, dataset, length):
        return dataset.loc[dataset['Title'].str.len() >= length].reset_index(drop=True).drop(columns=['Unnamed: 0'])

    def unbalanced_dataset(self, preprocessed_dataset,train_test_ratio = 0.1):
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        for index in preprocessed_dataset['DDC'].unique():
            sub_frame = preprocessed_dataset.loc[preprocessed_dataset['DDC'] == index]
            num_select = int(round(len(sub_frame) * (1 - train_test_ratio)))
            select_indices = np.random.choice(sub_frame.index, num_select, replace=False)
            sub_train = sub_frame.loc[sub_frame.index.isin(select_indices)]
            sub_test = sub_frame.loc[~sub_frame.index.isin(select_indices)]
            train_frame = pd.concat([train_frame,sub_train])
            test_frame = pd.concat([test_frame,sub_test])

        class_weight_dict = dict(enumerate(class_weight.compute_class_weight(
            'balanced',
            np.unique(train_frame['DDC'].to_numpy()),
            train_frame['DDC'].to_numpy())))

        return train_frame, test_frame, class_weight_dict

    def even_distribute_dataset(self,preprocessed_dataset, train_test_ratio=0.1):
        minimum = preprocessed_dataset['DDC'].value_counts()[-1]
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        for index in preprocessed_dataset['DDC'].unique():
            sub_frame = preprocessed_dataset.loc[preprocessed_dataset['DDC'] == index]
            select_indices = np.random.choice(sub_frame.index,
                                              minimum,
                                              replace=False)
            select_train = np.random.choice(select_indices,int(minimum * (1-train_test_ratio)),replace=False)
            sub_train = sub_frame.loc[sub_frame.index.isin(select_train)]
            remaining_indices = sub_frame.loc[~sub_frame.index.isin(select_train)].index
            sub_test = sub_frame.loc[sub_frame.index.isin(np.random.choice(remaining_indices, int(minimum * train_test_ratio),replace=False))]
            train_frame = pd.concat([train_frame,sub_train])
            test_frame = pd.concat([test_frame,sub_test])
        return train_frame, test_frame

    def generate_dataset(self, even=False, train_test_ratio=0.1):
        preprocessed_dataset = self.aggregate_to_level()
        preprocessed_dataset = self.remove_titles_below_length(preprocessed_dataset, length = 20)
        if even:
            return self.even_distribute_dataset(preprocessed_dataset,train_test_ratio), {}
        else:
            return self.unbalanced_dataset(preprocessed_dataset,train_test_ratio)


