import pandas as pd
import numpy as np
import os
from models.configs import config
from sklearn.utils import class_weight


class DatasetCreator:
    def __init__(self,
                 ddc_level=1,
                 min_title_length=20,
                 train_test_ratio=0.2,
                 balanced_classes=False):

        self.raw_dataset = pd.read_csv(os.path.join(os.path.abspath('../../src/dataprocessing'), 'data', 'datasets',
                                                    config.dataset_name), dtype='str')
        self.ddc_level = ddc_level
        self.min_title_length = min_title_length
        self.train_test_ratio = train_test_ratio
        self.balanced_classes = balanced_classes

    def aggregate_to_level(self):
        dataset = self.raw_dataset.copy()
        dataset['DDC'] = self.raw_dataset['DDC'].str[-self.ddc_level]
        return dataset

    def generate_dataset(self):
        dataset = self.aggregate_to_level()
        dataset = self.remove_titles_below_length(dataset=dataset, length=self.min_title_length)
        if self.balanced_classes:
            return self.even_distribute_dataset(dataset, self.train_test_ratio), {}
        else:
            return self.unbalanced_dataset(dataset, self.train_test_ratio)

    @staticmethod
    def remove_titles_below_length(dataset, length):
        return dataset.loc[dataset['Title'].str.len() >= length].reset_index(drop=True).drop(columns=['Unnamed: 0'])

    @staticmethod
    def unbalanced_dataset(preprocessed_dataset, train_test_ratio=0.1):
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        for index in preprocessed_dataset['DDC'].unique():
            sub_frame = preprocessed_dataset.loc[preprocessed_dataset['DDC'] == index]
            num_select = int(round(len(sub_frame) * (1 - train_test_ratio)))
            select_indices = np.random.choice(sub_frame.index, num_select, replace=False)
            sub_train = sub_frame.loc[sub_frame.index.isin(select_indices)]
            sub_test = sub_frame.loc[~sub_frame.index.isin(select_indices)]
            train_frame = pd.concat([train_frame, sub_train])
            test_frame = pd.concat([test_frame, sub_test])

        class_weight_dict = dict(
            enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(train_frame['DDC'].to_numpy()),
                                                        y=train_frame['DDC'].to_numpy())
                      )
        )

        return train_frame, test_frame, class_weight_dict

    @staticmethod
    def even_distribute_dataset(preprocessed_dataset, train_test_ratio=0.1):
        minimum_size = preprocessed_dataset['DDC'].value_counts()[-1]
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        for index in preprocessed_dataset['DDC'].unique():
            sub_frame = preprocessed_dataset.loc[preprocessed_dataset['DDC'] == index]
            select_indices = np.random.choice(sub_frame.index, minimum_size, replace=False)
            select_train = np.random.choice(select_indices, int(minimum_size * (1-train_test_ratio)), replace=False)
            sub_train = sub_frame.loc[sub_frame.index.isin(select_train)]
            remaining_indices = sub_frame.loc[~sub_frame.index.isin(select_train)].index
            sub_test = sub_frame.loc[sub_frame.index.isin(np.random.choice(remaining_indices, int(minimum_size * train_test_ratio), replace=False))]
            train_frame = pd.concat([train_frame, sub_train])
            test_frame = pd.concat([test_frame, sub_test])
        return train_frame, test_frame

