import pandas as pd
import numpy as np
import os
from src.model import config

class DatasetCreator:
    def __init__(self):
        self.raw_dataset = pd.read_csv(os.path.join(os.path.abspath('.'),'data','datasets',config.dataset_name),dtype='str')
        self.select_level = config.select_level

    def aggregate_to_level(self):
        return_dataset = self.raw_dataset.copy()
        return_dataset['DDC'] = self.raw_dataset['DDC'].str[-self.select_level]
        return return_dataset

    def even_distribute_dataset(self,preprocessed_dataset, train_test_ratio=0.1):
        minimum = preprocessed_dataset['DDC'].value_counts[-1]
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        for index in preprocessed_dataset['DDC'].unique():
            select_indices = np.random.choice(preprocessed_dataset.loc[preprocessed_dataset['DDC'] == index].index,
                                              minimum,
                                              replace=False)
            select_train = np.random.choice(select_indices,minimum * (1-train_test_ratio),replace=False)
            sub_train = preprocessed_dataset.loc[preprocessed_dataset.index.isin(select_train)]
            sub_test = preprocessed_dataset.loc[~preprocessed_dataset.index.isin(select_train)]
            train_frame = pd.concat([train_frame,sub_train])
            test_frame = pd.concat([test_frame,sub_test])
        return train_frame, test_frame

    def generate_dataset(self):
        preprocessed_dataset = self.aggregate_to_level()
        return self.even_distribute_dataset(preprocessed_dataset)


