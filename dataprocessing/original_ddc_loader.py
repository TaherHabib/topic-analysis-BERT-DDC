"""
Extracts the original 905 DDC classes from the classes.tsv file in the 'data' directory.
Only these 905 classes are used for training and analysis
"""

import pandas as pd


def load_classes_from_tsv(file_path):
    """
    loads all label IDs from the classes.tsv file
    :return:list of DDC classes in the scope of the models.
    :rtype: list object
    """
    classes_frame = pd.read_csv(file_path, delimiter='\t', names=['DDC'], dtype=str)
    return classes_frame['DDC'].to_list()


def create_ddc_label_lookup(classes_list):
    """
    creates a dictionary for label lookup. As output neurons from the network do not possess an inherent label,
    a lookup must be performed to assess the ddc class an input was associated with.
    :return: lookup dictionary  with {key:value} as {position_index defined in 'classes.tsv' (int) : DDC label (str)}
    :rtype: python 3 dictionary object
    """
    return {index: classes_list[index] for index in range(len(classes_list))}
