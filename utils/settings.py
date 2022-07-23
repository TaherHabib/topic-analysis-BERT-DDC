from pathlib import Path
import os

HF_bert_model_name = 'bert-base-multilingual-cased'

HF_distilbert_model_name = 'distilbert-base-multilingual-cased'


def get_project_root():
    """
    For setting the root of the project independent of OS.
    :return: root of the project
    """

    return os.path.abspath(Path(__file__).parent.parent)


def get_data_root():
    """
    For setting data directory independent of OS.
    :return: path to data directory
    """

    root = get_project_root()
    return os.path.join(root, 'data', 'data_')
