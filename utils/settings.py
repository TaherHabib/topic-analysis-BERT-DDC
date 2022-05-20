from pathlib import Path
import os


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


def get_gdrive_domain():
    """
    Returns the scheme and domain of google drive.
    :return: 2-tuple of strings
    """

    return 'https://drive.google.com/uc?id=', 'https://drive.google.com/drive/folders/'


def get_datafile_ids(download_trained_models=False, download_course_data=False):
    """
    Returns the gdrive ids and directory locations of all the required downloadable data files and folders.
    :return: 2-tuple of dictionaries
    """

    file_ids = {
        'bert_compressed_pooler_embeddings': {'id': '1wOZHY8ij2j24Retluk6UazcP0aN3a-y4', 'dir': 'model_data/'},
        'dataframe_embeddings_index': {'id': '1BkC6lVOHKFZO2XnV2TDdVpubZ5jTybbv', 'dir': 'model_data/'}
        }

    folder_ids = {
        'book_ddc_data': {'id': ['11aOHkVGWocMF3Fgx6dP8Il0JzoCjEdeh'], 'dir': ['datasets/book_ddc_data/']}
    }

    if download_trained_models:
        folder_ids['trained_models'] = {'id': ['1RVvTpzw4ccEzN-4xsO2mGV-DfBQZaEuk'],
                                        'dir': ['model_data/trained_models/']}
    if download_course_data:
        folder_ids['course_data'] = {'id': ['1qunWYgjhxA9DuekYk7k9YANsM-be3ouV',
                                            '1vsKc1GnPYI1gpzKR4RVlhjIZEJX8NFCO',
                                            '1gccj3w4hQje6QhCXG0w-is1-m27k0VGA'],
                                     'dir': ['datasets/course_data/db_dump',
                                             'datasets/course_data/k_annonymized',
                                             'datasets/course_data/p3_dump']}

    return file_ids, folder_ids
