import os
import gdown


class GDriveDownloader:

    def __init__(self,
                 data_root=None,
                 data_config_yml=None,
                 download_trained_models=False,
                 download_book_data=False,
                 download_course_data=False):

        self.data_root = data_root
        self.data_config = data_config_yml
        self.download_trained_models = download_trained_models
        self.download_book_data = download_book_data
        self.download_course_data = download_course_data

    @staticmethod
    def get_gdrive_domain():
        """
        Returns the scheme and domain of google drive.

        Returns
        -------
        2-tuple of strings

        """

        return 'https://drive.google.com/uc?id=', 'https://drive.google.com/drive/folders/'

    def download(self, verbose=True):
        """
        Downloads the file and folders from google drive provided in the data config (YAML) file

        Returns
        -------
        None

        """
        file_url_domain, folder_url_domain = self.get_gdrive_domain()

        for k, v in self.data_config.items():
            if type(v['ID']) is list:
                for idx, subfolder_ID in enumerate(v['ID']):
                    gdown.download_folder(url=folder_url_domain+subfolder_ID,
                                          quiet=not verbose,
                                          output=os.path.join(self.data_root, v['outdir'][idx]))
            else:
                gdown.download(url=file_url_domain+v['ID'],
                               quiet=not verbose,
                               output=os.path.join(self.data_root, v['outdir']))
    #
    #
    # def download_files(self):
    #     file_ids, folder_ids = gdd.get_datafile_ids(download_trained_models=args.download_trained_models,
    #                                                 download_course_data=args.download_course_data)
    #
    #     for file in file_ids.keys():
    #         gdown.download(url=gdrive_url_domain_file + file_ids[file]['id'],
    #                        quiet=not args.verbose,
    #                        output=os.path.join(data_path, file_ids[file]['dir']))
    #
    #     for folder in folder_ids.keys():
    #         for idx, subfolder_id in enumerate(folder_ids[folder]['id']):
    #             gdown.download_folder(url=gdrive_url_domain_folder + subfolder_id,
    #                                   quiet=not args.verbose,
    #                                   output=os.path.join(data_path, folder_ids[folder]['dir'][idx]))
    #

# def get_datafile_ids(download_trained_models=False, download_course_data=False):
#     """
#     Returns the gdrive ids and directory locations of all the required downloadable data files and folders.
#     :return: 2-tuple of dictionaries
#     """
#
#     file_ids = {
#         'bert_compressed_pooler_embeddings': {'id': '1wOZHY8ij2j24Retluk6UazcP0aN3a-y4', 'dir': 'model_data/'},
#         'dataframe_embeddings_index': {'id': '1BkC6lVOHKFZO2XnV2TDdVpubZ5jTybbv', 'dir': 'model_data/'}
#         }
#
#     folder_ids = {
#         'book_ddc_data': {'id': ['11aOHkVGWocMF3Fgx6dP8Il0JzoCjEdeh'], 'dir': ['datasets/book_ddc_data/']}
#     }
#
#     if download_trained_models:
#         folder_ids['trained_models'] = {'id': ['1RVvTpzw4ccEzN-4xsO2mGV-DfBQZaEuk'],
#                                         'dir': ['model_data/trained_models/']}
#     if download_course_data:
#         folder_ids['course_data'] = {'id': ['1qunWYgjhxA9DuekYk7k9YANsM-be3ouV',
#                                             '1vsKc1GnPYI1gpzKR4RVlhjIZEJX8NFCO',
#                                             '1gccj3w4hQje6QhCXG0w-is1-m27k0VGA'],
#                                      'dir': ['datasets/course_data/db_dump',
#                                              'datasets/course_data/k_annonymized',
#                                              'datasets/course_data/p3_dump']}
#
#     return file_ids, folder_ids


