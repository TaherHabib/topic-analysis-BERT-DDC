import os
import gdown


class GDriveDownloader:

    def __init__(self,
                 data_root=None,
                 data_config_yml=None,
                 download_trained_models=False,
                 download_book_data=False,
                 download_course_data=False
                 ):

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
