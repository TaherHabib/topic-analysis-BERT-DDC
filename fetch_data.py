import os
import yaml
import argparse
import logging
from utils import settings
from utils.data_utils import gdrive_data_download as gdd

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)


parser = argparse.ArgumentParser(description='Fetch data from cloud google drive')
parser.add_argument('-m', '--trained_models', dest='download_trained_models', action='store', nargs='?', const=True,
                    default=False, help='Whether to download trained classification head model weights. '
                                        'Specifying only short flag without any argument sets value to True.')
parser.add_argument('-b', '--book_data', dest='download_book_ddc_data', action='store', nargs='?', const=True,
                    default=False, help='Whether to download book titles data files (raw). Specifying only short flag '
                                        'without any argument sets value to True.')
parser.add_argument('-c', '--course_data', dest='download_course_data', action='store', nargs='?', const=True,
                    default=False, help='Whether to download course data files (raw). Specifying only short flag '
                                        'without any argument sets value to True.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store', nargs='?', const=True, default=False,
                    help='Verbosity of entire download and data setup process.')


if __name__ == '__main__':

    args = parser.parse_args()

    data_root = settings.get_data_root()
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    logger.info('Getting IDs of file and folders on Google Drive from YAML config file')
    project_root = settings.get_project_root()
    with open(os.path.join(project_root, 'configs', 'data_config.yml'), 'r') as f:
        data_config = yaml.safe_load(f)

    downloader = gdd.GDriveDownloader(data_root=data_root,
                                      data_config_yml=data_config,
                                      download_trained_models=args.download_trained_models,
                                      download_book_data=args.download_book_ddc_data,
                                      download_course_data=args.download_course_data)

    downloader.download(verbose=args.verbose)

    logger.info('Download of data files completed at data/data_.')










