from bash import bash
import os
import argparse
import logging
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import gdown

from src_utils import settings

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
parser.add_argument('-c', '--course_data', dest='download_course_data', action='store', nargs='?', const=True,
                    default=False, help='Whether to download course data files (raw). Specifying only short flag '
                                        'without any argument sets value to True.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store', nargs='?', const=True, default=True,
                    help='Verbosity of entire download and data setup process.')


def authenticate_login():
    gauth = GoogleAuth()

    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        # Authenticate if they're not there

        # This is what solved the issues:
        gauth.GetFlow()
        gauth.flow.params.update({'access_type': 'offline'})
        gauth.flow.params.update({'approval_prompt': 'force'})
        gauth.LocalWebserverAuth()

    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()

    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)


def bash_data_dvc_download(command, dataset):
    '''
        download data from google drive using dvc
    '''
    # define bash command

    dvc_pull = bash(f'wg')
    if dvc_pull.code == 0:
        print('Data download successful')
    else:
        print('Error encountered please see the log !!!')
        print(dvc_pull.stderr)


if __name__ == '__main__':

    args = parser.parse_args()

    data_path = settings.get_data_path()
    gdrive_url_domain_file, gdrive_url_domain_folder = settings.get_gdrive_domain()
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    logger.info('Getting IDs of file and folders on Google Drive')
    file_ids, folder_ids = settings.get_datafile_ids(download_trained_models=args.download_trained_models,
                                                     download_course_data=args.download_course_data)

    for file in file_ids.keys():
        gdown.download(url=gdrive_url_domain_file + file_ids[file]['id'],
                       quiet=not args.verbose,
                       output=os.path.join(data_path, file_ids[file]['dir']))

    for folder in folder_ids.keys():
        for idx, subfolder_id in enumerate(folder_ids[folder]['id']):
            gdown.download_folder(url=gdrive_url_domain_folder + subfolder_id,
                                  quiet=not args.verbose,
                                  output=os.path.join(data_path, folder_ids[folder]['dir'][idx]))

    logger.info('Download of data files completed at data/data_.')









