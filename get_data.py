from bash import bash
import os
from src_utils import settings

def structure_directory():
    '''
        Structure the data directory
    '''
    data_path = os.path.join(settings.get_project_root(),'data')
    os.makedirs(data_path)

def bash_data_dvc_download(dataset):
    '''
        download data from google drive using dvc
    '''
    # define bash command

    dvc_pull = bash(f'dvc pull data/{dataset}.dvc')
    if dvc_pull.code == 0:
        print('Data download successfull')
    else:
        print('Error encountered please see the log !!!')
        print(dvc_pull.stderr)


bash_data_dvc_download('demo.txt')