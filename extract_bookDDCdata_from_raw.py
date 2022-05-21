from dataprocessing.book_ddc_extractor import DDCBookExtractor
from utils import settings
import os
import pandas as pd

if __name__ == '__main__':

    project_root = settings.get_project_root()
    data_root = settings.get_data_root()

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    try:
        pd.read_csv(os.path.join(data_root, 'datasets', 'book_ddc_data', 'classes.tsv'),
                    delimiter='\t', names=['DDC'], dtype=str)
    except NotADirectoryError as e:
        print('Download the required files (directory "book_ddc_data/") from Google Drive first...')

    else:
        ddc_extractor = DDCBookExtractor(data_root=data_root)
        dataset = ddc_extractor.parse_collect_raw_data()  # DF 5 columns: [index, ISBN, Title, DDc, Description]

        # Save the full raw dataset
        dataset.to_csv(os.path.join(data_root, 'datasets', 'full_dataset.csv'))
