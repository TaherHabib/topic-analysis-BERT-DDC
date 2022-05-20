"""

Extracts book and DDC information from raw files and makes them usable for downstream tasks

"""

from os.path import join
import pandas as pd
import logging
import re
import csv
import json
from utils import settings

from models.preprocessing.original_ddc_loader import load_classes_from_tsv, create_ddc_label_lookup

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
logger.addHandler(handler)

project_root = settings.get_project_root()
data_root = settings.get_data_root()


class DDCBookExtractor:
    def __init__(self):

        self.data_path = join(data_root, 'datasets', 'book_ddc_data')
        self.classes = load_classes_from_tsv(join(self.data_path, 'classes.tsv'))
        self.class_position_hash = create_ddc_label_lookup(self.classes)

        self.uos_frame = None
        self.ub_frame = None
        self.crawled_frame = None
        self.TIB_frame = None

    def parse_collect_raw_data(self):
        """
        This function loads the raw data from files provided by the University of Bremen (UB)
        and university of Osnabrück (UOS). As the file format and parsing differ in both files,
        two sub-functions parse the data from both files individually.
        the parse_UOS_data function again contains two sub-functions that account for errors in the file.
        :return: pandas dataframe object containing all book entries from both libraries
        """
        def parse_UB_data():
            file_name = join(self.data_path, 'ddc.jsonl')
            label_dict = {'ISBN': [],
                          'Title': [],
                          'DDC': [],
                          'Description': []}
            with open(file_name, 'r', encoding='utf8') as infile:
                for line in infile:
                    entry = json.loads(line)
                    if 'isbn' not in entry:
                        isbnl = []
                    else:
                        isbnl = entry['isbn']
                    title = entry['title']
                    ddc = entry['ddc'][0]
                    if len(ddc) == 0:
                        continue
                    else:
                        ddc = re.sub("[^0-9]", "", ddc)
                    if not ddc.isdigit() or ddc == '':
                        continue

                    label_dict['ISBN'].append(isbnl)
                    label_dict['Title'].append(title)
                    label_dict['DDC'].append([ddc])
                    label_dict['Description'].append('')
            UB_frame = pd.DataFrame.from_dict(label_dict)
            return UB_frame

        def parse_TIB_data():
            tib_data = pd.read_csv(join(self.data_path, 'buchtitel_hannover.csv'), delimiter=',',
                                   names=['Title', 'DDC'])
            ddc_codes = tib_data['DDC']
            output_ddcs = []
            for ddc_index in ddc_codes:
                ddc = re.sub("[^0-9]", "", ddc_index)
                output_ddcs.append(ddc)
            tib_data['DDC'] = output_ddcs
            empty = pd.Series(['' for empty_index in range(len(tib_data))])
            tib_data['ISBN'] = empty
            tib_data['Description'] = empty
            return tib_data

        def parse_UOS_data():
            def clean_title(isbn, ddc, split_scope, description, existing_dict):
                # Sub function to title  from errors
                initial_entry = True
                subsplit_items = re.split('\n', split_scope)
                new_title = re.split('∏', subsplit_items[0])[0]
                existing_dict['ISBN'].append(isbn)
                existing_dict['Title'].append(new_title)
                existing_dict['DDC'].append(ddc)
                existing_dict['Description'].append(description)

                for sub_item in subsplit_items:

                    if initial_entry:
                        initial_entry = False
                        continue

                    category_split = re.split('∏', sub_item)
                    isbn_split = re.split('∐', category_split[0])[0]
                    ddc_split = re.split('∐', category_split[1])
                    sub_title = category_split[2]
                    if len(category_split) == 6:  # length of 6 indicates that there is a data field for description
                        sub_description = category_split[5]
                    else:
                        sub_description = ''
                    ddc_index = [re.sub("[^0-9]", "", ddc_index) for ddc_index in ddc_split]

                    if ddc_index[0] != '':
                        existing_dict['ISBN'].append(isbn_split)
                        existing_dict['Title'].append(sub_title)
                        existing_dict['DDC'].append(ddc_index)
                        existing_dict['Description'].append(sub_description)
                return existing_dict

            def clean_description(isbn, ddc, title, split_scope, existing_dict):
                # Sub function to clean description from errors
                initial_entry = True
                subsplit_items = re.split('\n', split_scope)
                new_description = subsplit_items[0]
                existing_dict['ISBN'].append(isbn)
                existing_dict['Title'].append(title)
                existing_dict['DDC'].append(ddc)
                existing_dict['Description'].append(new_description)

                for sub_item in subsplit_items:
                    if initial_entry:
                        initial_entry = False
                        continue

                    category_split = re.split('∏', sub_item)
                    isbn_split = re.split('∐', category_split[0])[0]
                    ddc_split = re.split('∐', category_split[1])
                    sub_title = category_split[2]
                    if len(category_split) == 6:  # length of 6 indicates that there is a data field for description
                        sub_description = category_split[5]
                    else:
                        sub_description = ''
                    ddc_index = [re.sub("[^0-9]", "", ddc_index) for ddc_index in ddc_split]
                    if ddc_index[0] != '':
                        existing_dict['ISBN'].append(isbn_split)
                        existing_dict['Title'].append(sub_title)
                        existing_dict['DDC'].append(ddc_index)
                        existing_dict['Description'].append(sub_description)
                return existing_dict

            ###

            file_name = join(self.data_path, 'bestand_0700_20181231_isbn_ddc.csv')

            existing_dict = {'ISBN': [],
                             'Title': [],
                             'DDC': [],
                             'Description': []}

            with open(file_name, 'r', encoding='utf8') as infile:
                reader = csv.reader(infile, delimiter='∏')
                for row in reader:
                    isbn = re.split("∐", row[0])
                    ddc = re.split("∐", row[1])
                    title = row[2]
                    if len(row) == 6:  # length of 6 indicates that there is a data field for description
                        description = row[5]
                    else:
                        description = ''

                    ddc = [re.sub("[^0-9]", "", sub_code) for sub_code in ddc]
                    ddc = list(dict.fromkeys(ddc))

                    if '∏' in title or "∐" in title:
                        existing_dict = clean_title(isbn, ddc, title, description, existing_dict)

                    elif '∏' in description or "∐" in description:
                        existing_dict = clean_description(isbn, ddc, title, description, existing_dict)

                    elif ddc[0] != '':
                        existing_dict['ISBN'].append(isbn)
                        existing_dict['Title'].append(title)
                        existing_dict['DDC'].append(ddc)
                        existing_dict['Description'].append(description)
                uos_frame = pd.DataFrame.from_dict(existing_dict)
            return uos_frame

        def parse_Crawled_data():
            file_name = join(self.data_path, 'crawled_ddcs.csv')
            crawled_dict = {'ISBN': [], 'Title': [], 'DDC': [], 'Description': []}
            with open(file_name) as infile:
                reader = csv.reader(infile)
                next(reader)
                for line in reader:
                    isbn = line[0]
                    ddc = line[1]
                    title = line[2]
                    description = line[4]
                    crawled_dict['ISBN'].append(isbn)
                    ddc = [re.sub("[^0-9]", "", ddc) for ddc_index in ddc]
                    ddc = list(dict.fromkeys(ddc))
                    crawled_dict['DDC'].append(ddc)
                    crawled_dict['Title'].append(title)
                    crawled_dict['Description'].append(description)
            crawled_frame = pd.DataFrame.from_dict(crawled_dict)
            return crawled_frame

        self.uos_frame = parse_UOS_data()
        self.ub_frame = parse_UB_data()
        self.crawled_frame = parse_Crawled_data()
        self.TIB_frame = parse_TIB_data()
        all_frames = [self.uos_frame, self.ub_frame, self.crawled_frame, self.TIB_frame]
        complete_frame = pd.concat(all_frames).explode(
            'DDC')  # remove this if you want all DDCs to be placed in one row
        complete_frame = complete_frame.dropna(subset=['DDC'], ).drop_duplicates(subset=['Title']).reset_index()
        return complete_frame


if __name__ == '__main__':

    ddc_extractor = DDCBookExtractor()
    dataset = ddc_extractor.parse_collect_raw_data()  # DF 5 columns: [index, ISBN, Title, DDc, Description]

    # Save the full raw dataset
    dataset.to_csv(join(data_root, 'datasets', 'full_dataset.csv'))
