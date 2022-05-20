import logging
import os
import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect_langs
from iso639 import languages

from models.preprocessing.original_ddc_loader import load_classes_from_tsv
from utils import settings

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
logger.addHandler(handler)

project_root = settings.get_project_root()
data_root = settings.get_data_root()


def language_detector(text):
    language_list = []
    lang_prob_list = []
    langs = detect_langs(text)
    confidence_prob = 0
    for lang in langs:
        confidence_prob += lang.prob
        lang_iso639 = lang.lang
        language_list.append(languages.get(alpha2=lang_iso639).name)
        lang_prob_list.append(lang.prob)
        if confidence_prob > 0.90:
            break
        else:
            continue
    return language_list, lang_prob_list


def extract_book_language(root_classes=None,
                          original_only=True,
                          include_root_class=True,
                          include_language=True,
                          include_lang_probs=False,
                          exclude=None,
                          aggregate_level=None,
                          random_seed=0,
                          save_to_disk=False):

    if root_classes:
        root_classes = root_classes
    else:
        root_classes = list(np.arange(10).astype('str'))  # all 10 root classes

    df_class_data = []
    original_classes = load_classes_from_tsv(os.path.join(data_root, 'datasets', 'book_ddc_data', 'classes.tsv'))

    for class_index in root_classes:
        logger.info('Loading data for class: {}'.format(class_index))
        data = pd.read_csv(os.path.join(data_root, 'model_data', 'pooler_output', 'samples_ddc_class_'+class_index+'.csv'),
                           usecols=['index', 'Title', 'DDC', 'Description'])
        if exclude:
            data = data.loc[~data['DDC'].isin(exclude)]
        if aggregate_level:
            data = data[data['DDC'].str.len() <= aggregate_level]

        # Filtering out data & embeddings for original DDC classes
        if original_only:
            # logger.info('Filtering out embeddings from the 905 original classes only')
            data = data[data['DDC'].isin(original_classes)].reset_index(drop=True)
        df_class_data.append(data)

    dataset = pd.concat(df_class_data).reset_index(drop=True)
    del df_class_data

    # Extracting language of books
    if include_language:
        DetectorFactory.seed = int(random_seed)
        logger.info('Including a \'language\' column (only books with titles greater than 20 characters)')
        dataset = dataset[dataset['Title'].str.len() >= 20].reset_index(drop=True)
        titles_language = []
        titles_language_probs = []
        for i in range(len(dataset)):
            lang_list, prob_list = language_detector(dataset['Title'][i])
            titles_language.append(lang_list)
            titles_language_probs.append(prob_list)
        dataset['language'] = titles_language
        if include_lang_probs:
            dataset['language_probs'] = titles_language_probs
        del titles_language, titles_language_probs

    # Including a column for 'root_class'
    if include_root_class:
        logger.info('Including the \'root_class\' column')
        titles_root_classes = []
        for i in range(len(dataset)):
            titles_root_classes.append(dataset['DDC'].iloc[i][0])
        dataset['root_class'] = titles_root_classes

    if save_to_disk:
        logger.info('Saving dataframe to disk (as a numpy array of type \'object\')...')
        df_filename = 'Classes{}_OriginalOnly{}_Lang{}_Probs{}'.format(''.join(root_classes),
                                                                       original_only,
                                                                       include_language,
                                                                       include_lang_probs)
        dataset.to_csv(os.path.join(data_root, 'datasets', df_filename + '.csv'))

    logger.info('Returning requested data...')
    return dataset


if __name__ == '__main__':
    # classes_to_extract = ['1']
    original_only = True
    include_root_class = True
    include_language = True
    include_lang_probs = True
    save_to_disk = True

    dl = extract_book_language(original_only=original_only,
                               include_root_class=include_root_class,
                               include_language=include_language,
                               include_lang_probs=include_lang_probs,
                               save_to_disk=save_to_disk)

    # TODO: Finish writing this script (complete usage is in jupyter notebook)
    # TODO: Use the correct data path in line 60 above
