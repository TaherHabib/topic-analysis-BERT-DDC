import os
import numpy as np
import yaml
import logging
import argparse
from dataprocessing.dataset_processing import DatasetCreator
from models.utils.DataLoader import SidBERTDataloader
from models import SidBERT, OmegaBERT
from utils import settings

# Set a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)


def generate_embeddings(dataloader=None, model=None, prune_at_layer='pooler_output'):

    embeddings = []

    for n, (batch_tokens, orig_indices) in enumerate(dataloader):
        if n % 1000 == 0:
            logger.info('Passing batch={} of {} batches'.format(n, len(dataloader)))

        # 'embeddings_' are of size (<size of batch>, <dimension of prune_at_layer's output>)
        embeddings_ = model.batch_get_pruned_model_output(batch_tokenized_data=batch_tokens, prune_at_layer=prune_at_layer)

        # Adding original indices from dataframe at 0th position, shifting embeddings one position to the right
        idx_embeddings = np.column_stack((orig_indices, embeddings_))

        embeddings.append(idx_embeddings)  # list of shape:
        # (<number of batches>, <size of batch>, <dimension of prune_at_layer's output>+1)

    # Reshaping 'embeddings' list into a numpy array sized:
    # (<total number of entries in dataset>, <dimension of prune_at_layer's output>+1>)
    # Total number of entries = number of batches * size of batch
    embeddings = np.array(embeddings)
    embeddings.reshape(embeddings.shape[0]*embeddings.shape[1], embeddings.shape[-1])

    np.savez_compressed(embeddings=embeddings,
                        file=os.path.join(data_root, 'model_data', 'model_embeddings_{}.npz'.format(prune_at_layer)))

    logger.info('Embeddings generation for the requested layer completed.')


parser = argparse.ArgumentParser(description='Generate Model Embeddings')
parser.add_argument('prune_at_layer', action='store', default=None, type=str,
                    help='Layer at which to prune the model to extract its output.')
parser.add_argument('dataset_filename', action='store', default=None, type=str,
                    help='Name of the dataset file containing all the book titles.')
parser.add_argument('-c', '--columns_to_use', dest='columns_to_use', action='append', required=True,
                    help='Columns to use for embeddings generation (should be columns with strings).')
parser.add_argument('-m', '--model_to_use', dest='model_to_use', action='store', required=True,
                    help='Columns to use for embeddings generation (should be columns with strings).')

if __name__ == '__main__':

    args = parser.parse_args()

    data_root = settings.get_data_root()
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    logger.info('Getting requisite data processing params from YAML config file')
    project_root = settings.get_project_root()
    with open(os.path.join(project_root, 'configs', 'train_config.yml'), 'r') as f:
        train_config = yaml.safe_load(f)

    logger.info('Creating dataset (containing only the titles and/or descriptions)...')
    dataset_, _, _ = DatasetCreator(dataset_filename=args.dataset_filename,
                                    aggregate_ddc_level=train_config['aggregate_ddc_level'],
                                    columns_to_use=args.columns_to_use,
                                    min_title_length=train_config['min_title_length']
                                    ).generate_dataset(embeddings_generator_mode=True)

    logger.info('Initializing Dataloader with batch size {} and sequence length {}'.format(train_config['batch_size'],
                                                                                           train_config['max_length']))
    dataloader = SidBERTDataloader(dataset=dataset_,
                                   batch_size=train_config['batch_size'],
                                   max_length=train_config['max_length'],
                                   embeddings_generator_mode=True)

    logger.info('Setting up the requisite model...')
    if args.model_to_use.lower() == 'omegabert':
        with open(os.path.join(project_root, 'configs', 'omegabert_config.yml'), 'r') as f:
            model_config = yaml.safe_load(f)
        model = OmegaBERT.OmegaBERT(freeze_bert_layers=model_config['freeze_bert_layers'],
                                    restore_model=True,
                                    sequence_max_length=train_config['max_length'],
                                    ddc_target_classes=sorted(list(dataset_['DDC'].unique())),
                                    leaky_relu_alpha=model_config['leaky_relu_alpha']
                                    )
    elif args.model_to_use.lower() == 'sidbert':
        with open(os.path.join(project_root, 'configs', 'sidbert_config.yml'), 'r') as f:
            model_config = yaml.safe_load(f)
        model = SidBERT.SidBERT(freeze_bert_layers=model_config['freeze_bert_layers'],
                                restore_model=True,
                                sequence_max_length=train_config['max_length'],
                                ddc_target_classes=sorted(list(dataset_['DDC'].unique()))
                                )
    else:
        raise ValueError('Please enter a valid model name (string)')

    # Checking if the correct layer name was entered for pruning the model
    try:
        assert (args.prune_at_layer in model_config['model_layers']), \
            '\'prune_at_layer\' argument was found to be of a wrong value'
    except AssertionError:
        raise ValueError('Please enter a valid layer name (string)')

    logger.info('Generating embeddings...')
    generate_embeddings(dataloader=dataloader, model=model, prune_at_layer=args.prune_at_layer)

    logger.info('Embeddings generated for the requested layer.')


# ==============CLASS SAMPLE COUNTS==============================
# Accounting for batch size=16 (3332 samples less than the whole)
# Total samples=1315988, Total samples in batches=1312656
# {'class_0': 109680,
# 'class_1': 56416,
# 'class_2': 61488,
# 'class_3': 367664,
# 'class_4': 41712,
# 'class_5': 177888,
# 'class_6': 235136,
# 'class_7': 86016,
# 'class_8': 102496,
# 'class_9': 74160}

# When counting from only the original 905 classes, total samples
# equal to 557871. Class-wise they are:
# {'class_0': 47863,
# 'class_1': 28743,
# 'class_2': 21250,
# 'class_3': 178358,
# 'class_4': 17558,
# 'class_5': 76874,
# 'class_6': 92990,
# 'class_7': 37346,
# 'class_8': 34278,
# 'class_9': 22611}
# ===============================================================







# def generate_bert_embeddings():
    # train_df, test_df, class_weight_dict = DatasetCreator(dataset_filename=config.dataset_filename,
    #                                                       aggregate_ddc_level=config.ddc_level,
    #                                                       min_title_length=config.min_title_length,
    #                                                       test_size=config.train_test_ratio,
    #                                                       balanced_class_distribution=config.balanced_classes).generate_final_dataset()
    # dataloader = DataLoader.SidBERTDataloader(dataset=train_df,
    #                                           batch_size=config.batch_size,
    #                                           max_length=config.max_length,
    #                                           generator_mode=True)
    # model = build_vanilla_bert()
    # for n, index in enumerate(dataloader):
    #     if n % 100 == 0:
    #         print(f'passing {n} of {len(dataloader)}')
    #     np.save(os.path.join(os.path.abspath('models'), 'data', 'embeddings', f'embeddings_batch_{n}.npy'),
    #             arr=model.predict(index))

# def compress_embeddings():
#     path = os.path.join(os.path.abspath('models'), 'data', 'embeddings')
#     num_embeddings = len(glob.glob(os.path.join(path, 'embeddings_batch_*')))
#     compress_array = np.zeros((num_embeddings, 64, 768))
#     for index in range(num_embeddings):
#         compress_array[index] = np.load(os.path.join(path, f'embeddings_batch_{index}.npy'))
#     np.savez_compressed(os.path.join(os.path.abspath('models'), 'data', 'compressed_embeddings_64_batch.npz'),
#                         compress_array)

# def build_vanilla_bert():
#     bert_model = transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased')
#     input_ids = tf.keras.layers.Input(shape=(config.max_length,), name='input_ids', dtype='int32')
#     input_masks_ids = tf.keras.layers.Input(shape=(config.max_length,), name='attention_mask', dtype='int32')
#     input_type_ids = tf.keras.layers.Input(shape=(config.max_length,), name='token_type_ids', dtype='int32')
#     out = bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids).pooler_output
#     model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids], outputs=out)
#     model.compile()
#     return model
