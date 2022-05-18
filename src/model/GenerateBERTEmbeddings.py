from src.preprocessing import DatasetProcessing, Dataloaders
import transformers
import tensorflow as tf
from src.model import config
import os
import glob
import h5py
import numpy as np

def generate_bert_embeddings():
    dataset, empty_test, inbalance = DatasetProcessing.DatasetCreator().generate_dataset(even=False, train_test_ratio=0)
    datlo = Dataloaders.SidBERTDataloader(batch_size=64, max_length=config.max_length, dataset=dataset, generator_mode=True)
    model = build_vanilla_bert()

    for n, index in enumerate(datlo):
        if n % 100 == 0:
            print(f'passing {n} of {len(datlo)}')
        np.save(os.path.join(os.path.abspath('.'),'data','embeddings',f'embeddings_batch_{n}.npy'),arr=model.predict(index))

def compress_embeddings():
    path = os.path.join(os.path.abspath('.'),'data','embeddings')
    num_embeddings = len(glob.glob(os.path.join(path,'embeddings_batch_*')))
    compress_array = np.zeros((num_embeddings,64,768))
    for index in range(num_embeddings):
        compress_array[index] = np.load(os.path.join(path,f'embeddings_batch_{index}.npy'))
    np.savez_compressed(os.path.join(os.path.abspath('.'),'data','compressed_embeddings_64_batch.npz'),compress_array)

def build_vanilla_bert():
    bert_model = transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased')
    input_ids = tf.keras.layers.Input(shape=(config.max_length,), name='input_ids', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(shape=(config.max_length,), name='attention_mask', dtype='int32')
    input_type_ids = tf.keras.layers.Input(shape=(config.max_length,), name='token_type_ids', dtype='int32')
    out = bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids).pooler_output
    model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids],
                           outputs=out)
    model.compile()
    return model