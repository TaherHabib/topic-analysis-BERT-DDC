import tensorflow as tf
import transformers
import numpy as np
import logging

class SidBERTDataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size = 16, max_length = 300):
        logger = logging.getLogger('Dataloader').setLevel('INFO')
        titles = dataset['Title'].to_list()
        self.classes = tf.keras.utils.to_categorical(dataset['DDC'].values)
        self.titles = np.asarray(titles, dtype='str')
        self.indices = np.arange(len(self.titles))
        self.batch_size = batch_size
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.max_length = max_length
        logging.info(f'Sucessfully initalized Dataloader with batch size {self.batch_size} and sequence length {self.max_length}')

    def __len__(self):
        return len(self.titles) // self.batch_size

    def __getitem__(self, index):
        local_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        sentence = self.titles[local_indices]
        classes = self.classes[local_indices]
        tokenized_encoding = self.tokenizer.batch_encode_plus(sentence,
                                              add_special_tokens=True,
                                              padding='max_length',
                                              max_length=self.max_length,
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_token_type_ids=True,
                                              pad_to_max_length=True,
                                              return_tensors="tf",).data
        return tokenized_encoding, classes