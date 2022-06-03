import tensorflow as tf
import transformers
import numpy as np
import logging
from utils.settings import HF_model_name

# Set a logger
logger = logging.getLogger('Dataloader')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)


class SidBERTDataloader(tf.keras.utils.Sequence):
    def __init__(self,
                 dataset=None,
                 batch_size=16,
                 max_length=300,
                 embeddings_generator_mode=False):

        self.classes = tf.keras.utils.to_categorical(dataset['DDC'].values)
        self.text_ = dataset['text_'].values
        self.orig_indices = dataset['orig_index'].values
        self.indices = np.arange(len(self.text_))
        self.embeddings_generator_mode = embeddings_generator_mode
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = transformers.BertTokenizer.from_pretrained(HF_model_name)
        logger.info(f'Successfully initialized Dataloader with batch size {self.batch_size} and sequence length {self.max_length}')

    def __len__(self):
        return len(self.text_) // self.batch_size

    def __getitem__(self, index):
        local_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        text = self.text_[local_indices]
        tokenized_encoding = self.tokenizer.batch_encode_plus(text,
                                                              add_special_tokens=True,
                                                              padding='max_length',
                                                              max_length=self.max_length,
                                                              truncation=True,
                                                              return_attention_mask=True,
                                                              return_token_type_ids=True,
                                                              pad_to_max_length=True,
                                                              return_tensors="tf",).data
        if self.embeddings_generator_mode:
            return tokenized_encoding, self.orig_indices[local_indices]
        else:
            return tokenized_encoding, self.classes[local_indices]

    def on_epoch_end(self):
        if not self.generator_mode:
            np.random.RandomState(42).shuffle(self.indices)

