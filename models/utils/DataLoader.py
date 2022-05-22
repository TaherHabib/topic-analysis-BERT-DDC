import tensorflow as tf
import transformers
import numpy as np
import logging

# Set a logger
logger = logging.getLogger('Dataloader')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
# logger.addHandler(handler)


class SidBERTDataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=16, max_length=300, generator_mode=False):
        self.classes = tf.keras.utils.to_categorical(dataset['DDC'].values)
        self.titles = dataset['Title'].values
        self.indices = np.arange(len(self.titles))
        self.generator_mode = generator_mode
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        logger.info(f'Successfully initialized Dataloader with batch size {self.batch_size} and sequence length {self.max_length}')

    def __len__(self):
        return len(self.titles) // self.batch_size

    def __getitem__(self, index):
        local_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        sentence = self.titles[local_indices]
        tokenized_encoding = self.tokenizer.batch_encode_plus(sentence,
                                                              add_special_tokens=True,
                                                              padding='max_length',
                                                              max_length=self.max_length,
                                                              truncation=True,
                                                              return_attention_mask=True,
                                                              return_token_type_ids=True,
                                                              pad_to_max_length=True,
                                                              return_tensors="tf",).data
        if self.generator_mode:
            return tokenized_encoding
        else:
            return tokenized_encoding, self.classes[local_indices]

    def on_epoch_end(self):
        if not self.generator_mode:
            np.random.RandomState(42).shuffle(self.indices)
