import tensorflow as tf
import transformers
import numpy as np
from utils.settings import HF_bert_model_name, HF_distilbert_model_name


class BERTDataloader(tf.keras.utils.Sequence):
    def __init__(self,
                 dataset=None,
                 bert_model_name=None,
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
        self.tokenizer = self.get_tokenizer(bert_model_name)

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
                                                              return_tensors="tf")
        if self.embeddings_generator_mode:
            return tokenized_encoding, self.orig_indices[local_indices]
        else:
            return tokenized_encoding, self.classes[local_indices]

    def on_epoch_end(self):
        if not self.generator_mode:
            np.random.RandomState(42).shuffle(self.indices)

    @staticmethod
    def get_tokenizer(model_name):

        if model_name == HF_bert_model_name:
            tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
        elif model_name == HF_distilbert_model_name:
            tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
        else:
            raise ValueError('Wrong model name found. Check available names of models from HuggingFace.')

        return tokenizer

