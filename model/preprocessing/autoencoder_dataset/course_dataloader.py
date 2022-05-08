import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import transformers
import logging


class CourseDataloader(k.utils.Sequence):
    """
    Dataloader to efficiently pass Course Data into Autoencoder network.
    :param text_only_mode: for generating BERT embeddings only
    :param generator_mode: whether to generate autoencoder input for single samples only

    :param titles: List of course titles to be pre-processed
    :param ddc_labels List of one-hot encoded ddc labels from model
    :param shuffle whether to shuffle the dataset if the model has finished training for one episode
    """
    def __init__(self, titles, ddc_labels, batch_size, shuffle=True,
                 text_only_mode = False,
                 generator_mode=False,
                 pooler_only=False):
        self.text_only_mode = text_only_mode
        self.generator_mode = generator_mode
        self.titles = titles
        self.ddc_labels = ddc_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bert_embedding_generator = BertEmbeddingModel(pooler_only)
        self.indices = np.arange(len(self.titles))
        if shuffle:
            self.on_epoch_end()
        logging.info('Course Dataloder initalized successfully.')

    def __len__(self):
        return len(self.titles) // self.batch_size

    def __getitem__(self, index):
        """
        Provides batches for training
        :param index: training index
        :return: batch of encoded titles and associated ddc code
        """
        local_indeces = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        titles = self.titles[local_indeces]
        encoded_titles = self.bert_embedding_generator.generate_single_embedding(titles)
        if self.text_only_mode:
            x = encoded_titles
        else:
            ddcs = np.array(self.ddc_labels[local_indeces], dtype='int32')
            x = [encoded_titles, ddcs]
        if self.generator_mode:
            return x
        else:
            return (x,x)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indices)


class BertEmbeddingModel:

    """
    Vanilla BERT model to encoder Course Title information into BERT embeddings and to use them downstream.
    """
    def __init__(self, pooler_only = True, input_length = 256):
        """
        :param pooler_only: Whether only the pooled output of BERT's CLS token is to be encoded.
        Not recommended as CLS tokens typically hold little information on actual natural language meaning.
        """
        self.pooler_only = pooler_only
        self.bert_model = transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.max_length = input_length
        self.embedding_generator = self._prepare_bert()

    def _prepare_bert(self):
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), name='input_token', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(self.max_length,), name='attention_mask', dtype='int32')
        input_type_ids = tf.keras.layers.Input(shape=(self.max_length,), name='token_type_ids',dtype='int32')
        output = self.bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids)

        model = tf.keras.models.Model(
            inputs=[input_ids, input_masks_ids, input_type_ids], outputs=output
        )
        return model

    def generate_single_embedding(self, sentence, batch_mode = True):
        if batch_mode:
            tokenized_sentence = self.bert_tokenizer.batch_encode_plus(sentence,
                                                                 add_special_tokens=True,
                                                                 padding='max_length',
                                                                 max_length=self.max_length,
                                                                 truncation=True,
                                                                 return_attention_mask=True,
                                                                 return_token_type_ids=True,
                                                                 pad_to_max_length=True,
                                                                 return_tensors="tf", )
        else:
            tokenized_sentence = self.bert_tokenizer.encode_plus(sentence,
                                                   add_special_tokens=True,
                                                   padding='max_length',
                                                   max_length=self.max_length,
                                                   truncation=True,
                                                   return_attention_mask=True,
                                                   return_token_type_ids=True,
                                                   pad_to_max_length=True,
                                                   return_tensors="tf", )
        input_ids = np.array(tokenized_sentence["input_ids"], dtype="int32")
        attention_masks = np.array(tokenized_sentence["attention_mask"], dtype="int32")
        token_type_ids = np.array(tokenized_sentence["token_type_ids"], dtype="int32")
        output = self.embedding_generator([input_ids,attention_masks,token_type_ids])
        if self.pooler_only:
            return output[1]
        else:
            return output[0]
