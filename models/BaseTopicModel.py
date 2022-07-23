import os
import transformers
import tensorflow as tf
from utils.settings import get_data_root, HF_bert_model_name, HF_distilbert_model_name
from utils.data_utils import original_ddc_loader as odl

data_root = get_data_root()


class BaseBERTModel:
    def __init__(self,
                 freeze_bert_layers=True,
                 sequence_max_length=300,
                 ddc_target_classes=None,
                 bert_model_name=None
                 ):

        self.bert_model_name = bert_model_name
        self.freeze_bert_layers = freeze_bert_layers

        # create label lookup table for label assignment from last classification layer
        self.ddc_target_classes = ddc_target_classes
        self.class_position_hash = odl.create_ddc_label_lookup(self.ddc_target_classes)

        # create tokenizer and set sequence length:
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model_name)
        self.max_length = sequence_max_length

        self.bert_output, self.model = self._build_model()

    def _build_model(self):

        input_ids = tf.keras.layers.Input(shape=(self.max_length,), name='input_ids', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(self.max_length,), name='attention_mask', dtype='int32')
        input_type_ids = tf.keras.layers.Input(shape=(self.max_length,), name='token_type_ids', dtype='int32')

        if self.bert_model_name == HF_bert_model_name:
            bert_model = transformers.TFBertModel.from_pretrained(self.bert_model_name)
            bert_output = bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids)
            model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids], outputs=bert_output)

        elif self.bert_model_name == HF_distilbert_model_name:
            bert_model = transformers.TFDistilBertModel.from_pretrained(self.bert_model_name)
            bert_output = bert_model(input_ids, attention_mask=input_masks_ids)
            model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=bert_output)

        else:
            raise ValueError('Wrong model name found. Check available names of models from HuggingFace.')

        if self.freeze_bert_layers:
            bert_model.trainable = False

        return bert_output, model

    def batch_get_pruned_model_output(self, batch_tokenized_data=None, batch_size=None, prune_at_layer=None):

        local_model = self.model
        if prune_at_layer == 'pooler_output':
            pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=self.bert_output.pooler_output)
        elif prune_at_layer == 'sequence_output':
            pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=self.bert_output.last_hidden_state)
        else:
            raise ValueError('Please enter a valid layer name for the base bert model: pooler_output OR sequence_output')

        return pruned_model.predict(x=batch_tokenized_data, batch_size=batch_size)

    def predict_single_example(self, sequence, top_n=1):
        """
        queries the models neural network to produce a DDC label assignment together with an associated probability
        sequence: String input that is to be classified
        top_n: number of DDC labels to be returned. Defaults to 1

        :return: dictionary with structure: key: DDC code value: probability
        :rtype: python 3 dictionary object
        """
        encoded_sequence = self.tokenizer.encode_plus(sequence,
                                                      add_special_tokens=True,
                                                      padding='max_length',
                                                      max_length=self.max_length,
                                                      truncation=True,
                                                      return_attention_mask=True,
                                                      return_token_type_ids=True,
                                                      pad_to_max_length=True,
                                                      return_tensors="tf")

        prediction_result = self.model.predict([encoded_sequence['input_ids'],
                                                encoded_sequence['attention_mask'],
                                                encoded_sequence['token_type_ids']])[0]

        # Extract 'top_n' entries from the sorted indices of the model's prediction probabilities
        max_ddcClasses_indices = prediction_result.argsort()[::-1][:top_n]  # contains 'top_n' position_indices with
        # max. probabilities outputted by the trained models
        max_ddcClasses_probs = prediction_result[max_ddcClasses_indices]
        max_ddcClasses_labels = [self.class_position_hash[index] for index in max_ddcClasses_indices]

        return dict(zip(max_ddcClasses_labels, max_ddcClasses_probs))
