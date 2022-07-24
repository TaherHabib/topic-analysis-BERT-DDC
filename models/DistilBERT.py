import glob
import os
import tensorflow as tf
import transformers
from models.utils import custom_decay
from utils.data_utils import original_ddc_loader as odl
from models.BaseTopicModel import BaseBERTModel
from utils.settings import get_data_root, HF_distilbert_model_name

data_root = get_data_root()


class DistilBERT:

    def __init__(self,
                 freeze_bert_layers=None,
                 restore_model=None,
                 sequence_max_length=300,
                 ddc_target_classes=None,
                 bert_model_name=None,
                 sequence_classifier=None
                 ):

        self.bert_model_name = bert_model_name
        self.freeze_bert_layers = freeze_bert_layers
        self.model_checkpoint_path = os.path.join(data_root, 'model_data', 'trained_models')

        # create label lookup table for label assignment from last classification layer
        self.ddc_target_classes = ddc_target_classes
        self.class_position_hash = odl.create_ddc_label_lookup(self.ddc_target_classes)

        # create tokenizer and set sequence length:
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(self.bert_model_name)
        self.max_length = sequence_max_length
        self.sequence_classifier = sequence_classifier

        self.bert_output, self.model = self._build_model(restore_model=restore_model)

    def _build_model(self, restore_model=None):

        input_ids = tf.keras.layers.Input(shape=(self.max_length,), name='input_ids', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(self.max_length,), name='attention_mask', dtype='int32')

        if self.sequence_classifier:
            distilbert_model = transformers.TFDistilBertForSequenceClassification.from_pretrained(self.bert_model_name)
        else:
            distilbert_model = transformers.TFDistilBertModel.from_pretrained(self.bert_model_name)

        if self.freeze_bert_layers:
            distilbert_model.trainable = False

        distilbert_output = distilbert_model(input_ids, attention_mask=input_masks_ids)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=distilbert_output)

        if restore_model:
            model.load_weights(os.path.join(self.model_checkpoint_path, '...'))
            return distilbert_output, model
        else:
            return distilbert_output, model

    def batch_get_pruned_model_output(self, batch_tokenized_data=None, prune_at_layer=None):

        local_model = self.model

        if self.sequence_classifier:
            if prune_at_layer == 'pooler_output':
                # TODO:
                pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=self.bert_output.pooler_output)
            elif prune_at_layer == 'classifier_output':
                pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=self.bert_output.logits)
            else:
                raise ValueError('Please enter a valid layer name for DistilBERTForSequenceClassification. '
                                 'Only: pooler_output OR classifier_output')
        else:
            if prune_at_layer == 'sequence_output':
                pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=self.bert_output.last_hidden_state)
            else:
                raise ValueError('Please enter a valid layer name for DistilBERT. Only: sequence_output')

        return pruned_model.predict(x=batch_tokenized_data.data)

    def _return_model_summary(self):
        return self.model.summary()




