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

        self.bert_output, self.model = self._build_model(restore_model=restore_model,
                                                         for_sequence_classification=sequence_classifier)

    def _build_model(self, restore_model=None, for_sequence_classification=None):

        if for_sequence_classification:
            distilbert_model = transformers.TFDistilBertForSequenceClassification.from_pretrained(self.bert_model_name)
        else:
            distilbert_model = transformers.TFDistilBertModel.from_pretrained(self.bert_model_name)

        distilbert_output = distilbert_model(input_ids, attention_mask=input_masks_ids)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids], outputs=bert_output)


        if self.freeze_bert_layers:
            bert_model.trainable = False

        return bert_output, model


