import transformers
import tensorflow as tf


class SemanticBERT:

    def __init__(self):
        self.model = None
        pass

    def build_model(self):
        bert_layer = transformers.TFBertForPreTraining.from_pretrained('bert-base-multilingual-cased')
        model = tf.keras.Model()
        pass

    def save_model(self):

        pass

    def load_model(self):
        pass

    def train_dataloader(self):
        model = self.model
