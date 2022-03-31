import tensorflow as tf
import transformers
import numpy as np
from os.path import abspath, join

from utils import settings
from preprocessing.original_DDCClass_loader import load_classes_from_tsv, \
    create_ddc_label_lookup


class SidBERTDataloader(tf.keras.utils.Sequence):
    def __init__(self, titles, batch_size=16, max_length=300):
        self.titles = np.asarray(titles, dtype='str')
        self.indices = np.arange(len(self.titles))
        self.batch_size = batch_size
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.max_length = max_length

    def __len__(self):
        return len(self.titles) // self.batch_size

    def __getitem__(self, index):
        local_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        sentence = self.titles[local_indices]
        tokenized_encoding = self.tokenizer.batch_encode_plus(sentence,
                                                              add_special_tokens=True,
                                                              padding='max_length',
                                                              max_length=self.max_length,
                                                              truncation=True,
                                                              return_attention_mask=True,
                                                              return_token_type_ids=True,
                                                              pad_to_max_length=True,
                                                              return_tensors="tf", ).data
        return tokenized_encoding


class SidBERT:
    """
    this class loads the trained BERT model, and provide api to call the model to get relevant ddc codes.
    All interactions related to database lookup operations for course retrieval are handled in
    recommender_backbone.py
    """
    def __init__(self):
        """
        Constructor of the class loads the trained model for prediction.
        Use configuration from BERT_CONF config.py to load the model
        """
        project_root = settings.get_project_root()
        data_path = join(project_root, 'src', 'data', 'SidBERT_data')
        # load models
        self.model_checkpoint_path = join(data_path, 'bert_models')

        # create label lookup table for label assignment from last classification layer
        self.classes = load_classes_from_tsv(join(data_path, 'bert_data', 'classes.tsv'))
        self.class_position_hash = create_ddc_label_lookup(self.classes)

        # create tokenizer and set sequence length:
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.max_length = 300

        # load trained model
        self.model = self._load_model()
        self.pruned_model = None
        self.loader = None

    def construct_dataloader(self, frame, batch_size):
        titles = frame['Title'].to_list()
        self.loader = SidBERTDataloader(titles=titles, batch_size=batch_size)
        return self.loader

    def _load_model(self):
        """
        Constructs model topology and loads weights from Checkpoint file. Topology needs to be changed
        manually every time a new model version is installed into the backend.
        :return: Tensorflow 2 keras model object containing the model architecture
        :rtype: tensorflow.keras.Model object
        """
        # Construct model topology
        bert_model = transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased')

        input_ids = tf.keras.layers.Input(shape=(300,), name='input_ids', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(300,), name='attention_mask', dtype='int32')
        input_type_ids = tf.keras.layers.Input(shape=(300,), name='token_type_ids', dtype='int32')
        out = bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids)
        sequence_output = out.last_hidden_state
        concat = tf.keras.layers.Dense(3000, activation='relu', name='dense_3000')(sequence_output)
        concat = tf.keras.layers.GlobalAveragePooling1D()(concat)
        dropout = tf.keras.layers.Dropout(0.35)(concat)
        dropout = tf.keras.layers.Dense(2048, activation='relu', name='dense_2048')(dropout)
        dropout = tf.keras.layers.Dropout(0.25)(dropout)
        output = tf.keras.layers.Dense(len(self.classes), activation="softmax", name='final')(dropout)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids],
                               outputs=output)

        # restore model weights from checkpoint file
        model.load_weights(join(self.model_checkpoint_path, 'new_training_latest_architecture'))
        return model

    def batch_get_pruned_model_output(self, dataset, batch_size, prune_at_layer=None):
        outputs = []
        local_model = self.model

        if type(prune_at_layer) is int:
            if prune_at_layer == 3:
                outputs.append(local_model.get_layer(index=prune_at_layer).output[1])  # to get pooler_output
            else:
                outputs.append(local_model.get_layer(index=prune_at_layer).output)
        else:
            if prune_at_layer == 'pooler_output':
                outputs.append(local_model.get_layer(index=3).output[1])  # to get pooler_output
            else:
                outputs.append(local_model.get_layer(name=prune_at_layer).output)

        self.pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=outputs)

        return self.pruned_model.predict(x=dataset, batch_size=batch_size)

    def predict_single_example(self, sequence, top_n=1):
        """
        queries the SidBERT neural network to produce a DDC label assignment together with an associated probability
        sequence: String input that is to be classified
        top_n: number of DDC labels to be returned. Defaults to 1

        :return: dictionary with structure: key: DDC code value: probability
        :rtype: python 3 dictionary object
        """
        encoded_sequence = self.tokenizer.encode_plus(sequence, add_special_tokens=True, padding='max_length',
                                                      max_length=self.max_length, truncation=True,
                                                      return_attention_mask=True,
                                                      return_token_type_ids=True, pad_to_max_length=True,
                                                      return_tensors="tf")

        prediction_result = self.model.predict([encoded_sequence['input_ids'],
                                                encoded_sequence['attention_mask'],
                                                encoded_sequence['token_type_ids']])[0]

        # Extract 'top_n' entries from the sorted indices of the model's prediction probabilities
        max_ddcClasses_indices = prediction_result.argsort()[::-1][:top_n]  # contains 'top_n' position_indices with
        # max. probabilities outputted by the trained model
        max_ddcClasses_probs = prediction_result[max_ddcClasses_indices]
        max_ddcClasses_labels = [self.class_position_hash[index] for index in max_ddcClasses_indices]

        return dict(zip(max_ddcClasses_labels, max_ddcClasses_probs))

    def _return_model_summary(self):
        return self.model.summary()
