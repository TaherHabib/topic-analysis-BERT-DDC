import glob
import os
import tensorflow as tf
import transformers
from models.utils import custom_decay
from utils.data_utils import original_ddc_loader as odl
from utils.settings import get_data_root, HF_model_name

data_root = get_data_root()


class OmegaBERT:
    def __init__(self,
                 freeze_bert_layers=True,
                 restore_model=False,
                 sequence_max_length=300,
                 ddc_target_classes=None,
                 leaky_relu_alpha=0.1):

        self.freeze_bert_layers = freeze_bert_layers
        self.model_checkpoint_path = os.path.join(data_root, 'model_data', 'trained_models', 'OmegaBERT')

        # create label lookup table for label assignment from last classification layer
        self.ddc_target_classes = ddc_target_classes
        self.class_position_hash = odl.create_ddc_label_lookup(self.ddc_target_classes)

        # create tokenizer and set sequence length:
        self.tokenizer = transformers.BertTokenizer.from_pretrained(HF_model_name)
        self.max_length = sequence_max_length

        self.bert_output, self.model = self._build_model(restore_model=restore_model, leaky_relu_alpha=leaky_relu_alpha)
        # self.model_index_id = self.get_model_index()

    def _build_model(self, restore_model=False, leaky_relu_alpha=0.1):

        """
        Constructs model topology and loads weights from Checkpoint file. Topology needs to be changed
        manually every time a new models version is installed into the backend.
        :return: Tensorflow 2 keras models object containing the models architecture
        :rtype: tensorflow.keras.Model object
        """

        # Construct models topology
        bert_model = transformers.TFBertModel.from_pretrained(HF_model_name)
        if self.freeze_bert_layers:
            bert_model.trainable = False

        input_ids = tf.keras.layers.Input(shape=(self.max_length,), name='input_ids', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(self.max_length,), name='attention_mask', dtype='int32')
        input_type_ids = tf.keras.layers.Input(shape=(self.max_length,), name='token_type_ids', dtype='int32')
        bert_output = bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids)
        glob_leaky = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)
        ch = tf.keras.layers.Dense(2048,activation=glob_leaky, name='dense_2048')(bert_output.pooler_output)
        ch = tf.keras.layers.BatchNormalization(name='b_norm_2048')(ch)
        ch = tf.keras.layers.Dense(1024,activation=glob_leaky, name='dense_1024')(ch)
        ch = tf.keras.layers.BatchNormalization(name='b_norm_1024')(ch)
        ch = tf.keras.layers.Dense(256,activation=glob_leaky, name='dense_256')(ch)
        output = tf.keras.layers.Dense(len(self.ddc_target_classes), activation="softmax", name='final')(ch)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids], outputs=output)

        if restore_model:
            model.load_weights(os.path.join(self.model_checkpoint_path, 'new_training_latest_architecture'))
            return bert_output, model
        else:
            return bert_output, model

    def batch_get_pruned_model_output(self, batch_tokenized_data=None, prune_at_layer=None):

        local_model = self.model
        if prune_at_layer == 'pooler_output':
            pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=self.bert_output.pooler_output)
        elif prune_at_layer == 'sequence_output':
            pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=self.bert_output.last_hidden_state)
        else:
            pruned_model = tf.keras.Model(inputs=local_model.inputs, outputs=local_model.get_layer(name=prune_at_layer).output)

        return pruned_model.predict(x=batch_tokenized_data)

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

        # Extract 'top_n' entries from the sorted indices of the models's prediction probabilities
        max_ddcClasses_indices = prediction_result.argsort()[::-1][:top_n]  # contains 'top_n' position_indices with
        # max. probabilities outputted by the trained models
        max_ddcClasses_probs = prediction_result[max_ddcClasses_indices]
        max_ddcClasses_labels = [self.class_position_hash[index] for index in max_ddcClasses_indices]

        return dict(zip(max_ddcClasses_labels, max_ddcClasses_probs))

    def _return_model_summary(self):
        return self.model.summary()






    def train_model(self, class_imbalance_dict, save_prefix=0):
        if config.train_to_convergence:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_acc',
                min_delta=config.delta_threshold,
                patience=config.patience,
                restore_best_weights=True
            )

        model = self.model
        model.summary()
        logging.info('Loading datasets')
        train, dev = self.get_dataloaders()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss="categorical_crossentropy",
            metrics=config.delta_metric,
        )
        if config.train_to_convergence:
            callbacks = [early_stopping]
        else:
            callbacks = []
        if config.use_custom_learning_schedule:
            callbacks.append(custom_decay.get_scheduler())
        history = model.fit(
            class_weight=class_imbalance_dict,
            x=train,
            validation_data=dev,
            epochs=config.training_epochs,
            use_multiprocessing=True,
            workers=-1,
            callbacks=callbacks
        )
        model.save_weights(f'./src/data/checkpoints/model_index_{save_prefix}')
        return history


# def get_model_index(self):
    #     model_list = glob.glob('../data/checkpoints/model_index_*.data-00000-of-00001')
    #     return len(model_list)
