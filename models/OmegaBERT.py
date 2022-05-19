import glob
from os.path import join
import tensorflow as tf
import transformers
import logging
from models.configs import config
from models.utils import custom_decay
from models.utils.DataLoaders import SidBERTDataloader
from src_utils import settings

# Set a logger
logger = logging.getLogger('OmegaBERT')
logger.setLevel('INFO')

project_root = settings.get_project_root()
data_path = ...


class OmegaBERT:
    def __init__(self, train, test, freeze_bert_layers=False):
        logging.getLogger('Model').setLevel('INFO')
        print(f'{tf.__version__}')
        self.freeze_bert_layers = freeze_bert_layers
        self.model = self.build_model()
        logging.info('model was initialized successfully.')
        self.model_index_id = self.get_model_index()
        self.train_dataset = train
        self.test_dataset = test

    def get_model_index(self):
        model_list = glob.glob('../data/checkpoints/model_index_*.data-00000-of-00001')
        return len(model_list)

    def get_dataloaders(self):
        return SidBERTDataloader(self.train_dataset), SidBERTDataloader(self.test_dataset)

    def build_model(self, restore_model=False):

        """
        Constructs model topology and loads weights from Checkpoint file. Topology needs to be changed
        manually every time a new models version is installed into the backend.
        :return: Tensorflow 2 keras models object containing the models architecture
        :rtype: tensorflow.keras.Model object
        """

        # Construct models topology
        bert_model = transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased')
        input_ids = tf.keras.layers.Input(shape=(config.max_length,), name='input_ids', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(config.max_length,), name='attention_mask', dtype='int32')
        input_type_ids = tf.keras.layers.Input(shape=(config.max_length,), name='token_type_ids', dtype='int32')
        out = bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids)
        if self.freeze_bert_layers:
            bert_model.trainable = False
        sequence_output = out.pooler_output
        glob_leaky = tf.keras.layers.LeakyReLU(alpha=0.15)
        ch1 = tf.keras.layers.Dense(1024,activation=glob_leaky,name='ch1')(sequence_output)
        ch2 = tf.keras.layers.BatchNormalization(name='b_norm_1')(ch1)
        ch3 = tf.keras.layers.Dense(512,activation=glob_leaky,name='ch2')(ch2)
        ch4 = tf.keras.layers.BatchNormalization(name='b_norm_2')(ch3)
        ch5 = tf.keras.layers.Dense(256,activation=glob_leaky,name='ch3')(ch4)
        ch6 = tf.keras.layers.Dense(128, activation=glob_leaky, name='ch4')(ch5)
        output = tf.keras.layers.Dense(config.classes, activation="softmax", name='dense_final')(ch6)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids],
                               outputs=output)
        return model

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