from src.preprocessing import Dataloaders, DatasetProcessing
import config
import glob
import numpy as np
import tensorflow as tf
import transformers
import logging


class TrainingWrapper:
    def __init__(self,train,test):
        logging.getLogger('Model').setLevel('INFO')
        self.model = self.build_model()
        logging.info('model was initialized successfully.')
        self.model_index_id = self.get_model_index()
        self.train_dataset = train
        self.test_dataset = test

    def get_model_index(self):
        model_list = glob.glob('../data/checkpoints/model_index_*.data-00000-of-00001')
        return len(model_list)

    def get_dataloaders(self):
        return Dataloaders.SidBERTDataloader(self.train_dataset), Dataloaders.SidBERTDataloader(self.test_dataset)

    def build_model(self):
        bert_model = transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased')
        input_ids = tf.keras.layers.Input(shape=(config.max_length,), name='input_ids', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(config.max_length,), name='attention_mask', dtype='int32')
        input_type_ids = tf.keras.layers.Input(shape=(config.max_length,), name='token_type_ids', dtype='int32')
        out = bert_model(input_ids, attention_mask=input_masks_ids, token_type_ids=input_type_ids)
        sequence_output = out.pooler_output
        output = tf.keras.layers.Dense(config.classes, activation="softmax", name='dense_final')(sequence_output)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids, input_type_ids],
                               outputs=output)
        return model

    def train_model(self):
        if config.train_to_convergence:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_acc',
                min_delta=config.delta_threshold,
                patience=config.patience,
                restore_best_weights=True
            )

        model = self.model
        model.summary()
        train, dev = self.get_dataloaders()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss="categorical_crossentropy",
            metrics=config.delta_metric,
        )
        if config.train_to_convergence:
            callbacks = [early_stopping]
        else:
            callbacks = None
        history = model.fit(
            x=train,
            validation_data=dev,
            epochs=config.training_epochs,
            use_multiprocessing=True,
            workers=-1,
            callbacks=callbacks
        )
        model.save_weights(f'../src/data/checkpoints/model_index_{self.model_index_id}')
        return history

