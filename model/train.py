from model import OmegaBERT
from model.preprocessing import dataset_processing
import logging

def main():
    logging.getLogger('main').setLevel('INFO')
    train, test, weight_dict = DatasetProcessing.DatasetCreator().generate_dataset(even=False, train_test_ratio=0.1)
    bert = OmegaBERT.TrainingWrapper(train=train, test=test, freeze_bert_layers=True)
    history = bert.train_model(weight_dict, save_prefix=0)
    return history