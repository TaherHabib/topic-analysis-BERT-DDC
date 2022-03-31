import transformers
from tensorflow.keras.utils import Sequence
import numpy as np

class SemanticDataloader(Sequence):
    def __init__(self, dataset, batch_size, max_length):
        self.max_length = max_length
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.dataset = dataset
        self.batch_size = batch_size
        self.preprocess_dataset()
        self.indices = np.arange(len(self.dataset))


    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        local_indeces = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        pass

    def on_epoch_end(self):
        pass

    def preprocess_dataset(self):
        unmasked_tokenized = self.tokenizer(self.dataset['title'].values,                                               add_special_tokens=True,
                                              padding='max_length',
                                              max_length=self.max_length,
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_token_type_ids=True,
                                              pad_to_max_length=True,
                                              return_tensors="tf",)
        masking_target = unmasked_tokenized['input_ids']
        masking_target = (masking_target != 101) & (masking_target != 0) & (masking_target != 103)
        # apply to unmasked tokens with probability values to identify masking targets
        masked_tokens = None
        return unmasked_tokenized, masked_tokens




