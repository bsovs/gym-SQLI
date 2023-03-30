import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoModel

from sqli_sim.envs._helper.error_message import TABLE_ACTION


class ErrorEncoder:
    def __init__(self, state, history_length, model_name='textattack/bert-base-uncased-MNLI'):
        # Make sure torch is working for macos
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print(x)
        else:
            print("MPS device not found.")

        self.state = state
        self.history_length = history_length
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, num_labels=2)
        print(self.model)

    def get_error_encoding(self, error_list):
        errors = error_list[-self.history_length:]
        error_strings = [' '.join(error) for error in errors]
        encoded_errors = self.tokenizer(error_strings, padding=True, truncation=True, return_tensors='pt')
        predictions = self.model(encoded_errors.input_ids, attention_mask=encoded_errors.attention_mask)[0]
        error_encoding = tf.reduce_mean(predictions.detach(), axis=0).numpy()

        # Pad or truncate the encoding to ensure a consistent shape
        if error_encoding.shape[0] < self.state.shape[0]:
            pad_size = self.state.shape[0] - error_encoding.shape[0]
            error_encoding = np.pad(error_encoding, ((0, pad_size), (0, 0)), mode='constant')
        elif error_encoding.shape[0] > self.state.shape[0]:
            error_encoding = error_encoding[:self.state.shape[0], :]

        return error_encoding


if __name__ == '__main__':
    error_encoder = ErrorEncoder(history_length=4, model_name='textattack/bert-base-uncased-MNLI')
    error_list = []
    for error in TABLE_ACTION:
        error_list.append(error)
        encodings = error_encoder.get_error_encoding(error_list)
        print(encodings)
