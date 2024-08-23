import numpy as np
import torch.nn as nn
from ml_collections import ConfigDict


def get_model_configs():
    config = ConfigDict()
    
    config.Linear = {
        'model_1': {'in_features': 5, 'out_features': 10, 'input_shape': (32, 5)},
        'model_2': {'in_features': 10, 'out_features': 20, 'input_shape': (32, 10)}
    }
    
    config.Conv2d = {
        'model_1': {'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'input_shape': (32, 3, 64, 64)},
        'model_2':  {'in_channels': 3, 'out_channels': 8, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'input_shape': (16, 3, 32, 32)},
        'model_3':  {'in_channels': 1, 'out_channels': 10, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'input_shape': (10, 1, 28, 28)}
    }
    
    config.ConvTranspose2d = {
        'model_1': {'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'input_shape': (32, 3, 64, 64)},
        'model_2':  {'in_channels': 3, 'out_channels': 8, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'input_shape': (16, 3, 32, 32)},
        'model_3':  {'in_channels': 1, 'out_channels': 10, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'input_shape': (10, 1, 28, 28)}
    }
    
    config.MaxPool2d = {
        'model_1': {'kernel_size': 2, 'stride': 2, 'padding': 0, 'input_shape': (32, 3, 64, 64)},
        'model_2': {'kernel_size': 3, 'stride': 2, 'padding': 1, 'input_shape': (16, 3, 32, 32)},
        'model_3': {'kernel_size': 2, 'stride': 2, 'padding': 0, 'input_shape': (10, 1, 28, 28)}
    }
    
    config.Embedding = {
        'model_1': {'num_embeddings': 100, 'embedding_dim': 10, 'input_shape': (32, 20)},
        'model_2': {'num_embeddings': 50, 'embedding_dim': 5, 'input_shape': (32, 15)}
    }
    
    config.RNN = {
        'model_1': {'input_size': 10, 'hidden_size': 20, 'batch_first': True, 'input_shape': (32, 5, 10)},
        'model_2': {'input_size': 20, 'hidden_size': 40, 'batch_first': True, 'input_shape': (32, 10, 20)}
    }
    
    config.LSTM = {
        'model_1': {'input_size': 10, 'hidden_size': 20, 'batch_first': True, 'input_shape': (32, 5, 10)},
        'model_2': {'input_size': 20, 'hidden_size': 40, 'batch_first': True,'input_shape': (32, 10, 20)}
    }
    
    # config.MultiheadAttention = {
    #     'model_1': {'embed_dim': 5, 'num_heads': 1, 'input_shape': (32, 5, 10)},
    #     'model_2': {'embed_dim': 10, 'num_heads': 2, 'input_shape': (32, 10, 20)}
    # }
    
    return config