import numpy as np
import torch.nn as nn
from ml_collections import ConfigDict


def get_model_configs():
    config = ConfigDict()
    
    config.linear = {
        'model_1': {'in_features': 5, 'out_features': 10},
        'model_2': {'in_features': 5, 'out_features': 20}
    }
    
    return config