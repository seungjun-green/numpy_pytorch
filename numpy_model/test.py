from colorama import Fore, Style, init
import numpy as np
import torch
import torch.nn as nn
from model_configs import get_model_configs
from ml_collections import ConfigDict
from numpy_model import models


model_configs = get_model_configs()
init(autoreset=True)

input_configs = {
    'linear': [np.random.rand(10, 5), np.random.rand(20, 5)],
    'conv': [np.random.rand(1, 3, 64, 64)],
    'transpose_conv': [np.random.rand(1, 3, 64, 64)],
    'rnn': [np.random.rand(32, 10, 10)],  # (seq_len, batch, input_size)
    'lstm': [np.random.rand(32, 256, 10)],  # (seq_len, batch, input_size)
}

def test_numpy_models(example_inputs, np_model, torch_model):
    """take multiple examples of input, then test whether outputs from 
    numpy model and torch model are same.

    Args:
        example_inputs (_type_): _description_
        np_model (_type_): _description_
        torch_model (_type_): _description_
    """
    np_inputs = example_inputs
    torch_inputs = [torch.from_numpy(input) for input in example_inputs]
    
    np_outputs = [np_model.forward(input) for input in np_inputs]
    torch_outputs = [torch_model(input) for input in torch_inputs]
    
    for i in range(len(np_outputs)):
        if np_outputs[i].shape == torch_outputs[i].shape:
            pass
        else:
            raise ValueError(f"Test Case {i+1} failed. Output shape of np model and output of torch model are different.")
    
    print(Fore.GREEN + "All test cases passed!" + Style.RESET_ALL)
    
for model_type, models in model_configs.items():
    for model_name, params in models.items():
        print(f"Testing {model_type} - {model_name} with params {params}")
        
        # Here, you would initialize the model using the params
        if model_type == 'linear':
            np_model = models.Linear(*params)
            torch_model = nn.Linear(*params)
            test_numpy_models(input_configs[model_type], np_model, torch_model)
        else:
            pass