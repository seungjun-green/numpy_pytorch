import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from colorama import Fore, Style, init
import numpy as np
import torch
import torch.nn as nn
from numpy_model.configs import get_model_configs
from ml_collections import ConfigDict
from numpy_model import models
from numpy_model import utils

model_configs = get_model_configs()
init(autoreset=True)


model_constructors = {
    'Linear': (models.Linear, nn.Linear),
    'Conv2d': (models.Conv2d, nn.Conv2d),
    'ConvTranspose2d': (models.ConvTranspose2d, nn.ConvTranspose2d),
    'MaxPool2d': (models.MaxPool2d, nn.MaxPool2d),
    'Embedding': (models.Embedding, nn.Embedding),
    'RNN': (models.RNN, nn.RNN),
    'LSTM': (models.LSTM, nn.LSTM),
    'MultiheadAttention': (models.MultiheadAttention, nn.MultiheadAttention),
}

def test_numpy_model(model_type, model_name, model_params, input_shape):
    """take multiple examples of input, then test whether outputs from  numpy model and torch model are same.
    """
    
    # define models
    np_constructor, torch_constructor = model_constructors[model_type]
    np_model = np_constructor(**model_params)
    torch_model = torch_constructor(**model_params)
    
    # define inputs
    if model_type == 'Embedding':
        np_input = np.random.randint(0, model_params['num_embeddings']-1, input_shape).astype(np.int64)
        torch_input = torch.tensor(np_input, dtype=torch.long)
    else:
        np_input = np.random.randn(*input_shape)
        torch_input = torch.tensor(np_input, dtype=torch.float32)

    # forward process
    np_output = np_model.forward(np_input)
    torch_output = torch_model(torch_input)
    
    flatten_np_output = utils.flatten_outputs(np_output)
    flatten_torch_output = utils.flatten_outputs(torch_output)
    
    for i in range(len(flatten_np_output)):
        assert flatten_np_output[i].shape == flatten_torch_output[i].detach().numpy().shape, f"Output shape mismatch for {model_type}-{model_name}"
        
def test():
    for model_type, test_configs in model_configs.items():
        for model_name, params in test_configs.items():
            model_params = {k: v for k, v in params.items() if k != 'input_shape'}
            test_numpy_model(model_type, model_name, model_params, params['input_shape'])
        
        print(Fore.GREEN + f"All test cases passed for {model_type}." + Style.RESET_ALL)
        
    print(Fore.GREEN + "Congratulations, All test cases are passed!" + Style.RESET_ALL)



if __name__== '__main__':
    test()