
import torch
from .Network import Network
from .Network import NetworkSmall

# note: very hacky, change to host on some website in the future 
import os
import sys 


MODEL_CLASS = {'large': Network, 'small': NetworkSmall}
MODEL_FILE = {'large': 'saved_model_cs_ws_ft_humansty.epoch_59.pth',
              'small': 'saved_model_cs_ws_ft_humansty.epoch_55.pth'}
FILE_PATH = {'large': os.path.join(os.path.split(__file__)[0], MODEL_FILE['large']), 
             'small': os.path.join(os.path.split(__file__)[0], MODEL_FILE['small'])}
FILE_HUGGINGFACE_URL = {'large': 'https://huggingface.co/aparekh2/kolossus/resolve/main/saved_model_cs_ws_ft_humansty.epoch_59.pth', 
                        'small': 'https://huggingface.co/aparekh2/kolossus/resolve/main/saved_model_cs_ws_ft_humansty.epoch_55.pth'}




def load_model(model='large'):
    if not os.path.isfile(os.path.join(os.path.split(__file__)[0], MODEL_FILE[model])):
        print(f"Downloading model from {FILE_HUGGINGFACE_URL[model]} to {FILE_PATH[model]}.")
        torch.hub.download_url_to_file(FILE_HUGGINGFACE_URL[model], FILE_PATH[model])

    model_weights = FILE_PATH[model]
    model = MODEL_CLASS[model]()
    model.load_state_dict(torch.load(model_weights))
    return model 
